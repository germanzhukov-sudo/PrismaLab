"""
Интеграция ЮKassa для приёма платежей.
Создание платежа и обработка вебхука.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import logging
import os
import uuid
from pathlib import Path
from urllib.parse import urlencode

from dotenv import load_dotenv

# Гарантируем загрузку .env до чтения переменных
_load_env = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_load_env)
from typing import Any

logger = logging.getLogger("prismalab.payment")

# Astria pack callback
ASTRIA_PACK_CALLBACK_SECRET = (os.getenv("PRISMALAB_ASTRIA_PACK_CALLBACK_SECRET") or "").strip()
# Уже доставленные паки по run_id — защита от дубля callback + fallback + polling
_pack_delivered: set[str] = set()
# Пак сейчас в процессе отправки пользователю (polling/recovery) — чтобы fallback/callback не дублировали финал
_pack_in_progress: set[str] = set()

# Публичные алиасы для межмодульной координации pack-flow.
# Используем ссылку на те же множества (не копии).
pack_delivered_set = _pack_delivered
pack_in_progress_set = _pack_in_progress


def _get_pack_callback_base_url() -> str:
    """Базовый URL для Astria pack callback. Из PRISMALAB_PUBLIC_URL или MINIAPP_URL."""
    url = (os.getenv("PRISMALAB_PUBLIC_URL") or "").strip()
    if url:
        return url.rstrip("/")
    mini = (os.getenv("MINIAPP_URL") or "").strip()
    if mini:
        # https://app.prismalab.ru/app -> https://app.prismalab.ru
        from urllib.parse import urlparse
        parsed = urlparse(mini)
        return f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else ""
    return ""


def _make_pack_callback_token(user_id: int, chat_id: int, pack_id: int, run_id: str) -> str:
    """HMAC-SHA256 подпись для callback URL."""
    if not ASTRIA_PACK_CALLBACK_SECRET:
        return ""
    msg = f"{user_id}:{chat_id}:{pack_id}:{run_id}"
    return hmac.new(
        ASTRIA_PACK_CALLBACK_SECRET.encode(),
        msg.encode(),
        hashlib.sha256,
    ).hexdigest()


def _verify_pack_callback_token(user_id: str, chat_id: str, pack_id: str, run_id: str, token: str) -> bool:
    """Проверка HMAC-подписи callback."""
    if not ASTRIA_PACK_CALLBACK_SECRET:
        return False
    try:
        uid, cid, pid = int(user_id), int(chat_id), int(pack_id)
        expected = _make_pack_callback_token(uid, cid, pid, run_id)
        return hmac.compare_digest(expected, token)
    except (ValueError, TypeError):
        return False


def build_pack_callback_url(user_id: int, chat_id: int, pack_id: int, run_id: str) -> str:
    """Собирает URL для prompts_callback Astria."""
    if not ASTRIA_PACK_CALLBACK_SECRET:
        return ""
    base = _get_pack_callback_base_url()
    if not base:
        return ""
    params = {"user_id": user_id, "chat_id": chat_id, "pack_id": pack_id, "run_id": run_id}
    params["token"] = _make_pack_callback_token(user_id, chat_id, pack_id, run_id)
    return f"{base}/webhooks/astria-pack?{urlencode(params)}"

# Выбор платёжной системы: "telegram" или "yookassa"
PAYMENT_PROVIDER = (os.getenv("PAYMENT_PROVIDER") or "telegram").strip().lower()

YOOKASSA_SHOP_ID = (os.getenv("YOOKASSA_SHOP_ID") or "").strip()
YOOKASSA_SECRET_KEY = (os.getenv("YOOKASSA_SECRET_KEY") or "").strip()
PAYMENT_TEST_AMOUNT = int(os.getenv("PAYMENT_TEST_AMOUNT") or "0")  # 0 = реальные цены, 10 = тест по 10 руб
YOOKASSA_RETURN_URL = (os.getenv("YOOKASSA_RETURN_URL") or "https://t.me/your_bot").strip()

# Telegram Payments (инвойс в Telegram, без webhook)
TELEGRAM_PROVIDER_TOKEN = (os.getenv("TELEGRAM_PROVIDER_TOKEN") or "").strip()

# Алерты
from prismalab.alerts import (
    alert_pack_error,
    alert_payment_error,
)
from prismalab.alerts import (
    alert_payment as send_payment_alert,
)


def use_yookassa() -> bool:
    """Использовать ЮKassa для платежей (по флагу PAYMENT_PROVIDER)."""
    return PAYMENT_PROVIDER == "yookassa" and is_yookassa_configured()


def use_telegram_payments() -> bool:
    """Использовать Telegram Payments (по флагу PAYMENT_PROVIDER)."""
    return PAYMENT_PROVIDER == "telegram" and is_telegram_payments_configured()
# Сумма инвойса в копейках. У Telegram для RUB минимум ~87.73 ₽ (8773 коп), ниже — BadRequest.
# Пока все инвойсы по 88 ₽ (тест). Потом заменить на реальные цены по тарифам.
INVOICE_AMOUNT_KOPECKS = int(os.getenv("INVOICE_AMOUNT_KOPECKS") or "8800")  # 88 руб (выше минимума Telegram)

# Префикс payload для PreCheckout (валидный платёж). Формат: pl:product_type:credits:user_id
INVOICE_PAYLOAD_PREFIX = "pl:"

# Цены продажи — через единый сервис prismalab.tariffs
# Не хардкодить цены здесь! Все PRICES_* перенесены в tariffs.py как _DEFAULT_*.
from prismalab.tariffs import get_price as _tariffs_get_price


def is_yookassa_configured() -> bool:
    return bool(YOOKASSA_SHOP_ID and YOOKASSA_SECRET_KEY)


def is_telegram_payments_configured() -> bool:
    """Оплата через Telegram Payments (инвойс в боте), без webhook."""
    return bool(TELEGRAM_PROVIDER_TOKEN)


def _amount_rub(store, product_type: str, credits: int) -> float:
    """Сумма в рублях для создания платежа. Читает цену из tariffs service."""
    if PAYMENT_TEST_AMOUNT > 0:
        return float(PAYMENT_TEST_AMOUNT)
    price = _tariffs_get_price(store, product_type, credits)
    if price is not None:
        return float(price)
    logger.warning("No price found for %s/%s, fallback 10", product_type, credits)
    return 10.0


def apply_test_amount(amount_rub: float) -> float:
    """
    Применяет глобальную тестовую сумму платежа, если она включена.
    Полезно для продуктов с динамической ценой (например, Astria packs).
    """
    if PAYMENT_TEST_AMOUNT > 0:
        return float(PAYMENT_TEST_AMOUNT)
    try:
        return max(1.0, float(amount_rub))
    except Exception:
        return 1.0


def create_payment(
    *,
    amount_rub: float,
    description: str,
    metadata: dict[str, str],
    return_url: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Создаёт платёж в ЮKassa через SDK.
    Возвращает (confirmation_url, payment_id) или (None, error_message).
    """
    if not is_yookassa_configured():
        return None, "Платёжная система не настроена"
    try:
        from yookassa import Configuration, Payment

        Configuration.configure(YOOKASSA_SHOP_ID, YOOKASSA_SECRET_KEY)
        idempotence_key = str(uuid.uuid4())
        amount_str = f"{amount_rub:.2f}"
        payment = Payment.create(
            {
                "amount": {"value": amount_str, "currency": "RUB"},
                "capture": True,
                "confirmation": {
                    "type": "redirect",
                    "return_url": return_url or YOOKASSA_RETURN_URL,
                },
                "description": description[:128],
                "metadata": {k: str(v)[:200] for k, v in metadata.items()},
            },
            idempotence_key,
        )
        url = None
        if payment.confirmation and hasattr(payment.confirmation, "confirmation_url"):
            url = payment.confirmation.confirmation_url
        elif hasattr(payment, "confirmation") and isinstance(payment.confirmation, dict):
            url = payment.confirmation.get("confirmation_url")
        return url, payment.id
    except Exception as e:
        logger.exception("Ошибка создания платежа ЮKassa: %s", e)
        return None, str(e)


def get_payment_status(payment_id: str) -> str | None:
    """Получить статус платежа по ID. Возвращает 'pending', 'succeeded', 'canceled' или None при ошибке."""
    if not is_yookassa_configured():
        return None
    try:
        from yookassa import Configuration, Payment

        Configuration.configure(YOOKASSA_SHOP_ID, YOOKASSA_SECRET_KEY)
        payment = Payment.find_one(payment_id)
        status = getattr(payment, "status", None)
        if status is None and hasattr(payment, "json"):
            j = getattr(payment, "json", None)
            if isinstance(j, dict):
                status = j.get("status")
        return status
    except Exception as e:
        logger.exception("Ошибка получения статуса платежа %s: %s", payment_id, e)
        return None


def get_payment_metadata(payment_id: str) -> dict:
    """Получить metadata платежа из ЮKassa. Возвращает dict (может быть пустым)."""
    if not is_yookassa_configured():
        return {}
    try:
        from yookassa import Configuration, Payment

        Configuration.configure(YOOKASSA_SHOP_ID, YOOKASSA_SECRET_KEY)
        payment = Payment.find_one(payment_id)
        meta = getattr(payment, "metadata", None)
        return dict(meta) if isinstance(meta, dict) else {}
    except Exception as e:
        logger.warning("Ошибка получения metadata платежа %s: %s", payment_id, type(e).__name__)
        return {}


def _pack_alert_details_from_metadata(metadata: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(metadata, dict):
        return None
    pack_id = str(metadata.get("pack_id") or "").strip()
    pack_title = str(metadata.get("pack_title") or "").strip()
    if not pack_title and pack_id.isdigit():
        # Fallback для старых платежей mini app, где title не писался в metadata.
        try:
            from prismalab.pack_offers import _find_pack_offer
            offer = _find_pack_offer(int(pack_id))
            if offer:
                pack_title = str(offer.get("title") or "").strip()
        except Exception:
            pass
    details = {
        "pack_id": pack_id,
        "pack_title": pack_title,
        "pack_class": str(metadata.get("pack_class") or "").strip(),
        "pack_num_images": str(metadata.get("pack_num_images") or metadata.get("credits") or "").strip(),
        "pack_cost_field": str(metadata.get("pack_cost_field") or "").strip(),
        "pack_cost_value": str(metadata.get("pack_cost_value") or "").strip(),
    }
    if any(details.values()):
        return details
    return None


def _yookassa_success_content(
    bot: Any, store: Any, user_id: int, product_type: str, credits: int
) -> tuple[str, Any]:
    """Текст и клавиатура после успешной оплаты ЮKassa — как в Telegram Payments."""
    if product_type == "fast":
        from prismalab.keyboards import _fast_style_choice_keyboard
        from prismalab.messages import (
            STYLE_EXAMPLES_FOOTER,
            _format_balance_express,
            _generations_count_fast,
        )
        profile = store.get_user(user_id)
        credits_now = _generations_count_fast(profile)
        gender = getattr(profile, "subject_gender", None) or "female"
        text = (
            f"Оплата получена ✅\n\n"
            f"{_format_balance_express(credits_now)}\n\n"
            f"Вернитесь в приложение для генерации или <b>выберите стиль</b> ниже 👇\n\n"
            f"{STYLE_EXAMPLES_FOOTER}"
        )
        kb = _fast_style_choice_keyboard(gender, include_tariffs=True, back_to_ready=True, page=0)
        # Add Mini App button if URL configured
        from prismalab.config import MINIAPP_URL
        if MINIAPP_URL:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
            existing_rows = list(kb.inline_keyboard) if kb else []
            new_rows = [[InlineKeyboardButton("⚡ Открыть Экспресс", web_app=WebAppInfo(url=MINIAPP_URL))]] + existing_rows
            kb = InlineKeyboardMarkup(new_rows)
        return text, kb

    if product_type == "persona_topup":
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo

        from prismalab.config import MINIAPP_URL
        from prismalab.messages import _format_balance_persona
        profile = store.get_user(user_id)
        new_total = profile.persona_credits_remaining
        text = (
            f"<b>Оплата получена</b> ✅\n\n"
            f"Выберите образ в приложении <b>Персона</b> 👇\n\n"
            f"{_format_balance_persona(new_total)}"
        )
        kb_rows = []
        if MINIAPP_URL:
            kb_rows.append([InlineKeyboardButton("✨ Персона", web_app=WebAppInfo(url=MINIAPP_URL))])
        kb_rows.append([InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")])
        kb = InlineKeyboardMarkup(kb_rows)
        return text, kb

    if product_type == "persona_create":
        from prismalab.keyboards import _persona_rules_keyboard
        from prismalab.messages import PERSONA_RULES_MESSAGE
        return PERSONA_RULES_MESSAGE, _persona_rules_keyboard()

    # fallback
    return "✅ Оплата получена!", None


async def poll_payment_status(
    payment_id: str,
    bot: Any,
    store: Any,
    user_id: int,
    chat_id: int,
    credits: int,
    product_type: str,
    amount_rub: float,
    timeout_seconds: int = 600,  # 10 минут
    poll_interval: int = 5,
    application: Any = None,
) -> None:
    """
    Поллинг статуса платежа. Проверяет каждые poll_interval секунд.
    При успехе — начисляет кредиты и отправляет сообщение.
    """
    import asyncio

    start_time = asyncio.get_event_loop().time()

    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout_seconds:
            logger.info("Таймаут поллинга платежа %s", payment_id)
            return

        status = get_payment_status(payment_id)

        if status == "succeeded":
            # Защита от дублирования: проверяем, не обработан ли уже
            if store.is_payment_processed(payment_id):
                logger.info("Платёж %s уже обработан ранее, пропускаем", payment_id)
                return

            logger.info("Платёж %s успешен, начисляем кредиты", payment_id)
            payment_log_id: int | None = None
            try:
                payment_log_id = store.log_payment(
                    user_id=user_id,
                    payment_id=payment_id,
                    payment_method="yookassa",
                    product_type=product_type,
                    credits=credits,
                    amount_rub=amount_rub,
                )
            except Exception as e:
                logger.exception("Ошибка записи платежа: %s", e)
                await asyncio.sleep(poll_interval)
                continue
            if payment_log_id is None:
                logger.info("Платёж %s уже записан параллельным обработчиком, пропускаем", payment_id)
                return

            # Начисляем кредиты
            if product_type == "fast":
                profile = store.get_user(user_id)
                new_total = profile.paid_generations_remaining + credits
                store.set_paid_generations_remaining(user_id, new_total)
            elif product_type == "persona_topup":
                profile = store.get_user(user_id)
                new_total = profile.persona_credits_remaining + credits
                store.set_persona_credits(user_id, new_total)
            elif product_type == "persona_create":
                store.set_persona_credits(user_id, credits)
                store.set_astria_lora_tune(user_id=user_id, tune_id=None)
            elif product_type == "persona_pack":
                # Пак — не начисляем кредиты, отправляем кнопку
                pass

            pack_alert_details: dict[str, str] | None = None
            if product_type == "persona_pack":
                # Для паков: при наличии Персоны — запускаем сразу; иначе просим 10 фото
                pack_id = "0"
                meta = get_payment_metadata(payment_id)
                pack_alert_details = _pack_alert_details_from_metadata(meta)
                if meta.get("pack_id"):
                    pack_id = str(meta["pack_id"])
                pack_id_int = int(pack_id) if pack_id.isdigit() else 0

                profile = store.get_user(user_id)
                has_persona = bool(
                    getattr(profile, "astria_lora_tune_id", None)
                    or getattr(profile, "astria_lora_pack_tune_id", None)
                )

                if has_persona and chat_id:
                    try:
                        from html import escape

                        from prismalab.handlers.packs import _run_persona_pack_generation
                        from prismalab.keyboards import _persona_training_keyboard
                        from prismalab.pack_offers import _find_pack_offer
                        offer = _find_pack_offer(pack_id_int)
                        if offer:
                            msg = await bot.send_message(
                                chat_id=int(chat_id),
                                text=f"Оплата получена ✅\n\nПриступаю к созданию фотосета <b>«{escape(offer['title'])}»</b>",
                                parse_mode="HTML",
                                reply_markup=_persona_training_keyboard(),
                            )
                            user_data = application._user_data[user_id] if application else {}
                            ctx = type("Context", (), {"bot": bot, "user_data": user_data, "application": application})()
                            coro = _run_persona_pack_generation(
                                context=ctx,
                                chat_id=int(chat_id),
                                user_id=user_id,
                                pack_id=pack_id_int,
                                offer=offer,
                                run_id=payment_id,
                                status_message_id=msg.message_id,
                            )
                            if application:
                                application.create_task(coro)
                            else:
                                asyncio.create_task(coro)
                        else:
                            has_persona = False
                    except Exception as e:
                        logger.warning("Poll: не удалось запустить пак напрямую для user %s: %s", user_id, e)
                        asyncio.get_event_loop().create_task(
                            alert_pack_error(
                                user_id,
                                pack_id=pack_id_int if pack_id_int > 0 else None,
                                pack_title=(offer.get("title") if 'offer' in locals() and isinstance(offer, dict) else None),
                                stage="payment_launch",
                                error=str(e),
                            )
                        )
                        has_persona = False

                if not has_persona:
                    # Нет Персоны — запускаем persona-flow после оплаты фотосета
                    try:
                        store.set_pending_pack_upload(user_id=user_id, pack_id=pack_id_int)
                    except Exception as e:
                        logger.warning("Не удалось установить pending_pack в БД: %s", e)

                    if application is not None:
                        try:
                            from prismalab.config import (
                                USERDATA_MODE,
                                USERDATA_PERSONA_PHOTOS,
                                USERDATA_PERSONA_SELECTED_PACK_ID,
                                USERDATA_PERSONA_UPLOAD_MSG_IDS,
                                USERDATA_PERSONA_WAITING_UPLOAD,
                            )
                            ud = application._user_data[user_id]
                            ud[USERDATA_MODE] = "persona"
                            ud[USERDATA_PERSONA_SELECTED_PACK_ID] = pack_id_int
                            ud[USERDATA_PERSONA_WAITING_UPLOAD] = False
                            ud[USERDATA_PERSONA_PHOTOS] = []
                            ud[USERDATA_PERSONA_UPLOAD_MSG_IDS] = []
                            logger.info("Установлен state persona rules для user %s (pack %s)", user_id, pack_id)
                        except Exception as e:
                            logger.warning("Не удалось установить user_data для пака: %s", e)

                    try:
                        from prismalab.keyboards import _persona_rules_keyboard
                        from prismalab.messages import PERSONA_RULES_MESSAGE
                        await bot.send_message(
                            chat_id=chat_id,
                            text=PERSONA_RULES_MESSAGE,
                            parse_mode="HTML",
                            reply_markup=_persona_rules_keyboard(),
                        )
                    except Exception as e:
                        logger.warning("Не удалось отправить сообщение о покупке пака: %s", e)
            else:
                # Текст и клавиатура как в Telegram Payments (bot.handle_successful_payment)
                msg_text, reply_markup = _yookassa_success_content(bot, store, user_id, product_type, credits)

                # Отправляем сообщение
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=msg_text,
                        parse_mode="HTML",
                        reply_markup=reply_markup,
                        disable_web_page_preview=True,
                    )
                except Exception as e:
                    logger.warning("Не удалось отправить сообщение об оплате: %s", e)

            # Алерт админу
            await send_payment_alert(user_id, amount_rub, credits, product_type, pack_details=pack_alert_details)

            return

        elif status == "canceled":
            logger.info("Платёж %s отменён", payment_id)
            try:
                await bot.send_message(chat_id=chat_id, text="❌ Платёж отменён или истёк.", parse_mode="HTML")
            except Exception:
                pass
            return

        # pending — ждём
        await asyncio.sleep(poll_interval)


async def handle_webhook(body: bytes, bot: Any, store: Any, application: Any = None) -> tuple[int, str]:
    """
    Обрабатывает вебхук от ЮKassa.
    Возвращает (http_status, response_body).
    """
    import json

    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        return 400, "Invalid JSON"

    event = data.get("event")
    if event != "payment.succeeded":
        return 200, "OK"

    payment_obj = data.get("object", {})
    payment_id = payment_obj.get("id")
    status = payment_obj.get("status")
    if status != "succeeded":
        return 200, "OK"

    metadata = payment_obj.get("metadata") or {}
    user_id_str = metadata.get("user_id")
    credits_str = metadata.get("credits")
    product_type = metadata.get("product_type", "")

    if not user_id_str or not credits_str:
        logger.warning("Платёж %s без user_id/credits в metadata", payment_id)
        return 200, "OK"

    try:
        user_id = int(user_id_str)
        credits = int(credits_str)
    except ValueError:
        return 200, "OK"

    # Защита от дублирования: проверяем, не обработан ли уже
    if store.is_payment_processed(payment_id):
        logger.info("Webhook: платёж %s уже обработан ранее, пропускаем", payment_id)
        return 200, "OK"

    # Логируем платёж для аналитики
    try:
        amount_rub = float(payment_obj.get("amount", {}).get("value", 0))
        payment_log_id = store.log_payment(
            user_id=user_id,
            payment_id=payment_id,
            payment_method="yookassa",
            product_type=product_type,
            credits=credits,
            amount_rub=amount_rub,
        )
        if payment_log_id is None:
            logger.info("Webhook: платёж %s уже записан, пропускаем дубль", payment_id)
            return 200, "OK"
    except Exception as e:
        logger.exception("Webhook: ошибка записи платежа %s: %s", payment_id, e)
        return 500, "Error"

    if product_type == "fast":
        profile = store.get_user(user_id)
        new_total = profile.paid_generations_remaining + credits
        store.set_paid_generations_remaining(user_id, new_total)
    elif product_type in ("persona_topup", "persona_create"):
        profile = store.get_user(user_id)
        if product_type == "persona_create":
            store.set_persona_credits(user_id, credits)
            store.set_astria_lora_tune(user_id=user_id, tune_id=None)  # пересоздание
        else:
            new_total = profile.persona_credits_remaining + credits
            store.set_persona_credits(user_id, new_total)
    elif product_type == "persona_pack":
        # Оплата фотопака
        pack_id = metadata.get("pack_id", "")
        chat_id = metadata.get("chat_id")
        pack_id_int = int(pack_id) if str(pack_id).isdigit() else 0

        profile = store.get_user(user_id)
        has_persona = bool(
            getattr(profile, "astria_lora_tune_id", None)
            or getattr(profile, "astria_lora_pack_tune_id", None)
        )

        if has_persona and chat_id:
            # Есть Персона — запускаем пак сразу, без загрузки фото
            try:
                from html import escape

                from prismalab.handlers.packs import _run_persona_pack_generation
                from prismalab.keyboards import _persona_training_keyboard
                from prismalab.pack_offers import _find_pack_offer
                offer = _find_pack_offer(pack_id_int)
                if offer:
                    msg = await bot.send_message(
                        chat_id=int(chat_id),
                        text=f"Оплата получена ✅\n\nПриступаю к созданию фотосета <b>«{escape(offer['title'])}»</b>",
                        parse_mode="HTML",
                        reply_markup=_persona_training_keyboard(),
                    )
                    user_data = application._user_data[user_id] if application else {}
                    ctx = type("Context", (), {"bot": bot, "user_data": user_data, "application": application})()
                    coro = _run_persona_pack_generation(
                        context=ctx,
                        chat_id=int(chat_id),
                        user_id=user_id,
                        pack_id=pack_id_int,
                        offer=offer,
                        run_id=payment_id,
                        status_message_id=msg.message_id,
                    )
                    if application:
                        application.create_task(coro)
                    else:
                        asyncio.create_task(coro)
                else:
                    has_persona = False  # fallback: попросим фото
            except Exception as e:
                logger.warning("Webhook: не удалось запустить пак напрямую для user %s: %s", user_id, e)
                asyncio.get_event_loop().create_task(
                    alert_pack_error(
                        user_id,
                        pack_id=pack_id_int if pack_id_int > 0 else None,
                        pack_title=(offer.get("title") if 'offer' in locals() and isinstance(offer, dict) else None),
                        stage="payment_launch",
                        error=str(e),
                    )
                )
                has_persona = False

        if not has_persona:
            # Нет Персоны — запускаем persona-flow после оплаты фотосета
            try:
                store.set_pending_pack_upload(user_id=user_id, pack_id=pack_id_int)
            except Exception as e:
                logger.warning("Webhook: не удалось установить pending_pack в БД: %s", e)

            if application is not None:
                try:
                    from prismalab.config import (
                        USERDATA_MODE,
                        USERDATA_PERSONA_PHOTOS,
                        USERDATA_PERSONA_SELECTED_PACK_ID,
                        USERDATA_PERSONA_UPLOAD_MSG_IDS,
                        USERDATA_PERSONA_WAITING_UPLOAD,
                    )
                    ud = application._user_data[user_id]
                    ud[USERDATA_MODE] = "persona"
                    ud[USERDATA_PERSONA_SELECTED_PACK_ID] = pack_id_int
                    ud[USERDATA_PERSONA_WAITING_UPLOAD] = False
                    ud[USERDATA_PERSONA_PHOTOS] = []
                    ud[USERDATA_PERSONA_UPLOAD_MSG_IDS] = []
                    logger.info("Webhook: установлен state persona rules для user %s (pack %s)", user_id, pack_id)
                except Exception as e:
                    logger.warning("Webhook: не удалось установить user_data для пака: %s", e)

            if chat_id:
                try:
                    from prismalab.keyboards import _persona_rules_keyboard
                    from prismalab.messages import PERSONA_RULES_MESSAGE
                    await bot.send_message(
                        chat_id=int(chat_id),
                        text=PERSONA_RULES_MESSAGE,
                        parse_mode="HTML",
                        reply_markup=_persona_rules_keyboard(),
                    )
                except Exception as e:
                    logger.warning("Не удалось отправить сообщение о покупке пака: %s", e)

        await send_payment_alert(
            user_id,
            amount_rub,
            credits,
            product_type,
            pack_details=_pack_alert_details_from_metadata(metadata),
        )
        return 200, "OK"
    else:
        logger.warning("Неизвестный product_type в платеже %s: %s", payment_id, product_type)
        return 200, "OK"

    msg_text, reply_markup = _yookassa_success_content(bot, store, user_id, product_type, credits)

    chat_id = metadata.get("chat_id")
    if chat_id:
        try:
            chat_id_int = int(chat_id)
            await bot.send_message(
                chat_id=chat_id_int,
                text=msg_text,
                parse_mode="HTML",
                reply_markup=reply_markup,
                disable_web_page_preview=True,
            )
        except Exception as e:
            logger.warning("Не удалось отправить сообщение об оплате: %s", e)

    # Алерт админу
    await send_payment_alert(user_id, amount_rub, credits, product_type, pack_details=None)

    return 200, "OK"


def run_webhook_server(bot: Any, store: Any, application: Any = None, bot_username: str | None = None) -> None:
    """Запускает HTTP-сервер на порту 8080: GET /health для healthcheck Docker; при наличии ЮKassa — POST /payment/webhook; /admin/* — админка."""
    import threading

    async def _serve() -> None:
        from aiohttp import web

        async def health_handler(_request: web.Request) -> web.Response:
            return web.Response(text="ok")

        app = web.Application()
        app.router.add_get("/health", health_handler)
        port = int(os.getenv("YOOKASSA_WEBHOOK_PORT") or "8080")

        if is_yookassa_configured():
            async def webhook_handler(request: web.Request) -> web.Response:
                if request.method != "POST":
                    return web.Response(status=405, text="Method not allowed")
                try:
                    body = await request.read()
                    status, text = await handle_webhook(body, bot, store, application=application)
                    return web.Response(status=status, text=text)
                except Exception as e:
                    logger.exception("Ошибка вебхука: %s", e)
                    return web.Response(status=500, text="Error")

            app.router.add_post("/payment/webhook", webhook_handler)
            logger.info("Вебхук ЮKassa слушает порт %s (path: /payment/webhook)", port)
        else:
            logger.info("Health endpoint слушает порт %s (path: /health)", port)

        # Astria pack prompts_callback — присылает готовые prompts
        if _get_pack_callback_base_url() and ASTRIA_PACK_CALLBACK_SECRET:
            async def astria_pack_webhook_handler(request: web.Request) -> web.Response:
                if request.method != "POST":
                    return web.Response(status=405, text="Method not allowed")
                try:
                    q = request.rel_url.query
                    user_id_s = q.get("user_id", "")
                    chat_id_s = q.get("chat_id", "")
                    pack_id_s = q.get("pack_id", "")
                    run_id_s = q.get("run_id", "")
                    token = q.get("token", "")
                    if not user_id_s or not chat_id_s or not pack_id_s or not run_id_s:
                        return web.Response(status=400, text="Missing user_id/chat_id/pack_id/run_id")
                    if not _verify_pack_callback_token(user_id_s, chat_id_s, pack_id_s, run_id_s, token):
                        logger.warning("Astria pack callback: invalid token")
                        return web.Response(status=403, text="Invalid token")
                    import time as _time
                    logger.info(
                        "Astria pack callback: получен run_id=%s user=%s pack=%s (ts=%.0f)",
                        run_id_s, user_id_s, pack_id_s, _time.time(),
                    )
                    if run_id_s in _pack_delivered:
                        logger.info("Astria pack callback: run_id=%s уже доставлено (polling/fallback)", run_id_s)
                        return web.Response(status=200, text="OK")
                    if run_id_s in _pack_in_progress:
                        logger.info("Astria pack callback: run_id=%s уже в процессе доставки (polling), пропускаем", run_id_s)
                        return web.Response(status=200, text="OK")
                    try:
                        from prismalab.bot import _pack_polling_active
                        if run_id_s in _pack_polling_active:
                            logger.info("Astria pack callback: run_id=%s polling активен, пропускаем callback", run_id_s)
                            return web.Response(status=200, text="OK")
                    except ImportError:
                        pass
                    _pack_in_progress.add(run_id_s)
                    body = await request.read()
                    prompts = json.loads(body.decode("utf-8")) if body else []
                    if not isinstance(prompts, list):
                        return web.Response(status=400, text="Expected JSON array")
                    from prismalab.astria_client import collect_prompt_image_urls
                    from prismalab.telegram_utils import _safe_send_document
                    from prismalab.pack_offers import _find_pack_offer
                    urls = collect_prompt_image_urls(prompts)
                    if not urls:
                        logger.info("Astria pack callback: no images in prompts")
                        return web.Response(status=200, text="OK")
                    user_id = int(user_id_s)
                    chat_id = int(chat_id_s)
                    pack_id = int(pack_id_s)
                    offer = _find_pack_offer(pack_id) or {}
                    pack_title = offer.get("title", "") or str(pack_id)
                    sent_count = 0
                    for i, url in enumerate(urls):
                        try:
                            out_bytes = await asyncio.to_thread(
                                lambda u=url: __import__("requests").get(u, timeout=90).content
                            )
                            if out_bytes:
                                bio = io.BytesIO(out_bytes)
                                bio.name = f"pack_{pack_id}_{sent_count + 1}.png"
                                caption = f"Фотосет «{pack_title}» ({sent_count + 1}/{len(urls)})" if sent_count == 0 else ""
                                await _safe_send_document(bot=bot, chat_id=chat_id, document=bio, caption=caption)
                                sent_count += 1
                                await asyncio.sleep(0.1)
                        except Exception as e:
                            logger.warning("Astria pack callback: download/send failed %s: %s", url[:50], e)
                    if sent_count > 0:
                        _pack_delivered.add(run_id_s)
                        logger.info(
                            "Astria pack callback: доставлено run_id=%s sent=%s (источник: callback)",
                            run_id_s, sent_count,
                        )
                    _pack_in_progress.discard(run_id_s)
                    try:
                        store.log_event(user_id, "pack_callback", {"pack_id": pack_id, "images_sent": sent_count})
                    except Exception:
                        pass
                    from prismalab.keyboards import _photoset_done_keyboard, _photoset_retry_keyboard
                    from prismalab.messages import _photoset_done_message
                    if sent_count > 0:
                        done_text = _photoset_done_message(include_gift=False)
                        await bot.send_message(
                            chat_id=chat_id,
                            text=done_text,
                            reply_markup=_photoset_done_keyboard(),
                            parse_mode="HTML",
                        )
                    else:
                        await bot.send_message(
                            chat_id=chat_id,
                            text="❌ Ошибка генерации фотосета. Попробуйте еще раз.",
                            reply_markup=_photoset_retry_keyboard(pack_id),
                        )
                        asyncio.get_event_loop().create_task(
                            alert_pack_error(
                                user_id,
                                pack_id=pack_id,
                                pack_title=pack_title,
                                stage="callback",
                                error="no images delivered",
                            )
                        )
                    return web.Response(status=200, text="OK")
                except json.JSONDecodeError as e:
                    _pack_in_progress.discard(run_id_s)
                    logger.warning("Astria pack callback: invalid JSON: %s", e)
                    return web.Response(status=400, text="Invalid JSON")
                except Exception as e:
                    _pack_in_progress.discard(run_id_s)
                    logger.exception("Astria pack callback error: %s", e)
                    return web.Response(status=500, text="Error")

            app.router.add_post("/webhooks/astria-pack", astria_pack_webhook_handler)
            logger.info("Astria pack callback слушает /webhooks/astria-pack")
        elif _get_pack_callback_base_url() and not ASTRIA_PACK_CALLBACK_SECRET:
            logger.warning("Astria pack callback отключен: не задан PRISMALAB_ASTRIA_PACK_CALLBACK_SECRET")

        # ASGI bridge для Starlette-приложений (админка, miniapp)
        from aiohttp_asgi import ASGIResource

        # Админка на том же порту по пути /admin (один вход, один порт)
        try:
            from prismalab.admin.app import create_admin_app
            admin_app = create_admin_app(store, bot=bot)
            # root_path="" — в scope уходит полный path (/admin/), иначе ASGI может резать и Starlette не матчит
            asgi_resource = ASGIResource(admin_app, root_path="")

            async def _admin_handler(request: web.Request) -> web.StreamResponse:
                # Fix: aiohttp_asgi/yarl ломается на Host с портом (localhost:8080)
                host = request.headers.get("Host", "")
                if ":" in host:
                    request = request.clone(headers={**request.headers, "Host": host.split(":")[0]})
                resp = await asgi_resource._handle(request)
                return resp

            # Тест: если /admin/ping отдаёт 200 — aiohttp матчит /admin/*, иначе проблема в роутере
            app.router.add_get("/admin/ping", lambda _: web.Response(text="admin ok"))
            app.router.add_route("*", "/admin", _admin_handler)
            app.router.add_route("*", "/admin/", _admin_handler)
            app.router.add_route("*", "/admin/{path:.*}", _admin_handler)
            logger.info("Админка доступна на порту %s (path: /admin/)", port)
        except ImportError as e:
            logger.warning("Админка не подключена (не установлены зависимости): %s", e)
        except Exception as e:
            logger.warning("Ошибка подключения админки: %s (%s)", type(e).__name__, e, exc_info=True)

        # Mini App на том же порту по пути /app (ASGI bridge → Starlette routes.py)
        try:
            from prismalab.miniapp.routes import create_miniapp, set_application, set_bot, set_bot_username, set_store as miniapp_set_store
            miniapp_set_store(store)
            set_bot(bot)
            set_application(application)
            set_bot_username(bot_username)

            miniapp_app = create_miniapp(store=store)
            miniapp_asgi = ASGIResource(miniapp_app, root_path="")

            async def _miniapp_handler(request: web.Request) -> web.StreamResponse:
                # Fix: aiohttp_asgi/yarl ломается на Host с портом (localhost:8080)
                host = request.headers.get("Host", "")
                if ":" in host:
                    request = request.clone(headers={**request.headers, "Host": host.split(":")[0]})
                return await miniapp_asgi._handle(request)

            app.router.add_route("*", "/app", _miniapp_handler)
            app.router.add_route("*", "/app/", _miniapp_handler)
            app.router.add_route("*", "/app/{path:.*}", _miniapp_handler)
            logger.info("Mini App доступен на порту %s (path: /app/, ASGI bridge)", port)
        except ImportError as e:
            logger.warning("Mini App не подключен: %s", e)
        except Exception as e:
            logger.warning("Ошибка подключения Mini App: %s (%s)", type(e).__name__, e, exc_info=True)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()

        # Background warm: прогреваем кеш Astria pack data при старте,
        # чтобы первый /api/auth не ждал 10-20 сек HTTP fan-out к Astria API.
        try:
            from prismalab.miniapp.services.photosets import get_packs_list
            astria_key = os.getenv("PRISMALAB_ASTRIA_API_KEY", "")

            async def _warm_pack_cache():
                try:
                    await get_packs_list(astria_api_key=astria_key, store=store)
                    logger.info("Astria pack cache warmed at startup")
                except Exception as e:
                    logger.warning("Failed to warm pack cache: %s", e)

            if astria_key:
                asyncio.create_task(_warm_pack_cache())
        except ImportError:
            pass

        while True:
            await asyncio.sleep(3600)

    def _thread_target() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_serve())
        finally:
            loop.close()

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()
