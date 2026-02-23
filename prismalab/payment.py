"""
Интеграция ЮKassa для приёма платежей.
Создание платежа и обработка вебхука.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Гарантируем загрузку .env до чтения переменных
_load_env = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_load_env)
from typing import Any

logger = logging.getLogger("prismalab.payment")

# Выбор платёжной системы: "telegram" или "yookassa"
PAYMENT_PROVIDER = (os.getenv("PAYMENT_PROVIDER") or "telegram").strip().lower()

YOOKASSA_SHOP_ID = (os.getenv("YOOKASSA_SHOP_ID") or "").strip()
YOOKASSA_SECRET_KEY = (os.getenv("YOOKASSA_SECRET_KEY") or "").strip()
PAYMENT_TEST_AMOUNT = int(os.getenv("PAYMENT_TEST_AMOUNT") or "0")  # 0 = реальные цены, 10 = тест по 10 руб
YOOKASSA_RETURN_URL = (os.getenv("YOOKASSA_RETURN_URL") or "https://t.me/your_bot").strip()

# Telegram Payments (инвойс в Telegram, без webhook)
TELEGRAM_PROVIDER_TOKEN = (os.getenv("TELEGRAM_PROVIDER_TOKEN") or "").strip()

# Алерт о платеже — используем модуль alerts
from prismalab.alerts import alert_payment as send_payment_alert


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

# Цены в рублях (для отображения и реальной оплаты если PAYMENT_TEST_AMOUNT=0)
PRICES_FAST = {5: 199, 10: 299, 30: 699}
PRICES_PERSONA_TOPUP = {10: 229, 20: 439, 30: 629}
PRICES_PERSONA_CREATE = {20: 599, 40: 999}


def is_yookassa_configured() -> bool:
    return bool(YOOKASSA_SHOP_ID and YOOKASSA_SECRET_KEY)


def is_telegram_payments_configured() -> bool:
    """Оплата через Telegram Payments (инвойс в боте), без webhook."""
    return bool(TELEGRAM_PROVIDER_TOKEN)


def _amount_rub(product_type: str, credits: int) -> float:
    """Сумма в рублях для создания платежа."""
    if PAYMENT_TEST_AMOUNT > 0:
        return float(PAYMENT_TEST_AMOUNT)
    if product_type == "fast":
        return float(PRICES_FAST.get(credits, 199))
    if product_type == "persona_topup":
        return float(PRICES_PERSONA_TOPUP.get(credits, 229))
    if product_type == "persona_create":
        return float(PRICES_PERSONA_CREATE.get(credits, 599))
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
        return payment.status
    except Exception as e:
        logger.warning("Ошибка получения статуса платежа %s: %s", payment_id, type(e).__name__)
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


def _yookassa_success_content(
    bot: Any, store: Any, user_id: int, product_type: str, credits: int
) -> tuple[str, Any]:
    """Текст и клавиатура после успешной оплаты ЮKassa — как в Telegram Payments."""
    if product_type == "fast":
        from prismalab.bot import (
            _format_balance_express,
            _fast_style_choice_keyboard,
            _generations_count_fast,
            STYLE_EXAMPLES_FOOTER,
        )
        profile = store.get_user(user_id)
        credits_now = _generations_count_fast(profile)
        gender = getattr(profile, "subject_gender", None) or "female"
        text = (
            f"Оплата получена ✅\n\n"
            f"{_format_balance_express(credits_now)}\n\n"
            f"<b>Выберите стиль</b> или введите <b>свой запрос</b> 👇\n\n"
            f"{STYLE_EXAMPLES_FOOTER}"
        )
        kb = _fast_style_choice_keyboard(gender, include_tariffs=True, back_to_ready=True, page=0)
        return text, kb

    if product_type == "persona_topup":
        from prismalab.bot import (
            _format_balance_persona,
            _persona_styles_keyboard,
            STYLE_EXAMPLES_FOOTER,
        )
        profile = store.get_user(user_id)
        new_total = profile.persona_credits_remaining
        gender = getattr(profile, "subject_gender", None) or "female"
        text = (
            f"Оплата получена ✅\n\n"
            f"<b>Выберите стиль</b> 👇\n\n"
            f"{_format_balance_persona(new_total)}\n\n"
            f"{STYLE_EXAMPLES_FOOTER}"
        )
        kb = _persona_styles_keyboard(gender, page=0)
        return text, kb

    if product_type == "persona_create":
        from prismalab.bot import PERSONA_RULES_MESSAGE, _persona_rules_keyboard
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
            try:
                store.log_payment(
                    user_id=user_id,
                    payment_id=payment_id,
                    payment_method="yookassa",
                    product_type=product_type,
                    credits=credits,
                    amount_rub=amount_rub,
                )
            except Exception as e:
                logger.exception("Ошибка записи платежа: %s", e)

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

            if product_type == "persona_pack":
                # Для паков: устанавливаем state загрузки фото и отправляем инструкцию
                pack_id = "0"
                meta = get_payment_metadata(payment_id)
                if meta.get("pack_id"):
                    pack_id = str(meta["pack_id"])
                pack_id_int = int(pack_id) if pack_id.isdigit() else 0

                # БД: fallback когда application.user_data не доходит (Mini App, другой поток)
                try:
                    store.set_pending_pack_upload(user_id=user_id, pack_id=pack_id_int)
                except Exception as e:
                    logger.warning("Не удалось установить pending_pack в БД: %s", e)

                # Устанавливаем user_data через application._user_data (defaultdict)
                # NB: application.user_data — read-only mappingproxy, писать можно только в _user_data
                if application is not None:
                    try:
                        from prismalab.bot import (
                            USERDATA_MODE,
                            USERDATA_PERSONA_SELECTED_PACK_ID,
                            USERDATA_PERSONA_PACK_WAITING_UPLOAD,
                            USERDATA_PERSONA_PACK_PHOTOS,
                            USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS,
                        )
                        ud = application._user_data[user_id]  # defaultdict(dict) — создаст если нет
                        ud[USERDATA_MODE] = "persona_pack_upload"
                        ud[USERDATA_PERSONA_SELECTED_PACK_ID] = pack_id_int
                        ud[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = True
                        ud[USERDATA_PERSONA_PACK_PHOTOS] = []
                        ud[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = []
                        logger.info("Установлен state persona_pack_upload для user %s (pack %s)", user_id, pack_id)
                    except Exception as e:
                        logger.warning("Не удалось установить user_data для пака: %s", e)

                try:
                    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                    kb = InlineKeyboardMarkup([
                        [InlineKeyboardButton("Сбросить фото пака", callback_data="pl_persona_pack_reset_photos")],
                    ])
                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"Оплата получена ✅\n\n"
                            f"<b>Для запуска фотопака нужно 10 фото</b>\n\n"
                            f"Отправьте 10 фото этого человека (можно все сразу или по одной)."
                        ),
                        parse_mode="HTML",
                        reply_markup=kb,
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
            await send_payment_alert(user_id, amount_rub, credits, product_type)

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
        store.log_payment(
            user_id=user_id,
            payment_id=payment_id,
            payment_method="yookassa",
            product_type=product_type,
            credits=credits,
            amount_rub=amount_rub,
        )
    except Exception:
        pass

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
        has_persona = bool(getattr(profile, "astria_lora_tune_id", None))

        if has_persona and chat_id:
            # Есть Персона — запускаем пак сразу, без загрузки фото
            try:
                from prismalab.bot import _run_persona_pack_generation, _find_pack_offer
                offer = _find_pack_offer(pack_id_int)
                if offer:
                    user_data = application._user_data.get(user_id, {}) if application else {}
                    ctx = type("Context", (), {"bot": bot, "user_data": user_data})()
                    coro = _run_persona_pack_generation(
                        context=ctx,
                        chat_id=int(chat_id),
                        user_id=user_id,
                        pack_id=pack_id_int,
                        offer=offer,
                    )
                    if application:
                        application.create_task(coro)
                    else:
                        asyncio.create_task(coro)
                    await bot.send_message(
                        chat_id=int(chat_id),
                        text=f"Оплата получена ✅\n\nЗапускаю пак «{offer['title']}».",
                        parse_mode="HTML",
                    )
                else:
                    has_persona = False  # fallback: попросим фото
            except Exception as e:
                logger.warning("Webhook: не удалось запустить пак напрямую для user %s: %s", user_id, e)
                has_persona = False

        if not has_persona:
            # Нет Персоны — просим загрузить 10 фото
            try:
                store.set_pending_pack_upload(user_id=user_id, pack_id=pack_id_int)
            except Exception as e:
                logger.warning("Webhook: не удалось установить pending_pack в БД: %s", e)

            if application is not None:
                try:
                    from prismalab.bot import (
                        USERDATA_MODE,
                        USERDATA_PERSONA_SELECTED_PACK_ID,
                        USERDATA_PERSONA_PACK_WAITING_UPLOAD,
                        USERDATA_PERSONA_PACK_PHOTOS,
                        USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS,
                    )
                    ud = application._user_data[user_id]
                    ud[USERDATA_MODE] = "persona_pack_upload"
                    ud[USERDATA_PERSONA_SELECTED_PACK_ID] = pack_id_int
                    ud[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = True
                    ud[USERDATA_PERSONA_PACK_PHOTOS] = []
                    ud[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = []
                    logger.info("Webhook: установлен state persona_pack_upload для user %s (pack %s)", user_id, pack_id)
                except Exception as e:
                    logger.warning("Webhook: не удалось установить user_data для пака: %s", e)

            if chat_id:
                try:
                    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                    kb = InlineKeyboardMarkup([
                        [InlineKeyboardButton("Сбросить фото пака", callback_data="pl_persona_pack_reset_photos")],
                    ])
                    await bot.send_message(
                        chat_id=int(chat_id),
                        text=(
                            f"Оплата получена ✅\n\n"
                            f"<b>Для запуска фотопака нужно 10 фото</b>\n\n"
                            f"Отправьте 10 фото этого человека (можно все сразу или по одной)."
                        ),
                        parse_mode="HTML",
                        reply_markup=kb,
                    )
                except Exception as e:
                    logger.warning("Не удалось отправить сообщение о покупке пака: %s", e)

        await send_payment_alert(user_id, amount_rub, credits, product_type)
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
    await send_payment_alert(user_id, amount_rub, credits, product_type)

    return 200, "OK"


def run_webhook_server(bot: Any, store: Any, application: Any = None) -> None:
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

        # ASGI bridge для Starlette-приложений (админка, miniapp)
        from aiohttp_asgi import ASGIResource

        # Админка на том же порту по пути /admin (один вход, один порт)
        try:
            from prismalab.admin.app import create_admin_app
            admin_app = create_admin_app(store)
            # root_path="" — в scope уходит полный path (/admin/), иначе ASGI может резать и Starlette не матчит
            asgi_resource = ASGIResource(admin_app, root_path="")

            async def _admin_handler(request: web.Request) -> web.StreamResponse:
                logger.info("Admin request: %s %s -> handling", request.method, request.path)
                resp = await asgi_resource._handle(request)
                logger.info("Admin response: %s", resp.status)
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

        # Mini App на том же порту по пути /app (прямые aiohttp handlers, без ASGIResource)
        try:
            from prismalab.miniapp import routes as miniapp_routes
            miniapp_routes.set_store(store)

            async def _miniapp_page(_request: web.Request) -> web.Response:
                """Отдаёт HTML Mini App."""
                from pathlib import Path
                html_path = Path(__file__).parent / "miniapp" / "templates" / "app.html"
                html = html_path.read_text(encoding="utf-8")
                return web.Response(text=html, content_type="text/html")

            async def _miniapp_api_auth(request: web.Request) -> web.Response:
                body = await request.json()
                # Прямой вызов API
                init_data = body.get("init_data", "")
                from prismalab.miniapp.auth import validate_init_data
                user = validate_init_data(init_data, miniapp_routes.BOT_TOKEN)
                if not user:
                    return web.json_response({"error": "Invalid init data"}, status=401)
                profile = miniapp_routes.get_store().get_user(user["user_id"])
                user_id = user["user_id"]
                owner_id = miniapp_routes.OWNER_ID
                if not owner_id or user_id != owner_id:
                    return web.json_response({"error": "Forbidden"}, status=403)
                return web.json_response({
                    "user_id": user_id,
                    "first_name": user["first_name"],
                    "credits": {"fast": profile.paid_generations_remaining, "free_used": profile.free_generation_used},
                    "gender": profile.subject_gender,
                    "packs_enabled": True,
                    "has_persona": bool(getattr(profile, "astria_lora_tune_id", None)),
                    "persona_credits": getattr(profile, "persona_credits_remaining", 0) or 0,
                })

            async def _miniapp_api_profile(request: web.Request) -> web.Response:
                """Сохранение пола в профиль (как в основном боте)."""
                try:
                    body = await request.json()
                except Exception:
                    return web.json_response({"error": "Invalid JSON"}, status=400)
                init_data = body.get("init_data", "")
                from prismalab.miniapp.auth import validate_init_data
                user = validate_init_data(init_data, miniapp_routes.BOT_TOKEN)
                if not user:
                    return web.json_response({"error": "Invalid init data"}, status=401)
                if not miniapp_routes.OWNER_ID or user["user_id"] != miniapp_routes.OWNER_ID:
                    return web.json_response({"error": "Forbidden"}, status=403)
                gender = body.get("gender")
                if gender not in ("male", "female"):
                    return web.json_response({"error": "Invalid gender"}, status=400)
                store = miniapp_routes.get_store()
                store.set_subject_gender(user["user_id"], gender)
                return web.json_response({"ok": True, "gender": gender})

            async def _miniapp_api_styles(request: web.Request) -> web.Response:
                user = _miniapp_get_user(request)
                if not user or not miniapp_routes.OWNER_ID or user["user_id"] != miniapp_routes.OWNER_ID:
                    return web.json_response({"error": "Forbidden"}, status=403)
                gender = request.query.get("gender", "female")
                styles = miniapp_routes.FAST_STYLES_FEMALE if gender == "female" else miniapp_routes.FAST_STYLES_MALE
                return web.json_response({"styles": styles, "gender": gender})

            async def _miniapp_api_generate(request: web.Request) -> web.Response:
                reader = await request.multipart()
                init_data = ""
                style_id = ""
                photo_bytes = b""
                async for part in reader:
                    if part.name == "init_data":
                        init_data = (await part.read()).decode()
                    elif part.name == "style_id":
                        style_id = (await part.read()).decode()
                    elif part.name == "photo":
                        photo_bytes = await part.read()

                from prismalab.miniapp.auth import validate_init_data
                user = validate_init_data(init_data, miniapp_routes.BOT_TOKEN)
                if not user:
                    return web.json_response({"error": "Unauthorized"}, status=401)
                if not miniapp_routes.OWNER_ID or user["user_id"] != miniapp_routes.OWNER_ID:
                    return web.json_response({"error": "Forbidden"}, status=403)

                user_id = user["user_id"]
                s = miniapp_routes.get_store()
                profile = s.get_user(user_id)
                has_free = not profile.free_generation_used
                has_paid = profile.paid_generations_remaining > 0
                if not has_free and not has_paid:
                    return web.json_response({"error": "no_credits"}, status=402)
                if len(photo_bytes) > 15 * 1024 * 1024:
                    return web.json_response({"error": "Photo too large"}, status=413)

                import uuid
                task_id = str(uuid.uuid4())[:8]
                miniapp_routes._generation_tasks[task_id] = {"status": "processing", "user_id": user_id, "style_id": style_id, "result_url": None, "error": None}
                asyncio.get_event_loop().create_task(
                    miniapp_routes._run_generation(task_id, user_id, style_id, photo_bytes, has_free, profile)
                )
                return web.json_response({"task_id": task_id, "status": "processing"})

            async def _miniapp_api_status(request: web.Request) -> web.Response:
                user = _miniapp_get_user(request)
                if not user or not miniapp_routes.OWNER_ID or user["user_id"] != miniapp_routes.OWNER_ID:
                    return web.json_response({"error": "Forbidden"}, status=403)
                task_id = request.match_info.get("task_id", "")
                task = miniapp_routes._generation_tasks.get(task_id)
                if not task:
                    return web.json_response({"error": "Task not found"}, status=404)
                if int(task.get("user_id") or 0) != int(user["user_id"]):
                    return web.json_response({"error": "Forbidden"}, status=403)
                response = {"task_id": task_id, "status": task["status"]}
                if task["status"] == "done":
                    response["result_url"] = task["result_url"]
                elif task["status"] == "error":
                    response["error"] = task["error"]
                return web.json_response(response)

            def _miniapp_get_user(request: web.Request):
                """Извлекает user из initData (header или query)."""
                init_data = request.headers.get("X-Telegram-Init-Data", "")
                if not init_data:
                    init_data = request.query.get("init_data", "")
                if not init_data:
                    return None
                from prismalab.miniapp.auth import validate_init_data
                return validate_init_data(init_data, miniapp_routes.BOT_TOKEN)

            async def _miniapp_api_packs(request: web.Request) -> web.Response:
                # Паки только для owner
                user = _miniapp_get_user(request)
                if not user or not miniapp_routes.OWNER_ID or user["user_id"] != miniapp_routes.OWNER_ID:
                    return web.json_response({"packs": []})
                offers = miniapp_routes._load_pack_offers()
                if not offers:
                    return web.json_response({"packs": []})
                result = []
                for offer in offers:
                    pack_data = await miniapp_routes._fetch_pack_data(offer["id"])
                    result.append({
                        "id": offer["id"],
                        "title": offer["title"],
                        "price_rub": offer["price_rub"],
                        "expected_images": offer["expected_images"],
                        "cover_url": pack_data["cover_url"],
                        "category": offer.get("category", "female"),
                    })
                return web.json_response({"packs": result})

            async def _miniapp_api_pack_detail(request: web.Request) -> web.Response:
                user = _miniapp_get_user(request)
                if not user or not miniapp_routes.OWNER_ID or user["user_id"] != miniapp_routes.OWNER_ID:
                    return web.json_response({"error": "Forbidden"}, status=403)
                pack_id_str = request.match_info.get("pack_id", "")
                try:
                    pack_id = int(pack_id_str)
                except (ValueError, TypeError):
                    return web.json_response({"error": "Invalid pack_id"}, status=400)
                offers = miniapp_routes._load_pack_offers()
                offer = next((o for o in offers if o["id"] == pack_id), None)
                if not offer:
                    return web.json_response({"error": "Pack not found"}, status=404)
                pack_data = await miniapp_routes._fetch_pack_data(pack_id)
                return web.json_response({
                    "id": offer["id"],
                    "title": offer["title"],
                    "price_rub": offer["price_rub"],
                    "expected_images": offer["expected_images"],
                    "cover_url": pack_data["cover_url"],
                    "examples": pack_data["examples"],
                })

            async def _miniapp_api_pack_buy(request: web.Request) -> web.Response:
                try:
                    body = await request.json()
                except Exception:
                    return web.json_response({"error": "Invalid JSON"}, status=400)
                init_data = body.get("init_data", "")
                from prismalab.miniapp.auth import validate_init_data
                user = validate_init_data(init_data, miniapp_routes.BOT_TOKEN)
                if not user:
                    return web.json_response({"error": "Unauthorized"}, status=401)
                if not miniapp_routes.OWNER_ID or user["user_id"] != miniapp_routes.OWNER_ID:
                    return web.json_response({"error": "Forbidden"}, status=403)
                pack_id_str = request.match_info.get("pack_id", "")
                try:
                    pack_id = int(pack_id_str)
                except (ValueError, TypeError):
                    return web.json_response({"error": "Invalid pack_id"}, status=400)
                offers = miniapp_routes._load_pack_offers()
                offer = next((o for o in offers if o["id"] == pack_id), None)
                if not offer:
                    return web.json_response({"error": "Pack not found"}, status=404)
                user_id = user["user_id"]
                price_rub = offer["price_rub"]
                amount = apply_test_amount(float(price_rub))
                miniapp_url = miniapp_routes.MINIAPP_URL
                return_url = miniapp_url.rstrip("/") + f"?pack_paid={pack_id}" if miniapp_url else ""
                if not miniapp_url:
                    logger.warning("MINIAPP_URL не задан — после оплаты пака пользователь не вернётся в Mini App")
                url, payment_id_or_err = create_payment(
                    amount_rub=amount,
                    description=f"PrismaLab — {offer['title']}",
                    metadata={
                        "user_id": str(user_id),
                        "chat_id": str(user_id),
                        "product_type": "persona_pack",
                        "credits": str(offer["expected_images"]),
                        "pack_id": str(pack_id),
                    },
                    return_url=return_url,
                )
                if not url:
                    return web.json_response({"error": "Payment creation failed"}, status=500)
                # Запускаем поллинг — bot и application доступны через замыкание run_webhook_server
                asyncio.get_event_loop().create_task(poll_payment_status(
                    payment_id=payment_id_or_err,
                    bot=bot,
                    store=store,
                    user_id=user_id,
                    chat_id=user_id,
                    credits=offer["expected_images"],
                    product_type="persona_pack",
                    amount_rub=amount,
                    application=application,
                ))
                return web.json_response({"payment_url": url, "payment_id": payment_id_or_err})

            # Статика Mini App
            from pathlib import Path
            miniapp_static_path = str(Path(__file__).parent / "miniapp" / "static")
            app.router.add_get("/app", _miniapp_page)
            app.router.add_get("/app/", _miniapp_page)
            app.router.add_post("/app/api/auth", _miniapp_api_auth)
            app.router.add_post("/app/api/profile", _miniapp_api_profile)
            app.router.add_get("/app/api/styles", _miniapp_api_styles)
            app.router.add_post("/app/api/generate", _miniapp_api_generate)
            app.router.add_get("/app/api/status/{task_id}", _miniapp_api_status)
            app.router.add_get("/app/api/packs", _miniapp_api_packs)
            app.router.add_get("/app/api/packs/{pack_id}", _miniapp_api_pack_detail)
            app.router.add_post("/app/api/packs/{pack_id}/buy", _miniapp_api_pack_buy)
            app.router.add_static("/app/static", miniapp_static_path)
            logger.info("Mini App доступен на порту %s (path: /app/)", port)
        except ImportError as e:
            logger.warning("Mini App не подключен: %s", e)
        except Exception as e:
            logger.warning("Ошибка подключения Mini App: %s (%s)", type(e).__name__, e, exc_info=True)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
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
