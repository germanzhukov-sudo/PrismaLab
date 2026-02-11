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
PRICES_FAST = {5: 169, 10: 309, 30: 690, 50: 990}
PRICES_PERSONA_TOPUP = {5: 269, 10: 499, 20: 899}
PRICES_PERSONA_CREATE = {10: 599, 20: 999}


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
        return float(PRICES_FAST.get(credits, 169))
    if product_type == "persona_topup":
        return float(PRICES_PERSONA_TOPUP.get(credits, 269))
    if product_type == "persona_create":
        return float(PRICES_PERSONA_CREATE.get(credits, 599))
    return 10.0


def create_payment(
    *,
    amount_rub: float,
    description: str,
    metadata: dict[str, str],
    return_url: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Создаёт платёж в ЮKassa.
    Возвращает (confirmation_url, payment_id) или (None, error_message).
    """
    if not is_yookassa_configured():
        return None, "ЮKassa не настроена (YOOKASSA_SHOP_ID, YOOKASSA_SECRET_KEY)"
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
        logger.exception("Ошибка получения статуса платежа %s: %s", payment_id, e)
        return None


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
                msg_text = f"✅ Оплата получена!\n\nДобавлено {credits} кредитов Экспресс-фото."
            elif product_type == "persona_topup":
                profile = store.get_user(user_id)
                new_total = profile.persona_credits_remaining + credits
                store.set_persona_credits(user_id, new_total)
                msg_text = f"✅ Оплата получена!\n\nДобавлено {credits} кредитов Персоны."
            elif product_type == "persona_create":
                store.set_persona_credits(user_id, credits)
                store.set_astria_lora_tune(user_id=user_id, tune_id=None)
                msg_text = f"✅ Оплата получена!\n\n{credits} кредитов Персоны. Загрузите 10 фото для обучения модели."
            else:
                msg_text = "✅ Оплата получена!"

            # Отправляем сообщение
            try:
                await bot.send_message(chat_id=chat_id, text=msg_text, parse_mode="HTML")
            except Exception as e:
                logger.warning("Не удалось отправить сообщение об оплате: %s", e)

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


async def handle_webhook(body: bytes, bot: Any, store: Any) -> tuple[int, str]:
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
        msg_text = f"Оплата получена ✅\n\nДобавлено {credits} кредитов Экспресс-фото. Можете выбирать стиль и загружать фото."
    elif product_type in ("persona_topup", "persona_create"):
        profile = store.get_user(user_id)
        if product_type == "persona_create":
            store.set_persona_credits(user_id, credits)
            store.set_astria_lora_tune(user_id=user_id, tune_id=None)  # пересоздание
            msg_text = f"Оплата получена ✅\n\n{credits} кредитов Персоны. Загрузите 10 фото для обучения модели."
        else:
            new_total = profile.persona_credits_remaining + credits
            store.set_persona_credits(user_id, new_total)
            msg_text = f"Оплата получена ✅\n\nДобавлено {credits} кредитов Персоны."
    else:
        logger.warning("Неизвестный product_type в платеже %s: %s", payment_id, product_type)
        return 200, "OK"

    chat_id = metadata.get("chat_id")
    if chat_id:
        try:
            chat_id_int = int(chat_id)
            await bot.send_message(chat_id=chat_id_int, text=msg_text, parse_mode="HTML")
        except Exception as e:
            logger.warning("Не удалось отправить сообщение об оплате: %s", e)

    return 200, "OK"


def run_webhook_server(bot: Any, store: Any) -> None:
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
                    status, text = await handle_webhook(body, bot, store)
                    return web.Response(status=status, text=text)
                except Exception as e:
                    logger.exception("Ошибка вебхука: %s", e)
                    return web.Response(status=500, text="Error")

            app.router.add_post("/payment/webhook", webhook_handler)
            logger.info("Вебхук ЮKassa слушает порт %s (path: /payment/webhook)", port)
        else:
            logger.info("Health endpoint слушает порт %s (path: /health)", port)

        # Админка на том же порту по пути /admin (один вход, один порт)
        try:
            from prismalab.admin.app import create_admin_app
            from aiohttp_asgi import ASGIResource
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
