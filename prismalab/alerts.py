"""
Система алертов для админа.
Отправляет уведомления через support-бота.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from pathlib import Path

_load_env = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_load_env)

logger = logging.getLogger("prismalab.alerts")

ALERT_ADMIN_ID = int(os.getenv("PRISMALAB_SUPPORT_ADMIN_ID") or "0")
SUPPORT_BOT_TOKEN = (os.getenv("PRISMALAB_SUPPORT_BOT_TOKEN") or "").strip()

# Время запуска бота (для алерта о рестарте)
_bot_start_time: datetime | None = None


async def _send_alert(text: str) -> None:
    """Базовая функция отправки алерта."""
    if not ALERT_ADMIN_ID or not SUPPORT_BOT_TOKEN:
        return
    try:
        from telegram import Bot
        bot = Bot(token=SUPPORT_BOT_TOKEN)
        await bot.send_message(chat_id=ALERT_ADMIN_ID, text=text, parse_mode="HTML")
        logger.info("Алерт отправлен")
    except Exception as e:
        logger.warning("Не удалось отправить алерт: %s", e)


async def alert_payment(user_id: int, amount_rub: float, credits: int, product_type: str) -> None:
    """Алерт о новом платеже."""
    product_names = {
        "fast": "Экспресс",
        "persona_topup": "Персона (пополнение)",
        "persona_create": "Создание Персоны",
    }
    product_name = product_names.get(product_type, product_type)

    text = (
        f"💰 <b>Новый платёж!</b>\n\n"
        f"Сумма: {amount_rub:.0f} ₽\n"
        f"Продукт: {product_name}\n"
        f"Кредиты: {credits}\n"
        f"Юзер: <a href=\"tg://user?id={user_id}\">{user_id}</a>"
    )
    await _send_alert(text)


async def alert_generation_error(user_id: int, error: str, generation_type: str = "express") -> None:
    """Алерт об ошибке генерации."""
    type_name = "Экспресс" if generation_type == "express" else "Персона"

    # Обрезаем ошибку если слишком длинная
    error_short = error[:200] + "..." if len(error) > 200 else error

    text = (
        f"⚠️ <b>Ошибка генерации!</b>\n\n"
        f"Тип: {type_name}\n"
        f"Юзер: <a href=\"tg://user?id={user_id}\">{user_id}</a>\n"
        f"Ошибка: <code>{error_short}</code>"
    )
    await _send_alert(text)


async def alert_slow_generation(user_id: int, duration_seconds: float, generation_type: str = "express") -> None:
    """Алерт о медленной генерации (> 5 минут)."""
    type_name = "Экспресс" if generation_type == "express" else "Персона"
    minutes = duration_seconds / 60

    text = (
        f"🐢 <b>Медленная генерация!</b>\n\n"
        f"Тип: {type_name}\n"
        f"Время: {minutes:.1f} мин\n"
        f"Юзер: <a href=\"tg://user?id={user_id}\">{user_id}</a>"
    )
    await _send_alert(text)


async def alert_bot_started() -> None:
    """Алерт о запуске/перезапуске бота."""
    global _bot_start_time

    now = datetime.now()

    # Если это первый запуск в этой сессии
    if _bot_start_time is None:
        _bot_start_time = now
        text = "🟢 <b>Бот запущен!</b>"
    else:
        # Перезапуск
        text = "🔄 <b>Бот перезапущен!</b>"
        _bot_start_time = now

    await _send_alert(text)


async def alert_daily_report(store) -> None:
    """Дневной отчёт со статистикой."""
    try:
        stats = store.get_dashboard_stats(period="day")

        text = (
            f"📊 <b>Отчёт за сегодня</b>\n\n"
            f"💰 Выручка: {stats.get('revenue', 0):.0f} ₽\n"
            f"🧾 Платежей: {stats.get('payments_count', 0)}\n"
            f"👥 Новых юзеров: {stats.get('new_users', 0)}\n"
            f"🖼 Генераций: {stats.get('generations_count', 0)}"
        )

        # Добавляем маржу если есть
        if stats.get('margin'):
            text += f"\n📈 Маржа: {stats.get('margin', 0):.0f} ₽"

        await _send_alert(text)
    except Exception as e:
        logger.warning("Не удалось отправить дневной отчёт: %s", e)
