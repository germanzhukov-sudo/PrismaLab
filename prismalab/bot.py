#!/usr/bin/env python3
"""
PrismaLab — Telegram-бот стилизации фото (Astria, KIE).

Запуск (из корня репозитория):
  python3 -m prismalab.bot
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import io
import logging
from datetime import timedelta
from pathlib import Path
import os
import secrets
import sys
import threading
import time
from typing import Any

import requests
from PIL import Image, ImageOps
from telegram import BotCommand, BotCommandScopeChat, BotCommandScopeDefault, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, LabeledPrice, Update, WebAppInfo
from telegram.constants import ChatAction
from telegram.error import BadRequest, TimedOut
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PreCheckoutQueryHandler,
    filters,
)

from prismalab.astria_client import (
    AstriaError,
    create_tune_from_pack as astria_create_tune_from_pack,
    download_first_image_bytes as astria_download_first_image_bytes,
    get_pack as astria_get_pack,
    get_tune_prompt_ids as astria_get_tune_prompt_ids,
    run_prompt_and_wait as astria_run_prompt_and_wait,
    wait_pack_images as astria_wait_pack_images,
)
from prismalab.kie_client import (
    KieError,
    download_image_bytes as kie_download_image_bytes,
    run_task_and_wait as kie_run_task_and_wait,
    upload_file_base64 as kie_upload_file_base64,
)
from prismalab.settings import load_settings  # сначала загружаем .env
from prismalab.payment import (
    INVOICE_AMOUNT_KOPECKS,
    INVOICE_PAYLOAD_PREFIX,
    PRICES_PERSONA_TOPUP,
    TELEGRAM_PROVIDER_TOKEN,
    create_payment,
    is_telegram_payments_configured,
    is_yookassa_configured,
    use_yookassa,
    use_telegram_payments,
    poll_payment_status,
    get_payment_status,
    apply_test_amount,
    _amount_rub,
)
from prismalab.persona_prompts import PERSONA_STYLE_PROMPTS
from prismalab.styles import STYLES, get_style
from prismalab.storage import PrismaLabStore
from prismalab.alerts import alert_generation_error, alert_slow_generation, alert_daily_report, alert_payment_error
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("prismalab")
# Важно: httpx на INFO логирует URL Telegram API, где есть токен бота.
logging.getLogger("httpx").setLevel(logging.WARNING)


# Блокировка генерации на уровне user_id (атомарная, вместо USERDATA_JOB_LOCK)
_user_locks: dict[int, asyncio.Lock] = {}
_lock_dict_mutex = threading.Lock()


def _get_user_lock(user_id: int) -> asyncio.Lock:
    with _lock_dict_mutex:
        if user_id not in _user_locks:
            _user_locks[user_id] = asyncio.Lock()
        return _user_locks[user_id]


async def _acquire_user_generation_lock(user_id: int) -> asyncio.Lock | None:
    """Пытается захватить lock. Возвращает lock при успехе, None если уже занят."""
    lock = _get_user_lock(user_id)
    # timeout=0.05: даём шанс acquire выполниться; при занятом lock — TimeoutError
    try:
        await asyncio.wait_for(lock.acquire(), timeout=0.05)
        return lock
    except asyncio.TimeoutError:
        return None


USERDATA_PHOTO_FILE_IDS = "prismalab_photo_file_ids"
USERDATA_ASTRIA_FACEID_FILE_IDS = "prismalab_astria_faceid_file_ids"
USERDATA_ASTRIA_LORA_FILE_IDS = "prismalab_astria_lora_file_ids"
USERDATA_NANO_BANANA_FILE_IDS = "prismalab_nano_banana_file_ids"
USERDATA_MODE = "prismalab_mode"  # normal | fast | persona | persona_pack_upload | astria_faceid | astria_lora
USERDATA_JOB_LOCK = "prismalab_job_lock"  # deprecated: используем _acquire_user_generation_lock
USERDATA_PROMPT_STRENGTH = "prismalab_prompt_strength"
USERDATA_USE_PERSONAL = "prismalab_use_personal"
USERDATA_SUBJECT_GENDER = "prismalab_subject_gender"  # male | female | None
USERDATA_PERSONA_WAITING_UPLOAD = "prismalab_persona_waiting_upload"  # bool, ждём 10 фото
USERDATA_PERSONA_PHOTOS = "prismalab_persona_photos"  # list of file_id для 10 фото Персоны
USERDATA_PERSONA_CREDITS = "prismalab_persona_credits"  # 10 или 20
USERDATA_PERSONA_TRAINING_STATUS = "prismalab_persona_training"  # "training" | "done" | "error"
USERDATA_FAST_SELECTED_STYLE = "prismalab_fast_selected_style"  # style_id когда ждём фото
USERDATA_FAST_CUSTOM_PROMPT = "prismalab_fast_custom_prompt"  # текст промпта при style_id == "custom"
USERDATA_FAST_LAST_MSG_ID = "prismalab_fast_last_msg_id"  # id сообщения "Загрузите фото" для удаления
USERDATA_FAST_STYLE_MSG_ID = "prismalab_fast_style_msg_id"  # id сообщения со стилями (для удаления при смене пола)
USERDATA_FAST_PERSONA_MSG_ID = "prismalab_fast_persona_msg_id"  # id первого сообщения (Персона) в двухсообщенном экране тарифов
USERDATA_FAST_STYLE_PAGE = "prismalab_fast_style_page"  # страница стилей для возврата после генерации
USERDATA_PERSONA_UPLOAD_MSG_IDS = "prismalab_persona_upload_msg_ids"  # id сообщений «Фото X/10» для удаления при 10-м фото
USERDATA_PERSONA_STYLE_MSG_ID = "prismalab_persona_style_msg_id"  # id сообщения со стилями (для удаления при смене пола)
USERDATA_PERSONA_SELECTED_STYLE = "prismalab_persona_selected_style"  # (style_id, label) когда 0 кредитов, ждём докупки
USERDATA_PERSONA_STYLE_PAGE = "prismalab_persona_style_page"  # страница стилей для возврата после генерации
USERDATA_PERSONA_RECREATING = "prismalab_persona_recreating"  # True — в процессе пересоздания, удалять старую только при оплате
USERDATA_PERSONA_PACK_WAITING_UPLOAD = "prismalab_persona_pack_waiting_upload"  # bool, ждём 10 фото для пака
USERDATA_PERSONA_PACK_PHOTOS = "prismalab_persona_pack_photos"  # list of file_id для pak-run (10 фото)
USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS = "prismalab_persona_pack_upload_msg_ids"  # id прогресс-сообщений upload пака
USERDATA_PROFILE_DELETE_JOB = "prismalab_profile_delete_job"  # Job для автоудаления профиля через 10 сек
USERDATA_GETFILEID_EXPECTING_PHOTO = "prismalab_getfileid_expecting_photo"  # owner вызвал /getfileid, ждём фото
USERDATA_EXAMPLES_MEDIA_IDS = "prismalab_examples_media_ids"  # id сообщений текущего альбома (для удаления при навигации)
USERDATA_EXAMPLES_NAV_MSG_ID = "prismalab_examples_nav_msg_id"  # id сообщения с кнопками навигации
USERDATA_EXAMPLES_PAGE = "prismalab_examples_page"  # последняя просмотренная страница альбомов
USERDATA_EXAMPLES_INTRO_MSG_ID = "prismalab_examples_intro_msg_id"  # id сообщения с intro (для удаления при возврате)
USERDATA_PERSONA_SELECTED_PACK_ID = "prismalab_persona_selected_pack_id"  # выбранный pack_id для оплаты

# Единое сообщение об ошибке для пользователя (без технических деталей)
USER_FRIENDLY_ERROR = "Произошла ошибка. Кредит не списали. Попробуйте ещё раз."

# Лимит размера изображения: 15 МБ (Telegram до 20 МБ для фото, 50 МБ для документов)
MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024

OWNER_ID = int(os.getenv("PRISMALAB_OWNER_ID") or "0")

# URL Mini App (задаётся после покупки домена + SSL)
MINIAPP_URL = os.getenv("MINIAPP_URL", "")

# Ограничение доступа для dev-режима: если задан ALLOWED_USERS, только эти юзеры могут пользоваться ботом
_allowed_users_str = os.getenv("ALLOWED_USERS", "")
ALLOWED_USERS: set[int] = set(int(x.strip()) for x in _allowed_users_str.split(",") if x.strip().isdigit()) if _allowed_users_str else set()


def _dev_skip_pack_payment() -> bool:
    """
    Dev-флаг для теста паков без оплаты.
    В проде должен быть выключен.
    """
    raw = (os.getenv("PRISMALAB_DEV_SKIP_PACK_PAYMENT") or "").strip().lower()
    return _is_dev_runtime() and raw in {"1", "true", "yes", "y"}


def _dev_pack_train_from_images() -> bool:
    """
    Dev-флаг: запускать паки через загрузку 10 фото (без tune_ids).
    Нужен как обходной путь, если Astria не отдает prompts в режиме tune_ids.
    """
    raw = (os.getenv("PRISMALAB_DEV_PACKS_TRAIN_FROM_IMAGES") or "").strip().lower()
    return _is_dev_runtime() and raw in {"1", "true", "yes", "y"}


def _is_dev_runtime() -> bool:
    """
    Жестко считаем dev только при TABLE_PREFIX=dev_*
    """
    prefix = (os.getenv("TABLE_PREFIX") or "").strip().lower()
    return prefix.startswith("dev_")


def _guard_dev_only_flags() -> None:
    """
    Fail-fast защита: если dev-флаги включили не в dev-среде — не стартуем.
    Это исключает случайный запуск dev-поведения на проде.
    """
    if _is_dev_runtime():
        return
    bad: list[str] = []
    for key in ("PRISMALAB_DEV_SKIP_PACK_PAYMENT", "PRISMALAB_DEV_PACKS_TRAIN_FROM_IMAGES"):
        raw = (os.getenv(key) or "").strip().lower()
        if raw in {"1", "true", "yes", "y"}:
            bad.append(key)
    if bad:
        raise RuntimeError(
            "Dev-only flags are enabled outside dev runtime: "
            + ", ".join(bad)
            + ". Disable them or set TABLE_PREFIX=dev_."
        )


# Паки, которые всегда в списке (Mini App + бот)
_DEFAULT_PACK_OFFERS: list[dict[str, Any]] = [
    {"id": 248, "title": "Dog Art", "price_rub": 990.0, "expected_images": 16, "class_name": "dog"},
    {"id": 682, "title": "Cat Meowgic", "price_rub": 990.0, "expected_images": 43, "class_name": "cat"},
    {"id": 593, "title": "Kids Halloween", "price_rub": 790.0, "expected_images": 19, "class_name": "boy"},
    {"id": 859, "title": "Kids Holiday", "price_rub": 790.0, "expected_images": 40, "class_name": "girl"},
    {"id": 2152, "title": "Nordic Girl", "price_rub": 790.0, "expected_images": 44, "class_name": "girl"},
    {"id": 2501, "title": "Newborn Dreams", "price_rub": 790.0, "expected_images": 80, "class_name": "girl"},
]


def _pack_offers() -> list[dict[str, Any]]:
    """
    Конфиг паков: env PRISMALAB_ASTRIA_PACK_OFFERS + _DEFAULT_PACK_OFFERS.
    """
    seen_ids: set[int] = set()
    offers: list[dict[str, Any]] = []

    raw = (os.getenv("PRISMALAB_ASTRIA_PACK_OFFERS") or "").strip()
    if raw:
        try:
            items = json.loads(raw)
            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    try:
                        pack_id = int(it.get("id"))
                        title = str(it.get("title") or f"Пак #{pack_id}")
                        price_rub = float(it.get("price_rub"))
                        expected_images = int(it.get("expected_images") or 0)
                        class_name_raw = str(it.get("class_name") or "").strip().lower()
                        class_name = class_name_raw if class_name_raw in {"man", "woman", "boy", "girl", "dog", "cat"} else ""
                        seen_ids.add(pack_id)
                        offers.append({
                            "id": pack_id,
                            "title": title,
                            "price_rub": max(1.0, price_rub),
                            "expected_images": max(0, expected_images),
                            "class_name": class_name,
                        })
                    except Exception:
                        continue
        except Exception:
            logger.warning("PRISMALAB_ASTRIA_PACK_OFFERS: невалидный JSON")

    for p in _DEFAULT_PACK_OFFERS:
        if p["id"] not in seen_ids:
            offers.append(dict(p))
            seen_ids.add(p["id"])

    return offers


def _find_pack_offer(pack_id: int) -> dict[str, Any] | None:
    for offer in _pack_offers():
        if int(offer.get("id") or 0) == int(pack_id):
            return offer
    return None

store = PrismaLabStore()
# Инициализируем таблицы для аналитики (payments, user_events) при старте
store.init_admin_tables()


async def _delete_profile_job_callback(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Job: удалить сообщение профиля через 10 сек."""
    job = context.job
    if not job or not job.data:
        return
    chat_id = job.data.get("chat_id")
    message_id = job.data.get("message_id")
    if chat_id is None or message_id is None:
        return
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception as e:
        logger.warning("Не удалось удалить сообщение профиля: chat_id=%s msg_id=%s err=%s", chat_id, message_id, e)


def _cancel_profile_delete_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отменить запланированное удаление профиля."""
    job = context.user_data.pop(USERDATA_PROFILE_DELETE_JOB, None)
    if job:
        try:
            job.schedule_removal()
        except Exception:
            pass


async def _update_fast_style_message(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, new_msg: Any
) -> None:
    """Удалить предыдущее сообщение со стилями (если есть) и сохранить ID нового."""
    old_id = context.user_data.get(USERDATA_FAST_STYLE_MSG_ID)
    if old_id is not None and old_id != new_msg.message_id:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=old_id)
        except Exception:
            pass
    context.user_data[USERDATA_FAST_STYLE_MSG_ID] = new_msg.message_id


def _schedule_profile_delete(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_id: int, user_id: int
) -> None:
    """Отменить предыдущий job и запланировать удаление профиля через 10 сек."""
    _cancel_profile_delete_job(context)
    job = context.application.job_queue.run_once(
        _delete_profile_job_callback,
        when=timedelta(seconds=10),
        data={"chat_id": chat_id, "message_id": message_id},
        chat_id=user_id,
        user_id=user_id,
        name=f"profile_delete_{user_id}",
    )
    context.user_data[USERDATA_PROFILE_DELETE_JOB] = job


def _round_to_64(x: float) -> int:
    return max(64, int(round(x / 64.0)) * 64)

def _ceil_to_64(x: int) -> int:
    # Всегда округляем ВВЕРХ, чтобы не ужимать и не терять детали лица.
    return max(64, ((max(1, int(x)) + 63) // 64) * 64)


def _prepare_image_for_photomaker(image_bytes: bytes) -> bytes:
    """
    Для identity-preserving моделей лучше НЕ делать центр-кроп (может срезать волосы/контур лица).
    Делаем мягкий resize (без обрезки), максимум по длинной стороне 1024, и сохраняем PNG.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _prepare_image_for_instantid(image_bytes: bytes) -> tuple[bytes, int, int]:
    """
    InstantID / SDXL обычно требуют ширину/высоту кратные 64.
    Делаем resize без кропа + добавляем поля до кратности 64 (без обрезки лица).
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    w, h = img.size
    tw = _ceil_to_64(w)
    th = _ceil_to_64(h)

    # ВАЖНО: ImageOps.pad может КРОПАТЬ, поэтому делаем canvas вручную (только padding).
    canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    x = (tw - w) // 2
    y = (th - h) // 2
    canvas.paste(img, (x, y))

    out = io.BytesIO()
    canvas.save(out, format="PNG", optimize=True)
    return out.getvalue(), tw, th


def _prepare_image_for_instantid_zoom(image_bytes: bytes, *, zoom: float) -> tuple[bytes, int, int]:
    """
    Подстраховка для обычных фото (сжатых Telegram):
    если детектор лица не справляется, пробуем чуть приблизить центр кадра.
    """
    try:
        z = float(zoom)
    except Exception:
        z = 1.0
    if z <= 1.0:
        return _prepare_image_for_instantid(image_bytes)

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)

    w, h = img.size
    cw = max(64, int(round(w / z)))
    ch = max(64, int(round(h / z)))
    left = max(0, (w - cw) // 2)
    top = max(0, (h - ch) // 2)
    cropped = img.crop((left, top, left + cw, top + ch))

    w2, h2 = cropped.size
    tw = _ceil_to_64(w2)
    th = _ceil_to_64(h2)
    canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    x = (tw - w2) // 2
    y = (th - h2) // 2
    canvas.paste(cropped, (x, y))

    out = io.BytesIO()
    canvas.save(out, format="PNG", optimize=True)
    return out.getvalue(), tw, th


def _postprocess_output(style_id: str, out_bytes: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(out_bytes))
        img.load()
    except Exception as e:
        raise ValueError("API вернул не изображение. Попробуй ещё раз.") from e
    if style_id != "noir":
        return out_bytes
    try:
        img = img.convert("L")
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception as e:
        raise ValueError("Не смог обработать результат изображения.") from e

def _guess_aspect_ratio(w: int, h: int) -> str:
    # простая квантизация под типичные aspect_ratio Flux
    r = w / h if h else 1.0
    if r > 1.6:
        return "16:9"
    if r > 1.35:
        return "3:2"
    if r > 1.15:
        return "4:3"
    if r < 0.62:
        return "9:16"
    if r < 0.75:
        return "2:3"
    if r < 0.87:
        return "3:4"
    return "1:1"


def _format_strength(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")

def _subject_prompt_prefix(g: str | None) -> str:
    if g == "male":
        return "photorealistic portrait photo of an adult man, male"
    if g == "female":
        return "photorealistic portrait photo of an adult woman, female"
    return "photorealistic portrait photo of a person"


def _instantid_subject(g: str | None) -> str:
    # Для InstantID лучше не “портретить” промптом, а описывать “человек + сцена/стиль”.
    if g == "male":
        return "a man"
    if g == "female":
        return "a woman"
    return "a person"


def _subject_negative_lock(g: str | None) -> str:
    if g == "male":
        return "female, woman, girl, breasts, cleavage, lipstick, makeup, dress"
    if g == "female":
        return "male, man, boy, beard, mustache"
    return ""

def _is_personal_enabled(context: ContextTypes.DEFAULT_TYPE) -> bool:
    # None => считаем включено (если у пользователя вообще есть персональная модель)
    return context.user_data.get(USERDATA_USE_PERSONAL) is not False


def _personal_label(enabled: bool) -> str:
    return "⭐️ Персональная: Вкл" if enabled else "⭐️ Персональная: Выкл"

async def _safe_get_file_bytes(
    bot: Any,
    file_id: str,
    *,
    max_retries: int = 2,  # Уменьшил до 2 попыток
    timeout: int = 20,  # Уменьшил до 20 секунд
) -> bytes:
    """
    Безопасное скачивание файла из Telegram с обработкой таймаутов и retry.
    ПРОСТАЯ версия без лишних обёрток.
    """
    logger.info(f"[СКАЧИВАНИЕ] Начинаю скачивание файла {file_id[:15]}...")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"[СКАЧИВАНИЕ] Попытка {attempt + 1}/{max_retries}: вызываю get_file...")
            tg_file = await bot.get_file(file_id, read_timeout=timeout, write_timeout=timeout, connect_timeout=timeout)
            
            logger.info(f"[СКАЧИВАНИЕ] get_file OK, вызываю download_as_bytearray...")
            image_bytes = bytes(await tg_file.download_as_bytearray(read_timeout=timeout, write_timeout=timeout, connect_timeout=timeout))
            
            logger.info(f"[СКАЧИВАНИЕ] ✅ Файл скачан! Размер: {len(image_bytes)} байт")
            return image_bytes
            
        except (TimedOut, asyncio.TimeoutError) as e:
            error_type = "TimedOut" if isinstance(e, TimedOut) else "asyncio.TimeoutError"
            logger.warning(f"[СКАЧИВАНИЕ] ❌ {error_type} на попытке {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                wait_time = 3
                logger.info(f"[СКАЧИВАНИЕ] Жду {wait_time}с перед повтором...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"[СКАЧИВАНИЕ] ❌ Таймаут после {max_retries} попыток")
                raise
        except Exception as e:
            logger.warning(f"[СКАЧИВАНИЕ] ❌ Ошибка {type(e).__name__}: {e} на попытке {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                wait_time = 3
                logger.info(f"[СКАЧИВАНИЕ] Жду {wait_time}с перед повтором...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"[СКАЧИВАНИЕ] ❌ Ошибка после {max_retries} попыток: {type(e).__name__}: {e}")
                raise


async def _safe_edit_status(bot: Any, chat_id: int, message_id: int, text: str, **kwargs: Any) -> None:
    """edit_message_text с подавлением 'Message to edit not found' и 'message is not modified'."""
    try:
        await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, **kwargs)
    except BadRequest as e:
        err = str(e).lower()
        if "message to edit not found" in err or "message is not modified" in err:
            logger.debug("edit_status пропущен: %s", e)
        else:
            raise


async def _safe_send_document(
    bot: Any,
    chat_id: int,
    document: io.BytesIO,
    caption: str,
    *,
    max_retries: int = 3,
    timeout: int = 90,
) -> None:
    """
    Безопасная отправка документа с обработкой таймаутов и retry.
    Если отправка документа не удалась, пробует отправить как фото.
    """
    document.seek(0)
    
    for attempt in range(max_retries):
        try:
            await bot.send_document(
                chat_id=chat_id,
                document=document,
                caption=caption,
                read_timeout=timeout,
                write_timeout=timeout,
                connect_timeout=timeout,
            )
            return  # Успешно отправлено
        except TimedOut:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.warning(f"Таймаут при отправке документа (попытка {attempt + 1}/{max_retries}), жду {wait_time}с...")
                await asyncio.sleep(wait_time)
                document.seek(0)
            else:
                logger.warning(f"Таймаут при отправке документа после {max_retries} попыток, пробую отправить как фото...")
                # Fallback: пробуем отправить как фото
                document.seek(0)
                try:
                    await bot.send_photo(
                        chat_id=chat_id,
                        photo=document,
                        caption=caption,
                        read_timeout=timeout,
                        write_timeout=timeout,
                        connect_timeout=timeout,
                    )
                    return
                except Exception as photo_err:
                    logger.error(f"Ошибка при отправке фото (fallback): {photo_err}")
                    raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.warning(f"Ошибка при отправке документа (попытка {attempt + 1}/{max_retries}): {e}, жду {wait_time}с...")
                await asyncio.sleep(wait_time)
                document.seek(0)
            else:
                logger.error(f"Ошибка при отправке документа после {max_retries} попыток: {e}")
                # Последняя попытка: пробуем фото
                document.seek(0)
                try:
                    await bot.send_photo(
                        chat_id=chat_id,
                        photo=document,
                        caption=caption,
                        read_timeout=timeout,
                        write_timeout=timeout,
                        connect_timeout=timeout,
                    )
                    return
                except Exception as photo_err:
                    logger.error(f"Ошибка при отправке фото (fallback): {photo_err}")
                    raise


def _generations_count_fast(profile: Any) -> int:
    """Сколько генераций у юзера в режиме «Быстрое фото» (только бесплатная + платные)."""
    total = (0 if profile.free_generation_used else 1) + profile.paid_generations_remaining
    return total


def _generations_line(profile: Any) -> str:
    """Строка про остаток генераций для экрана /start."""
    if profile.paid_generations_remaining > 0:
        n = profile.paid_generations_remaining
        if n % 10 == 1 and n % 100 != 11:
            word = "генерация"
        elif n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
            word = "генерации"
        else:
            word = "генераций"
        return f"У вас осталось {n} {word}"
    if not profile.free_generation_used:
        return "У вас есть 1 бесплатная генерация"
    return "У вас 0 генераций"


def _fast_generations_line(profile: Any) -> str:
    """Фраза «У вас есть N генерация/генераций» для экрана Быстрое фото."""
    n = _generations_count_fast(profile)
    if n == 0:
        return "У вас 0 генераций"
    if n % 10 == 1 and n % 100 != 11:
        word = "генерация"
    elif n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        word = "генерации"
    else:
        word = "генераций"
    return f"У вас есть {n} {word}"


def _fast_credits_word(n: int) -> str:
    """Склонение: 1 кредит, 2-4 кредита, 5+ кредитов."""
    if n % 10 == 1 and n % 100 != 11:
        return "кредит"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return "кредита"
    return "кредитов"


def _format_balance_express(credits: int) -> str:
    """Единый формат: 💳 Баланс Экспресс-фото: N кредит/кредита/кредитов."""
    w = _fast_credits_word(credits)
    return f"💳 <b>Баланс Экспресс-фото:</b> {credits} {w}"


def _format_balance_persona(credits: int, *, emoji: str = "💳") -> str:
    """Единый формат: [emoji] Баланс Персоны: N кредит/кредита/кредитов. По умолчанию 💳, на главном экране ✨."""
    w = _fast_credits_word(credits)
    return f"{emoji} <b>Баланс Персоны:</b> {credits} {w}"


STYLE_EXAMPLES_FOOTER = 'Примеры фото → <a href="https://t.me/prismalab_styles/8">в канале образов</a>'


def _start_message_text(profile: Any) -> str:
    """Текст экрана главного меню (для /menu и по «Назад»)."""
    fast_credits = _generations_count_fast(profile)
    persona_credits = getattr(profile, "persona_credits_remaining", 0) or 0
    return (
        "Привет! Это бот для ваших крутых фотосессий 😎\n\n"
        "<b>Мы тонко настраиваем нейросеть</b> под ваше лицо, чтобы результат был максимально похож и выглядел дорого\n\n"
        "<b>За этим качеством – сотни тестов и настроек. И мы по-настоящему гордимся результатом</b>\n\n"
        "<b>Есть два способа сделать фото:</b>\n\n"
        "1) ✨ <b>Персона</b> (с вас 10 фото) – наша фирменная фишка: "
        "нейросеть учится на ваших фото и выдаёт кадры уровня «это я, только в кино»\n\n"
        "Реальные фотосессии могут начать казаться бессмысленными 🙂\n\n"
        "2) ⚡️ <b>Экспресс-фото</b> (с вас 1 фото) – как у большинства ботов в Telegram.\n"
        "По одному фото нейросети сложнее точно сохранить лицо, поэтому есть элемент случайности.\n"
        "Зато если исходник удачный – результат может получиться <b>очень сильным:</b> красиво, аккуратно и иногда прямо с первого раза\n\n"
        "<b>Начнём?</b>\n\n"
        "<b>1 кредит = 1 создание фото</b>\n\n"
        f"{_format_balance_express(fast_credits)}\n\n"
        f"{_format_balance_persona(persona_credits, emoji='✨')}"
    )


def _fast_gender_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура выбора пола для Быстрое фото: Женский, Мужской, Назад."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Женский", callback_data="pl_fast_gender:female"),
            InlineKeyboardButton("Мужской", callback_data="pl_fast_gender:male"),
        ],
        [InlineKeyboardButton("Назад", callback_data="pl_fast_back")],
    ])


# --- Персона ---

PERSONA_INTRO_MESSAGE = """<b>Вот где начинается магия</b> ✨

Вы загружаете <b>10 качественных фото</b> – и мы обучаем <b>персональную модель</b> под вашу внешность: черты лица, мимика, нюансы

После обучения вы просто выбираете стиль и получаете кадры уровня <b>«это я, только в кино»</b>: узнаваемо, стабильно, красиво.

<b>Самые частые отзывы:</b>

— «Это я — только смелее и увереннее»
— «Фото из жизни, где всё сложилось»
— «Самая крутая версия меня»

<b>Тарифы:</b>
• <b>Персона + 20 кредитов – 599 ₽</b>
• <b>Персона + 40 кредитов – 999 ₽</b>"""


def _persona_intro_keyboard(user_id: int = 0) -> InlineKeyboardMarkup:
    """Клавиатура вводного экрана Персоны: тарифы и Назад."""
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton("✨ 599 руб – 20 фото", callback_data="pl_persona_buy:20")],
        [InlineKeyboardButton("✨ 999 руб – 40 фото", callback_data="pl_persona_buy:40")],
    ]
    if _pack_offers() and OWNER_ID and user_id == OWNER_ID:
        rows.append([InlineKeyboardButton("🎬 Готовые фотопаки", callback_data="pl_persona_packs")])
    rows.append([InlineKeyboardButton("Назад", callback_data="pl_fast_back")])
    return InlineKeyboardMarkup(rows)


PERSONA_PACKS_MESSAGE = """<b>Готовые фотопаки</b>

Пак = фиксированная цена за серию готовых кадров.
После оплаты запускаем генерацию автоматически и присылаем весь результат в чат."""


def _persona_packs_keyboard() -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for offer in _pack_offers():
        pack_id = int(offer["id"])
        title = str(offer["title"])
        price = float(offer["price_rub"])
        expected_images = int(offer.get("expected_images") or 0)
        if expected_images > 0:
            label = f"{title} — {int(price)} ₽ ({expected_images} фото)"
        else:
            label = f"{title} — {int(price)} ₽"
        rows.append([InlineKeyboardButton(label, callback_data=f"pl_persona_pack_buy:{pack_id}")])
    rows.append([InlineKeyboardButton("Назад", callback_data="pl_persona_back")])
    return InlineKeyboardMarkup(rows)


def _persona_gender_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура выбора пола для Персоны: Женский, Мужской, Назад."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Женский", callback_data="pl_persona_gender:female"),
            InlineKeyboardButton("Мужской", callback_data="pl_persona_gender:male"),
        ],
        [InlineKeyboardButton("Назад", callback_data="pl_fast_back")],
    ])


def _persona_tariff_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура тарифов Персоны (создание): 599/20, 999/40, Назад."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✨ 599 руб – 20 фото", callback_data="pl_persona_buy:20")],
        [InlineKeyboardButton("✨ 999 руб – 40 фото", callback_data="pl_persona_buy:40")],
        [InlineKeyboardButton("Назад", callback_data="pl_persona_back")],
    ])


PERSONA_ERROR_MESSAGE = "Произошла ошибка, кредит не списали. Попробуйте ещё раз"
PERSONA_CREDITS_OUT_MESSAGE = """<b>Кредиты закончились, а Персона осталась</b> ✅

Модель уже обучена, поэтому эти тарифы <b>дешевле</b>

Выберите пакет и продолжим 👇"""


def _persona_credits_out_content(profile: Any) -> tuple[str, InlineKeyboardMarkup]:
    """Текст и клавиатура при 0 кредитах Персоны. Если есть кредиты в Экспрессе — добавляем строку и кнопку."""
    fast_credits = _generations_count_fast(profile)
    if fast_credits > 0:
        text = PERSONA_CREDITS_OUT_MESSAGE + f"\n\nИли перейдите в <b>Экспресс-фото</b>, там ещё есть <b>{fast_credits} {_fast_credits_word(fast_credits)}</b>"
        kb = _persona_credits_out_keyboard(with_express=True, profile=profile)
    else:
        text = PERSONA_CREDITS_OUT_MESSAGE
        kb = _persona_credits_out_keyboard()
    return text, kb


def _persona_credits_out_keyboard(*, with_express: bool = False, profile: Any | None = None) -> InlineKeyboardMarkup:
    """Клавиатура при закончившихся кредитах: тарифы, [Экспресс-фото], Главное меню."""
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton("✨ 10 кредитов – 229 руб", callback_data="pl_persona_topup_buy:10")],
        [InlineKeyboardButton("✨ 20 кредитов – 439 руб", callback_data="pl_persona_topup_buy:20")],
        [InlineKeyboardButton("✨ 30 кредитов – 629 руб", callback_data="pl_persona_topup_buy:30")],
    ]
    if with_express:
        rows.append([InlineKeyboardButton(_express_button_label(profile), callback_data="pl_start_fast")])
    rows.append([InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")])
    return InlineKeyboardMarkup(rows)


PERSONA_RECREATE_CONFIRM_MESSAGE = """⚠️ <b>Текущая модель «Персона» будет удалена</b>

Если вы согласны – нажмите «Продолжить»"""


def _persona_recreate_confirm_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура подтверждения пересоздания: Продолжить, Главное меню."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Продолжить", callback_data="pl_persona_recreate_confirm")],
        [InlineKeyboardButton("Главное меню", callback_data="pl_persona_recreate_cancel")],
    ])


PERSONA_TOPUP_MESSAGE = """Докупить кредиты для Персоны:

• 10 кредитов – 229 ₽
• 20 кредитов – 439 ₽
• 30 кредитов – 629 ₽"""


def _persona_topup_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура докупки кредитов: 10/229, 20/439, 30/629, Назад."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✨ 10 кредитов – 229 руб", callback_data="pl_persona_topup_buy:10")],
        [InlineKeyboardButton("✨ 20 кредитов – 439 руб", callback_data="pl_persona_topup_buy:20")],
        [InlineKeyboardButton("✨ 30 кредитов – 629 руб", callback_data="pl_persona_topup_buy:30")],
        [InlineKeyboardButton("Назад", callback_data="pl_persona_show_credits_out")],
    ])


def _persona_topup_pay_keyboard(credits: int) -> InlineKeyboardMarkup:
    """Кнопка «Оплатить» для докупки кредитов."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Оплатить", callback_data=f"pl_persona_topup_confirm:{credits}", api_kwargs={"style": "success"})],
        [InlineKeyboardButton("Назад", callback_data="pl_persona_show_credits_out")],
    ])


def _persona_pay_confirm_keyboard(credits: int) -> InlineKeyboardMarkup:
    """Кнопка «Оплатить» перед переходом на платёжку."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Оплатить", callback_data=f"pl_persona_confirm_pay:{credits}", api_kwargs={"style": "success"})],
        [InlineKeyboardButton("Назад", callback_data="pl_persona_back")],
    ])


PERSONA_RULES_MESSAGE = """<b>Оплата получена</b> ✅

<b>А теперь самое важное</b>
Мы с вами хотим один результат – чтобы фотосессия получилась реально выдающейся. Поэтому сначала внимательно прочитайте правила загрузки фото

<b>Для качественного обучения модели необходимо:</b>

✅ <b>Количество:</b> 10 фотографий

✅ <b>Качество:</b>
• чёткие, не размытые снимки хорошего качества
• хорошее освещение без пересветов и глубоких теней
• лицо хорошо видно и оно в фокусе

✅ <b>Разнообразие ракурсов:</b>
• анфас (прямо в камеру)
• 3/4 (слегка повернуто)
• профиль

✅ <b>Разнообразие выражений:</b>
• с улыбкой
• серьёзное
• нейтральное

❌ <b>НЕ загружайте:</b>
• групповые фото
• размытые или тёмные снимки
• фото с фильтрами или сильной обработкой
• селфи с искажением от широкоугольной камеры
• фото в очках (максимум 2–3 из всех)
• фото с закрытым лицом (шапки, маски, волосы и т.д.)

💡 <b>Совет:</b> чем качественнее и разнообразнее фото, тем круче и стабильнее получится фотосессия"""


def _persona_rules_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура после правил: Всё понятно, погнали!"""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Всё понятно, погнали!", callback_data="pl_persona_got_it")],
    ])


PERSONA_UPLOAD_WAIT_MESSAGE = """<b>Загружайте фото – жду с нетерпением!</b> 😄

Можно отправить <b>все сразу</b> или <b>по одной</b>"""


PERSONA_PACK_UPLOAD_WAIT_MESSAGE = """<b>Для запуска фотопака нужно 10 фото</b>

Отправьте 10 фото этого человека (можно все сразу или по одной)."""


PERSONA_TRAINING_MESSAGE = """Все 10 фото получил ✅

Отправляю модель на обучение

Можно закрыть чат – я обязательно напишу, когда всё будет готово. Обычно это занимает около 10 минут, иногда чуть дольше."""


def _persona_training_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура после отправки на обучение: Проверить статус Персоны."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Проверить статус Персоны", callback_data="pl_persona_check_status")],
    ])


# Стили Персоны: женские и мужские (label, style_id)
PERSONA_STYLES_FEMALE = [
    ("Свадебный образ", "wedding"),
    ("Барби", "barbie"),
    ("Вечерний Гламур", "evening_glamour"),
    ("Волшебный лес", "magic_forest"),
    ("Дымка и тайна", "smoke_mystery"),
    ("Дымовая завеса", "smoke_veil"),
    ("Неоновый киберпанк", "neon_cyberpunk"),
    ("Городской переулок", "city_alley"),
    ("Утро в постели", "morning_bed"),
    ("Монахиня в клубе", "nun_club"),
    ("Задумчивый арлекин", "thoughtful_arlekin"),
    ("Чёрно-белая интимность", "bw_intimacy"),
    ("Туман и меланхолия", "fog_melancholy"),
    ("Ведьма на Хэллоуин", "halloween_witch"),
    ("Силуэт в дверном проёме", "doorway_silhouette"),
    ("Ночное окно", "night_window_smoke"),
    ("Белый фон", "white_background"),
    ("Мокрое окно", "wet_window"),
    ("Голливудская классика", "hollywood_classic"),
    ("Драматический свет", "dramatic_light"),
    ("Городской нуар", "city_noir"),
    ("Чёрно-белая рефлексия", "bw_reflection"),
    ("Ретро 50-х", "retro_50s"),
    ("Сепия fashion", "sepia_fashion"),
    ("Арт-деко у бассейна", "artdeco_pool"),
    ("Греческая королева", "greek_queen"),
    ("Воздушная фигура", "airy_figure"),
    ("Бальный зал", "ballroom"),
    ("Взгляд в душу", "soul_gaze"),
    ("Студийный дым", "studio_smoke"),
    ("Шёлковая роскошь", "silk_luxury"),
    ("Пиджак и тень", "blazer_shadow"),
    ("Клеопатра", "cleopatra"),
    ("Морской ветер", "sea_breeze"),
    ("Old money", "old_money"),
    ("Лавандовое бьюти", "lavender_beauty"),
    ("Серебряная иллюзия", "silver_illusion"),
    ("Белоснежная чистота", "white_purity"),
    ("Бордовый бархат", "burgundy_velvet"),
    ("Серый кашемир", "grey_cashmere"),
    ("Чёрная сетка", "black_mesh"),
    ("Лавандовый шёлк", "lavender_silk"),
    ("Ванна с лепестками", "bath_petals"),
    ("Дождливое окно", "rainy_window"),
    ("Джазовый бар", "jazz_bar"),
    ("Пикник на пледе", "picnic_blanket"),
    ("Художественная студия", "art_studio"),
    ("Уют зимнего камина", "winter_fireplace"),
]
PERSONA_STYLES_MALE = [
    ("Ночной бар", "night_bar"),
    ("В костюме у окна", "suit_window"),
    ("Прогулка в парке", "park_walk"),
    ("Утренний кофе", "morning_coffee"),
    ("Лесной портрет", "forest_portrait"),
    ("Ночной клуб", "night_club"),
    ("Мастерская художника", "artist_workshop"),
    ("Силуэт на закате", "sunset_silhouette"),
    ("Байкер", "biker"),
    ("Пилот", "pilot"),
    ("Библиотека одиночества", "library_solitude"),
    ("Туманный берег", "foggy_shore"),
    ("Городской спорт", "city_sport"),
    ("Радость на пляже", "beach_joy"),
    ("Силуэт в дверях", "door_silhouette"),
    ("Пианист в баре", "pianist_bar"),
    ("Свечи и бархат", "candles_velvet"),
    ("Дождливый вечер", "rainy_evening"),
    ("Ночная крыша", "night_rooftop"),
    ("Контраст теней", "shadow_contrast"),
    ("Белый фон", "white_background_male"),
    ("Дымная мистика", "smoky_mystery"),
    ("Улицы Нью-Йорка", "nyc_streets"),
    ("На рыбалке", "fishing"),
    ("Стильная лестница", "stylish_stairs"),
]


PERSONA_STYLES_PER_PAGE = 8


def _persona_styles_keyboard(gender: str, page: int = 0, user_id: int = 0) -> InlineKeyboardMarkup:
    """25 стилей для Персоны: по 8 на страницу (8+8+9), 1 кнопка в ряд, навигация стрелками."""
    styles = PERSONA_STYLES_FEMALE if gender == "female" else PERSONA_STYLES_MALE
    total = len(styles)
    total_pages = (total + PERSONA_STYLES_PER_PAGE - 1) // PERSONA_STYLES_PER_PAGE
    page = max(0, min(page, total_pages - 1)) if total_pages else 0

    start = page * PERSONA_STYLES_PER_PAGE
    end = min(start + PERSONA_STYLES_PER_PAGE, total)
    page_styles = styles[start:end]

    rows = [[InlineKeyboardButton(label, callback_data=f"pl_persona_style:{sid}")] for label, sid in page_styles]

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton("← Пред", callback_data=f"pl_persona_page:{page - 1}"))
    nav_buttons.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="pl_persona_page:noop"))
    if page < total_pages - 1:
        nav_buttons.append(InlineKeyboardButton("След →", callback_data=f"pl_persona_page:{page + 1}"))
    if nav_buttons:
        rows.append(nav_buttons)

    if _pack_offers() and OWNER_ID and user_id == OWNER_ID:
        rows.append([InlineKeyboardButton("🎬 Готовые фотопаки", callback_data="pl_persona_packs")])

    return InlineKeyboardMarkup(rows)


def _persona_style_prompt(style_id: str, label: str) -> str:
    """Промпт для стиля Персоны. Использует PERSONA_STYLE_PROMPTS или генерирует из label."""
    if style_id in PERSONA_STYLE_PROMPTS:
        return PERSONA_STYLE_PROMPTS[style_id]
    return (
        "IDENTICAL FACE AND FEATURES from reference photo, same skin tone, ultra high detail face. "
        f"Professional portrait, {label}, natural lighting, sharp focus on face, photorealistic."
    )


# Тексты экрана «Тарифы» при 0 кредитов (2 сообщения: призыв к Персоне → тарифы Экспресс)
FAST_TARIFFS_PERSONA_MESSAGE = """<b>Баланс Экспресс-фото:</b> 0 кредитов

Хотите результат, который пересылают в чаты и ставят на аватар? Выбирайте тариф <b>Персона</b> ✨"""

FAST_TARIFFS_TARIFFS_MESSAGE = """Остаёмся в <b>Экспрессе</b>? Выберите удобный пакет 👇"""


def _payment_yookassa_keyboard(url: str, back_callback: str) -> InlineKeyboardMarkup:
    """Клавиатура экрана оплаты ЮKassa: Оплатить + Назад."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💳 Оплатить", url=url)],
        [InlineKeyboardButton("Назад", callback_data=back_callback)],
    ])


def _fast_tariff_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура «тарифы»: пакеты 5, 10, 30 + Персона + Назад."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚡️ 5 за 199 руб", callback_data="pl_fast_buy:5")],
        [InlineKeyboardButton("⚡️ 10 за 299 руб", callback_data="pl_fast_buy:10")],
        [InlineKeyboardButton("⚡️ 30 за 699 руб", callback_data="pl_fast_buy:30")],
        [InlineKeyboardButton("✨ Персона", callback_data="pl_start_persona")],
        [InlineKeyboardButton("Назад", callback_data="pl_fast_back")],
    ])


def _fast_tariff_persona_only_keyboard() -> InlineKeyboardMarkup:
    """Только кнопка Персона (первое сообщение при 0 кредитах)."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✨ Персона", callback_data="pl_start_persona")],
    ])


def _fast_tariff_packages_keyboard(*, back_callback: str = "pl_fast_back") -> InlineKeyboardMarkup:
    """Пакеты 5, 10, 30 + Назад (без Персоны)."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚡️ 5 за 199 руб", callback_data="pl_fast_buy:5")],
        [InlineKeyboardButton("⚡️ 10 за 299 руб", callback_data="pl_fast_buy:10")],
        [InlineKeyboardButton("⚡️ 30 за 699 руб", callback_data="pl_fast_buy:30")],
        [InlineKeyboardButton("Назад", callback_data=back_callback)],
    ])


async def _send_fast_tariffs_two_messages(
    bot: Any, chat_id: int, context: ContextTypes.DEFAULT_TYPE, *, edit_message: Any = None, back_callback: str = "pl_fast_back"
) -> None:
    """Двухсообщенный экран тарифов: призыв к Персоне + тарифы Экспресс. При выборе Персоны второе удаляется, при Назад — первое."""
    if edit_message:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=edit_message.message_id,
            text=FAST_TARIFFS_PERSONA_MESSAGE,
            reply_markup=_fast_tariff_persona_only_keyboard(),
            parse_mode="HTML",
        )
        persona_msg_id = edit_message.message_id
    else:
        msg1 = await bot.send_message(
            chat_id=chat_id,
            text=FAST_TARIFFS_PERSONA_MESSAGE,
            reply_markup=_fast_tariff_persona_only_keyboard(),
            parse_mode="HTML",
        )
        persona_msg_id = msg1.message_id
    msg = await bot.send_message(
        chat_id=chat_id,
        text=FAST_TARIFFS_TARIFFS_MESSAGE,
        reply_markup=_fast_tariff_packages_keyboard(back_callback=back_callback),
        parse_mode="HTML",
    )
    context.user_data[USERDATA_FAST_PERSONA_MSG_ID] = persona_msg_id
    context.user_data[USERDATA_FAST_STYLE_MSG_ID] = msg.message_id


def _fast_upload_keyboard() -> InlineKeyboardMarkup:
    """Только «Назад» после «Загрузите фото» в Быстрое фото."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Назад", callback_data="pl_fast_back")],
    ])


FAST_CUSTOM_PROMPT_REQUEST_MESSAGE = """✏️ <b>Свой запрос</b>

Напишите текстом описание картинки — это будет <b>запрос для нашей нейросети</b>. Лучше на английском.

Например: <b>Woman in red dress at sunset, beach background, photorealistic</b>

Отправьте <b>одно сообщение</b> с описанием 👇"""

FAST_PHOTO_RULES_MESSAGE = """Жду фото! Но сначала прочитайте правила

• <b>Селфи крупным планом</b> – лицо занимает большую часть кадра
• <b>Смотрите в камеру</b> – без сильных поворотов головы
• <b>Хороший свет</b> – дневной у окна или ровный комнатный, без жёстких теней
• <b>Без фильтров и масок</b> – никаких бьюти-эффектов
• <b>Чёткое фото</b> – без размытости и сильного "шума"
• <b>Без очков и головных уборов</b>

<b>Чем лучше исходники</b> – тем точнее сходство и красивее результат ❤️"""


def _fast_upload_or_change_keyboard() -> InlineKeyboardMarkup:
    """Загрузить фото, Поменять стиль."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Загрузить фото", callback_data="pl_fast_upload_photo"),
            InlineKeyboardButton("Поменять стиль", callback_data="pl_fast_change_style"),
        ],
    ])


def _fast_ready_to_upload_text(credits: int, style_label: str, *, after_payment: bool = False) -> str:
    """Текст экрана «готов к загрузке»: баланс + выбранный стиль."""
    parts = []
    if after_payment:
        parts.append("<b>Оплата получена ✅</b>")
    parts.append(_format_balance_express(credits))
    parts.append(f"Выбранный стиль: <b>{style_label}</b>")
    parts.append("Загрузите фото или поменяйте стиль, если хотите")
    return "\n\n".join(parts)


def _fast_style_screen_text(credits: int, credits_word: str, *, has_photo: bool = False) -> str:
    """Единый текст экрана выбора стиля: с фото (после генерации/возврат) или без."""
    balance = _format_balance_express(credits)
    credit_line = f"{balance}\n\n<b>1 кредит = 1 фото</b>"
    if has_photo:
        base = f"<b>Можете не загружать фото заново</b> – просто выберите другой стиль для этого же снимка\n\nЕсли хотите – <b>загрузите новое</b> 👇\n\n{credit_line}"
    else:
        base = f"{credit_line}\n\n<b>Выберите стиль</b> или введите <b>свой запрос</b> 👇"
    return f"{base}\n\n{STYLE_EXAMPLES_FOOTER}"


def _fast_style_label(style_id: str) -> str:
    """Подпись стиля для Экспресс-фото; для custom возвращает «Свой запрос»."""
    if style_id == "custom":
        return "Свой запрос"
    return next((l for l, s in FAST_STYLES_MALE + FAST_STYLES_FEMALE if s == style_id), style_id)


FAST_STYLES_PER_PAGE = 8


def _fast_style_choice_keyboard(
    gender: str,
    *,
    include_tariffs: bool = True,
    back_to_ready: bool = False,
    from_profile: bool = False,
    page: int = 0,
) -> InlineKeyboardMarkup:
    """Стили по страницам (как в Персоне) + Свой запрос + навигация + Тарифы/Назад."""
    styles = FAST_STYLES_FEMALE if gender == "female" else FAST_STYLES_MALE
    total = len(styles)
    total_pages = max(1, (total + FAST_STYLES_PER_PAGE - 1) // FAST_STYLES_PER_PAGE)
    page = max(0, min(page, total_pages - 1))

    start = page * FAST_STYLES_PER_PAGE
    end = min(start + FAST_STYLES_PER_PAGE, total)
    page_styles = styles[start:end]

    rows = [[InlineKeyboardButton(label, callback_data=f"pl_fast_style:{sid}")] for label, sid in page_styles]
    rows.append([InlineKeyboardButton("✏️ Свой запрос", callback_data="pl_fast_style:custom")])

    # ctx: 0=main(pl_fast_back), 1=back_to_ready(pl_fast_show_ready), 2=from_profile(pl_profile)
    ctx = 2 if from_profile else (1 if back_to_ready else 0)
    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton("← Пред", callback_data=f"pl_fast_page:{page - 1}:{ctx}"))
    nav_buttons.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="pl_fast_page:noop"))
    if page < total_pages - 1:
        nav_buttons.append(InlineKeyboardButton("След →", callback_data=f"pl_fast_page:{page + 1}:{ctx}"))
    if nav_buttons:
        rows.append(nav_buttons)

    if from_profile:
        back_data = "pl_profile"
    else:
        back_data = "pl_fast_show_ready" if back_to_ready else "pl_fast_back"
    if include_tariffs:
        rows.append([
            InlineKeyboardButton("✨ Персона", callback_data="pl_start_persona"),
            InlineKeyboardButton("Назад", callback_data=back_data),
        ])
    else:
        rows.append([InlineKeyboardButton("Назад", callback_data=back_data)])
    return InlineKeyboardMarkup(rows)


# Стили для «Быстрое фото»: мужские и женские (label, style_id)
FAST_STYLES_MALE = [
    ("Ночной бар", "night_bar"),
    ("В костюме у окна", "suit_window"),
    ("Прогулка в парке", "park_walk"),
    ("Утренний кофе", "morning_coffee"),
    ("Лесной портрет", "forest_portrait"),
    ("Ночной клуб", "night_club"),
    ("Мастерская художника", "artist_workshop"),
    ("Силуэт на закате", "sunset_silhouette"),
    ("Байкер", "biker"),
    ("Пилот", "pilot"),
]
FAST_STYLES_FEMALE = [
    ("Свадебный образ", "wedding"),
    ("Мокрое окно", "wet_window"),
    ("Вечерний гламур", "evening_glamour"),
    ("Неоновый киберпанк", "neon_cyberpunk"),
    ("Драматический свет", "dramatic_light"),
    ("Городской нуар", "city_noir"),
    ("Студийный дым", "studio_smoke"),
    ("Чёрно-белая рефлексия", "bw_reflection"),
    ("Бальный зал", "ballroom"),
    ("Греческая королева", "greek_queen"),
    ("Мокрая рубашка", "wet_shirt"),
    ("Клеопатра", "cleopatra"),
    ("Old money", "old_money"),
    ("Лавандовое бьюти", "lavender_beauty"),
    ("Серебряная иллюзия", "silver_illusion"),
    ("Белоснежная чистота", "white_purity"),
    ("Бордовый бархат", "burgundy_velvet"),
    ("Серый кашемир", "grey_cashmere"),
    ("Чёрная сетка", "black_mesh"),
    ("Лавандовый шёлк", "lavender_silk"),
    ("Шёлковое бельё в отеле", "silk_lingerie_hotel"),
    ("Ванна с лепестками", "bath_petals"),
    ("Шампанское на балконе", "champagne_balcony"),
    ("Дождливое окно", "rainy_window"),
    ("Кофе в отеле", "coffee_hotel"),
    ("Джазовый бар", "jazz_bar"),
    ("Пикник на пледе", "picnic_blanket"),
    ("Художественная студия", "art_studio"),
    ("Уют зимнего камина", "winter_fireplace"),
]

# Промпты для Быстрого фото и Персоны — единый файл persona_prompts.py (_persona_style_prompt)


def _fast_styles_keyboard(gender: str) -> InlineKeyboardMarkup:
    """10 стилей для Быстрое фото в зависимости от пола."""
    styles = FAST_STYLES_FEMALE if gender == "female" else FAST_STYLES_MALE
    rows = [[InlineKeyboardButton(label, callback_data=f"pl_fast_style:{sid}")] for label, sid in styles]
    return InlineKeyboardMarkup(rows)


SUPPORT_BOT_USERNAME = "prismalab_support_bot"

HELP_MESSAGE = "Нужна помощь? Напишите в поддержку – ответим как можно скорее"

# Путь к JSON с альбомами примеров (создаётся по /getfileid)
_EXAMPLES_ALBUMS_PATH = Path(__file__).resolve().parent.parent / "examples_albums.json"


def _load_examples_albums() -> list[dict[str, Any]]:
    """Загрузить альбомы из JSON."""
    try:
        if _EXAMPLES_ALBUMS_PATH.exists():
            data = json.loads(_EXAMPLES_ALBUMS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except Exception as e:
        logger.warning("Не удалось загрузить examples_albums.json: %s", e)
    return []


def _save_examples_albums(albums: list[dict[str, Any]]) -> None:
    """Сохранить альбомы в JSON."""
    _EXAMPLES_ALBUMS_PATH.write_text(json.dumps(albums, ensure_ascii=False, indent=2), encoding="utf-8")


def _express_button_label(profile: Any | None) -> str:
    """Подпись кнопки Экспресс-фото: «(1 фото бесплатно)» только если бесплатная генерация не потрачена."""
    if profile and not getattr(profile, "free_generation_used", True):
        return "⚡️ Экспресс-фото (1 фото бесплатно)"
    return "⚡️ Экспресс-фото"


def _start_keyboard(profile: Any | None = None) -> InlineKeyboardMarkup:
    """Клавиатура экрана /start: Mini App (только owner), Быстрое фото, Персона, Тарифы, Примеры, FAQ."""
    rows: list[list[InlineKeyboardButton]] = []
    # Кнопка Mini App Studio — только для owner (боевая аппка пока в тесте)
    user_id = getattr(profile, "user_id", None) if profile else None
    if MINIAPP_URL and MINIAPP_URL.startswith("https://") and OWNER_ID and user_id == OWNER_ID:
        rows.append([InlineKeyboardButton("🎨 Открыть студию", web_app=WebAppInfo(url=MINIAPP_URL))])
    rows.extend([
        [InlineKeyboardButton("✨ Персона", callback_data="pl_start_persona")],
        [InlineKeyboardButton(_express_button_label(profile), callback_data="pl_start_fast")],
        [InlineKeyboardButton("Тарифы", callback_data="pl_start_tariffs")],
        [InlineKeyboardButton("Примеры работ", callback_data="pl_start_examples")],
        [InlineKeyboardButton("А точно ли получится круто?", callback_data="pl_start_faq")],
    ])
    return InlineKeyboardMarkup(rows)


def _profile_text(profile: Any) -> str:
    """Текст экрана Профиль. Пол в конце — динамическая строка с эмодзи."""
    fast_credits = _generations_count_fast(profile)
    persona_credits = getattr(profile, "persona_credits_remaining", 0) or 0
    personas_count = 1 if getattr(profile, "astria_lora_tune_id", None) else 0
    gender = profile.subject_gender or "female"
    gender_label = "Женский 👩" if gender == "female" else "Мужской 👨"
    return (
        f"{_format_balance_express(fast_credits)}\n\n"
        f"{_format_balance_persona(persona_credits)}\n\n"
        f"• <b>Созданных персон</b> – {personas_count}\n\n"
        f"Пол – {gender_label}"
    )


def _profile_keyboard(profile: Any) -> InlineKeyboardMarkup:
    """Клавиатура Профиля: Изменить пол, Экспресс-фото, Персона."""
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton("Изменить пол", callback_data="pl_profile_toggle_gender")],
        [InlineKeyboardButton("✨ Персона", callback_data="pl_start_persona")],
        [InlineKeyboardButton(_express_button_label(profile), callback_data="pl_profile_fast_tariffs")],
    ]
    return InlineKeyboardMarkup(rows)


def _fast_tariff_keyboard_from_profile() -> InlineKeyboardMarkup:
    """Тарифы экспресс-фото при переходе из Профиля: пакеты 5, 10, 30 + Персона + Назад -> Профиль."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚡️ 5 за 199 руб", callback_data="pl_fast_buy:5")],
        [InlineKeyboardButton("⚡️ 10 за 299 руб", callback_data="pl_fast_buy:10")],
        [InlineKeyboardButton("⚡️ 30 за 699 руб", callback_data="pl_fast_buy:30")],
        [InlineKeyboardButton("✨ Персона", callback_data="pl_start_persona")],
        [InlineKeyboardButton("Назад", callback_data="pl_profile")],
    ])


def _get_prompt_strength(settings, context: ContextTypes.DEFAULT_TYPE) -> float:
    v = context.user_data.get(USERDATA_PROMPT_STRENGTH)
    try:
        if v is not None:
            return max(0.1, min(0.95, float(v)))
    except Exception:
        pass
    return float(settings.prompt_strength)


TIPS_MESSAGE = (
    "📋 <b>Шпаргалка</b>\n\n"
    "<b>Деплой на сервер:</b>\n"
    "<code>cd ~/PrismaLab && ./deploy.sh</code>\n\n"
    "<b>Логи:</b>\n"
    "<code>ssh root@194.87.133.7 \"cd /root/PrismaLab && docker compose logs -f\"</code>"
)


async def tips_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /tips: шпаргалка с командами деплоя и логов (только для owner)."""
    user_id = update.effective_user.id if update.effective_user else None
    if OWNER_ID and user_id != OWNER_ID:
        await update.message.reply_text("Команда недоступна.")
        return
    await update.message.reply_text(TIPS_MESSAGE, parse_mode="HTML")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = int(update.effective_user.id) if update.effective_user else 0
    # Dev-режим: только разрешённые юзеры
    if ALLOWED_USERS and user_id not in ALLOWED_USERS:
        return
    _clear_persona_flow_state(context)
    profile = store.get_user(user_id)
    # Логируем событие старта для аналитики
    try:
        store.log_event(user_id, "start")
    except Exception:
        pass
    await update.message.reply_text(
        _start_message_text(profile),
        reply_markup=_start_keyboard(profile),
        parse_mode="HTML",
    )


async def newpersona_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /newpersona: если есть персона — подтверждение пересоздания, иначе — сразу флоу Персоны."""
    user_id = int(update.effective_user.id) if update.effective_user else 0
    _clear_persona_flow_state(context)
    chat_id = update.message.chat_id if update.message else 0
    bot = context.bot
    intro_msg_id = context.user_data.pop(USERDATA_EXAMPLES_INTRO_MSG_ID, None)
    media_ids = context.user_data.pop(USERDATA_EXAMPLES_MEDIA_IDS, None) or []
    nav_msg_id = context.user_data.pop(USERDATA_EXAMPLES_NAV_MSG_ID, None)
    if intro_msg_id is not None or media_ids or nav_msg_id is not None:
        to_delete = []
        if intro_msg_id is not None and intro_msg_id != (update.message.message_id if update.message else None):
            to_delete.append(intro_msg_id)
        to_delete.extend(media_ids)
        if nav_msg_id is not None:
            to_delete.append(nav_msg_id)
        if to_delete:
            await asyncio.gather(*[bot.delete_message(chat_id=chat_id, message_id=mid) for mid in to_delete], return_exceptions=True)
    profile = store.get_user(user_id)
    if not getattr(profile, "astria_lora_tune_id", None):
        context.user_data[USERDATA_MODE] = "persona"
        known_gender = getattr(profile, "subject_gender", None) or context.user_data.get(USERDATA_SUBJECT_GENDER)
        if known_gender not in ("male", "female"):
            await update.message.reply_text(
                "Выбери пол – так нейросеть точнее настроит результат",
                reply_markup=_persona_gender_keyboard(),
            )
        else:
            context.user_data[USERDATA_SUBJECT_GENDER] = known_gender
            await update.message.reply_text(
                PERSONA_INTRO_MESSAGE,
                reply_markup=_persona_intro_keyboard(),
                parse_mode="HTML",
            )
        return
    await update.message.reply_text(
        PERSONA_RECREATE_CONFIRM_MESSAGE,
        reply_markup=_persona_recreate_confirm_keyboard(),
        parse_mode="HTML",
    )


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /menu — то же, что /start (главное меню)."""
    await start_command(update, context)


async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /profile — экран Профиль."""
    user_id = int(update.effective_user.id) if update.effective_user else 0
    profile = store.get_user(user_id)
    msg = await update.message.reply_text(
        _profile_text(profile),
        reply_markup=_profile_keyboard(profile),
        parse_mode="HTML",
    )
    _schedule_profile_delete(context, msg.chat_id, msg.message_id, user_id)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /help — Помощь и ссылка на бот поддержки."""
    support_url = f"https://t.me/{SUPPORT_BOT_USERNAME}"
    await update.message.reply_text(
        HELP_MESSAGE,
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("Написать в поддержку", url=support_url)],
            [InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")],
        ]),
    )


async def getfileid_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /getfileid: ответь на фото и получи file_id. С аргументом — сохранить в альбом.
    Пример: /getfileid Экспресс-фото
    Только для owner."""
    user_id = update.effective_user.id if update.effective_user else None
    if OWNER_ID and user_id != OWNER_ID:
        await update.message.reply_text("Команда недоступна.")
        return
    reply = update.message.reply_to_message
    caption = (context.args or [])
    caption_str = " ".join(caption).strip() if caption else ""
    if not reply:
        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Анастасия", callback_data="pl_getfileid_album:Анастасия"),
                InlineKeyboardButton("Мария", callback_data="pl_getfileid_album:Мария"),
            ],
            [
                InlineKeyboardButton("Наталья", callback_data="pl_getfileid_album:Наталья"),
                InlineKeyboardButton("Иван", callback_data="pl_getfileid_album:Иван"),
            ],
        ])
        await update.message.reply_text(
            "1. Выбери альбом\n2. Отправь фото",
            reply_markup=kb,
        )
        return
    fid: str | None = None
    if reply.photo:
        fid = reply.photo[-1].file_id
    elif reply.document and (reply.document.mime_type or "").startswith("image/"):
        fid = reply.document.file_id
    if not fid:
        await update.message.reply_text("Ответь на фото (или изображение как файл)")
        return
    if caption_str:
        albums = _load_examples_albums()
        found = next((a for a in albums if (a.get("caption") or "").strip() == caption_str), None)
        if found:
            ids_list = found.setdefault("file_ids", [])
            if len(ids_list) < 10:
                ids_list.append(fid)
                _save_examples_albums(albums)
                await update.message.reply_text(f"Добавлено в альбом «{caption_str}» ({len(ids_list)}/10)")
            else:
                await update.message.reply_text(f"В альбоме «{caption_str}» уже 10 фото — максимум")
        else:
            albums.append({"caption": caption_str, "file_ids": [fid]})
            _save_examples_albums(albums)
            await update.message.reply_text(f"Создан альбом «{caption_str}» и добавлено фото")
    else:
        await update.message.reply_text(f"<code>{fid}</code>", parse_mode="HTML")


async def handle_getfileid_album_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка выбора альбома для /getfileid."""
    query = update.callback_query
    if not query or not query.data:
        return
    user_id = query.from_user.id if query.from_user else 0
    if OWNER_ID and user_id != OWNER_ID:
        await query.answer("Недоступно")
        return
    await query.answer()
    _, album_name = query.data.split(":", 1)
    context.user_data[USERDATA_GETFILEID_EXPECTING_PHOTO] = album_name
    await query.edit_message_text(f"Теперь отправь фото — добавлю в «{album_name}»")


def _fast_after_gender_content(profile: "UserProfile", gender: str | None = None, *, has_photo: bool = False) -> tuple[str | None, InlineKeyboardMarkup | None]:
    """Текст и клавиатура после выбора пола. При 0 кредитов — (None, None), вызывающий делает двухсообщенный экран."""
    g = gender or getattr(profile, "subject_gender", None) or "female"
    credits = _generations_count_fast(profile)
    credits_word = _fast_credits_word(credits)
    has_generations = credits > 0
    if has_generations:
        text = _fast_style_screen_text(credits, credits_word, has_photo=has_photo)
        return text, _fast_style_choice_keyboard(g, include_tariffs=True)
    return None, None


async def handle_start_fast_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Быстрое фото»: если пол уже известен — скип выбора пола, иначе выбор пола → загрузка фото или тарифы."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    known_gender = getattr(profile, "subject_gender", None) or context.user_data.get(USERDATA_SUBJECT_GENDER)
    if known_gender in ("male", "female"):
        context.user_data[USERDATA_SUBJECT_GENDER] = known_gender
        context.user_data[USERDATA_MODE] = "fast"
        has_photo = bool(context.user_data.get(USERDATA_PHOTO_FILE_IDS))
        text, reply_markup = _fast_after_gender_content(profile, gender=known_gender, has_photo=has_photo)
        if text is not None:
            extra = {"parse_mode": "HTML", "disable_web_page_preview": True}
            await query.edit_message_text(text, reply_markup=reply_markup, **extra)
            context.user_data[USERDATA_FAST_STYLE_MSG_ID] = query.message.message_id
        else:
            await _send_fast_tariffs_two_messages(context.bot, query.message.chat_id, context, edit_message=query.message)
        return
    await query.edit_message_text(
        "Выбери пол – так нейросеть точнее настроит результат",
        reply_markup=_fast_gender_keyboard(),
    )


async def handle_start_persona_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «✨ Персона»: если модель уже есть — сразу стили, иначе вводный экран."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    _cancel_profile_delete_job(context)
    # Удалить второе сообщение («Остаёмся в Экспрессе?» + тарифы), если пришли из двухсообщенного экрана после генерации
    tariffs_msg_id = context.user_data.pop(USERDATA_FAST_STYLE_MSG_ID, None)
    if tariffs_msg_id is not None and tariffs_msg_id != query.message.message_id:
        try:
            await context.bot.delete_message(chat_id=query.message.chat_id, message_id=tariffs_msg_id)
        except Exception:
            pass
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    context.user_data[USERDATA_MODE] = "persona"
    logger.info("handle_start_persona: user_id=%s astria_lora_tune_id=%s", user_id, getattr(profile, "astria_lora_tune_id", None))

    if getattr(profile, "astria_lora_tune_id", None):
        credits = profile.persona_credits_remaining
        if credits <= 0:
            text, kb = _persona_credits_out_content(profile)
            await query.edit_message_text(text, reply_markup=kb, parse_mode="HTML")
            return
        gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
        await query.edit_message_text(
            f"<b>Выберите стиль из списка ниже</b> 👇\n\n{_format_balance_persona(credits)}\n\n{STYLE_EXAMPLES_FOOTER}",
            reply_markup=_persona_styles_keyboard(gender, page=0),
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        context.user_data[USERDATA_PERSONA_STYLE_MSG_ID] = query.message.message_id
        context.user_data[USERDATA_PERSONA_STYLE_PAGE] = 0
        return

    # Персоны нет, но кредиты есть — значит оплатил, но фото не загрузил (или обучение в процессе)
    credits = getattr(profile, "persona_credits_remaining", 0) or 0
    pending = getattr(profile, "astria_lora_tune_id_pending", None)
    if credits > 0:
        if pending:
            # Обучение в процессе (бот мог рестартовать) — показать «Проверить статус»
            context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "training"
            await query.edit_message_text(
                PERSONA_TRAINING_MESSAGE,
                reply_markup=_persona_training_keyboard(),
                parse_mode="HTML",
            )
            return
        # Показать экран загрузки фото (как после оплаты)
        await query.edit_message_text(
            PERSONA_RULES_MESSAGE,
            reply_markup=_persona_rules_keyboard(),
            parse_mode="HTML",
        )
        return

    # Персоны нет и кредитов нет: если пол неизвестен — выбор пола, иначе — intro с тарифами
    known_gender = getattr(profile, "subject_gender", None) or context.user_data.get(USERDATA_SUBJECT_GENDER)
    if known_gender not in ("male", "female"):
        await query.edit_message_text(
            "Выбери пол – так нейросеть точнее настроит результат",
            reply_markup=_persona_gender_keyboard(),
        )
        return
    context.user_data[USERDATA_SUBJECT_GENDER] = known_gender
    await query.edit_message_text(
        PERSONA_INTRO_MESSAGE,
        reply_markup=_persona_intro_keyboard(),
        parse_mode="HTML",
    )


def _examples_intro_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура под intro: кнопка «Смотреть примеры»."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Смотреть примеры", callback_data="pl_examples_show_albums")],
    ])


def _examples_nav_keyboard(page: int, total: int) -> InlineKeyboardMarkup:
    """Клавиатура навигации по альбомам примеров."""
    channel_url = (os.getenv("PRISMALAB_EXAMPLES_CHANNEL_URL") or "https://t.me/prismalab_styles/8").strip()
    rows: list[list[InlineKeyboardButton]] = []
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("←", callback_data=f"pl_examples_page:{page - 1}"))
    if page < total - 1:
        nav.append(InlineKeyboardButton("→", callback_data=f"pl_examples_page:{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton("Канал с образами", url=channel_url)])
    rows.append([InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")])
    return InlineKeyboardMarkup(rows)


async def _show_examples_page(
    bot: Any,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    page: int,
    *,
    delete_previous: bool = False,
    nav_msg_id_to_delete: int | None = None,
) -> None:
    """Показать альбом примеров на странице page (0-based)."""
    albums = [a for a in _load_examples_albums() if (a.get("file_ids") or [])]
    total = len(albums)
    if total == 0:
        return
    page = max(0, min(page, total - 1))
    album = albums[page]
    caption = (album.get("caption") or "")[:1024]
    file_ids = (album.get("file_ids") or [])[:10]
    if not file_ids:
        return
    if delete_previous:
        to_delete = list(context.user_data.pop(USERDATA_EXAMPLES_MEDIA_IDS, []) or [])
        if nav_msg_id_to_delete is not None:
            to_delete.append(nav_msg_id_to_delete)
        await asyncio.gather(*[
            bot.delete_message(chat_id=chat_id, message_id=mid) for mid in to_delete
        ], return_exceptions=True)
    media = [InputMediaPhoto(media=fid, caption=caption if i == 0 else None) for i, fid in enumerate(file_ids)]
    sent = await bot.send_media_group(chat_id=chat_id, media=media)
    context.user_data[USERDATA_EXAMPLES_MEDIA_IDS] = [m.message_id for m in sent]
    nav_text = f"{page + 1} альбом из {total}"
    kb = _examples_nav_keyboard(page, total)
    nav_msg = await bot.send_message(chat_id=chat_id, text=nav_text, reply_markup=kb)
    context.user_data[USERDATA_EXAMPLES_NAV_MSG_ID] = nav_msg.message_id
    context.user_data[USERDATA_EXAMPLES_PAGE] = page


async def handle_start_examples_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Примеры работ»: текст + альбомы + навигация. Сохраняет страницу при уходе, восстанавливает при возврате."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    chat_id = query.message.chat_id if query.message else 0
    bot = context.bot

    # При возврате удаляем старые intro и media, чтобы не дублировать
    intro_msg_id = context.user_data.pop(USERDATA_EXAMPLES_INTRO_MSG_ID, None)
    media_ids = context.user_data.pop(USERDATA_EXAMPLES_MEDIA_IDS, None) or []
    to_delete = []
    if intro_msg_id is not None and intro_msg_id != query.message.message_id:
        to_delete.append(intro_msg_id)
    to_delete.extend(media_ids)
    if to_delete:
        await asyncio.gather(*[bot.delete_message(chat_id=chat_id, message_id=mid) for mid in to_delete], return_exceptions=True)

    intro = (
        "<b>Примеры персональных фотосессий</b>\n\n"
        "Ниже – результаты работы <b>Персоны</b> ✨\n"
        "Каждый пример – <b>реальный человек</b>, его исходные фото и результат после обучения персональной модели\n"
        "Все карусели размещены <b>с разрешения людей</b>, которые сами поделились результатами\n\n"
        "Мы не делаем абстрактную «красивую картинку»\n"
        "Мы усиливаем внешний вид: на снимках – вы, просто <b>спокойнее, смелее и свободнее</b>\n\n"
        "Если вы тоже решите поделиться результатами, мы <b>с радостью подарим приятный бонус</b> 🤍\n\n"
        "<b>Листайте, присматривайтесь и представляйте себя. Всё остальное сделаем мы</b>"
    )
    albums = [a for a in _load_examples_albums() if (a.get("file_ids") or [])]
    if not albums:
        user_id = int(query.from_user.id) if query.from_user else 0
        profile = store.get_user(user_id)
        empty_kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✨ Персона", callback_data="pl_start_persona"),
                InlineKeyboardButton(_express_button_label(profile), callback_data="pl_start_fast"),
            ],
            [InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")],
        ])
        await query.edit_message_text(
            intro + "\n\n<b>Хотите попробовать?</b>",
            reply_markup=empty_kb,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    else:
        await query.edit_message_text(
            intro,
            reply_markup=_examples_intro_keyboard(),
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    context.user_data[USERDATA_EXAMPLES_INTRO_MSG_ID] = query.message.message_id


async def handle_examples_show_albums_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Смотреть примеры»: показываем альбомы и навигацию под intro."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    chat_id = query.message.chat_id if query.message else 0
    bot = context.bot
    saved_page = context.user_data.get(USERDATA_EXAMPLES_PAGE, 0)
    nav_msg_id = context.user_data.get(USERDATA_EXAMPLES_NAV_MSG_ID)
    await _show_examples_page(
        bot, chat_id, context, saved_page,
        delete_previous=True,
        nav_msg_id_to_delete=nav_msg_id,
    )


async def handle_examples_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Навигация по альбомам примеров (Вперёд / Назад)."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    try:
        page = int(query.data.split(":")[1])
    except (IndexError, ValueError):
        return
    chat_id = query.message.chat_id if query.message else 0
    bot = context.bot
    nav_msg_id = context.user_data.get(USERDATA_EXAMPLES_NAV_MSG_ID)
    await _show_examples_page(bot, chat_id, context, page, delete_previous=True, nav_msg_id_to_delete=nav_msg_id)


TARIFFS_MESSAGE = """<b>Тарифы PrismaLab</b>

Напомним: у нас два формата, и они разные: <b>Персона</b> и <b>Экспресс-фото</b>

В каждом формате <b>1 кредит = 1 фото</b>

✨ <b>Персона</b> (с вас 10 фото)

Флагманский формат для тех, кто хочет <b>стабильный «вау»-результат с максимальным сходством</b>
Вы загружаете 10 фото, мы создаём персональную модель, и дальше вы получаете <b>сильно и красиво – раз за разом</b>

Создание Персоны оплачивается отдельно и уже включает кредиты:

• <b>Персона + 20 кредитов – 599 ₽</b>
• <b>Персона + 40 кредитов – 999 ₽</b>

Когда Персона уже создана, докупать кредиты выгоднее:

• <b>10 кредитов – 229 ₽</b>
• <b>20 кредитов – 439 ₽</b>
• <b>30 кредитов – 629 ₽</b>

⚡ <b>Экспресс-фото</b> (с вас 1 фото)

Быстрый способ попробовать: <b>одно исходное фото → выбранный стиль → результат</b>

Пакеты:

• <b>5 кредитов – 199 ₽</b>
• <b>10 кредитов – 299 ₽</b>
• <b>30 кредитов – 699 ₽</b>

Крутого вам творчества! ❤️"""


async def handle_start_tariffs_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Тарифы»: текст тарифов и кнопки Экспресс, Персона, Назад."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("✨ Персона", callback_data="pl_start_persona")],
        [InlineKeyboardButton(_express_button_label(profile), callback_data="pl_start_fast")],
        [InlineKeyboardButton("Назад", callback_data="pl_fast_back")],
    ])
    await query.edit_message_text(TARIFFS_MESSAGE, reply_markup=kb, parse_mode="HTML")


async def handle_start_faq_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «А точно ли получится круто?»."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    text = (
        "<b>Точно ли получится круто?</b>\n\n"
        "Круто получится – вопрос насколько и каким способом. <b>Формата два</b>\n\n"
        "✨ <b>Персона (с вас 10 фото)</b>\n\n"
        "<b>Главное отличие PrismaLab</b>. Мы обучаем модель на ваших фото, и лицо сохраняется максимально точно: мимика, черты, ощущение «это я». <b>Результат – как полноценная фотосессия.</b>\n"
        "Бывает, наша нейросеть шалит с пальцами и мелкими деталями – за такие фото <b>мы начисляем дополнительные кредиты</b>\n\n"
        "⚡️ <b>Быстрое фото (с вас 1 фото)</b>\n\n"
        "Загружаете один снимок, выбираете стиль, получаете результат.\n"
        "Есть элемент случайности, но при хорошем исходнике <b>наша нейросеть выжмет максимум</b>: красиво, аккуратно и часто очень эффектно\n\n"
        "Иногда <b>Экспресс-фото с первого раза выдаёт шедевр</b> – особенно если исходное фото удачное.\n"
        "А если хочется <b>стабильно максимального сходства</b> и результата «как фотосессия» – выбирайте <b>Персону</b>\n\n"
        "Самый простой путь – попробовать <b>Экспресс-фото</b>, а если хотите включить «вау-режим» надолго – перейти на <b>Персону</b>"
    )
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("✨ Персона", callback_data="pl_start_persona")],
        [InlineKeyboardButton(_express_button_label(profile), callback_data="pl_start_fast")],
        [InlineKeyboardButton("Примеры работ", callback_data="pl_start_examples")],
        [InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")],
    ])
    await query.edit_message_text(text, reply_markup=kb, parse_mode="HTML")


async def handle_help_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Помощь»: текст и ссылка на бот поддержки."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    support_url = f"https://t.me/{SUPPORT_BOT_USERNAME}"
    await query.edit_message_text(
        HELP_MESSAGE,
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("Написать в поддержку", url=support_url)],
            [InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")],
        ]),
    )


async def handle_profile_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Профиль»: баланс, персоны, пол (динамическая строка с эмодзи)."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    chat_id = query.message.chat_id if query.message else 0
    bot = context.bot
    persona_msg_id = context.user_data.pop(USERDATA_FAST_PERSONA_MSG_ID, None)
    if persona_msg_id is not None and persona_msg_id != query.message.message_id:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=persona_msg_id)
        except Exception:
            pass
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    await query.edit_message_text(
        _profile_text(profile),
        reply_markup=_profile_keyboard(profile),
        parse_mode="HTML",
    )
    _schedule_profile_delete(context, query.message.chat_id, query.message.message_id, int(query.from_user.id) if query.from_user else 0)


async def handle_profile_toggle_gender_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Изменить пол»: сразу переключаем пол и обновляем текст."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    current = profile.subject_gender or "female"
    new_gender = "male" if current == "female" else "female"
    store.set_subject_gender(user_id, new_gender)
    context.user_data[USERDATA_SUBJECT_GENDER] = new_gender
    profile = store.get_user(user_id)
    # Удаляем сообщение со стилями Экспресс-фото (могло остаться от другого пола)
    style_msg_id = context.user_data.pop(USERDATA_FAST_STYLE_MSG_ID, None)
    if style_msg_id is not None:
        try:
            await context.bot.delete_message(chat_id=query.message.chat_id, message_id=style_msg_id)
        except Exception:
            pass
    # Удаляем сообщение со стилями Персоны (могло остаться от другого пола)
    persona_style_msg_id = context.user_data.pop(USERDATA_PERSONA_STYLE_MSG_ID, None)
    if persona_style_msg_id is not None:
        try:
            await context.bot.delete_message(chat_id=query.message.chat_id, message_id=persona_style_msg_id)
        except Exception:
            pass
    await query.edit_message_text(
        _profile_text(profile),
        reply_markup=_profile_keyboard(profile),
        parse_mode="HTML",
    )
    _schedule_profile_delete(context, query.message.chat_id, query.message.message_id, user_id)


async def handle_profile_fast_tariffs_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Экспресс-фото» из Профиля: при наличии кредитов — стили, иначе тарифы."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    _cancel_profile_delete_job(context)
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    if _generations_count_fast(profile) > 0:
        context.user_data[USERDATA_MODE] = "fast"
        gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or profile.subject_gender or "female"
        credits = _generations_count_fast(profile)
        credits_word = _fast_credits_word(credits)
        has_photo = bool(context.user_data.get(USERDATA_PHOTO_FILE_IDS))
        text = _fast_style_screen_text(credits, credits_word, has_photo=has_photo)
        page = context.user_data.get(USERDATA_FAST_STYLE_PAGE, 0)
        kb = _fast_style_choice_keyboard(gender, include_tariffs=True, from_profile=True, page=page)
        await query.edit_message_text(text, reply_markup=kb, parse_mode="HTML", disable_web_page_preview=True)
        context.user_data[USERDATA_FAST_STYLE_MSG_ID] = query.message.message_id
    else:
        await _send_fast_tariffs_two_messages(
            context.bot, query.message.chat_id, context, edit_message=query.message, back_callback="pl_profile"
        )


async def handle_persona_create_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Создать персону» (из профиля): если пол неизвестен — выбор пола, иначе — intro с тарифами."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    _cancel_profile_delete_job(context)
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    known_gender = getattr(profile, "subject_gender", None) or context.user_data.get(USERDATA_SUBJECT_GENDER)
    if known_gender in ("male", "female"):
        context.user_data[USERDATA_SUBJECT_GENDER] = known_gender
        context.user_data[USERDATA_MODE] = "persona"
        await query.edit_message_text(
            PERSONA_INTRO_MESSAGE,
            reply_markup=_persona_intro_keyboard(),
            parse_mode="HTML",
        )
    else:
        await query.edit_message_text(
            "Выбери пол – так нейросеть точнее настроит результат",
            reply_markup=_persona_gender_keyboard(),
        )


async def handle_persona_gender_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """После выбора пола в «Персона»: сохраняем в профиль, показываем intro с тарифами."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    _, gender = query.data.split(":", 1)
    context.user_data[USERDATA_SUBJECT_GENDER] = gender
    user_id = int(query.from_user.id) if query.from_user else 0
    store.set_subject_gender(user_id, gender)
    context.user_data[USERDATA_MODE] = "persona"
    await query.edit_message_text(
        PERSONA_INTRO_MESSAGE,
        reply_markup=_persona_intro_keyboard(),
        parse_mode="HTML",
    )


def _persona_pack_class_name(gender: str | None) -> str:
    return "man" if gender == "male" else "woman"


def _persona_lora_name(gender: str | None) -> str:
    """
    Имя класса для обучения персональной LoRA.
    По умолчанию сохраняем историческое поведение: person.
    Для dev-экспериментов можно включить man/woman через env.
    """
    mode = (os.getenv("PRISMALAB_PERSONA_LORA_NAME_MODE") or "person").strip().lower()
    if mode in {"gender", "class", "man_woman"}:
        return _persona_pack_class_name(gender)
    return "person"


def _resolve_pack_class_name(offer: dict[str, Any], gender: str | None) -> str:
    custom = str(offer.get("class_name") or "").strip().lower()
    if custom in {"man", "woman", "boy", "girl", "dog", "cat"}:
        return custom
    return _persona_pack_class_name(gender)


def _pack_classes_text(classes: list[str]) -> str:
    mapping = {"man": "man (мужчина)", "woman": "woman (женщина)", "boy": "boy (мальчик)", "girl": "girl (девочка)", "dog": "dog (собакой)", "cat": "cat (кошкой)"}
    labels: list[str] = []
    for cls in classes:
        labels.append(mapping.get(cls, cls))
    return ", ".join(labels)


def _pack_wait_timeout_seconds(expected_images: int) -> int:
    """
    Таймаут ожидания паков:
    - можно переопределить через PRISMALAB_ASTRIA_PACK_MAX_SECONDS
    - по умолчанию масштабируем от количества изображений
    """
    raw = (os.getenv("PRISMALAB_ASTRIA_PACK_MAX_SECONDS") or "").strip()
    if raw:
        try:
            return max(900, int(raw))
        except Exception:
            pass
    expected = max(1, int(expected_images))
    # 30 минут минимум, дальше ~75 секунд на изображение.
    return max(1800, expected * 75)


async def _run_persona_pack_generation(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    pack_id: int,
    offer: dict[str, Any],
    train_file_ids: list[str] | None = None,
) -> None:
    gen_lock = await _acquire_user_generation_lock(user_id)
    if gen_lock is None:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Сейчас уже идет генерация. Повторите запуск пака через минуту.",
        )
        return

    status_msg = await context.bot.send_message(
        chat_id=chat_id,
        text=f"🎬 Запускаю пак «{offer['title']}»...",
    )
    try:
        settings = load_settings()
        if not settings.astria_api_key:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_msg.message_id,
                text="❌ Сервис генерации не настроен.",
            )
            return
        profile = store.get_user(user_id)
        class_name = _resolve_pack_class_name(offer, profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER))
        pack = await astria_get_pack(api_key=settings.astria_api_key, pack_id=pack_id)
        expected = int(offer.get("expected_images") or 0)
        try:
            if isinstance(pack.costs, dict) and pack.costs and class_name not in pack.costs:
                available_classes = [k for k, v in pack.costs.items() if isinstance(v, dict)]
                if available_classes:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_msg.message_id,
                        text=(
                            "❌ Этот пак не поддерживает тип вашей Персоны.\n"
                            f"Доступные классы: {_pack_classes_text(available_classes)}."
                        ),
                    )
                    return
            class_cost = pack.costs.get(class_name) if isinstance(pack.costs, dict) else None
            if isinstance(class_cost, dict):
                expected = int(class_cost.get("num_images") or expected)
        except Exception:
            pass
        expected = max(1, expected)

        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_msg.message_id,
            text=f"🎬 Пак «{offer['title']}»: запускаю генерацию ({expected} фото)...",
        )
        title = f"{offer['title']} user:{user_id} ts:{int(time.time())}"
        known_prompt_ids: set[str] = set()
        if train_file_ids:
            image_bytes_list: list[bytes] = []
            for fid in train_file_ids[:10]:
                img_bytes = await _safe_get_file_bytes(context.bot, fid)
                image_bytes_list.append(_prepare_image_for_photomaker(img_bytes))
            tune = await astria_create_tune_from_pack(
                api_key=settings.astria_api_key,
                pack_id=pack_id,
                title=title[:120],
                name=class_name,
                image_bytes_list=image_bytes_list,
            )
        else:
            lora_tune_id_raw = profile.astria_lora_tune_id
            if not lora_tune_id_raw:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_msg.message_id,
                    text="❌ У вас еще нет обученной Персоны. Сначала обучите Персону (10 фото).",
                )
                return
            try:
                lora_tune_id = int(str(lora_tune_id_raw))
            except ValueError:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_msg.message_id,
                    text="❌ Не удалось определить ID вашей Персоны. Напишите в поддержку.",
                )
                return
            try:
                known_prompt_ids = await astria_get_tune_prompt_ids(
                    api_key=settings.astria_api_key,
                    tune_id=str(lora_tune_id),
                )
            except Exception as e:
                logger.warning("Не удалось получить существующие prompts для tune %s: %s", lora_tune_id, e)
            tune = await astria_create_tune_from_pack(
                api_key=settings.astria_api_key,
                pack_id=pack_id,
                title=title[:120],
                name=class_name,
                tune_ids=[lora_tune_id],
            )
        logger.info(
            "pack: astria create_tune_from_pack response: tune_id=%s status=%s raw=%s",
            tune.tune_id,
            tune.status,
            tune.raw,
        )

        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_msg.message_id,
            text="⏳ Пак запущен. Жду готовые изображения...",
        )
        wait_timeout = _pack_wait_timeout_seconds(expected)
        urls = await astria_wait_pack_images(
            api_key=settings.astria_api_key,
            tune_id=tune.tune_id,
            expected_images=expected,
            known_prompt_ids=known_prompt_ids,
            max_seconds=wait_timeout,
            poll_seconds=6.0,
        )
        if not urls:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_msg.message_id,
                text="❌ Пак завершился без изображений. Попробуйте еще раз.",
            )
            return

        # Сохраняем tune_id как Persona, если у пользователя её ещё нет (пак создал tune из загруженных фото)
        if train_file_ids and tune.tune_id and not profile.astria_lora_tune_id:
            try:
                store.set_astria_lora_tune(user_id=user_id, tune_id=tune.tune_id)
                logger.info("pack: сохранён tune_id %s как Persona для user %s", tune.tune_id, user_id)
            except Exception as e:
                logger.warning("pack: не удалось сохранить tune_id для user %s: %s", user_id, e)

        total = min(len(urls), expected)
        sent_count = 0
        pack_download_timeout = 90.0
        for i, url in enumerate(urls[:total], start=1):
            out_bytes = None
            for attempt in range(3):
                try:
                    out_bytes = await astria_download_first_image_bytes(
                        [url],
                        api_key=settings.astria_api_key,
                        timeout_s=pack_download_timeout,
                    )
                    break
                except (AstriaError, requests.RequestException) as e:
                    if attempt < 2:
                        logger.warning(
                            "Pack image %s/%s download attempt %s failed: %s, retrying...",
                            i,
                            total,
                            attempt + 1,
                            e,
                        )
                        await asyncio.sleep(2.0)
                    else:
                        logger.warning(
                            "Pack image %s/%s download failed after 3 attempts, skipping: %s",
                            i,
                            total,
                            e,
                        )
            if out_bytes:
                bio = io.BytesIO(out_bytes)
                bio.name = f"pack_{pack_id}_{sent_count + 1}.png"
                caption = f"Пак «{offer['title']}» ({sent_count + 1}/{total})" if sent_count == 0 else ""
                await _safe_send_document(
                    bot=context.bot,
                    chat_id=chat_id,
                    document=bio,
                    caption=caption,
                )
                sent_count += 1
                await asyncio.sleep(0.1)

        store.log_event(
            user_id,
            "pack_generation",
            {"pack_id": pack_id, "pack_title": offer["title"], "images_sent": sent_count},
        )
        page = context.user_data.get(USERDATA_PERSONA_STYLE_PAGE, 0)
        gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
        done_text = f"✅ Пак «{offer['title']}» готов. Отправлено {sent_count} фото."
        if sent_count < total:
            done_text += f" (не удалось скачать {total - sent_count})"
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_msg.message_id,
            text=done_text,
            reply_markup=_persona_styles_keyboard(gender, page),
        )
    except AstriaError as e:
        logger.exception("Ошибка генерации пака %s (Astria): %s", pack_id, e)
        details = str(e).strip()
        short = details[:300] if details else "Ошибка генерации"
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_msg.message_id,
                text=f"❌ Ошибка генерации пака: {short}",
            )
        except Exception:
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Ошибка генерации пака: {short}")
    except Exception as e:
        logger.exception("Ошибка генерации пака %s: %s", pack_id, e)
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_msg.message_id,
                text="❌ Ошибка генерации пака. Попробуйте еще раз.",
            )
        except Exception:
            await context.bot.send_message(chat_id=chat_id, text="❌ Ошибка генерации пака. Попробуйте еще раз.")
    finally:
        gen_lock.release()


async def _poll_persona_pack_payment_and_run(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    payment_id: str,
    user_id: int,
    chat_id: int,
    pack_id: int,
    offer: dict[str, Any],
    timeout_seconds: int = 900,
    poll_interval: int = 5,
) -> None:
    started = time.monotonic()
    while True:
        if time.monotonic() - started > timeout_seconds:
            logger.info("Таймаут поллинга оплаты пака %s", payment_id)
            return
        status = await asyncio.to_thread(get_payment_status, payment_id)
        if status == "succeeded":
            if store.is_payment_processed(payment_id):
                return
            amount_rub = apply_test_amount(float(offer["price_rub"]))
            expected_images = int(offer.get("expected_images") or 0)
            store.log_payment(
                user_id=user_id,
                payment_id=payment_id,
                payment_method="yookassa",
                product_type="persona_pack",
                credits=max(1, expected_images),
                amount_rub=amount_rub,
            )
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Оплата получена ✅\n\nЗапускаю пак «{offer['title']}».",
                parse_mode="HTML",
            )
            await _run_persona_pack_generation(
                context=context,
                chat_id=chat_id,
                user_id=user_id,
                pack_id=pack_id,
                offer=offer,
            )
            return
        if status == "canceled":
            await context.bot.send_message(chat_id=chat_id, text="❌ Платеж отменен или истек.")
            return
        await asyncio.sleep(poll_interval)


async def handle_persona_packs_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()
    offers = _pack_offers()
    if not offers:
        await query.edit_message_text(
            "Паки пока не настроены. Добавьте PRISMALAB_ASTRIA_PACK_OFFERS в .env dev-бота.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="pl_persona_back")]]),
        )
        return
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    if not profile.astria_lora_tune_id and not _dev_pack_train_from_images():
        await query.edit_message_text(
            "Паки доступны после обучения Персоны.\n\nСначала оплатите «Персона» и загрузите 10 фото.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="pl_persona_back")]]),
        )
        return
    await query.edit_message_text(
        PERSONA_PACKS_MESSAGE,
        reply_markup=_persona_packs_keyboard(),
        parse_mode="HTML",
    )


async def handle_persona_pack_buy_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    logger.info("persona_pack_buy click: user_id=%s data=%s", user_id, query.data)

    try:
        parts = query.data.split(":", 1)
        if len(parts) != 2:
            await query.edit_message_text("Некорректная команда пакета.", reply_markup=_persona_packs_keyboard())
            return
        pack_id = int(parts[1])
    except Exception:
        await query.edit_message_text("Некорректный pack_id.", reply_markup=_persona_packs_keyboard())
        return

    offer = _find_pack_offer(pack_id)
    logger.info("persona_pack_buy: parsed pack_id=%s offer_found=%s", pack_id, bool(offer))
    if not offer:
        await query.edit_message_text("Пак не найден в конфиге.", reply_markup=_persona_packs_keyboard())
        return
    if not use_yookassa():
        logger.info("persona_pack_buy: yookassa disabled")
        await query.edit_message_text(
            "Для паков в dev включена только оплата по ссылке.",
            reply_markup=_persona_packs_keyboard(),
        )
        return

    chat_id = query.message.chat_id if query.message else 0
    logger.info("persona_pack_buy: loading profile user_id=%s", user_id)
    profile = store.get_user(user_id)
    logger.info("persona_pack_buy: profile loaded lora_tune_id=%s", getattr(profile, "astria_lora_tune_id", None))
    if not profile.astria_lora_tune_id and not _dev_pack_train_from_images():
        await query.edit_message_text("Сначала обучите Персону (10 фото).", reply_markup=_persona_intro_keyboard())
        return

    settings = load_settings()
    logger.info("persona_pack_buy: settings loaded astria_key=%s", bool(settings.astria_api_key))
    if not settings.astria_api_key:
        await query.edit_message_text("❌ Сервис генерации не настроен.", reply_markup=_persona_packs_keyboard())
        return

    logger.info("persona_pack_buy: edit progress message")
    await query.edit_message_text("⏳ Проверяю пак и создаю ссылку оплаты…")

    class_name = _resolve_pack_class_name(offer, profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER))
    logger.info("persona_pack_buy: fetching pack from astria pack_id=%s class=%s", pack_id, class_name)
    try:
        pack = await asyncio.wait_for(
            astria_get_pack(api_key=settings.astria_api_key, pack_id=pack_id),
            timeout=35.0,
        )
        logger.info("persona_pack_buy: astria pack fetched pack_id=%s", pack_id)
    except Exception as e:
        logger.warning("Не удалось получить pack %s перед оплатой: %s", pack_id, e)
        await query.edit_message_text(
            "❌ Не удалось проверить конфигурацию пака. Попробуйте еще раз.",
            reply_markup=_persona_packs_keyboard(),
        )
        return
    if isinstance(pack.costs, dict) and pack.costs and class_name not in pack.costs:
        available_classes = [k for k, v in pack.costs.items() if isinstance(v, dict)]
        available_text = _pack_classes_text(available_classes) if available_classes else "не определены"
        await query.edit_message_text(
            (
                "Этот пак не подходит для вашей Персоны.\n"
                f"Доступные классы: {available_text}."
            ),
            reply_markup=_persona_packs_keyboard(),
        )
        return

    if _dev_pack_train_from_images():
        context.user_data[USERDATA_PERSONA_SELECTED_PACK_ID] = pack_id
        context.user_data[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = True
        context.user_data[USERDATA_PERSONA_PACK_PHOTOS] = []
        context.user_data[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = []
        context.user_data[USERDATA_MODE] = "persona_pack_upload"
        await query.edit_message_text(
            PERSONA_PACK_UPLOAD_WAIT_MESSAGE,
            reply_markup=_persona_pack_upload_keyboard(),
            parse_mode="HTML",
        )
        return

    if _dev_skip_pack_payment():
        logger.info("persona_pack_buy: dev skip payment enabled, run pack directly user_id=%s pack_id=%s", user_id, pack_id)
        await query.edit_message_text(
            "🧪 Dev-режим: оплату пропускаю, запускаю пак сразу.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="pl_persona_packs")]]),
        )
        await _run_persona_pack_generation(
            context=context,
            chat_id=chat_id,
            user_id=user_id,
            pack_id=pack_id,
            offer=offer,
        )
        return

    amount = apply_test_amount(float(offer["price_rub"]))
    me = await context.bot.get_me()
    return_url = f"https://t.me/{me.username}" if me and me.username else None
    logger.info("persona_pack_buy: creating payment amount=%s", amount)
    try:
        url, payment_id = await asyncio.wait_for(
            asyncio.to_thread(
                create_payment,
                amount_rub=amount,
                description=f"Пак: {offer['title']}",
                metadata={
                    "user_id": str(user_id),
                    "chat_id": str(chat_id),
                    "credits": str(max(1, int(offer.get("expected_images") or 1))),
                    "product_type": "persona_pack",
                    "pack_id": str(pack_id),
                },
                return_url=return_url,
            ),
            timeout=35.0,
        )
        logger.info("persona_pack_buy: payment created payment_id=%s has_url=%s", payment_id, bool(url))
    except asyncio.TimeoutError:
        await query.edit_message_text(
            "❌ Таймаут создания платежа. Попробуйте еще раз.",
            reply_markup=_persona_packs_keyboard(),
        )
        return
    except Exception as e:
        logger.warning("Ошибка create_payment для pack %s: %s", pack_id, e)
        asyncio.create_task(alert_payment_error(user_id, "persona_pack", str(e)))
        await query.edit_message_text(
            "❌ Ошибка оплаты. Попробуйте еще раз.",
            reply_markup=_persona_packs_keyboard(),
        )
        return
    if not (url and payment_id):
        err = str(payment_id or "unknown")
        asyncio.create_task(alert_payment_error(user_id, "persona_pack", err))
        if "network error" in err.lower() or "readtimeout" in err.lower() or "connecttimeout" in err.lower():
            await query.edit_message_text(
                "❌ Не удалось связаться с платёжной системой (сеть/таймаут). Попробуйте еще раз через 1-2 минуты.",
                reply_markup=_persona_packs_keyboard(),
            )
        else:
            await query.edit_message_text(f"❌ Ошибка оплаты: {payment_id}", reply_markup=_persona_packs_keyboard())
        return

    context.user_data[USERDATA_PERSONA_SELECTED_PACK_ID] = pack_id
    await query.edit_message_text(
        (
            f"<b>{offer['title']}</b>\n"
            f"Цена: <b>{int(amount)} ₽</b>\n\n"
            "Оплатите, после этого бот автоматически запустит пак и пришлет готовые фото."
        ),
        reply_markup=_payment_yookassa_keyboard(url, "pl_persona_packs"),
        parse_mode="HTML",
    )
    asyncio.create_task(
        _poll_persona_pack_payment_and_run(
            context=context,
            payment_id=payment_id,
            user_id=user_id,
            chat_id=chat_id,
            pack_id=pack_id,
            offer=offer,
        )
    )


async def handle_persona_buy_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Выбор тарифа Персоны: сразу инвойс/ссылка/симуляция (без экрана «Нажмите Оплатить»)."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    _, count_str = query.data.split(":", 1)
    credits = int(count_str) if count_str in ("20", "40") else 20
    context.user_data[USERDATA_PERSONA_CREDITS] = credits
    user_id = int(query.from_user.id) if query.from_user else 0
    chat_id = query.message.chat_id if query.message else 0

    if use_yookassa():
        amount = _amount_rub("persona_create", credits)
        me = await context.bot.get_me()
        return_url = f"https://t.me/{me.username}" if me and me.username else None
        url, payment_id = create_payment(
            amount_rub=amount,
            description=f"Персона: {credits} кредитов",
            metadata={
                "user_id": str(user_id),
                "chat_id": str(chat_id),
                "credits": str(credits),
                "product_type": "persona_create",
            },
            return_url=return_url,
        )
        if url and payment_id:
            await query.edit_message_text(
                f"<b>Создание Персоны + {credits} кредитов за {amount:.0f} ₽</b>\n\nОплатите удобным способом, после оплаты загрузите 10 фото",
                parse_mode="HTML",
                reply_markup=_payment_yookassa_keyboard(url, "pl_persona_back"),
            )
            asyncio.create_task(poll_payment_status(
                payment_id=payment_id,
                bot=context.bot,
                store=store,
                user_id=user_id,
                chat_id=chat_id,
                credits=credits,
                product_type="persona_create",
                amount_rub=amount,
            ))
            return
        else:
            logger.warning("Ошибка создания платежа (persona_create): %s", payment_id)
            asyncio.create_task(alert_payment_error(user_id, "persona_create", str(payment_id)))
            await query.edit_message_text(f"❌ Ошибка оплаты: {payment_id}")
            return

    if use_telegram_payments():
        amount = _amount_rub("persona_create", credits)
        amount_kop = max(8800, int(amount * 100))  # минимум 88 ₽ для Telegram
        payload = f"pl:persona_create:{credits}:{user_id}"
        try:
            await context.bot.send_invoice(
                chat_id=chat_id,
                title="Персона: создание",
                description=f"Персона: {credits} кредитов",
                payload=payload,
                provider_token=TELEGRAM_PROVIDER_TOKEN,
                currency="RUB",
                prices=[LabeledPrice(label="Оплата", amount=amount_kop)],
            )
        except BadRequest as e:
            logger.exception("send_invoice (persona_create) BadRequest: %s", e)
            await query.edit_message_text("❌ Не удалось отправить счёт. Проверьте в BotFather → Payments, что подключена платёжная система.")
        return

    store.set_persona_credits(user_id, credits)
    if context.user_data.pop(USERDATA_PERSONA_RECREATING, None):
        store.set_astria_lora_tune(user_id=user_id, tune_id=None)
    context.user_data[USERDATA_MODE] = "persona"
    await query.edit_message_text(
        PERSONA_RULES_MESSAGE,
        reply_markup=_persona_rules_keyboard(),
        parse_mode="HTML",
    )


async def handle_persona_back_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Назад» из экранов Персоны (тарифы, выбор пола): возврат на вводный экран Персоны."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    context.user_data[USERDATA_MODE] = "persona"
    context.user_data.pop(USERDATA_PERSONA_WAITING_UPLOAD, None)
    context.user_data.pop(USERDATA_PERSONA_PHOTOS, None)
    context.user_data.pop(USERDATA_PERSONA_CREDITS, None)
    await query.edit_message_text(
        PERSONA_INTRO_MESSAGE,
        reply_markup=_persona_intro_keyboard(),
        parse_mode="HTML",
    )


async def handle_persona_show_credits_out_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать экран «кредиты закончились» (Назад из докупки или подтверждения)."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    text, kb = _persona_credits_out_content(profile)
    await query.edit_message_text(text, reply_markup=kb, parse_mode="HTML")


async def handle_persona_topup_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Докупить кредиты»: показываем тарифы 10/229, 20/439, 30/629."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    await query.edit_message_text(
        PERSONA_TOPUP_MESSAGE,
        reply_markup=_persona_topup_keyboard(),
    )


async def handle_persona_topup_buy_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Выбор пакета докупки: сразу инвойс/ссылка/симуляция (без экрана «Нажмите Оплатить»)."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    _, count_str = query.data.split(":", 1)
    credits = int(count_str) if count_str in ("10", "20", "30") else 10
    context.user_data[USERDATA_PERSONA_CREDITS] = credits
    user_id = int(query.from_user.id) if query.from_user else 0
    chat_id = query.message.chat_id if query.message else 0

    if use_yookassa():
        amount = _amount_rub("persona_topup", credits)
        me = await context.bot.get_me()
        return_url = f"https://t.me/{me.username}" if me and me.username else None
        url, payment_id = create_payment(
            amount_rub=amount,
            description=f"Докупка Персона: {credits} кредитов",
            metadata={
                "user_id": str(user_id),
                "chat_id": str(chat_id),
                "credits": str(credits),
                "product_type": "persona_topup",
            },
            return_url=return_url,
        )
        if url and payment_id:
            await query.edit_message_text(
                f"<b>{credits} кредитов Персона за {amount:.0f} ₽</b>\n\nОплатите удобным способом, кредиты зачислим моментально",
                parse_mode="HTML",
                reply_markup=_payment_yookassa_keyboard(url, "pl_persona_show_credits_out"),
            )
            asyncio.create_task(poll_payment_status(
                payment_id=payment_id,
                bot=context.bot,
                store=store,
                user_id=user_id,
                chat_id=chat_id,
                credits=credits,
                product_type="persona_topup",
                amount_rub=amount,
            ))
            return
        else:
            logger.warning("Ошибка создания платежа (persona_topup): %s", payment_id)
            asyncio.create_task(alert_payment_error(user_id, "persona_topup", str(payment_id)))
            await query.edit_message_text(f"❌ Ошибка оплаты: {payment_id}")
            return

    if use_telegram_payments():
        amount = _amount_rub("persona_topup", credits)
        amount_kop = max(8800, int(amount * 100))  # минимум 88 ₽ для Telegram
        payload = f"pl:persona_topup:{credits}:{user_id}"
        try:
            await context.bot.send_invoice(
                chat_id=chat_id,
                title="Докупка кредитов Персоны",
                description=f"{credits} кредитов Персоны",
                payload=payload,
                provider_token=TELEGRAM_PROVIDER_TOKEN,
                currency="RUB",
                prices=[LabeledPrice(label="Оплата", amount=amount_kop)],
            )
        except BadRequest as e:
            logger.exception("send_invoice (persona_topup) BadRequest: %s", e)
            await query.edit_message_text("❌ Не удалось отправить счёт. Проверьте в BotFather → Payments, что подключена платёжная система.")
        return

    profile = store.get_user(user_id)
    new_total = profile.persona_credits_remaining + credits
    store.set_persona_credits(user_id, new_total)
    profile = store.get_user(user_id)
    gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"

    selected = context.user_data.pop(USERDATA_PERSONA_SELECTED_STYLE, None)
    gen_lock = await _acquire_user_generation_lock(user_id) if selected else None
    if selected and gen_lock is None:
        selected = None

    if selected and gen_lock is not None:
        style_id, label = selected
        prompt = _persona_style_prompt(style_id, label)
        status_text = f"<b>Оплата получена</b> ✅\n\n{_format_balance_persona(new_total)}\n\nВыбран стиль: «{label}»\n\n🎨 <i>Создаю изображение...</i>"
        await query.edit_message_text(status_text, parse_mode="HTML")
        settings = load_settings()

        async def runner() -> None:
            try:
                await _run_style_job(
                    bot=context.bot,
                    chat_id=query.message.chat_id,
                    photo_file_ids=[],
                    style_id=style_id,
                    settings=settings,
                    status_message_id=query.message.message_id,
                    prompt_strength=0.7,
                    user_id=user_id,
                    subject_gender=gender,
                    use_personal_requested=False,
                    test_prompt=None,
                    lora_prompt_override=prompt,
                    style_title_override=label,
                    is_persona_style=True,
                    context=context,
                )
            finally:
                gen_lock.release()

        context.application.create_task(runner())
    else:
        text = f"<b>Оплата получена</b> ✅\n\n<b>Выберите стиль</b> 👇\n\n{_format_balance_persona(new_total)}\n\n{STYLE_EXAMPLES_FOOTER}"
        await query.edit_message_text(
            text,
            reply_markup=_persona_styles_keyboard(gender, page=0),
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        context.user_data[USERDATA_PERSONA_STYLE_MSG_ID] = query.message.message_id
    context.user_data[USERDATA_PERSONA_STYLE_PAGE] = 0


async def handle_persona_topup_confirm_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Оплата докупки: Telegram Payments (инвойс) или ЮKassa (ссылка) или симуляция."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    _, count_str = query.data.split(":", 1)
    credits = int(count_str) if count_str in ("10", "20", "30") else 10
    user_id = int(query.from_user.id) if query.from_user else 0
    chat_id = query.message.chat_id if query.message else 0

    if use_yookassa():
        amount = _amount_rub("persona_topup", credits)
        me = await context.bot.get_me()
        return_url = f"https://t.me/{me.username}" if me and me.username else None
        url, payment_id = create_payment(
            amount_rub=amount,
            description=f"Докупка Персона: {credits} кредитов",
            metadata={
                "user_id": str(user_id),
                "chat_id": str(chat_id),
                "credits": str(credits),
                "product_type": "persona_topup",
            },
            return_url=return_url,
        )
        if url and payment_id:
            await query.edit_message_text(
                f"<b>{credits} кредитов Персона за {amount:.0f} ₽</b>\n\nОплатите удобным способом, кредиты зачислим моментально",
                parse_mode="HTML",
                reply_markup=_payment_yookassa_keyboard(url, "pl_persona_show_credits_out"),
            )
            asyncio.create_task(poll_payment_status(
                payment_id=payment_id,
                bot=context.bot,
                store=store,
                user_id=user_id,
                chat_id=chat_id,
                credits=credits,
                product_type="persona_topup",
                amount_rub=amount,
            ))
            return
        else:
            logger.warning("Ошибка создания платежа (persona_topup): %s", payment_id)
            asyncio.create_task(alert_payment_error(user_id, "persona_topup", str(payment_id)))
            await query.edit_message_text(f"❌ Ошибка оплаты: {payment_id}")
            return

    if use_telegram_payments():
        amount = _amount_rub("persona_topup", credits)
        amount_kop = max(8800, int(amount * 100))  # минимум 88 ₽ для Telegram
        payload = f"pl:persona_topup:{credits}:{user_id}"
        try:
            await context.bot.send_invoice(
                chat_id=chat_id,
                title="Докупка кредитов Персоны",
                description=f"{credits} кредитов Персоны",
                payload=payload,
                provider_token=TELEGRAM_PROVIDER_TOKEN,
                currency="RUB",
                prices=[LabeledPrice(label="Оплата", amount=amount_kop)],
            )
        except BadRequest as e:
            logger.exception("send_invoice (persona_topup) BadRequest: %s", e)
            await query.edit_message_text("❌ Не удалось отправить счёт. Проверьте в BotFather → Payments, что подключена платёжная система.")
        return

    profile = store.get_user(user_id)
    new_total = profile.persona_credits_remaining + credits
    store.set_persona_credits(user_id, new_total)
    profile = store.get_user(user_id)
    gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"

    selected = context.user_data.pop(USERDATA_PERSONA_SELECTED_STYLE, None)
    gen_lock = await _acquire_user_generation_lock(user_id) if selected else None
    if selected and gen_lock is None:
        selected = None  # не запускаем, если уже идёт генерация

    if selected and gen_lock is not None:
        style_id, label = selected
        prompt = _persona_style_prompt(style_id, label)
        status_text = f"<b>Оплата получена</b> ✅\n\n{_format_balance_persona(new_total)}\n\nВыбран стиль: «{label}»\n\n🎨 <i>Создаю изображение...</i>"
        await query.edit_message_text(status_text, parse_mode="HTML")
        settings = load_settings()

        async def runner() -> None:
            try:
                await _run_style_job(
                    bot=context.bot,
                    chat_id=query.message.chat_id,
                    photo_file_ids=[],
                    style_id=style_id,
                    settings=settings,
                    status_message_id=query.message.message_id,
                    prompt_strength=0.7,
                    user_id=user_id,
                    subject_gender=gender,
                    use_personal_requested=False,
                    test_prompt=None,
                    lora_prompt_override=prompt,
                    style_title_override=label,
                    is_persona_style=True,
                    context=context,
                )
            finally:
                gen_lock.release()

        context.application.create_task(runner())
    else:
        text = f"<b>Оплата получена</b> ✅\n\n<b>Выберите стиль</b> 👇\n\n{_format_balance_persona(new_total)}\n\n{STYLE_EXAMPLES_FOOTER}"
        await query.edit_message_text(
            text,
            reply_markup=_persona_styles_keyboard(gender, page=0),
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        context.user_data[USERDATA_PERSONA_STYLE_MSG_ID] = query.message.message_id
    context.user_data[USERDATA_PERSONA_STYLE_PAGE] = 0


async def handle_persona_recreate_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Создать новую Персону»: показываем подтверждение."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    chat_id = query.message.chat_id if query.message else 0
    bot = context.bot
    intro_msg_id = context.user_data.pop(USERDATA_EXAMPLES_INTRO_MSG_ID, None)
    media_ids = context.user_data.pop(USERDATA_EXAMPLES_MEDIA_IDS, None) or []
    nav_msg_id = context.user_data.pop(USERDATA_EXAMPLES_NAV_MSG_ID, None)
    if intro_msg_id is not None or media_ids or nav_msg_id is not None:
        to_delete = []
        if intro_msg_id is not None and intro_msg_id != query.message.message_id:
            to_delete.append(intro_msg_id)
        to_delete.extend(media_ids)
        if nav_msg_id is not None:
            to_delete.append(nav_msg_id)
        if to_delete:
            await asyncio.gather(*[bot.delete_message(chat_id=chat_id, message_id=mid) for mid in to_delete], return_exceptions=True)
    await query.edit_message_text(
        PERSONA_RECREATE_CONFIRM_MESSAGE,
        reply_markup=_persona_recreate_confirm_keyboard(),
        parse_mode="HTML",
    )


async def handle_persona_recreate_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """«Главное меню» из подтверждения пересоздания: сбрасываем флоу, возвращаем в главное меню."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    _clear_persona_flow_state(context)
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    await query.edit_message_text(
        _start_message_text(profile),
        reply_markup=_start_keyboard(profile),
        parse_mode="HTML",
    )


def _clear_persona_flow_state(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сброс состояния флоу Персоны при выходе/отмене."""
    for key in (
        USERDATA_MODE,
        USERDATA_PERSONA_WAITING_UPLOAD,
        USERDATA_PERSONA_PHOTOS,
        USERDATA_PERSONA_CREDITS,
        USERDATA_PERSONA_TRAINING_STATUS,
        USERDATA_PERSONA_UPLOAD_MSG_IDS,
        USERDATA_PERSONA_STYLE_MSG_ID,
        USERDATA_PERSONA_STYLE_PAGE,
        USERDATA_PERSONA_SELECTED_STYLE,
        USERDATA_PERSONA_SELECTED_PACK_ID,
        USERDATA_PERSONA_PACK_WAITING_UPLOAD,
        USERDATA_PERSONA_PACK_PHOTOS,
        USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS,
        USERDATA_PERSONA_RECREATING,
    ):
        context.user_data.pop(key, None)


async def handle_persona_recreate_confirm_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Подтверждение: НЕ удаляем модель, переходим к тарифам. Удаление только при оплате."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    context.user_data[USERDATA_PERSONA_RECREATING] = True
    profile = store.get_user(user_id)
    known_gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
    context.user_data[USERDATA_SUBJECT_GENDER] = known_gender
    context.user_data[USERDATA_MODE] = "persona"
    await query.edit_message_text(
        PERSONA_INTRO_MESSAGE,
        reply_markup=_persona_intro_keyboard(),
        parse_mode="HTML",
    )


async def handle_persona_confirm_pay_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Оплатить»: Telegram Payments (инвойс) или ЮKassa (ссылка) или симуляция."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    _, count_str = query.data.split(":", 1)
    credits = int(count_str) if count_str in ("20", "40") else 20
    context.user_data[USERDATA_PERSONA_CREDITS] = credits
    user_id = int(query.from_user.id) if query.from_user else 0
    chat_id = query.message.chat_id if query.message else 0

    if use_yookassa():
        amount = _amount_rub("persona_create", credits)
        me = await context.bot.get_me()
        return_url = f"https://t.me/{me.username}" if me and me.username else None
        url, payment_id = create_payment(
            amount_rub=amount,
            description=f"Персона: {credits} кредитов",
            metadata={
                "user_id": str(user_id),
                "chat_id": str(chat_id),
                "credits": str(credits),
                "product_type": "persona_create",
            },
            return_url=return_url,
        )
        if url and payment_id:
            await query.edit_message_text(
                f"<b>Создание Персоны + {credits} кредитов за {amount:.0f} ₽</b>\n\nОплатите удобным способом, после оплаты загрузите 10 фото",
                parse_mode="HTML",
                reply_markup=_payment_yookassa_keyboard(url, "pl_persona_back"),
            )
            asyncio.create_task(poll_payment_status(
                payment_id=payment_id,
                bot=context.bot,
                store=store,
                user_id=user_id,
                chat_id=chat_id,
                credits=credits,
                product_type="persona_create",
                amount_rub=amount,
            ))
            return
        else:
            logger.warning("Ошибка создания платежа (persona_create): %s", payment_id)
            asyncio.create_task(alert_payment_error(user_id, "persona_create", str(payment_id)))
            await query.edit_message_text(f"❌ Ошибка оплаты: {payment_id}")
            return

    if use_telegram_payments():
        amount = _amount_rub("persona_create", credits)
        amount_kop = max(8800, int(amount * 100))  # минимум 88 ₽ для Telegram
        payload = f"pl:persona_create:{credits}:{user_id}"
        try:
            await context.bot.send_invoice(
                chat_id=chat_id,
                title="Персона: создание",
                description=f"Персона: {credits} кредитов",
                payload=payload,
                provider_token=TELEGRAM_PROVIDER_TOKEN,
                currency="RUB",
                prices=[LabeledPrice(label="Оплата", amount=amount_kop)],
            )
        except BadRequest as e:
            logger.exception("send_invoice (persona_create) BadRequest: %s", e)
            await query.edit_message_text("❌ Не удалось отправить счёт. Проверьте в BotFather → Payments, что подключена платёжная система.")
        return

    store.set_persona_credits(user_id, credits)
    if context.user_data.pop(USERDATA_PERSONA_RECREATING, None):
        store.set_astria_lora_tune(user_id=user_id, tune_id=None)
    context.user_data[USERDATA_MODE] = "persona"
    await query.edit_message_text(
        PERSONA_RULES_MESSAGE,
        reply_markup=_persona_rules_keyboard(),
        parse_mode="HTML",
    )


async def handle_persona_got_it_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Всё понятно, погнали!»: переходим к загрузке 10 фото."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    context.user_data[USERDATA_PERSONA_WAITING_UPLOAD] = True
    context.user_data[USERDATA_PERSONA_PHOTOS] = []
    context.user_data[USERDATA_MODE] = "persona"
    await query.edit_message_text(PERSONA_UPLOAD_WAIT_MESSAGE, parse_mode="HTML")


def _persona_upload_keyboard() -> InlineKeyboardMarkup:
    """Кнопка «Сбросить и начать заново» при загрузке фото Персоны."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Сбросить и начать заново", callback_data="pl_persona_reset_photos")],
    ])


def _persona_pack_upload_keyboard() -> InlineKeyboardMarkup:
    """Кнопки при загрузке фото для запуска пака."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Сбросить фото пака", callback_data="pl_persona_pack_reset_photos")],
        [InlineKeyboardButton("Назад к пакам", callback_data="pl_persona_packs")],
    ])


async def handle_persona_reset_photos_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сброс загруженных фото Персоны."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    context.user_data[USERDATA_PERSONA_PHOTOS] = []
    context.user_data[USERDATA_PERSONA_WAITING_UPLOAD] = True
    context.user_data[USERDATA_PERSONA_UPLOAD_MSG_IDS] = []
    await query.edit_message_text(PERSONA_UPLOAD_WAIT_MESSAGE, parse_mode="HTML")


async def handle_miniapp_pack_upload_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка из Mini App после оплаты пака: начинаем загрузку фото."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    # Формат: pl_pack_upload:{pack_id}:{credits}
    parts = query.data.split(":")
    pack_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    credits = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 20

    context.user_data[USERDATA_PERSONA_SELECTED_PACK_ID] = pack_id
    context.user_data[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = True
    context.user_data[USERDATA_PERSONA_PACK_PHOTOS] = []
    context.user_data[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = []
    context.user_data[USERDATA_MODE] = "persona_pack_upload"

    await query.edit_message_text(
        PERSONA_PACK_UPLOAD_WAIT_MESSAGE,
        reply_markup=_persona_pack_upload_keyboard(),
        parse_mode="HTML",
    )


async def handle_persona_pack_reset_photos_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сброс загруженных фото для pack-run в dev-режиме."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    context.user_data[USERDATA_PERSONA_PACK_PHOTOS] = []
    context.user_data[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = True
    context.user_data[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = []
    context.user_data[USERDATA_MODE] = "persona_pack_upload"
    await query.edit_message_text(
        PERSONA_PACK_UPLOAD_WAIT_MESSAGE,
        reply_markup=_persona_pack_upload_keyboard(),
        parse_mode="HTML",
    )


async def handle_persona_check_status_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Проверить статус Персоны»: проверяет статус, при pending — опрашивает tune (восстановление после рестарта)."""
    query = update.callback_query
    if not query:
        return
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    pending_tune_id = getattr(profile, "astria_lora_tune_id_pending", None)

    # Восстановление: есть pending tune (бот рестартовал во время обучения)
    if pending_tune_id:
        await query.answer("Проверяю статус…", show_alert=False)
        try:
            from prismalab.astria_client import _get_tune, _timeout_s
            settings = load_settings()
            last = await asyncio.to_thread(
                _get_tune,
                api_key=settings.astria_api_key,
                tune_id=pending_tune_id,
                timeout_s=_timeout_s(30.0),
            )
            status = str(last.get("status") or last.get("state") or "").lower()
            trained_at = last.get("trained_at")
            if status in {"completed", "succeeded", "ready", "trained", "finished"} or trained_at:
                store.set_astria_lora_tune(user_id=user_id, tune_id=pending_tune_id)
                context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "done"
                credits = profile.persona_credits_remaining
                gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
                text = f"Готово! 🎉 Персональная модель обучена\n\nТеперь выберите стиль из списка ниже – у вас {credits} {_fast_credits_word(credits)}\n\n{STYLE_EXAMPLES_FOOTER}"
                await query.edit_message_text(
                    text,
                    reply_markup=_persona_styles_keyboard(gender),
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
                return
            if status in {"failed", "error", "cancelled"}:
                store.clear_astria_lora_tune_pending(user_id)
                context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "error"
                await query.edit_message_text(
                    "При обучении возникла ошибка. Загрузите 10 фото заново или напишите в поддержку.",
                    parse_mode="HTML",
                )
                return
        except Exception as e:
            logger.warning("Ошибка проверки pending tune %s: %s", pending_tune_id, e)

    status = context.user_data.get(USERDATA_PERSONA_TRAINING_STATUS) or "training"
    if status == "training":
        await query.answer(
            "Модель ещё обучается ⏳ Обычно это занимает около 10 минут. Напишу, когда будет готово.",
            show_alert=True,
        )
    elif status == "error":
        await query.answer(
            "При обучении возникла ошибка. Напиши нам в поддержку — разберёмся.",
            show_alert=True,
        )
    else:
        await query.answer("Модель готова! Скоро появится возможность генерировать.", show_alert=True)


async def handle_persona_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Переключение страницы стилей Персоны (← Пред / След →)."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    _, page_str = query.data.split(":", 1)
    if page_str == "noop":
        return
    try:
        page = int(page_str)
    except ValueError:
        return
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    credits = profile.persona_credits_remaining
    gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
    text = f"<b>Выберите стиль из списка ниже</b> 👇\n\n{_format_balance_persona(credits)}\n\n{STYLE_EXAMPLES_FOOTER}"
    await query.edit_message_text(
        text,
        reply_markup=_persona_styles_keyboard(gender, page),
        parse_mode="HTML",
        disable_web_page_preview=True,
    )
    context.user_data[USERDATA_PERSONA_STYLE_MSG_ID] = query.message.message_id
    context.user_data[USERDATA_PERSONA_STYLE_PAGE] = page


async def handle_persona_style_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Выбор стиля в Персоне: генерация через Astria LoRA, списание кредита."""
    query = update.callback_query
    if not query or not query.data or "pl_persona_style:" not in query.data:
        return
    _, style_id = query.data.split(":", 1)
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    credits = profile.persona_credits_remaining
    if credits <= 0:
        await query.answer()
        gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
        label = next((l for l, s in (PERSONA_STYLES_FEMALE if gender == "female" else PERSONA_STYLES_MALE) if s == style_id), style_id)
        context.user_data[USERDATA_PERSONA_SELECTED_STYLE] = (style_id, label)
        text, kb = _persona_credits_out_content(profile)
        await query.edit_message_text(text, reply_markup=kb, parse_mode="HTML")
        return
    gen_lock = await _acquire_user_generation_lock(user_id)
    if gen_lock is None:
        await query.answer("Пожалуйста, немного подождите. Ещё обрабатываю прошлый запрос", show_alert=True)
        return
    await query.answer()
    context.user_data.pop(USERDATA_PERSONA_STYLE_MSG_ID, None)
    try:
        await context.bot.delete_message(chat_id=query.message.chat_id, message_id=query.message.message_id)
    except Exception:
        pass
    gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
    label = next((l for l, s in (PERSONA_STYLES_FEMALE if gender == "female" else PERSONA_STYLES_MALE) if s == style_id), style_id)
    prompt = _persona_style_prompt(style_id, label)
    status_msg = await context.bot.send_message(chat_id=query.message.chat_id, text="🎨 <i>Создаю изображение...</i>", parse_mode="HTML")
    settings = load_settings()

    async def runner() -> None:
        try:
            await _run_style_job(
                bot=context.bot,
                chat_id=query.message.chat_id,
                photo_file_ids=[],
                style_id=style_id,
                settings=settings,
                status_message_id=status_msg.message_id,
                prompt_strength=0.7,
                user_id=user_id,
                subject_gender=gender,
                use_personal_requested=False,
                test_prompt=None,
                lora_prompt_override=prompt,
                style_title_override=label,
                is_persona_style=True,
                context=context,
            )
        finally:
            gen_lock.release()

    context.application.create_task(runner())


async def handle_fast_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Переключение страницы стилей Экспресс-фото (← Пред / След →)."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    parts = (query.data or "").split(":")
    if len(parts) < 2 or parts[1] == "noop":
        return
    try:
        page = int(parts[1])
    except ValueError:
        return
    ctx = int(parts[2]) if len(parts) > 2 else 0  # 0=main, 1=back_to_ready, 2=from_profile

    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or profile.subject_gender or "female"
    credits = _generations_count_fast(profile)
    credits_word = _fast_credits_word(credits)
    has_photo = bool(context.user_data.get(USERDATA_PHOTO_FILE_IDS))
    text = _fast_style_screen_text(credits, credits_word, has_photo=has_photo)

    include_tariffs = credits > 0
    back_to_ready = ctx == 1
    from_profile = ctx == 2
    context.user_data[USERDATA_FAST_STYLE_PAGE] = page
    reply_markup = _fast_style_choice_keyboard(
        gender, include_tariffs=include_tariffs, back_to_ready=back_to_ready, from_profile=from_profile, page=page
    )
    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML", disable_web_page_preview=True)


async def handle_fast_gender_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """После выбора пола в «Быстрое фото»: сохраняем пол в профиль, затем загрузка фото или тарифы."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    # pl_fast_gender:female или pl_fast_gender:male
    _, gender = query.data.split(":", 1)
    context.user_data[USERDATA_SUBJECT_GENDER] = gender
    user_id = int(query.from_user.id) if query.from_user else 0
    store.set_subject_gender(user_id, gender)  # запоминаем пол один раз (смена потом в профиле)
    profile = store.get_user(user_id)
    context.user_data[USERDATA_MODE] = "fast"
    has_photo = bool(context.user_data.get(USERDATA_PHOTO_FILE_IDS))
    text, reply_markup = _fast_after_gender_content(profile, gender=gender, has_photo=has_photo)
    if text is not None:
        extra = {"parse_mode": "HTML", "disable_web_page_preview": True}
        await query.edit_message_text(text, reply_markup=reply_markup, **extra)
        context.user_data[USERDATA_FAST_STYLE_MSG_ID] = query.message.message_id
    else:
        await _send_fast_tariffs_two_messages(context.bot, query.message.chat_id, context, edit_message=query.message)


async def handle_fast_back_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Назад» / «Главное меню»: возврат на стартовый экран, сброс флоу Персоны при выходе."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    chat_id = query.message.chat_id if query.message else 0
    bot = context.bot

    # Выход из Примеров работ: удаляем intro и media, nav редактируем в главное меню (PAGE сохраняем для возврата)
    intro_msg_id = context.user_data.pop(USERDATA_EXAMPLES_INTRO_MSG_ID, None)
    media_ids = context.user_data.pop(USERDATA_EXAMPLES_MEDIA_IDS, None) or []
    context.user_data.pop(USERDATA_EXAMPLES_NAV_MSG_ID, None)
    if intro_msg_id is not None or media_ids:
        to_delete = []
        if intro_msg_id is not None and intro_msg_id != query.message.message_id:
            to_delete.append(intro_msg_id)
        to_delete.extend(media_ids)
        if to_delete:
            await asyncio.gather(*[bot.delete_message(chat_id=chat_id, message_id=mid) for mid in to_delete], return_exceptions=True)

    context.user_data.pop(USERDATA_FAST_SELECTED_STYLE, None)
    context.user_data.pop(USERDATA_FAST_CUSTOM_PROMPT, None)
    context.user_data.pop(USERDATA_FAST_LAST_MSG_ID, None)
    context.user_data.pop(USERDATA_FAST_STYLE_MSG_ID, None)
    persona_msg_id = context.user_data.pop(USERDATA_FAST_PERSONA_MSG_ID, None)
    if persona_msg_id is not None and persona_msg_id != query.message.message_id:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=persona_msg_id)
        except Exception:
            pass
    _clear_persona_flow_state(context)
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    text = _start_message_text(profile)
    kb = _start_keyboard(profile)
    try:
        await query.edit_message_text(text, reply_markup=kb, parse_mode="HTML")
    except BadRequest as e:
        logger.warning("handle_fast_back: edit_message_text failed (%s), sending new message", e)
        try:
            await query.message.reply_text(text, reply_markup=kb, parse_mode="HTML")
        except Exception as send_err:
            logger.error("handle_fast_back: send failed: %s", send_err)
            raise


async def handle_fast_show_tariffs_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Назад» с экрана оплаты Экспресс: возврат к выбору тарифа. Редактируем текущее сообщение — первое (Персона) уже в чате."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    await query.edit_message_text(
        FAST_TARIFFS_TARIFFS_MESSAGE,
        reply_markup=_fast_tariff_packages_keyboard(),
        parse_mode="HTML",
    )
    context.user_data[USERDATA_FAST_STYLE_MSG_ID] = query.message.message_id


async def handle_fast_buy_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Нажатие на пакет генераций (5/10/30). Telegram Payments (инвойс) или ЮKassa или симуляция."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    _, count_str = query.data.split(":", 1)
    try:
        count = int(count_str)
    except ValueError:
        count = 5
    if count not in (5, 10, 30):
        count = 5
    user_id = int(query.from_user.id) if query.from_user else 0
    chat_id = query.message.chat_id if query.message else 0

    if use_yookassa():
        amount = _amount_rub("fast", count)
        me = await context.bot.get_me()
        return_url = f"https://t.me/{me.username}" if me and me.username else None
        url, payment_id = create_payment(
            amount_rub=amount,
            description=f"Экспресс-фото: {count} кредитов",
            metadata={
                "user_id": str(user_id),
                "chat_id": str(chat_id),
                "credits": str(count),
                "product_type": "fast",
            },
            return_url=return_url,
        )
        if url and payment_id:
            await query.edit_message_text(
                f"<b>{count} кредитов Экспресс за {amount:.0f} ₽</b>\n\nОплатите удобным способом, кредиты зачислим моментально",
                parse_mode="HTML",
                reply_markup=_payment_yookassa_keyboard(url, "pl_fast_show_tariffs"),
            )
            # Запускаем поллинг статуса в фоне
            asyncio.create_task(poll_payment_status(
                payment_id=payment_id,
                bot=context.bot,
                store=store,
                user_id=user_id,
                chat_id=chat_id,
                credits=count,
                product_type="fast",
                amount_rub=amount,
            ))
            return
        else:
            logger.warning("Ошибка создания платежа (fast): %s", payment_id)
            asyncio.create_task(alert_payment_error(user_id, "fast", str(payment_id)))
            await query.edit_message_text(f"❌ Ошибка оплаты: {payment_id}")
            return

    if use_telegram_payments():
        payload = f"pl:fast:{count}:{user_id}"
        try:
            await context.bot.send_invoice(
                chat_id=chat_id,
                title="Экспресс-фото",
                description=f"Экспресс-фото: {count} кредитов",
                payload=payload,
                provider_token=TELEGRAM_PROVIDER_TOKEN,
                currency="RUB",
                prices=[LabeledPrice(label="Оплата", amount=INVOICE_AMOUNT_KOPECKS)],
            )
        except BadRequest as e:
            logger.exception("send_invoice (fast) BadRequest: %s", e)
            await query.edit_message_text("❌ Не удалось отправить счёт. Проверьте в BotFather → Payments, что подключена платёжная система.")
        return

    profile = store.get_user(user_id)
    new_total = profile.paid_generations_remaining + count
    store.set_paid_generations_remaining(user_id, new_total)
    context.user_data[USERDATA_MODE] = "fast"
    profile = store.get_user(user_id)
    gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or profile.subject_gender or "female"
    credits = _generations_count_fast(profile)
    photo_file_ids = list(context.user_data.get(USERDATA_PHOTO_FILE_IDS, []))
    selected_style = context.user_data.pop(USERDATA_FAST_SELECTED_STYLE, None)
    context.user_data.pop(USERDATA_FAST_CUSTOM_PROMPT, None)
    if photo_file_ids:
        # Если был выбран стиль при 0 кредитов — сразу генерируем
        if selected_style and selected_style != "custom":
            style_label = _fast_style_label(selected_style)
            prompt = _persona_style_prompt(selected_style, style_label)
            chat_id = query.message.chat_id
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=query.message.message_id)
            except Exception:
                pass
            status_msg = await context.bot.send_message(chat_id=chat_id, text="🎨 <i>Создаю изображение...</i>", parse_mode="HTML")
            context.application.create_task(
                _run_fast_generation_impl(
                    context=context,
                    chat_id=chat_id,
                    user_id=user_id,
                    style_id=selected_style,
                    style_label=style_label,
                    prompt=prompt,
                    photo_file_ids=photo_file_ids,
                    profile=profile,
                    status_msg=status_msg,
                )
            )
            return
        page = context.user_data.get(USERDATA_FAST_STYLE_PAGE, 0)
        text = f"Оплата получена ✅\n\n{_format_balance_express(credits)}\n\n<b>1 кредит = 1 фото</b>\n\n<b>Выберите стиль</b> или введите <b>свой запрос</b> 👇\n\n{STYLE_EXAMPLES_FOOTER}"
        await query.edit_message_text(
            text,
            reply_markup=_fast_style_choice_keyboard(gender, include_tariffs=True, back_to_ready=True, page=page),
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        context.user_data[USERDATA_FAST_STYLE_MSG_ID] = query.message.message_id
    else:
        styles = FAST_STYLES_FEMALE if gender == "female" else FAST_STYLES_MALE
        style_label, style_id = styles[0]
        context.user_data[USERDATA_FAST_SELECTED_STYLE] = style_id
        text = _fast_ready_to_upload_text(credits, style_label, after_payment=True)
        await query.edit_message_text(
            text,
            reply_markup=_fast_upload_or_change_keyboard(),
            parse_mode="HTML",
        )


async def handle_pre_checkout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Подтверждение перед списанием (Telegram Payments)."""
    try:
        query = update.pre_checkout_query
        if not query:
            return
        payload = (query.invoice_payload or "").strip()
        if payload.startswith(INVOICE_PAYLOAD_PREFIX):
            await query.answer(ok=True)
        else:
            await query.answer(ok=False, error_message="Неизвестный тип платежа")
    except Exception as e:
        logger.exception("PreCheckout ошибка: %s", e)
        if update.pre_checkout_query:
            try:
                await update.pre_checkout_query.answer(ok=False, error_message="Ошибка. Попробуйте позже.")
            except Exception:
                pass
        raise


async def handle_successful_payment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Успешная оплата через Telegram Payments: начисляем кредиты."""
    try:
        msg = update.message
        if not msg or not msg.successful_payment:
            return
        payload = (msg.successful_payment.invoice_payload or "").strip()
        if not payload.startswith(INVOICE_PAYLOAD_PREFIX):
            return
        parts = payload.split(":")
        if len(parts) != 4:  # pl, product_type, credits, user_id
            logger.warning("Неверный payload успешной оплаты: %s", payload)
            return
        _, product_type, credits_str, user_id_str = parts
        try:
            credits = int(credits_str)
            user_id = int(user_id_str)
        except ValueError:
            logger.warning("Не удалось распарсить payload: %s", payload)
            return

        # Логируем платёж для аналитики
        try:
            amount_rub = float(msg.successful_payment.total_amount) / 100
            store.log_payment(
                user_id=user_id,
                payment_id=msg.successful_payment.telegram_payment_charge_id,
                payment_method="telegram",
                product_type=product_type,
                credits=credits,
                amount_rub=amount_rub,
            )
            logger.info("Платёж записан: user=%s, amount=%.2f, type=%s", user_id, amount_rub, product_type)
        except Exception as e:
            logger.exception("Ошибка записи платежа в БД: %s", e)

        if product_type == "fast":
            context.user_data[USERDATA_MODE] = "fast"
            profile = store.get_user(user_id)
            new_total = profile.paid_generations_remaining + credits
            store.set_paid_generations_remaining(user_id, new_total)
            text = f"Оплата получена ✅\n\n{_format_balance_express(_generations_count_fast(store.get_user(user_id)))}\n\n<b>1 кредит = 1 фото</b>\n\n<b>Выберите стиль</b> или введите <b>свой запрос</b> 👇\n\n{STYLE_EXAMPLES_FOOTER}"
            gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or store.get_user(user_id).subject_gender or "female"
            reply_msg = await msg.reply_text(
                text,
                reply_markup=_fast_style_choice_keyboard(gender, include_tariffs=True, back_to_ready=True, page=0),
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            context.user_data[USERDATA_FAST_STYLE_MSG_ID] = reply_msg.message_id
            context.user_data[USERDATA_FAST_STYLE_PAGE] = 0
        elif product_type == "persona_topup":
            context.user_data[USERDATA_MODE] = "persona"
            profile = store.get_user(user_id)
            new_total = profile.persona_credits_remaining + credits
            store.set_persona_credits(user_id, new_total)
            gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
            text = f"Оплата получена ✅\n\n<b>Выберите стиль</b> 👇\n\n{_format_balance_persona(new_total)}\n\n{STYLE_EXAMPLES_FOOTER}"
            await msg.reply_text(
                text,
                reply_markup=_persona_styles_keyboard(gender, page=0),
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        elif product_type == "persona_create":
            context.user_data[USERDATA_MODE] = "persona"
            store.set_persona_credits(user_id, credits)
            if context.user_data.pop(USERDATA_PERSONA_RECREATING, None):
                store.set_astria_lora_tune(user_id=user_id, tune_id=None)
            await msg.reply_text(
                PERSONA_RULES_MESSAGE,
                reply_markup=_persona_rules_keyboard(),
                parse_mode="HTML",
            )
        else:
            logger.warning("Неизвестный product_type в оплате: %s", product_type)
    except Exception as e:
        logger.exception("handle_successful_payment ошибка: %s", e)
        raise


async def handle_fast_upload_photo_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Загрузить фото»: показываем правила загрузки."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    await query.edit_message_text(
        FAST_PHOTO_RULES_MESSAGE,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="pl_fast_show_ready")]]),
        parse_mode="HTML",
    )


async def handle_fast_change_style_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Поменять стиль»: показываем выбор стиля."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or profile.subject_gender or "female"
    credits = _generations_count_fast(profile)
    credits_word = _fast_credits_word(credits)
    has_photo = bool(context.user_data.get(USERDATA_PHOTO_FILE_IDS))
    page = context.user_data.get(USERDATA_FAST_STYLE_PAGE, 0)
    text = _fast_style_screen_text(credits, credits_word, has_photo=has_photo)
    await query.edit_message_text(
        text,
        reply_markup=_fast_style_choice_keyboard(gender, include_tariffs=True, back_to_ready=True, page=page),
        parse_mode="HTML",
        disable_web_page_preview=True,
    )
    context.user_data[USERDATA_FAST_STYLE_MSG_ID] = query.message.message_id


async def handle_fast_show_ready_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Возврат на экран «Загрузить фото / Поменять стиль» (из правил)."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or profile.subject_gender or "female"
    style_id = context.user_data.get(USERDATA_FAST_SELECTED_STYLE)
    if not style_id:
        styles = FAST_STYLES_FEMALE if gender == "female" else FAST_STYLES_MALE
        style_label, style_id = styles[0]
        context.user_data[USERDATA_FAST_SELECTED_STYLE] = style_id
    else:
        style_label = _fast_style_label(style_id)
    credits = _generations_count_fast(profile)
    text = _fast_ready_to_upload_text(credits, style_label, after_payment=False)
    await query.edit_message_text(
        text,
        reply_markup=_fast_upload_or_change_keyboard(),
        parse_mode="HTML",
    )


async def _run_fast_generation_impl(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    style_id: str,
    style_label: str,
    prompt: str,
    photo_file_ids: list[str],
    profile: Any,
    status_msg: Any,
    status_prefix: str = "",
) -> None:
    """Выполняет генерацию Быстрое фото и отправляет результат + сообщение с кнопками стилей."""
    status_message_id = status_msg.message_id
    bot = context.bot
    settings = load_settings()
    total_timeout = settings.kie_max_seconds + 120  # KIE + буфер на загрузку/скачивание
    prefix = status_prefix or ""

    async def _do_generation() -> None:
        if not settings.kie_api_key:
            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=f"{prefix}{USER_FRIENDLY_ERROR}")
            return
        status_text = f"{prefix}🎨 <i>Создаю изображение...</i>"
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=status_text, parse_mode="HTML")
        except BadRequest as e:
            if "message is not modified" not in str(e).lower():
                raise
            photo_bytes = await _safe_get_file_bytes(bot, photo_file_ids[-1])
            if not photo_bytes:
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=f"{prefix}{USER_FRIENDLY_ERROR}")
                return
            random_id = secrets.token_hex(8)
            uploaded_url = None
            for upload_attempt in range(3):
                try:
                    uploaded_url = await asyncio.to_thread(
                        kie_upload_file_base64,
                        api_key=settings.kie_api_key,
                        image_bytes=photo_bytes,
                        file_name=f"fast_{random_id}.jpg",
                        upload_path="user-uploads",
                        timeout_s=90.0,
                    )
                    break
                except Exception as up_e:
                    err_str = str(up_e).lower()
                    if "connection" in err_str or "remote" in err_str or "disconnected" in err_str:
                        logger.warning("Быстрое фото: загрузка в KIE попытка %s/3: %s", upload_attempt + 1, up_e)
                        if upload_attempt < 2:
                            await asyncio.sleep(4.0)
                        else:
                            raise
                    else:
                        raise
            if not uploaded_url:
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=f"{prefix}{USER_FRIENDLY_ERROR}")
                return
            kie_result = None
            for gen_attempt in range(2):
                try:
                    kie_result = await kie_run_task_and_wait(
                        api_key=settings.kie_api_key,
                        model="seedream/4.5-edit",
                        prompt=prompt,
                        image_input=[uploaded_url],
                        aspect_ratio="1:1",
                    quality="basic",
                    output_format="jpg",
                    max_seconds=settings.kie_max_seconds,
                        poll_seconds=3.0,
                    )
                    break
                except Exception as gen_e:
                    err_str = str(gen_e).lower()
                    if "connection" in err_str or "remote" in err_str or "disconnected" in err_str:
                        logger.warning("Быстрое фото: вызов Seedream попытка %s/2: %s", gen_attempt + 1, gen_e)
                        if gen_attempt < 1:
                            await asyncio.sleep(5.0)
                        else:
                            raise
                    else:
                        raise
            if not kie_result.image_url:
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=f"{prefix}{USER_FRIENDLY_ERROR}")
                return
            image_bytes = await asyncio.to_thread(kie_download_image_bytes, kie_result.image_url, timeout_s=30.0)
            bio = io.BytesIO(image_bytes)
            bio.name = f"fast_{style_id}.jpg"
            await bot.delete_message(chat_id=chat_id, message_id=status_message_id)
        await _safe_send_document(bot=bot, chat_id=chat_id, document=bio, caption=f"Экспресс-фото: {style_label}")
        # Логируем успешную генерацию для аналитики
        try:
            store.log_event(user_id, "generation", {"mode": "fast", "style": style_id, "provider": "kie"})
        except Exception:
            pass
        if not profile.free_generation_used:
            store.spend_free_generation(user_id)
        elif profile.paid_generations_remaining > 0:
            store.set_paid_generations_remaining(user_id, profile.paid_generations_remaining - 1)
        updated_profile = store.get_user(user_id)
        credits = _generations_count_fast(updated_profile)
        gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
        context.user_data.pop(USERDATA_FAST_SELECTED_STYLE, None)  # чтобы загрузка нового фото → выбор стиля, а не авто-генерация
        if credits <= 0:
            await _send_fast_tariffs_two_messages(bot, chat_id, context)
        else:
            credits_word = _fast_credits_word(credits)
            page = context.user_data.get(USERDATA_FAST_STYLE_PAGE, 0)
            text = f"<b>Можете не загружать фото заново</b> – просто выберите другой стиль для этого же снимка\n\nЕсли хотите – <b>загрузите новое</b> 👇\n\n{_format_balance_express(credits)}\n\n<b>1 кредит = 1 фото</b>\n\n{STYLE_EXAMPLES_FOOTER}"
            reply_markup = _fast_style_choice_keyboard(gender, include_tariffs=True, back_to_ready=True, page=page)
            msg = await bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            await _update_fast_style_message(context, chat_id, msg)

    try:
        start_time = time.time()
        await asyncio.wait_for(_do_generation(), timeout=float(total_timeout))
        duration = time.time() - start_time
        # Алерт о медленной генерации (> 5 минут)
        if duration > 300:
            await alert_slow_generation(user_id, duration, "express")
    except asyncio.TimeoutError:
        logger.warning("Быстрое фото: таймаут %sс (стиль %s)", total_timeout, style_id)
        await alert_generation_error(user_id, f"Таймаут {total_timeout}с", "express")
        try:
            await bot.edit_message_text(
                chat_id=chat_id, message_id=status_message_id,
                text=f"{prefix}{USER_FRIENDLY_ERROR}",
            )
        except Exception:
            await bot.send_message(chat_id=chat_id, text=USER_FRIENDLY_ERROR)
        except KieError as e:
            logger.error("Быстрое фото KIE: %s", e)
        await alert_generation_error(user_id, str(e), "express")
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=f"{prefix}{USER_FRIENDLY_ERROR}")
        except Exception:
            await bot.send_message(chat_id=chat_id, text=USER_FRIENDLY_ERROR)
        except Exception as e:
            logger.exception("Быстрое фото (стиль %s): %s", style_id, e)
        await alert_generation_error(user_id, str(e), "express")
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=f"{prefix}{USER_FRIENDLY_ERROR}")
        except Exception:
            try:
                await bot.send_message(chat_id=chat_id, text=USER_FRIENDLY_ERROR)
            except Exception as send_err:
                logger.error("Быстрое фото: не удалось отправить сообщение об ошибке: %s", send_err)
    # Lock освобождается в caller через gen_lock.release()


async def handle_fast_style_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Выбор стиля: либо «загрузите фото» (если фото нет), либо генерация с имеющимся фото. Для «Свой запрос» — сначала ввод текста."""
    query = update.callback_query
    if not query or not query.data or "pl_fast_style:" not in query.data:
        return
    await query.answer()
    _, style_id = query.data.split(":", 1)
    style_label = _fast_style_label(style_id)
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    if _generations_count_fast(profile) <= 0:
        context.user_data[USERDATA_FAST_SELECTED_STYLE] = style_id
        if style_id == "custom":
            context.user_data.pop(USERDATA_FAST_CUSTOM_PROMPT, None)  # не использовать старый промпт после пополнения
        await _send_fast_tariffs_two_messages(context.bot, query.message.chat_id, context, edit_message=query.message)
        return
    photo_file_ids = list(context.user_data.get(USERDATA_PHOTO_FILE_IDS, []))
    custom_prompt = (context.user_data.get(USERDATA_FAST_CUSTOM_PROMPT) or "").strip()

    if style_id == "custom":
        context.user_data[USERDATA_FAST_SELECTED_STYLE] = style_id
        context.user_data.pop(USERDATA_FAST_CUSTOM_PROMPT, None)  # каждый раз запрашиваем новый запрос
        custom_prompt = (context.user_data.get(USERDATA_FAST_CUSTOM_PROMPT) or "").strip()
        if not custom_prompt:
            await query.edit_message_text(
                FAST_CUSTOM_PROMPT_REQUEST_MESSAGE,
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="pl_fast_change_style")]]),
                parse_mode="HTML",
            )
            return
        if not photo_file_ids:
            await query.edit_message_text(
                FAST_PHOTO_RULES_MESSAGE,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("Назад", callback_data="pl_fast_change_style"),
                ]]),
                parse_mode="HTML",
            )
            return
        gen_lock = await _acquire_user_generation_lock(user_id)
        if gen_lock is None:
            await query.edit_message_text("Пожалуйста, немного подождите. Ещё обрабатываю прошлый запрос")
            return
        prompt = custom_prompt
        chat_id = query.message.chat_id
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=query.message.message_id)
        except Exception:
            pass
        status_msg = await context.bot.send_message(chat_id=chat_id, text="🎨 <i>Создаю изображение...</i>", parse_mode="HTML")

        async def _run_and_release() -> None:
            try:
                await _run_fast_generation_impl(
                    context=context,
                    chat_id=chat_id,
                    user_id=user_id,
                    style_id=style_id,
                    style_label=style_label,
                    prompt=prompt,
                    photo_file_ids=photo_file_ids,
                    profile=profile,
                    status_msg=status_msg,
                )
            finally:
                gen_lock.release()

        context.application.create_task(_run_and_release())
        return

    if not photo_file_ids:
        context.user_data[USERDATA_FAST_SELECTED_STYLE] = style_id
        await query.edit_message_text(
            FAST_PHOTO_RULES_MESSAGE,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("Назад", callback_data="pl_fast_change_style"),
            ]]),
            parse_mode="HTML",
        )
        return
    gen_lock = await _acquire_user_generation_lock(user_id)
    if gen_lock is None:
        await query.edit_message_text("Пожалуйста, немного подождите. Ещё обрабатываю прошлый запрос")
        return
    prompt = _persona_style_prompt(style_id, style_label)
    chat_id = query.message.chat_id
    gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or profile.subject_gender or "female"
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=query.message.message_id)
    except Exception:
        pass
    status_msg = await context.bot.send_message(chat_id=chat_id, text="🎨 <i>Создаю изображение...</i>", parse_mode="HTML")

    async def _run_and_release() -> None:
        try:
            await _run_fast_generation_impl(
                context=context,
                chat_id=chat_id,
                user_id=user_id,
                style_id=style_id,
                style_label=style_label,
                prompt=prompt,
                photo_file_ids=photo_file_ids,
                profile=profile,
                status_msg=status_msg,
            )
        finally:
            gen_lock.release()

    context.application.create_task(_run_and_release())


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Приём своего запроса в Экспресс-фото (стиль «Свой запрос»)."""
    if not update.message or not update.message.text:
        return
    user_id = int(update.effective_user.id) if update.effective_user else 0
    if ALLOWED_USERS and user_id not in ALLOWED_USERS:
        return
    mode = context.user_data.get(USERDATA_MODE) or "normal"
    if mode != "fast":
        return
    selected_style = context.user_data.get(USERDATA_FAST_SELECTED_STYLE)
    if selected_style == "custom":
        text = (update.message.text or "").strip()
        if not text:
            await update.message.reply_text("Напишите непустое описание картинки.")
            return
        context.user_data[USERDATA_FAST_CUSTOM_PROMPT] = text[:2000]
        photo_file_ids = list(context.user_data.get(USERDATA_PHOTO_FILE_IDS, []))
        if photo_file_ids:
            user_id = int(update.effective_user.id) if update.effective_user else 0
            profile = store.get_user(user_id)
            if _generations_count_fast(profile) <= 0:
                chat_id = update.effective_chat.id if update.effective_chat else 0
                await _send_fast_tariffs_two_messages(context.bot, chat_id, context)
                return
            gen_lock = await _acquire_user_generation_lock(user_id)
            if gen_lock is None:
                await update.message.reply_text("Пожалуйста, немного подождите. Ещё обрабатываю прошлый запрос")
                return
            chat_id = update.effective_chat.id if update.effective_chat else 0
            status_msg = await update.message.reply_text("🎨 <i>Создаю изображение...</i>", parse_mode="HTML")

            async def _run_and_release() -> None:
                try:
                    await _run_fast_generation_impl(
                        context=context,
                        chat_id=chat_id,
                        user_id=user_id,
                        style_id="custom",
                        style_label="Свой запрос",
                        prompt=text[:2000],
                        photo_file_ids=photo_file_ids,
                        profile=profile,
                        status_msg=status_msg,
                    )
                finally:
                    gen_lock.release()

            context.application.create_task(_run_and_release())
        else:
            await update.message.reply_text(
                "Запрос принят ✅ Теперь загрузите фото по правилам ниже.\n\n" + FAST_PHOTO_RULES_MESSAGE,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("Назад", callback_data="pl_fast_change_style"),
                ]]),
                parse_mode="HTML",
            )
    else:
        await update.message.reply_text(
            "Чтобы отправить свой запрос, нажмите «✏️ Свой запрос» в меню стилей.",
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return

    user_id = int(update.effective_user.id) if update.effective_user else 0
    if ALLOWED_USERS and user_id not in ALLOWED_USERS:
        return
    album_name = context.user_data.pop(USERDATA_GETFILEID_EXPECTING_PHOTO, None)
    if OWNER_ID and user_id == OWNER_ID and album_name:
        photo = update.message.photo[-1]
        albums = _load_examples_albums()
        found = next((a for a in albums if (a.get("caption") or "").strip() == album_name), None)
        if found:
            ids_list = found.setdefault("file_ids", [])
            if len(ids_list) < 10:
                ids_list.append(photo.file_id)
                _save_examples_albums(albums)
                await update.message.reply_text(f"Добавлено в «{album_name}» ({len(ids_list)}/10)")
            else:
                await update.message.reply_text(f"В «{album_name}» уже 10 фото — максимум")
        else:
            albums.append({"caption": album_name, "file_ids": [photo.file_id]})
            _save_examples_albums(albums)
            await update.message.reply_text(f"Создан альбом «{album_name}» и добавлено фото")
        return

    mode = context.user_data.get(USERDATA_MODE) or "normal"
    # Fallback: Mini App оплата — user_data не доходит из webhook/poll, проверяем БД
    if mode != "persona_pack_upload":
        pending_pack_id = store.get_pending_pack_upload(user_id)
        if pending_pack_id is not None:
            context.user_data[USERDATA_MODE] = "persona_pack_upload"
            context.user_data[USERDATA_PERSONA_SELECTED_PACK_ID] = pending_pack_id
            context.user_data[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = True
            context.user_data[USERDATA_PERSONA_PACK_PHOTOS] = []
            context.user_data[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = []
            mode = "persona_pack_upload"
    logger.info(f"[Photo Handler] Режим: {mode}, user {update.effective_user.id}")
    photo = update.message.photo[-1]  # самое большое

    if photo.file_size is not None and photo.file_size > MAX_IMAGE_SIZE_BYTES:
        await update.message.reply_text(
            f"Слишком большой файл ({photo.file_size // (1024 * 1024)} МБ). Максимум 15 МБ. Сожмите фото или отправьте в меньшем разрешении."
        )
        return

    if mode == "persona_pack_upload" and context.user_data.get(USERDATA_PERSONA_PACK_WAITING_UPLOAD):
        ids = list(context.user_data.get(USERDATA_PERSONA_PACK_PHOTOS, []))
        if len(ids) >= 10:
            await update.message.reply_text("Уже загружено 10 фото для пака. Нажмите «Сбросить фото пака» или дождитесь результата.")
            return
        ids.append(photo.file_id)
        context.user_data[USERDATA_PERSONA_PACK_PHOTOS] = ids
        count = len(ids)
        if count < 10:
            text = f"Фото для пака {count}/10 получено. Осталось {10 - count}."
            msg = await update.message.reply_text(text, reply_markup=_persona_pack_upload_keyboard())
            upload_msg_ids = list(context.user_data.get(USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS, []))
            upload_msg_ids.append(msg.message_id)
            context.user_data[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = upload_msg_ids
        else:
            context.user_data[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = False
            context.user_data[USERDATA_MODE] = "persona"
            store.clear_pending_pack_upload(user_id)
            await update.message.reply_text("Все 10 фото получил ✅\n\nЗапускаю генерацию пака…")
            pack_id = int(context.user_data.get(USERDATA_PERSONA_SELECTED_PACK_ID) or 0)
            offer = _find_pack_offer(pack_id)
            if not offer:
                await update.message.reply_text("❌ Не удалось найти выбранный пак. Откройте «Готовые фотопаки» и выберите заново.")
                return
            context.application.create_task(
                _run_persona_pack_generation(
                    context=context,
                    chat_id=update.effective_chat.id,
                    user_id=user_id,
                    pack_id=pack_id,
                    offer=offer,
                    train_file_ids=list(ids),
                )
            )
            return

    if mode == "persona" and context.user_data.get(USERDATA_PERSONA_WAITING_UPLOAD):
        ids = list(context.user_data.get(USERDATA_PERSONA_PHOTOS, []))
        if len(ids) >= 10:
            await update.message.reply_text("Уже загружено 10 фото. Жду или нажми «Сбросить и начать заново».")
            return
        ids.append(photo.file_id)
        context.user_data[USERDATA_PERSONA_PHOTOS] = ids
        count = len(ids)
        if count < 10:
            text = f"Фото {count}/10 получено. Осталось {10 - count}."
            kb = _persona_upload_keyboard() if count >= 1 else None
            msg = await update.message.reply_text(text, reply_markup=kb)
            upload_msg_ids = list(context.user_data.get(USERDATA_PERSONA_UPLOAD_MSG_IDS, []))
            upload_msg_ids.append(msg.message_id)
            context.user_data[USERDATA_PERSONA_UPLOAD_MSG_IDS] = upload_msg_ids
        else:
            chat_id = update.effective_chat.id
            for msg_id in context.user_data.get(USERDATA_PERSONA_UPLOAD_MSG_IDS, []):
                try:
                    await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
                except Exception as e:
                    logger.warning("Не удалось удалить сообщение загрузки: chat_id=%s msg_id=%s err=%s", chat_id, msg_id, e)
            context.user_data[USERDATA_PERSONA_UPLOAD_MSG_IDS] = []
            context.user_data[USERDATA_PERSONA_WAITING_UPLOAD] = False
            context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "training"
            context.user_data[USERDATA_ASTRIA_LORA_FILE_IDS] = list(ids)
            await update.message.reply_text(
                PERSONA_TRAINING_MESSAGE,
                reply_markup=_persona_training_keyboard(),
            )
            user_id = int(update.effective_user.id) if update.effective_user else 0
            context.application.create_task(
                _start_astria_lora(context, update.effective_chat.id, user_id, from_persona=True)
            )
        return

    # Режим Persona: оплатил, но не нажал «Всё понятно» — напомнить про правила
    if mode == "persona" and not context.user_data.get(USERDATA_PERSONA_WAITING_UPLOAD):
        user_id = int(update.effective_user.id) if update.effective_user else 0
        profile = store.get_user(user_id)
        credits = getattr(profile, "persona_credits_remaining", 0) or 0
        pending = getattr(profile, "astria_lora_tune_id_pending", None)
        if credits > 0 and not profile.astria_lora_tune_id:
            # Есть pending (обучение шло, бот рестартовал) — показать «Проверить статус»
            if pending:
                context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "training"
                await update.message.reply_text(
                    PERSONA_TRAINING_MESSAGE,
                    reply_markup=_persona_training_keyboard(),
                )
                return
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("Да, всё понятно!", callback_data="pl_persona_got_it")],
            ])
            await update.message.reply_text("Правила прочитали? 🫶", reply_markup=kb)
            return

    # Режим Persona (превью): показать редирект в Персону или Экспресс
    if mode == "persona":
        user_id = int(update.effective_user.id) if update.effective_user else 0
        profile = store.get_user(user_id)
        if profile.astria_lora_tune_id:
            text = (
                "Если хотите создать новую Персону, нажмите <b>«Создать персону»</b> и загрузите <b>10 фото</b>\n\n"
                "Или перейдите в <b>«Экспресс-фото»</b> для быстрой стилизации одного снимка"
            )
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("Создать новую Персону", callback_data="pl_persona_recreate")],
                [InlineKeyboardButton(_express_button_label(profile), callback_data="pl_start_fast")],
                [InlineKeyboardButton("Назад", callback_data="pl_start_persona")],
            ])
        else:
            text = (
                "Сначала создайте <b>Персону</b> и следуйте инструкциям\n\n"
                "Или перейдите в раздел <b>Экспресс-фото</b>"
            )
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("✨ Персона", callback_data="pl_persona_create")],
                [InlineKeyboardButton(_express_button_label(profile), callback_data="pl_start_fast")],
                [InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")],
            ])
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")
        return

    # Режим fast или fallback: есть генерации — обрабатываем как Быстрое фото
    user_id = int(update.effective_user.id) if update.effective_user else 0
    profile = store.get_user(user_id)
    use_fast = mode == "fast" or _generations_count_fast(profile) > 0
    if use_fast:
        context.user_data[USERDATA_MODE] = "fast"
        gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or profile.subject_gender or "female"
        context.user_data[USERDATA_SUBJECT_GENDER] = gender
        selected_style = context.user_data.pop(USERDATA_FAST_SELECTED_STYLE, None)
        if selected_style:
            if _generations_count_fast(profile) <= 0:
                context.user_data[USERDATA_FAST_SELECTED_STYLE] = selected_style
                chat_id = update.effective_chat.id if update.effective_chat else 0
                await _send_fast_tariffs_two_messages(context.bot, chat_id, context)
                return
            # Стиль уже выбран — сохраняем фото и запускаем генерацию
            gen_lock = await _acquire_user_generation_lock(user_id)
            if gen_lock is None:
                await update.message.reply_text("Пожалуйста, немного подождите. Ещё обрабатываю прошлый запрос")
                return
            style_id = selected_style
            style_label = _fast_style_label(style_id)
            if style_id == "custom":
                prompt = (context.user_data.get(USERDATA_FAST_CUSTOM_PROMPT) or "").strip()
                if not prompt:
                    gen_lock.release()
                    await update.message.reply_text(
                        "Сначала напишите текстом описание картинки (выберите «✏️ Свой запрос» и отправьте сообщение).",
                    )
                    context.user_data[USERDATA_FAST_SELECTED_STYLE] = style_id
                    return
            else:
                prompt = _persona_style_prompt(style_id, style_label)
            context.user_data[USERDATA_PHOTO_FILE_IDS] = [photo.file_id]
            last_msg_id = context.user_data.pop(USERDATA_FAST_LAST_MSG_ID, None)
            chat_id = update.effective_chat.id if update.effective_chat else 0
            if last_msg_id:
                try:
                    await context.bot.delete_message(chat_id=chat_id, message_id=last_msg_id)
                except Exception:
                    pass
            status_msg = await update.message.reply_text("🎨 <i>Создаю изображение...</i>", parse_mode="HTML")

            async def _run_and_release() -> None:
                try:
                    await _run_fast_generation_impl(
                        context=context,
                        chat_id=chat_id,
                        user_id=user_id,
                        style_id=style_id,
                        style_label=style_label,
                        prompt=prompt,
                        photo_file_ids=[photo.file_id],
                        profile=profile,
                        status_msg=status_msg,
                    )
                finally:
                    gen_lock.release()

            context.application.create_task(_run_and_release())
        else:
            # Новое фото — показываем выбор стиля
            context.user_data[USERDATA_PHOTO_FILE_IDS] = [photo.file_id]
            credits = _generations_count_fast(profile)
            credits_word = _fast_credits_word(credits)
            page = context.user_data.get(USERDATA_FAST_STYLE_PAGE, 0)
            text = f"Отлично! <b>Выберите стиль</b> или введите <b>свой запрос</b> 👇\n\n{_format_balance_express(credits)}\n\n<b>1 кредит = 1 фото</b>\n\n{STYLE_EXAMPLES_FOOTER}"
            msg = await update.message.reply_text(
                text,
                reply_markup=_fast_style_choice_keyboard(gender, include_tariffs=True, page=page),
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            chat_id = update.effective_chat.id if update.effective_chat else 0
            await _update_fast_style_message(context, chat_id, msg)
        return

    user_id = int(update.effective_user.id) if update.effective_user else 0
    profile = store.get_user(user_id)
    await update.message.reply_text(
        "Перед загрузкой фото нужно выбрать раздел: <b>Экспресс-фото</b> или <b>Персона</b>",
        reply_markup=_start_keyboard(profile),
        parse_mode="HTML",
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.document:
        return
    user_id = int(update.effective_user.id) if update.effective_user else 0
    if ALLOWED_USERS and user_id not in ALLOWED_USERS:
        return
    doc = update.message.document
    if not (doc.mime_type or "").startswith("image/"):
        await update.message.reply_text("Пришли картинку (файл должен быть изображением).")
        return
    if doc.file_size is not None and doc.file_size > MAX_IMAGE_SIZE_BYTES:
        await update.message.reply_text(
            f"Слишком большой файл ({doc.file_size // (1024 * 1024)} МБ). Максимум 15 МБ. Сожмите изображение."
        )
        return

    mode = context.user_data.get(USERDATA_MODE) or "normal"
    if mode != "persona_pack_upload":
        pending_pack_id = store.get_pending_pack_upload(user_id)
        if pending_pack_id is not None:
            context.user_data[USERDATA_MODE] = "persona_pack_upload"
            context.user_data[USERDATA_PERSONA_SELECTED_PACK_ID] = pending_pack_id
            context.user_data[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = True
            context.user_data[USERDATA_PERSONA_PACK_PHOTOS] = []
            context.user_data[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = []
            mode = "persona_pack_upload"
    if mode == "persona_pack_upload" and context.user_data.get(USERDATA_PERSONA_PACK_WAITING_UPLOAD):
        ids = list(context.user_data.get(USERDATA_PERSONA_PACK_PHOTOS, []))
        if len(ids) >= 10:
            await update.message.reply_text("Уже загружено 10 фото для пака. Нажмите «Сбросить фото пака» или дождитесь результата.")
            return
        ids.append(doc.file_id)
        context.user_data[USERDATA_PERSONA_PACK_PHOTOS] = ids
        count = len(ids)
        if count < 10:
            text = f"Фото для пака {count}/10 получено. Осталось {10 - count}."
            msg = await update.message.reply_text(text, reply_markup=_persona_pack_upload_keyboard())
            upload_msg_ids = list(context.user_data.get(USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS, []))
            upload_msg_ids.append(msg.message_id)
            context.user_data[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = upload_msg_ids
        else:
            context.user_data[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = False
            context.user_data[USERDATA_MODE] = "persona"
            store.clear_pending_pack_upload(user_id)
            await update.message.reply_text("Все 10 фото получил ✅\n\nЗапускаю генерацию пака…")
            pack_id = int(context.user_data.get(USERDATA_PERSONA_SELECTED_PACK_ID) or 0)
            offer = _find_pack_offer(pack_id)
            if not offer:
                await update.message.reply_text("❌ Не удалось найти выбранный пак. Откройте «Готовые фотопаки» и выберите заново.")
                return
            context.application.create_task(
                _run_persona_pack_generation(
                    context=context,
                    chat_id=update.effective_chat.id,
                    user_id=user_id,
                    pack_id=pack_id,
                    offer=offer,
                    train_file_ids=list(ids),
                )
            )
            return

    if mode == "persona" and context.user_data.get(USERDATA_PERSONA_WAITING_UPLOAD):
        ids = list(context.user_data.get(USERDATA_PERSONA_PHOTOS, []))
        if len(ids) >= 10:
            await update.message.reply_text("Уже загружено 10 фото. Жду или нажми «Сбросить и начать заново».")
            return
        ids.append(doc.file_id)
        context.user_data[USERDATA_PERSONA_PHOTOS] = ids
        count = len(ids)
        if count < 10:
            text = f"Фото {count}/10 получено. Осталось {10 - count}."
            kb = _persona_upload_keyboard() if count >= 1 else None
            msg = await update.message.reply_text(text, reply_markup=kb)
            upload_msg_ids = list(context.user_data.get(USERDATA_PERSONA_UPLOAD_MSG_IDS, []))
            upload_msg_ids.append(msg.message_id)
            context.user_data[USERDATA_PERSONA_UPLOAD_MSG_IDS] = upload_msg_ids
        else:
            chat_id = update.effective_chat.id
            for msg_id in context.user_data.get(USERDATA_PERSONA_UPLOAD_MSG_IDS, []):
                try:
                    await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
                except Exception as e:
                    logger.warning("Не удалось удалить сообщение загрузки: chat_id=%s msg_id=%s err=%s", chat_id, msg_id, e)
            context.user_data[USERDATA_PERSONA_UPLOAD_MSG_IDS] = []
            context.user_data[USERDATA_PERSONA_WAITING_UPLOAD] = False
            context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "training"
            context.user_data[USERDATA_ASTRIA_LORA_FILE_IDS] = list(ids)
            await update.message.reply_text(
                PERSONA_TRAINING_MESSAGE,
                reply_markup=_persona_training_keyboard(),
            )
            user_id = int(update.effective_user.id) if update.effective_user else 0
            context.application.create_task(
                _start_astria_lora(context, update.effective_chat.id, user_id, from_persona=True)
            )
        return

    # Режим Persona (превью): показать редирект в Персону или Экспресс
    if mode == "persona":
        _user_id = int(update.effective_user.id) if update.effective_user else 0
        _profile = store.get_user(_user_id)
        if _profile.astria_lora_tune_id:
            text = (
                "Если хотите создать новую Персону, нажмите <b>«Создать персону»</b> и загрузите <b>10 фото</b>\n\n"
                "Или перейдите в <b>«Экспресс-фото»</b> для быстрой стилизации одного снимка"
            )
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("Создать новую Персону", callback_data="pl_persona_recreate")],
                [InlineKeyboardButton(_express_button_label(_profile), callback_data="pl_start_fast")],
                [InlineKeyboardButton("Назад", callback_data="pl_start_persona")],
            ])
        else:
            text = (
                "Сначала создайте <b>Персону</b> и следуйте инструкциям\n\n"
                "Или перейдите в раздел <b>Экспресс-фото</b>"
            )
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("✨ Персона", callback_data="pl_persona_create")],
                [InlineKeyboardButton(_express_button_label(_profile), callback_data="pl_start_fast")],
                [InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")],
            ])
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")
        return

    # Режим fast или fallback: есть генерации — обрабатываем как Быстрое фото
    _user_id = int(update.effective_user.id) if update.effective_user else 0
    _profile = store.get_user(_user_id)
    _use_fast = mode == "fast" or _generations_count_fast(_profile) > 0
    if _use_fast:
        context.user_data[USERDATA_MODE] = "fast"
        gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or _profile.subject_gender or "female"
        context.user_data[USERDATA_SUBJECT_GENDER] = gender
        selected_style = context.user_data.pop(USERDATA_FAST_SELECTED_STYLE, None)
        if selected_style:
            if _generations_count_fast(_profile) <= 0:
                context.user_data[USERDATA_FAST_SELECTED_STYLE] = selected_style
                _chat_id = update.effective_chat.id if update.effective_chat else 0
                await _send_fast_tariffs_two_messages(context.bot, _chat_id, context)
                return
            gen_lock = await _acquire_user_generation_lock(_user_id)
            if gen_lock is None:
                await update.message.reply_text("Пожалуйста, немного подождите. Ещё обрабатываю прошлый запрос")
                return
            style_id = selected_style
            style_label = _fast_style_label(style_id)
            if style_id == "custom":
                prompt = (context.user_data.get(USERDATA_FAST_CUSTOM_PROMPT) or "").strip()
                if not prompt:
                    gen_lock.release()
                    await update.message.reply_text(
                        "Сначала напишите текстом описание картинки (выберите «✏️ Свой запрос» и отправьте сообщение).",
                    )
                    context.user_data[USERDATA_FAST_SELECTED_STYLE] = style_id
                    return
            else:
                prompt = _persona_style_prompt(style_id, style_label)
            context.user_data[USERDATA_PHOTO_FILE_IDS] = [doc.file_id]
            last_msg_id = context.user_data.pop(USERDATA_FAST_LAST_MSG_ID, None)
            _chat_id = update.effective_chat.id if update.effective_chat else 0
            if last_msg_id:
                try:
                    await context.bot.delete_message(chat_id=_chat_id, message_id=last_msg_id)
                except Exception:
                    pass
            status_msg = await update.message.reply_text("🎨 <i>Создаю изображение...</i>", parse_mode="HTML")

            async def _run_and_release() -> None:
                try:
                    await _run_fast_generation_impl(
                        context=context,
                        chat_id=_chat_id,
                        user_id=_user_id,
                        style_id=style_id,
                        style_label=style_label,
                        prompt=prompt,
                        photo_file_ids=[doc.file_id],
                        profile=_profile,
                        status_msg=status_msg,
                    )
                finally:
                    gen_lock.release()

            context.application.create_task(_run_and_release())
        else:
            context.user_data[USERDATA_PHOTO_FILE_IDS] = [doc.file_id]
            credits = _generations_count_fast(_profile)
            credits_word = _fast_credits_word(credits)
            page = context.user_data.get(USERDATA_FAST_STYLE_PAGE, 0)
            text = f"Отлично! <b>Выберите стиль</b> или введите <b>свой запрос</b> 👇\n\n{_format_balance_express(credits)}\n\n<b>1 кредит = 1 фото</b>\n\n{STYLE_EXAMPLES_FOOTER}"
            msg = await update.message.reply_text(
                text,
                reply_markup=_fast_style_choice_keyboard(gender, include_tariffs=True, page=page),
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            chat_id = update.effective_chat.id if update.effective_chat else 0
            await _update_fast_style_message(context, chat_id, msg)
        return

    user_id = int(update.effective_user.id) if update.effective_user else 0
    profile = store.get_user(user_id)
    await update.message.reply_text(
        "Перед загрузкой фото нужно выбрать раздел: <b>Экспресс-фото</b> или <b>Персона</b>",
        reply_markup=_start_keyboard(profile),
        parse_mode="HTML",
    )


async def handle_prompt_strength_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()

    settings = load_settings()
    action = (query.data or "").split(":", 1)[1] if ":" in (query.data or "") else ""
    if action == "noop":
        return

    cur = _get_prompt_strength(settings, context)
    step = 0.05
    if action == "down":
        # “Похожесть” = меньше перерисовки => уменьшаем prompt_strength
        cur = max(0.1, cur - step)
    elif action == "up":
        cur = min(0.95, cur + step)

    context.user_data[USERDATA_PROMPT_STRENGTH] = cur
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    try:
        await query.edit_message_reply_markup(
            reply_markup=_start_keyboard(profile)
        )
    except Exception:
        pass


async def handle_gender_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()

    data = query.data or ""
    action = data.split(":", 1)[1] if ":" in data else ""
    if action == "male":
        context.user_data[USERDATA_SUBJECT_GENDER] = "male"
    elif action == "female":
        context.user_data[USERDATA_SUBJECT_GENDER] = "female"
    else:
        context.user_data.pop(USERDATA_SUBJECT_GENDER, None)

    settings = load_settings()
    ps = _get_prompt_strength(settings, context)
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(user_id)
    try:
        await query.edit_message_reply_markup(
            reply_markup=_start_keyboard(profile)
        )
    except Exception:
        pass


async def handle_personal_toggle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()
    uid = int(query.from_user.id) if query.from_user else 0
    profile = store.get_user(uid) if uid else None
    has_personal = bool(profile and profile.personal_model_version and profile.personal_trigger_word)
    if not has_personal:
        await query.message.reply_text("У тебя ещё нет персональной модели. Нажми «10 фото».")
        return

    enabled = _is_personal_enabled(context)
    context.user_data[USERDATA_USE_PERSONAL] = (not enabled)

    settings = load_settings()
    ps = _get_prompt_strength(settings, context)
    try:
        await query.edit_message_reply_markup(
            reply_markup=_start_keyboard(profile)
        )
    except Exception:
        pass


async def handle_reset_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    import logging
    logger = logging.getLogger("prismalab")
    
    query = update.callback_query
    if not query:
        logger.warning("handle_reset_callback вызван без query")
        return
    
    try:
        await query.answer()
        user_id = update.effective_user.id
        logger.info(f"Очистка данных для user {user_id}")
        
        # Полная очистка user_data
        context.user_data.pop(USERDATA_PHOTO_FILE_IDS, None)
        context.user_data.pop(USERDATA_ASTRIA_FACEID_FILE_IDS, None)
        context.user_data.pop(USERDATA_ASTRIA_LORA_FILE_IDS, None)
        context.user_data.pop(USERDATA_NANO_BANANA_FILE_IDS, None)
        context.user_data.pop(USERDATA_MODE, None)
        context.user_data.pop(USERDATA_PROMPT_STRENGTH, None)
        context.user_data.pop(USERDATA_SUBJECT_GENDER, None)
        context.user_data.pop(USERDATA_USE_PERSONAL, None)
        
        # Очищаем данные из базы (storage) - полная очистка
        try:
            store.clear_user_data(user_id=user_id)
            logger.info(f"Данные из storage очищены для user {user_id}")
        except Exception as e:
            logger.error(f"Ошибка при очистке storage для user {user_id}: {e}", exc_info=True)
        
        # Сброс фото обычно означает “работаем по новым фото”, а не по старой персональной модели
        context.user_data[USERDATA_USE_PERSONAL] = False
        await query.edit_message_text("✅ Все фото и данные сброшены. Отправь новое фото.")
        logger.info(f"Очистка завершена для user {user_id}")
    except Exception as e:
        logger.error("Ошибка в handle_reset_callback: %s", e, exc_info=True)
        try:
            await query.answer(USER_FRIENDLY_ERROR, show_alert=True)
        except Exception:
            pass


# Функция handle_nano_banana_multi_callback удалена - больше не используется
# Replicate train10 (_start_train10) удалён


async def _start_astria_lora(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int, from_persona: bool = False
) -> None:
    """Создаёт LoRA tune из 10 фото через Astria API. from_persona=True — поток Персоны."""
    logger.info(f"[LoRA] ========== НАЧАЛО создания LoRA для user {user_id} ==========")
    gen_lock = await _acquire_user_generation_lock(user_id)
    if gen_lock is None:
        logger.warning(f"[LoRA] ❌ Lock уже занят для user {user_id}, пропускаю")
        return
    
    try:
        settings = load_settings()
        logger.info(f"[LoRA] Настройки загружены, проверяю API ключ...")
        if not settings.astria_api_key:
            logger.error("[LoRA] ❌ Нет API ключа Astria")
            await context.bot.send_message(chat_id=chat_id, text=USER_FRIENDLY_ERROR)
            return
        logger.info(f"[LoRA] API ключ найден (длина: {len(settings.astria_api_key)})")
        
        file_ids = list(context.user_data.get(USERDATA_ASTRIA_LORA_FILE_IDS, []))
        logger.info(f"[LoRA] Получено file_ids: {len(file_ids)}")
        if len(file_ids) < 10:
            logger.warning(f"[LoRA] ❌ Недостаточно фото: {len(file_ids)}/10")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Нужно 10 фото. Сейчас {len(file_ids)}/10. Отправь ещё {10 - len(file_ids)} фото."
            )
            return
        
        # Исторически обучение шло с class name = "person".
        # Для dev можно переключить на man/woman через PRISMALAB_PERSONA_LORA_NAME_MODE.
        name = "person"
        if from_persona:
            profile_for_name = store.get_user(user_id)
            gender_for_name = profile_for_name.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER)
            name = _persona_lora_name(gender_for_name)
        
        # Скачиваем все 10 фото (с обработкой таймаутов)
        image_bytes_list = []
        if not from_persona:
            await context.bot.send_message(chat_id=chat_id, text="Скачиваю 10 фото…")
        for idx, fid in enumerate(file_ids, 1):
            try:
                image_bytes = await _safe_get_file_bytes(context.bot, fid)
                # Подготавливаем фото
                image_bytes = _prepare_image_for_photomaker(image_bytes)
                image_bytes_list.append(image_bytes)
                if not from_persona and idx % 3 == 0:  # Обновляем статус каждые 3 фото
                    await context.bot.send_message(chat_id=chat_id, text=f"Скачано {idx}/10 фото…")
            except Exception as e:
                logger.error(f"Ошибка при скачивании фото {idx}/10: {e}")
                await context.bot.send_message(chat_id=chat_id, text=USER_FRIENDLY_ERROR)
                raise
        
        if not from_persona:
            await context.bot.send_message(chat_id=chat_id, text="Создаю LoRA tune…\n\nЭто может занять 10–60 минут (иногда до 2 часов при загрузке). Я напишу, когда будет готово.")
        
        logger.info(f"[LoRA] Начинаю создание LoRA tune через Astria API...")
        from prismalab.astria_client import create_lora_tune_and_wait
        
        logger.info(f"[LoRA] Вызываю create_lora_tune_and_wait с параметрами: name={name}, base_tune_id=1504944, preset=flux-lora-portrait")
        result = await create_lora_tune_and_wait(
            api_key=settings.astria_api_key,
            name=name,
            title=f"LoRA user {user_id}",
            image_bytes_list=image_bytes_list,
            base_tune_id="1504944",  # Flux1.dev из галереи (проверенная конфигурация для LoRA)
            preset="flux-lora-portrait",  # Рекомендуется для людей
            on_created=lambda tid: store.set_astria_lora_tune_pending(user_id=user_id, tune_id=tid),
            max_seconds=7200,  # До 2 часов на training (увеличено с 1 часа)
            poll_seconds=15.0,
        )
        
        # Проверяем, что модель действительно создана как LoRA
        model_type = result.raw.get("model_type") or "unknown"
        if model_type.lower() != "lora":
            logger.error(f"⚠️ ВНИМАНИЕ: Astria создал tune {result.tune_id} как '{model_type}', а не как 'lora'!")
            await context.bot.send_message(chat_id=chat_id, text=USER_FRIENDLY_ERROR)
            return
        
        store.set_astria_lora_tune(user_id=user_id, tune_id=result.tune_id)
        context.user_data.pop(USERDATA_ASTRIA_LORA_FILE_IDS, None)
        
        if from_persona:
            context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "done"
            profile = store.get_user(user_id)
            credits = profile.persona_credits_remaining
            gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
            text = f"Готово! 🎉 Персональная модель обучена\n\nТеперь выберите стиль из списка ниже – у вас {credits} {_fast_credits_word(credits)}\n\n{STYLE_EXAMPLES_FOOTER}"
            await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=_persona_styles_keyboard(gender),
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        else:
            context.user_data[USERDATA_MODE] = "normal"
            logger.info(f"✅ LoRA {result.tune_id} успешно создана и сохранена (model_type='{model_type}')")
            profile = store.get_user(user_id)
            await context.bot.send_message(
                chat_id=chat_id,
                text="✅ Готово! LoRA модель создана на Flux1.dev.\n"
                f"ID модели: {result.tune_id}\n"
                "Теперь нажми «✨ Персона» — я буду генерировать сцены с высоким качеством.",
                reply_markup=_start_keyboard(profile),
            )
    except AstriaError as e:
        if from_persona:
            context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "error"
        store.clear_astria_lora_tune_pending(user_id)
        logger.error("Astria LoRA error: %s", e, exc_info=True)
        msg = "Что-то пошло не так, персона не создалась. Загрузите фото заново или напишите в поддержку." if from_persona else USER_FRIENDLY_ERROR
        await context.bot.send_message(chat_id=chat_id, text=msg)
    except Exception as e:
        if from_persona:
            context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "error"
        store.clear_astria_lora_tune_pending(user_id)
        logger.error("Astria LoRA error: %s", e, exc_info=True)
        msg = "Что-то пошло не так, персона не создалась. Загрузите фото заново или напишите в поддержку." if from_persona else USER_FRIENDLY_ERROR
        await context.bot.send_message(chat_id=chat_id, text=msg)
    finally:
        gen_lock.release()


async def _run_style_job(
    *,
    bot,
    chat_id: int,
    photo_file_ids: list[str],
    style_id: str,
    settings,
    status_message_id: int,
    prompt_strength: float,
    user_id: int,
    subject_gender: str | None,
    use_personal_requested: bool,
    test_prompt: str | None = None,
    lora_prompt_override: str | None = None,
    style_title_override: str | None = None,
    is_persona_style: bool = False,
    context: ContextTypes.DEFAULT_TYPE | None = None,
) -> None:
    try:
        use_test_prompt = test_prompt is not None
        if use_test_prompt:
            from prismalab.styles import StylePreset
            style = StylePreset(
                id="test",
                title="Тест",
                prompt="",
                negative_prompt="",
            )
        elif is_persona_style and lora_prompt_override:
            from prismalab.styles import StylePreset
            style = StylePreset(
                id=style_id,
                title=style_title_override or style_id,
                prompt=lora_prompt_override,
                negative_prompt="",
            )
        else:
            style = get_style(style_id)
            if not style:
                err_msg = PERSONA_ERROR_MESSAGE if is_persona_style else "Неизвестный стиль."
                extra = {}
                if is_persona_style and context:
                    profile = store.get_user(user_id)
                    gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
                    page = context.user_data.get(USERDATA_PERSONA_STYLE_PAGE, 0)
                    extra["reply_markup"] = _persona_styles_keyboard(gender, page)
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=err_msg, **extra)
                return

        await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        refs: list[bytes] = []
        selected_ids = list(photo_file_ids[-4:])
        for fid in selected_ids:
            ref_bytes = await _safe_get_file_bytes(bot, fid)
            refs.append(ref_bytes)

        user_profile = store.get_user(user_id)
        needs_photo = not (user_profile and (user_profile.astria_lora_tune_id or user_profile.astria_tune_id))
        if user_profile and (user_profile.astria_lora_tune_id or user_profile.astria_tune_id):
            logger.info(f"[ASTRIA] LoRA/FaceID найден, фото не требуется.")
        if not refs and needs_photo:
            err_msg = PERSONA_ERROR_MESSAGE if is_persona_style else "Не нашёл фото для обработки."
            extra = {}
            if is_persona_style and context:
                profile = store.get_user(user_id)
                gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
                page = context.user_data.get(USERDATA_PERSONA_STYLE_PAGE, 0)
                extra["reply_markup"] = _persona_styles_keyboard(gender, page)
            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=err_msg, **extra)
            return

        primary_ref = refs[-1] if refs else None
        profile_for_keyboard = store.get_user(user_id)
        has_personal_model = bool(profile_for_keyboard and profile_for_keyboard.personal_model_version and profile_for_keyboard.personal_trigger_word)
        personal_enabled = bool(has_personal_model and use_personal_requested)

        # Только Astria (Replicate удалён)
        if True:
            if not settings.astria_api_key:
                err_msg = PERSONA_ERROR_MESSAGE if is_persona_style else USER_FRIENDLY_ERROR
                extra = {}
                if is_persona_style and context:
                    profile = store.get_user(user_id)
                    gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
                    page = context.user_data.get(USERDATA_PERSONA_STYLE_PAGE, 0)
                    extra["reply_markup"] = _persona_styles_keyboard(gender, page)
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=err_msg, **extra)
                return
            # Проверяем, есть ли у пользователя FaceID или LoRA tune
            user_profile = store.get_user(user_id)
            astria_tune_id = user_profile.astria_tune_id if user_profile else None
            astria_lora_tune_id = user_profile.astria_lora_tune_id if user_profile else None
            
            # Определяем, какой tune использовать (приоритет LoRA, если есть)
            use_lora = astria_lora_tune_id is not None
            active_tune_id = astria_lora_tune_id if use_lora else astria_tune_id
            logger.info(f"[ASTRIA Generate] astria_lora_tune_id={astria_lora_tune_id}, astria_tune_id={astria_tune_id}, use_lora={use_lora}, active_tune_id={active_tune_id}")
            logger.info(f"[ASTRIA Generate] base_model будет Flux1.dev (1504944) для LoRA" if use_lora else f"[ASTRIA Generate] base_model будет Realistic Vision (690204) для FaceID")
            
            if not active_tune_id:
                err_msg = PERSONA_ERROR_MESSAGE if is_persona_style else (
                    "❌ Для генерации нужно сначала создать FaceID или LoRA tune.\n\n"
                    "• 📸 FaceID (1 фото) - быстро, для одной сцены\n"
                    "• 🎯 LoRA (10 фото) - качественно, для множества сцен"
                )
                extra = {}
                if is_persona_style and context:
                    profile = store.get_user(user_id)
                    gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
                    page = context.user_data.get(USERDATA_PERSONA_STYLE_PAGE, 0)
                    extra["reply_markup"] = _persona_styles_keyboard(gender, page)
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=err_msg, **extra)
                return
            # Определяем тип генерации: FaceID или LoRA
            subj = _instantid_subject(subject_gender)
            
            if use_lora:
                # LoRA генерация: используем <lora:{tune_id}:strength> в промпте
                # Inference на базовой модели Flux1.dev (1504944)
                lora_weight = 1.1  # Вес LoRA для сходства (1.0–1.2)
                
                # Промпт для Astria LoRA (lora_prompt_override — для Персоны)
                # LoRA обучается с name="person", поэтому всегда "ohwx person"
                ohwx_token = "ohwx person"
                if lora_prompt_override:
                    english_prompt = lora_prompt_override
                else:
                    english_prompt = """IDENTICAL FACE AND FEATURES from reference photo, same skin tone, ultra high detail face. A professional waist-up portrait of airline pilot in cockpit, wearing crisp uniform with epaulettes and tie. Soft diffused light from cockpit windows creates even illumination on his face. He sits in captain's seat with hands on controls or arms relaxed, looking at camera with confident composed expression. Instrument panels and aviation equipment softly blurred in background. Natural skin texture, sharp eyes, commercial aviation professional photography style"""
                text = (
                    f"<lora:{active_tune_id}:{lora_weight}> "
                    f"{ohwx_token}, {english_prompt}"
                ).strip()
                neg = "blurry, low quality, deformed face, bad anatomy, cartoon, cgi, plastic skin, overly smooth skin"
                
                # Для LoRA используем базовую модель Flux1.dev (1504944) - та же, на которой создавалась LoRA
                base_model_tune_id = "1504944"  # Flux1.dev из галереи
                logger.info(f"[ASTRIA LoRA] Использую базовую модель Flux1.dev (1504944) для inference с LoRA tune_id={active_tune_id}")
            else:
                # FaceID генерация: используем <faceid:{tune_id}:strength> в промпте
                # Inference на базовой модели Realistic Vision V5.1 (690204)
                # Единый промпт для тестирования (про сад)
                faceid_weight = 1.0  # Preset "MAX FACE": дефолт 1.0 для упора в лицо
                
                # Промпт на английском, FaceID тег в начале (как в галерее Astria)
                # Промпт про смеющуюся девушку для Astria FaceID с акцентом на реалистичность
                english_prompt = (
                    "ultra realistic photograph, not illustration, not painting, not digital art, real photo, "
                    "same reference female character, candid laugh, head turned left, eyes squinting, looking away from camera, "
                    "tropical greenery background, golden hour, realistic photo, 35mm, shallow depth of field, "
                    "identity preserved, natural expression change, no face morphing, no distortion, correct anatomy, "
                    "no extra fingers, no text, natural skin pores, authentic photography, documentary style, candid photo, "
                    "professional photography, photorealistic, high detail, no CGI, no 3D render, no digital painting, no illustration style"
                )
                
                text = (
                    f"<faceid:{active_tune_id}:{faceid_weight}> "
                    f"{english_prompt}"
                ).strip()
                
                # Негативный промпт по рекомендациям Astria
                neg = (
                    "blurry, low quality, deformed face, bad anatomy, cartoon, cgi, plastic skin, overly smooth skin"
                )
                
                # Для FaceID используем базовую модель Realistic Vision V5.1 (690204)
                base_model_tune_id = settings.astria_tune_id  # 690204 - Realistic Vision V5.1 из галереи

            # Логируем промпт для отладки
            tune_type = "LoRA" if use_lora else "FaceID"
            logger.info(f"Astria {tune_type} промпт для стиля {style.id} (полный, длина {len(text)}): {text}")
            if use_lora:
                logger.info(f"Astria промпт - количество тегов LoRA: {text.count('<lora:')}")
            else:
                logger.info(f"Astria промпт - количество тегов FaceID: {text.count('<faceid:')}")
            
            title = style_title_override or style.title
            if not is_persona_style:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message_id,
                    text="Создаю фотографию, обычно это занимает около 10 секунд.",
                )
            # text-to-image: input_image_bytes=None, denoising_strength не нужен
            # Добавляем параметры для улучшения качества сцен
            # НЕ используем style="Photographic" - он может конфликтовать со стилями пользователя
            # Для винтажных стилей используем color_grading и film_grain
            import secrets
            random_seed = secrets.randbelow(2**32)  # 0 to 2^32-1 для разнообразия
            
            # Для винтажных стилей используем цветокоррекцию, для остальных - нет
            use_color_grading = None
            use_film_grain = False
            if style_id in {"vintage_film", "noir", "nyc_70s"}:
                use_color_grading = "Film Portra"
                use_film_grain = True
            
            # Параметры для генерации
            if use_lora:
                # Для LoRA (Flux): по документации Flux не поддерживает negative_prompt и weighted prompts
                # Параметры настроены для максимальной реалистичности (меньше "рисованности")
                use_cfg_scale = 3.0  # Ещё уменьшено для более естественного результата
                use_steps = 40  # Увеличено для более детального и реалистичного результата
                
                # Для LoRA: только super_resolution, face_correct/face_swap/inpaint_faces выключены
                use_face_correct = False
                use_face_swap = False
                use_inpaint_faces = False
                use_seed = random_seed  # Поддерживается для Flux1.dev
                logger.info(f"[ASTRIA] LoRA на Flux1.dev: face_correct=false, face_swap=false, inpaint_faces=false, hires_fix=true, seed={use_seed}")
                super_resolution = True  # Для лучшего качества
                # Для Flux negative_prompt не поддерживается
                neg = None  # Flux не поддерживает negative_prompt
            else:
                # Preset "MAX FACE" для FaceID (упор на качество лица)
                # Обязательные: cfg_scale=3, face_correct=true, face_swap=true
                # Улучшают лицо: super_resolution=true
                # inpaint_faces НЕ поддерживается для FaceID (только для LoRA)
                use_seed = random_seed  # Для FaceID seed поддерживается
                if use_test_prompt:
                    # Preset "MAX FACE" для тестового промпта
                    use_cfg_scale = 3.0
                    use_steps = 40  # 35-45 для реализма лица
                else:
                    # Preset "MAX FACE" для FaceID: cfg_scale=3 (обязательно)
                    use_cfg_scale = 3.0
                    # Steps 35-45 для реализма лица (по рекомендациям)
                    use_steps = min(settings.astria_steps, 45) if settings.astria_steps else 40
                    if use_steps < 35:
                        use_steps = 35  # Минимум 35 для качества лица
                
                # Preset "MAX FACE": обязательные параметры
                use_face_correct = True  # Обязательно для FaceID
                use_face_swap = True  # Обязательно для FaceID
                # super_resolution=true почти всегда улучшает лицо
                super_resolution = True
                # inpaint_faces НЕ поддерживается для FaceID (Astria возвращает 422)
                # Для full-body/long-shot можно использовать только LoRA с inpaint_faces
                use_inpaint_faces = None
                logger.info(f"[ASTRIA] FaceID MAX FACE: cfg_scale={use_cfg_scale}, steps={use_steps}, inpaint_faces={use_inpaint_faces} (None - не поддерживается)")
            # КРИТИЧНО: По документации Astria:
            # - FaceID inference на базовой модели Realistic Vision V5.1 (690204)
            # - LoRA inference на базовой модели Flux1.dev (1504944) - та же, на которой создавалась LoRA
            # Tune ID (FaceID/LoRA) используется только в промпте как <faceid:...> или <lora:...>
            logger.info(f"[ASTRIA] Отправка запроса: use_lora={use_lora}, inpaint_faces={use_inpaint_faces}, tune_type={tune_type}")
            astria_res = await astria_run_prompt_and_wait(
                api_key=settings.astria_api_key,
                tune_id=base_model_tune_id,  # Базовая модель из галереи, НЕ FaceID/LoRA tune!
                text=text,
                negative_prompt=neg,  # None для LoRA (Flux не поддерживает)
                input_image_bytes=None,  # text-to-image, не img2img
                cfg_scale=use_cfg_scale,
                steps=use_steps,
                denoising_strength=None,  # не используется для text-to-image
                super_resolution=super_resolution,
                hires_fix=True if use_lora else None,  # LoRA: пробуем True; FaceID: не поддерживается
                face_correct=use_face_correct if use_lora else use_face_correct,
                face_swap=use_face_swap if use_lora else use_face_swap,
                inpaint_faces=use_inpaint_faces,  # True для LoRA, None для FaceID (не поддерживается)
                style=None,  # не используем встроенный style
                color_grading=use_color_grading,
                film_grain=use_film_grain,
                seed=use_seed,  # None для Flux 2 Pro LoRA, random_seed для остальных
                max_seconds=settings.astria_max_seconds,
                poll_seconds=2.0,
            )
            if not is_persona_style:
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text="Скачиваю результат…")
            out_bytes = await astria_download_first_image_bytes(astria_res.images, api_key=settings.astria_api_key)
            out_bytes = _postprocess_output(style_id, out_bytes)

            bio = io.BytesIO(out_bytes)
            bio.name = f"{settings.app_name}_{style.id}.png"
            if not is_persona_style:
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text="Отправляю в Telegram…")
            tune_type_label = "LoRA" if use_lora else "FaceID"
            
            persona_caption = f"Персона: «{style.title}»" if is_persona_style else f"Готово ({tune_type_label}): «{style.title}»"
            await _safe_send_document(
                bot=bot,
                chat_id=chat_id,
                document=bio,
                caption=persona_caption,
            )
            # Логируем успешную генерацию для аналитики
            try:
                store.log_event(user_id, "generation", {"mode": "persona" if is_persona_style else "fast", "style": style_id, "provider": "astria"})
            except Exception:
                pass

            if is_persona_style and context:
                try:
                    await bot.delete_message(chat_id=chat_id, message_id=status_message_id)
                except Exception:
                    pass
                credits = store.decrement_persona_credits(user_id)
                profile = store.get_user(user_id)
                if credits <= 0:
                    text, reply_markup = _persona_credits_out_content(profile)
                else:
                    gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
                    page = context.user_data.get(USERDATA_PERSONA_STYLE_PAGE, 0)
                    credits_word = _fast_credits_word(credits)
                    text = f"<b>Готово!</b>\n\nМожете выбрать <b>другой стиль</b> или снова <b>попробовать тот же</b>\n\n{_format_balance_persona(credits)}"
                    reply_markup = _persona_styles_keyboard(gender, page)
                msg = await bot.send_message(
                        chat_id=chat_id,
                    text=text,
                    reply_markup=reply_markup,
                    parse_mode="HTML",
                    )
                context.user_data[USERDATA_PERSONA_STYLE_MSG_ID] = msg.message_id
            else:
                profile = store.get_user(user_id)
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message_id,
                    text="Готово. Хочешь другой стиль?",
                    reply_markup=_start_keyboard(profile),
                )
            return

        # НОВЫЙ ПАЙПЛАЙН: “сцена → face swap → upscale”
        # Идея: сцену рисуем “как кино”, а лицо возвращаем face-swap'ом, чтобы не получался “другой человек”.
        if provider == "replicate" and scene_swap:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text=f"Генерирую сцену (реализм) для «{style.title}»…",
            )

            subj = _instantid_subject(subject_gender)
            scene_prompt = f"{subj}, {style.prompt}".replace("portrait", "").replace("  ", " ").strip(" ,")
            if style_id == "noir":
                scene_prompt = f"{scene_prompt}, black and white, monochrome, grayscale, no color"

            # База для позы/кадра: используем входное фото, но с высокой силой перерисовки, чтобы фон реально поменялся.
            base_bytes = _prepare_image_for_photomaker(primary_ref)
            img = Image.open(io.BytesIO(base_bytes))
            out_w, out_h = img.size
            data_uri = build_data_uri(base_bytes, filename="input.png")

            ps_scene = max(0.75, min(0.95, float(prompt_strength)))

            flux_input: dict[str, Any] = {}
            flux_input["image"] = data_uri
            # Для теста добавляем больше деталей сцены, для обычных стилей - стандартные детали
            if use_test_prompt:
                flux_input["prompt"] = f"{scene_prompt}, professional photography, high quality, detailed scene, sharp focus, cinematic lighting, photorealistic, natural lighting, atmospheric depth, rich details"
            else:
                flux_input["prompt"] = f"{scene_prompt}, photorealistic, cinematic lighting, high detail"
            flux_input["prompt_strength"] = ps_scene
            flux_input.setdefault("num_inference_steps", settings.flux_num_inference_steps)
            flux_input.setdefault("guidance", settings.flux_guidance)
            flux_input.setdefault("num_outputs", 1)
            flux_input.setdefault("output_format", "png")
            flux_input.setdefault("aspect_ratio", _guess_aspect_ratio(out_w, out_h))

            # 1) Пытаемся Flux (обычно сцены выглядят лучше), но он может быть отключён на Replicate.
            # Для теста сразу используем SDXL, так как Flux часто отключён
            use_sdxl_directly = use_test_prompt  # Для теста сразу SDXL
            scene_res = None
            
            if not use_sdxl_directly:
                try:
                    scene_res = await run_prediction_and_wait(
                        api_token=settings.replicate_api_token,
                        model_version=settings.replicate_flux_model_version,
                        model_input=flux_input,
                        max_seconds=settings.replicate_max_seconds,
                        poll_seconds=2.0,
                    )
                except ReplicateError as e:
                    s = str(e).lower()
                    error_msg = str(e)
                    # Проверяем разные варианты ошибки "version disabled"
                    if ("version disabled" in s or 
                        '"title":"version disabled"' in s or
                        "disabled" in s or
                        "недоступен" in s.lower() or
                        "отключён" in s.lower()):
                        logger.warning(f"Flux отключён, переключаюсь на SDXL: {error_msg}")
                        use_sdxl_directly = True
                    else:
                        raise
            
            # Если Flux не сработал или для теста - используем SDXL
            if use_sdxl_directly or scene_res is None:
                if not use_sdxl_directly:
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_message_id,
                        text="Flux на Replicate сейчас отключён. Пробую запасной генератор сцены (SDXL)…",
                    )
                prepared_bytes, sw, sh = _prepare_image_for_sdxl(primary_ref)
                sdxl_uri = build_data_uri(prepared_bytes, filename="input.png")
                sdxl_input = dict(settings.replicate_base_input)
                sdxl_input[settings.replicate_image_key] = sdxl_uri
                # Для теста добавляем больше деталей сцены, для обычных стилей - стандартные детали
                if use_test_prompt:
                    sdxl_input[settings.replicate_prompt_key] = f"{scene_prompt}, professional photography, high quality, detailed scene, sharp focus, cinematic lighting, photorealistic, natural lighting, atmospheric depth, rich details"
                else:
                    sdxl_input[settings.replicate_prompt_key] = f"{scene_prompt}, photorealistic, cinematic lighting, high detail"
                neg = (style.negative_prompt or "").strip()
                extra_neg = "lowres, blurry, jpeg artifacts, text, watermark, logo, bad anatomy, deformed"
                combined_neg = f"{neg}, {extra_neg}" if neg else extra_neg
                sdxl_input[settings.replicate_negative_prompt_key] = combined_neg
                sdxl_input.setdefault("width", sw)
                sdxl_input.setdefault("height", sh)
                sdxl_input.setdefault("num_outputs", 1)
                sdxl_input.setdefault("guidance_scale", 7.5)
                sdxl_input.setdefault("num_inference_steps", 50)
                sdxl_input.setdefault("apply_watermark", False)
                sdxl_input["prompt_strength"] = ps_scene
                scene_res = await run_prediction_and_wait(
                    api_token=settings.replicate_api_token,
                    model_version=settings.replicate_sdxl_model_version,
                    model_input=sdxl_input,
                    max_seconds=settings.replicate_max_seconds,
                    poll_seconds=2.0,
                )
            if scene_res.status != "succeeded":
                msg = f"Не получилось сгенерировать сцену (status={scene_res.status})."
                if scene_res.error:
                    msg += f"\nОшибка: {scene_res.error}"
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=msg)
                return

            scene_url = _first_output_url(scene_res.output)
            if not scene_url:
                scene_bytes = await download_output_image_bytes(scene_res.output)
                scene_url = build_data_uri(scene_bytes, filename="scene.png")

            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text="Фиксирую лицо (face swap)…")
            source_uri = build_data_uri(base_bytes, filename="source.png")
            swap_res = await run_prediction_and_wait(
                api_token=settings.replicate_api_token,
                model_version=settings.faceswap_model_version,
                model_input={
                    "input_image": scene_url,
                    "source_image": source_uri,
                    "inputface_index": 0,
                    "sourceface_index": 0,
                },
                max_seconds=settings.replicate_max_seconds,
                poll_seconds=2.0,
            )
            if swap_res.status != "succeeded":
                msg = f"Не получилось зафиксировать лицо (status={swap_res.status})."
                if swap_res.error:
                    msg += f"\nОшибка: {swap_res.error}"
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=msg)
                return

            swap_url = _first_output_url(swap_res.output)
            if not swap_url:
                swap_bytes = await download_output_image_bytes(swap_res.output)
                swap_url = build_data_uri(swap_bytes, filename="swap.png")

            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text="Улучшаю качество (upscale)…")
            up_res = await run_prediction_and_wait(
                api_token=settings.replicate_api_token,
                model_version=settings.upscale_model_version,
                model_input={
                    "image": swap_url,
                    "scale": float(settings.upscale_scale),
                    "face_enhance": bool(settings.upscale_face_enhance),
                },
                max_seconds=settings.replicate_max_seconds,
                poll_seconds=2.0,
            )
            if up_res.status != "succeeded":
                msg = f"Не получилось улучшить качество (status={up_res.status})."
                if up_res.error:
                    msg += f"\nОшибка: {up_res.error}"
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=msg)
                return

            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text="Скачиваю результат…")
            out_bytes = await download_output_image_bytes(up_res.output)
            out_bytes = _postprocess_output(style_id, out_bytes)
            bio = io.BytesIO(out_bytes)
            bio.name = f"{settings.app_name}_{style.id}.png"
            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text="Отправляю в Telegram…")
            await _safe_send_document(
                bot=bot,
                chat_id=chat_id,
                document=bio,
                caption=f"Готово (сцена + лицо): «{style.title}»",
            )
            profile = store.get_user(user_id)
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text="Готово. Хочешь другой стиль на это же фото?",
                reply_markup=_start_keyboard(profile),
            )
            return

        # Если у пользователя есть персональная модель (после режима "10 фото") — используем её (если включено).
        # Можно отключить, выставив PRISMALAB_DISABLE_PERSONAL=true.
        # ДЛЯ ТЕСТА: принудительно отключаем персональную модель, используем InstantID
        profile = store.get_user(user_id)
        disable_personal = (os.getenv("PRISMALAB_DISABLE_PERSONAL") or "").strip().lower() in {"1", "true", "yes", "y"}
        use_personal = (
            (not disable_personal)
            and (not use_test_prompt)  # ДЛЯ ТЕСТА: отключаем персональную модель
            and use_personal_requested
            and bool(profile.personal_model_version)
            and bool(profile.personal_trigger_word)
        )
        # ДЛЯ ТЕСТА: принудительно используем InstantID с правильной версией модели
        if use_test_prompt:
            engine = "instantid"
            model_version_to_use = settings.replicate_instantid_model_version  # Используем именно InstantID версию
            logger.info(f"ТЕСТ: принудительно используем engine=instantid, model_version={model_version_to_use}")
        else:
            model_version_to_use = settings.replicate_model_version
            engine = getattr(settings, "engine", "instantid")
            if use_personal:
                engine = "personal"
                model_version_to_use = str(profile.personal_model_version)
            logger.info(f"Обычный режим: engine={engine}, model_version={model_version_to_use}, use_personal={use_personal}")

        gender_prefix = _subject_prompt_prefix(subject_gender)
        gender_neg = _subject_negative_lock(subject_gender)
        if engine == "personal":
            # Персональная модель после LoRA training (текст-в-изображение)
            trigger = str(profile.personal_trigger_word)
            prompt = f"{gender_prefix}, {trigger}, {style.prompt}, high quality"
            if style_id == "noir":
                prompt = f"{prompt}, black and white, monochrome, grayscale, no color"

            # аспект берём из первого рефа (если есть)
            base_bytes = _prepare_image_for_photomaker(primary_ref)
            img = Image.open(io.BytesIO(base_bytes))
            out_w, out_h = img.size

            model_input = dict(settings.replicate_base_input)
            model_input["prompt"] = prompt
            model_input.setdefault("num_outputs", 1)
            model_input.setdefault("output_format", "png")
            model_input.setdefault("aspect_ratio", _guess_aspect_ratio(out_w, out_h))
            # типичные параметры для flux dev LoRA моделей
            model_input.setdefault("num_inference_steps", 28)
            model_input.setdefault("guidance_scale", 7.5)
            model_input.setdefault("lora_scale", 1.0)

        elif engine == "sdxl":
            prepared_bytes, out_w, out_h = _prepare_image_for_sdxl(primary_ref)
            data_uri = build_data_uri(prepared_bytes, filename="input.png")
            model_input = dict(settings.replicate_base_input)
            model_input[settings.replicate_image_key] = data_uri
            model_input[settings.replicate_prompt_key] = f"{gender_prefix}, {style.prompt}, high quality"
            neg = style.negative_prompt.strip()
            extra_neg = "lowres, blurry, jpeg artifacts, text, watermark, logo, bad anatomy, deformed"
            combined_neg = f"{neg}, {extra_neg}" if neg else extra_neg
            if gender_neg:
                combined_neg = f"{combined_neg}, {gender_neg}"
            model_input[settings.replicate_negative_prompt_key] = combined_neg

            # Дефолты SDXL (не перетираем, если заданы через PRISMALAB_REPLICATE_BASE_INPUT_JSON)
            model_input.setdefault("width", out_w)
            model_input.setdefault("height", out_h)
            model_input.setdefault("num_outputs", 1)
            model_input.setdefault("guidance_scale", 7.5)
            model_input.setdefault("num_inference_steps", 50)
            model_input.setdefault("apply_watermark", False)

            # Важно для img2img: удерживает сходство с исходным фото.
            model_input["prompt_strength"] = float(prompt_strength)
        elif engine == "flux":
            # Flux dev: img2img, без negative_prompt
            base_bytes = _prepare_image_for_photomaker(primary_ref)
            img = Image.open(io.BytesIO(base_bytes))
            out_w, out_h = img.size
            data_uri = build_data_uri(base_bytes, filename="input.png")

            model_input = dict(settings.replicate_base_input)
            model_input["image"] = data_uri
            # Для Flux лучше короткие “человеческие” промпты
            model_input["prompt"] = f"{gender_prefix}, {style.prompt}"
            model_input["prompt_strength"] = float(prompt_strength)
            model_input.setdefault("num_inference_steps", settings.flux_num_inference_steps)
            model_input.setdefault("guidance", settings.flux_guidance)
            model_input.setdefault("num_outputs", 1)
            model_input.setdefault("output_format", "png")
            model_input.setdefault("aspect_ratio", _guess_aspect_ratio(out_w, out_h))
        elif engine == "flux_ultra":
            # Flux 1.1 Pro Ultra: другие поля (image_prompt вместо image)
            base_bytes = _prepare_image_for_photomaker(primary_ref)
            img = Image.open(io.BytesIO(base_bytes))
            out_w, out_h = img.size
            image_prompt = build_data_uri(base_bytes, filename="input.png")

            prompt = f"{gender_prefix}, {style.prompt}"
            if style_id == "noir":
                prompt = f"{prompt}, black and white, monochrome, grayscale, no color"

            # Маппим наш prompt_strength (0..1) в image_prompt_strength (0..1):
            # меньше prompt_strength => больше “похожести” => сильнее влияние reference фото
            image_prompt_strength = max(0.1, min(1.0, 1.0 - float(prompt_strength)))

            model_input = dict(settings.replicate_base_input)
            model_input["prompt"] = prompt
            model_input["image_prompt"] = image_prompt
            model_input["image_prompt_strength"] = image_prompt_strength
            model_input["raw"] = bool(settings.flux_ultra_raw)
            model_input["safety_tolerance"] = int(settings.flux_ultra_safety_tolerance)
            model_input["output_format"] = "png"
            model_input["aspect_ratio"] = _guess_aspect_ratio(out_w, out_h)
        elif engine == "flux_ultra_swap":
            # Pipeline: Flux Ultra -> FaceSwap -> Upscale
            base_bytes = _prepare_image_for_photomaker(primary_ref)
            img = Image.open(io.BytesIO(base_bytes))
            out_w, out_h = img.size
            image_prompt = build_data_uri(base_bytes, filename="input.png")

            prompt = f"{gender_prefix}, {style.prompt}"
            if style_id == "noir":
                prompt = f"{prompt}, black and white, monochrome, grayscale, no color"

            image_prompt_strength = max(0.1, min(1.0, 1.0 - float(prompt_strength)))

            # 1) Generate scene via Flux Ultra
            flux_input = dict(settings.replicate_base_input)
            flux_input["prompt"] = prompt
            flux_input["image_prompt"] = image_prompt
            flux_input["image_prompt_strength"] = image_prompt_strength
            flux_input["raw"] = bool(settings.flux_ultra_raw)
            flux_input["safety_tolerance"] = int(settings.flux_ultra_safety_tolerance)
            flux_input["output_format"] = "png"
            flux_input["aspect_ratio"] = _guess_aspect_ratio(out_w, out_h)

            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text=f"Генерирую сцену (FLUX Ultra) для «{style.title}»…",
            )
            flux_res = await run_prediction_and_wait(
                api_token=settings.replicate_api_token,
                model_version=settings.replicate_model_version,  # flux-ultra version
                model_input=flux_input,
                max_seconds=settings.replicate_max_seconds,
                poll_seconds=2.0,
            )
            if flux_res.status != "succeeded":
                msg = f"Не получилось сгенерировать сцену (status={flux_res.status})."
                if flux_res.error:
                    msg += f"\nОшибка: {flux_res.error}"
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=msg)
                return

            flux_out_url = flux_res.output
            if not isinstance(flux_out_url, str) or not flux_out_url:
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text="FLUX Ultra вернул неожиданный output.")
                return

            # 2) Face swap
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text="Фиксирую лицо (face swap)…",
            )
            source_uri = build_data_uri(base_bytes, filename="source.png")
            swap_res = await run_prediction_and_wait(
                api_token=settings.replicate_api_token,
                model_version=settings.faceswap_model_version,
                model_input={
                    "input_image": flux_out_url,
                    "source_image": source_uri,
                    "inputface_index": 0,
                    "sourceface_index": 0,
                },
                max_seconds=settings.replicate_max_seconds,
                poll_seconds=2.0,
            )
            if swap_res.status != "succeeded":
                msg = f"Не получилось зафиксировать лицо (status={swap_res.status})."
                if swap_res.error:
                    msg += f"\nОшибка: {swap_res.error}"
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=msg)
                return

            swap_out_url = swap_res.output
            if not isinstance(swap_out_url, str) or not swap_out_url:
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text="Face swap вернул неожиданный output.")
                return

            # 3) Upscale
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text="Улучшаю качество (upscale)…",
            )
            up_res = await run_prediction_and_wait(
                api_token=settings.replicate_api_token,
                model_version=settings.upscale_model_version,
                model_input={
                    "image": swap_out_url,
                    "scale": float(settings.upscale_scale),
                    "face_enhance": bool(settings.upscale_face_enhance),
                },
                max_seconds=settings.replicate_max_seconds,
                poll_seconds=2.0,
            )
            if up_res.status != "succeeded":
                msg = f"Не получилось улучшить качество (status={up_res.status})."
                if up_res.error:
                    msg += f"\nОшибка: {up_res.error}"
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=msg)
                return

            # переопределяем result/output на финал пайплайна
            result_output_for_send = up_res.output
            if not isinstance(result_output_for_send, str) or not result_output_for_send:
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text="Upscale вернул неожиданный output.")
                return
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text="Скачиваю результат…",
            )
            out_bytes = await download_output_image_bytes(result_output_for_send)
            out_bytes = _postprocess_output(style_id, out_bytes)
            bio = io.BytesIO(out_bytes)
            bio.name = f"{settings.app_name}_{style.id}.png"

            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text="Отправляю в Telegram…",
            )
            await _safe_send_document(
                bot=bot,
                chat_id=chat_id,
                document=bio,
                caption=f"Готово (face swap + upscale): «{style.title}»",
            )
            profile = store.get_user(user_id)
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text="Готово. Хочешь другой вариант? (будет другая сцена, но лицо сохраню)",
                reply_markup=_start_keyboard(profile),
            )
            return
        elif engine == "instantid":
            # InstantID: identity-preserving
            subj = _instantid_subject(subject_gender)
            # style.prompt у нас часто содержит слово “portrait”, которое иногда ухудшает сцену.
            style_prompt = (style.prompt or "").replace("portrait", "").replace("  ", " ").strip(" ,")
            prompt = f"{subj}, {style_prompt}, photorealistic, detailed, high quality"
            if style_id == "noir":
                prompt = f"{prompt}, black and white, monochrome, grayscale, no color"

            # Маппим “похожесть/стиль” (prompt_strength) в силу identity: чем меньше prompt_strength, тем сильнее identity.
            identity_boost = max(0.0, min(1.0, 1.0 - float(prompt_strength)))
            # Держим скейлы в разумных пределах, чтобы стиль не “умирал”.
            ip_scale = max(0.35, min(0.8, float(settings.instantid_ip_adapter_scale) + identity_boost * 0.3))
            cn_scale = max(0.45, min(0.9, float(settings.instantid_controlnet_conditioning_scale) + identity_boost * 0.3))

            # Авто-повтор: если “лицо не найдено”, пробуем ещё раз с небольшим приближением.
            face_zooms = [1.0, 1.35, 1.6]
            last_face_err: str | None = None
            for idx, z in enumerate(face_zooms):
                base_bytes, out_w, out_h = _prepare_image_for_instantid_zoom(primary_ref, zoom=z)
                data_uri = build_data_uri(base_bytes, filename="input.png")
                model_input = dict(settings.replicate_base_input)
                # InstantID требует input_image, а не image
                model_input["input_image"] = data_uri
                # Также пробуем image на случай, если модель использует его
                if "image" not in model_input:
                    model_input["image"] = data_uri
                model_input["prompt"] = prompt
                style_neg = (style.negative_prompt or "").strip()
                base_neg = (
                    "blurry, lowres, jpeg artifacts, low quality, out of focus, text, watermark, logo, "
                    "distorted features, asymmetric face, duplicate faces, multiple people, ugly, disfigured, "
                    "bad anatomy, deformed face"
                )
                model_input["negative_prompt"] = f"{style_neg}, {base_neg}" if style_neg else base_neg
                model_input["ip_adapter_scale"] = ip_scale
                model_input["controlnet_conditioning_scale"] = cn_scale
                # guidance/steps теперь по дефолтам ниже/выше соответственно (см. settings.py), но env может переопределить
                model_input["guidance_scale"] = settings.instantid_guidance_scale
                model_input["num_inference_steps"] = settings.instantid_num_inference_steps
                model_input["width"] = out_w
                model_input["height"] = out_h

                if idx == 0:
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_message_id,
                        text=f"Генерирую в стиле «{style.title}». Обычно это занимает до пары минут…",
                    )
                else:
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_message_id,
                        text="Плохо вижу лицо — пробую автоматически приблизить…",
                    )

                res = await run_prediction_and_wait(
                    api_token=settings.replicate_api_token,
                    model_version=model_version_to_use,
                    model_input=model_input,
                    max_seconds=settings.replicate_max_seconds,
                    poll_seconds=2.0,
                )

                if res.status == "succeeded":
                    # Для paid-продукта качество критично: сразу улучшаем резкость/детали апскейлом.
                    out_for_send = res.output
                    out_url = _first_output_url(out_for_send)
                    if out_url:
                        await bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=status_message_id,
                            text="Улучшаю качество (upscale)…",
                        )
                        up_res = await run_prediction_and_wait(
                            api_token=settings.replicate_api_token,
                            model_version=settings.upscale_model_version,
                            model_input={
                                "image": out_url,
                                "scale": float(settings.upscale_scale),
                                "face_enhance": bool(settings.upscale_face_enhance),
                            },
                            max_seconds=settings.replicate_max_seconds,
                            poll_seconds=2.0,
                        )
                        if up_res.status == "succeeded":
                            out_for_send = up_res.output

                    await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text="Скачиваю результат…")
                    out_bytes = await download_output_image_bytes(out_for_send)
                    out_bytes = _postprocess_output(style_id, out_bytes)
                    bio = io.BytesIO(out_bytes)
                    bio.name = f"{settings.app_name}_{style.id}.png"
                    await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text="Отправляю в Telegram…")
                    await _safe_send_document(
                        bot=bot,
                        chat_id=chat_id,
                        document=bio,
                        caption=f"Готово (без сжатия): «{style.title}»",
                    )
                    profile = store.get_user(user_id)
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_message_id,
                        text="Готово. Хочешь другой стиль на это же фото?",
                        reply_markup=_start_keyboard(profile),
                    )
                    return

                err_text = str(res.error or "").strip()
                if "face detector could not find a face" in err_text.lower():
                    last_face_err = err_text
                    continue
                if "nsfw content detected" in err_text.lower():
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_message_id,
                        text=(
                            "Фильтр безопасности Replicate решил, что запрос слишком 18+.\n\n"
                            "Что делать:\n"
                            "- попробуй стиль «Ресторан (safe)»\n"
                            "- убери слова типа mini/off-shoulder/lingerie и акцент на теле\n"
                            "- оставь “classy, fully clothed, non-sexual”\n"
                        ),
                    )
                    return

                msg = f"Не получилось обработать изображение (status={res.status})."
                if err_text:
                    msg += f"\nОшибка: {err_text}"
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=msg)
                return

            # Всё ещё не нашли лицо
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text=(
                    "Я всё ещё не смог найти лицо на фото.\n\n"
                    "Попробуй другое селфи: лицо крупнее и по центру, без сильной тени."
                    + (f"\n\n(Тех. причина: {last_face_err})" if last_face_err else "")
                ),
            )
            return
        else:
            # PhotoMaker: identity-preserving для людей
            data_uris = [build_data_uri(_prepare_image_for_photomaker(b), filename="input.png") for b in refs]
            model_input = dict(settings.replicate_base_input)
            model_input["input_image"] = data_uris[0]
            if len(data_uris) > 1:
                model_input["input_image2"] = data_uris[1]
            if len(data_uris) > 2:
                model_input["input_image3"] = data_uris[2]
            if len(data_uris) > 3:
                model_input["input_image4"] = data_uris[3]
            # Обязательный триггер "img" в промпте
            if subject_gender == "male":
                model_input["prompt"] = f"A photo of a man img, {style.prompt}"
            elif subject_gender == "female":
                model_input["prompt"] = f"A photo of a woman img, {style.prompt}"
            else:
                model_input["prompt"] = f"A photo of a person img, {style.prompt}"
            model_input["negative_prompt"] = (
                "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
                "cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry"
            )
            model_input["num_steps"] = settings.photomaker_num_steps
            model_input["guidance_scale"] = settings.photomaker_guidance_scale
            model_input["style_strength_ratio"] = settings.photomaker_style_strength_ratio
            model_input["num_outputs"] = 1

        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message_id,
            text=f"Генерирую в стиле «{style.title}». Обычно это занимает до пары минут…",
        )

        result = await run_prediction_and_wait(
            api_token=settings.replicate_api_token,
            model_version=model_version_to_use,
            model_input=model_input,
            max_seconds=settings.replicate_max_seconds,
            poll_seconds=2.0,
        )

        if result.status != "succeeded":
            msg = f"Не получилось обработать изображение (status={result.status})."
            err_text = str(result.error or "").strip()
            if err_text:
                # Частая причина для InstantID: не нашлось лицо
                if "face detector could not find a face" in err_text.lower():
                    msg = (
                        "Я не смог найти лицо на фото.\n\n"
                        "Пришли другое фото, где:\n"
                        "- лицо видно крупно (селфи/портрет)\n"
                        "- одно лицо в кадре\n"
                        "- нормальный свет, без сильной тени\n"
                        "- без маски/закрытого лица (очки/капюшон могут мешать)"
                    )
                else:
                    msg += f"\nОшибка: {err_text}"
            await _safe_edit_status(bot, chat_id, status_message_id, text=msg)
            return

        await _safe_edit_status(bot, chat_id, status_message_id, text="Скачиваю результат…")
        out_bytes = await download_output_image_bytes(result.output)
        out_bytes = _postprocess_output(style_id, out_bytes)
        bio = io.BytesIO(out_bytes)
        bio.name = f"{settings.app_name}_{style.id}.png"

        # Отправляем как документ, чтобы Telegram не "пережал" качество.
        await _safe_edit_status(bot, chat_id, status_message_id, text="Отправляю в Telegram…")
        await _safe_send_document(
            bot=bot,
            chat_id=chat_id,
            document=bio,
            caption=f"Готово (без сжатия): «{style.title}»",
        )
        profile = store.get_user(user_id)
        await _safe_edit_status(
            bot, chat_id, status_message_id,
            text="Готово. Хочешь другой стиль на это же фото?",
            reply_markup=_start_keyboard(profile),
        )

    except AstriaError as e:
        logger.warning("Astria error: %s", e)
        gen_type = "persona" if is_persona_style else "express"
        await alert_generation_error(user_id, str(e), gen_type)
        err_msg = PERSONA_ERROR_MESSAGE if is_persona_style else USER_FRIENDLY_ERROR
        if is_persona_style and context:
            profile = store.get_user(user_id)
            gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
            page = context.user_data.get(USERDATA_PERSONA_STYLE_PAGE, 0)
            await _safe_edit_status(
                bot, chat_id, status_message_id, text=err_msg,
                reply_markup=_persona_styles_keyboard(gender, page),
            )
        else:
            await _safe_edit_status(bot, chat_id, status_message_id, text=err_msg)
    except Exception as e:
        logger.error("Ошибка PrismaLab job: %s", e, exc_info=True)
        gen_type = "persona" if is_persona_style else "express"
        await alert_generation_error(user_id, str(e), gen_type)
        try:
            if is_persona_style:
                err_text = PERSONA_ERROR_MESSAGE
                if context:
                    profile = store.get_user(user_id)
                    gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
                    page = context.user_data.get(USERDATA_PERSONA_STYLE_PAGE, 0)
                    await _safe_edit_status(
                        bot, chat_id, status_message_id, text=err_text,
                        reply_markup=_persona_styles_keyboard(gender, page),
                    )
                else:
                    await _safe_edit_status(bot, chat_id, status_message_id, text=err_text)
            else:
                err_text = USER_FRIENDLY_ERROR
                await _safe_edit_status(bot, chat_id, status_message_id, text=err_text)
        except Exception:
            pass


async def handle_kie_test_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик для тестирования KIE API (разные модели)"""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    
    # Парсим модель из callback_data: "pl_kie_test:model_name" или "pl_kie_test:model_name:param" или "pl_kie_test" (дефолт)
    data = query.data or ""
    upscale_factor_from_button = None
    if ":" in data:
        parts = data.split(":", 2)
        model_name = parts[1]
        # Если есть третий параметр - это upscale_factor для Topaz
        if len(parts) > 2:
            upscale_factor_from_button = parts[2]
    else:
        model_name = "nano-banana-pro"  # Дефолт
    
    # Названия моделей и их лейблы
    model_labels = {
        "google/imagen4-ultra": "🚀 Imagen4 Ultra",
        "flux-2/pro-image-to-image": "⚡ Flux Pro Img2Img",
        "flux-2/flex-image-to-image": "⚡ Flux Flex Img2Img",
        "flux-2/flex-text-to-image": "⚡ Flux Flex T2I",
        "flux-2/pro-text-to-image": "⚡ Flux Pro T2I",
        "ideogram/v3-text-to-image": "🎨 Ideogram V3 T2I",
        "ideogram/v3-edit": "✏️ Ideogram V3 Edit",
        "ideogram/v3-remix": "🔄 Ideogram V3 Remix",
        "ideogram/character": "👤 Character",
        "ideogram/character-edit": "✏️ Character Edit",
        "ideogram/character-remix": "🔄 Character Remix",
        "topaz/image-upscale": "🔍 Topaz Upscale",
        "recraft/crisp-upscale": "🔍 Crisp Upscale",
        "nano-banana-pro": "🍌 Nano Banana Pro",
        "seedream/4.5-edit": "🎨 Seedream 4.5",
        "seedream/4.5-text-to-image": "🎨 Seedream 4.5",
    }
    # Для отображения используем короткое имя
    model_label = model_labels.get(model_name, model_name)
    
    settings = load_settings()
    if not settings.kie_api_key:
        await query.edit_message_text(
            USER_FRIENDLY_ERROR,
        )
        return
    
    # Для Nano Banana проверяем оба источника фото (обычные и multi)
    # Всегда используем обычные фото (USERDATA_PHOTO_FILE_IDS)
    photo_file_ids = list(context.user_data.get(USERDATA_PHOTO_FILE_IDS, []))
    
    if not photo_file_ids:
        await query.edit_message_text("Сначала отправь фото, потом нажимай кнопку.")
        return
    
    user_id = int(query.from_user.id) if query.from_user else 0
    gen_lock = await _acquire_user_generation_lock(user_id)
    if gen_lock is None:
        await query.edit_message_text("Пожалуйста, немного подождите. Ещё обрабатываю прошлый запрос")
        return
    
    status_msg = await query.message.reply_text(f"{model_label} Запускаю генерацию…")
    
    async def runner():
        try:
            bot = context.bot
            chat_id = query.message.chat_id
            status_message_id = status_msg.message_id
            
            # Скачиваем фото пользователя
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text="📥 Скачиваю твоё фото…",
            )
            
            # Берём последнее (самое свежее) фото из списка
            photo_file_id = photo_file_ids[-1] if photo_file_ids else None
            if not photo_file_id:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message_id,
                    text="❌ Не найдено фото. Отправь фото заново.",
                )
                return
            photo_bytes = await _safe_get_file_bytes(bot, photo_file_id)
            
            # Загружаем фото в KIE для получения публичного URL
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text="📤 Загружаю фото…",
            )
            
            import secrets
            random_id = secrets.token_hex(8)
            uploaded_url = await asyncio.to_thread(
                kie_upload_file_base64,
                api_key=settings.kie_api_key,
                image_bytes=photo_bytes,
                file_name=f"user_photo_{random_id}.jpg",
                upload_path="user-uploads",
                timeout_s=60.0,
            )
            
            logger.info(f"KIE: фото загружено, URL: {uploaded_url}")
            
            # Промпт для Character, Seedream и Nano Banana
            if "ideogram/character" in model_name.lower() or "seedream/4.5-text-to-image" in model_name.lower() or "nano-banana-pro" in model_name.lower():
                test_prompt = (
                    """IDENTICAL FACE AND FEATURES from reference photo, same skin tone, ultra high detail face. A dramatic urban portrait bathed in cool blue neon light from unseen city signs. The woman wears a sleek black turtleneck, rain droplets glistening on her shoulders. Hard side lighting carves sharp shadows across her cheekbones and jawline, while distant street lights create bokeh orbs in the background. Her direct, unflinching gaze holds both vulnerability and strength, captured in crisp high-contrast that emphasizes every subtle facial detail"""
                )
            else:
                # Единый промпт для остальных моделей (для удобства тестирования)
                # Используем нейтральный язык, чтобы избежать NSFW фильтров
                test_prompt = (
                    "Place the person from the uploaded portrait, wearing a casual white blouse, in a peaceful garden setting. "
                    "The scene should feature vibrant green plants and colorful flowers, with soft sunlight filtering through the leaves. "
                    "The person should be sitting on a wooden bench, holding a book and smiling gently. "
                    "The background should be filled with lush greenery, with a serene, tranquil atmosphere. "
                    "Golden afternoon light should highlight the person's face and create soft shadows on the ground, "
                    "adding a peaceful, reflective mood to the scene. Professional photography, family-friendly, safe for work."
                )
            negative_prompt = (
                "blurry, lowres, bad quality, low quality, jpeg artifacts, out of focus, "
                "deformed face, face distortion, changed facial features, different person, "
                "asymmetrical eyes, duplicated face, two faces, multiple people, "
                "plastic skin, doll face, over-smooth skin, "
                "cartoon, anime, illustration, text, watermark, logo"
            )
            
            # Для обратной совместимости оставляем старую логику, но она не используется
            if False:  # Отключено - используем единый промпт выше
                # Промпт для остальных моделей - используем NYC для обеих моделей
                nyc_style = get_style("nyc_70s")
                if nyc_style:
                    # Используем промпт NYC для обеих моделей
                    test_prompt = f"{nyc_style.prompt}, preserve the person's face and identity from the reference image, photorealistic, high detail"
                    negative_prompt = nyc_style.negative_prompt or None
                else:
                    # Fallback, если стиль не найден
                    test_prompt = (
                        "Cinematic vintage 35mm film photo, 1970s New York City street scene at night after rain. "
                        "A single person standing near a classic 1970s American car parked on the curb. "
                        "Wet asphalt, warm street lamps, soft reflections on the road, subtle fog in the distance. "
                        "Brick buildings with fire escapes, vintage shop signs, blurred taxi lights, distant skyline. "
                        "Kodak film look: natural skin texture, soft warm tones, slightly faded colors, authentic film grain, "
                        "dust, tiny scratches, subtle light leaks. "
                        "Candid documentary photo, realistic, photorealistic, high detail. "
                        "85mm lens, shallow depth of field, cinematic lighting. "
                        "Keep the same person and recognizable facial features. Preserve the person's face and identity from the reference image."
                    )
                    negative_prompt = (
                        "blurry, lowres, bad quality, low quality, jpeg artifacts, out of focus, "
                        "deformed face, face distortion, changed facial features, different person, "
                        "asymmetrical eyes, duplicated face, two faces, multiple people, "
                        "plastic skin, doll face, over-smooth skin, "
                        "cartoon, anime, illustration, text, watermark, logo"
                    )
            
            logger.info(f"KIE: используем промпт для {model_name} (длина {len(test_prompt)}): {test_prompt[:100]}...")
            
            # Инициализируем переменные для всех моделей
            use_upscale_factor = None
            
            # Для upscale моделей определяем коэффициент ДО использования
            if "upscale" in model_name.lower() or "topaz" in model_name.lower() or "recraft" in model_name.lower():
                if "topaz" in model_name.lower():
                    use_upscale_factor = upscale_factor_from_button if upscale_factor_from_button else "2"
            
            # Показываем промпт пользователю
            prompt_preview = test_prompt[:100] + "..." if len(test_prompt) > 100 else test_prompt
            # Определяем режим работы модели
            if "upscale" in model_name.lower() or "topaz" in model_name.lower() or "recraft" in model_name.lower():
                # Для upscale показываем коэффициент
                factor_text = f" ({use_upscale_factor}x)" if use_upscale_factor else ""
                mode_text = f"upscale{factor_text}"
            elif "ideogram/character" in model_name.lower():
                mode_text = "character generation"
            elif "image-to-image" in model_name.lower() or "edit" in model_name.lower() or "remix" in model_name.lower():
                mode_text = "image-to-image"
            else:
                mode_text = "text-to-image"
            # Для upscale не показываем промпт (его нет)
            status_text = f"{model_label} Генерирую ({mode_text})\n\n"
            if not ("upscale" in model_name.lower() or "topaz" in model_name.lower() or "recraft" in model_name.lower()):
                status_text += f"📝 Промпт: {prompt_preview}\n\n"
            status_text += f"⏱ Это может занять 30-60 секунд."
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text=status_text,
            )
            
            # Параметры для разных моделей
            # Используем локальную переменную для модели, которая может быть изменена
            use_model_name = model_name
            if "google/imagen4-ultra" in model_name.lower():
                # Imagen4 Ultra - только text-to-image
                use_resolution = None
                use_quality = None
                use_aspect_ratio = "1:1"
                use_image_input = None  # Не поддерживает image-to-image
                logger.info(f"KIE: используем Imagen4 Ultra (text-to-image only)")
            elif "flux-2" in model_name.lower():
                # Flux-2 модели
                use_resolution = "2K"
                use_quality = None
                use_aspect_ratio = "1:1"  # Flux-2 ТРЕБУЕТ aspect_ratio (обязательный)
                if "image-to-image" in model_name.lower():
                    use_image_input = [uploaded_url]  # Используется как input_urls
                else:
                    use_image_input = None  # Text-to-image не требует изображения
                logger.info(f"KIE: используем Flux-2 модель {model_name}")
            elif "ideogram/v3" in model_name.lower():
                # Ideogram V3 модели
                use_resolution = None
                use_quality = None
                use_aspect_ratio = None  # Используется image_size
                if "edit" in model_name.lower() or "remix" in model_name.lower():
                    use_image_input = [uploaded_url]  # Используется как image_url (один URL)
                else:
                    use_image_input = None  # Text-to-image не требует изображения
                logger.info(f"KIE: используем Ideogram V3 модель {model_name}")
            elif "ideogram/character" in model_name.lower():
                # Ideogram Character модели используют reference_image_urls для сохранения лица
                use_resolution = None  # Не используется для Ideogram
                use_quality = None  # Не используется для Ideogram
                use_aspect_ratio = None  # Ideogram использует image_size вместо aspect_ratio
                use_image_input = [uploaded_url]  # Передаём как reference_image_urls
                logger.info(f"KIE: используем Ideogram Character модель {model_name} с reference_image_urls")
            elif "upscale" in model_name.lower() or "topaz" in model_name.lower() or "recraft" in model_name.lower():
                # Upscale модели - улучшают качество исходного фото (рефа) перед генерацией
                use_resolution = None
                use_quality = None
                use_aspect_ratio = None
                use_image_input = [uploaded_url]  # Используется как image_url или image
                # Для Topaz коэффициент уже определён выше (в начале функции), для Recraft не нужен
                logger.info(f"KIE: используем Upscale модель {model_name} (factor: {use_upscale_factor}) для улучшения исходного фото")
            elif "seedream" in model_name.lower():
                # Seedream 4.5 параметры
                use_resolution = None  # Не используется для Seedream
                use_quality = "high"  # basic (2K) или high (4K)
                use_aspect_ratio = "1:1"
                # Для image-to-image используем edit версию, для text-to-image - без image_urls
                # Если это text-to-image, но нужен reference - переключаемся на edit
                if "text-to-image" in model_name.lower() and uploaded_url:
                    # Text-to-image не поддерживает image_urls, переключаемся на edit
                    logger.info(f"KIE: text-to-image не поддерживает image_urls, переключаюсь на edit версию")
                    use_model_name = "seedream/4.5-edit"
                    use_image_input = [uploaded_url]
                elif "edit" in model_name.lower():
                    use_image_input = [uploaded_url]
                else:
                    use_image_input = None
            else:
                # Nano Banana и другие
                use_resolution = "2K"
                use_quality = None
                use_aspect_ratio = "1:1"
                # Для всех моделей используем одно фото
                use_image_input = [uploaded_url]
            
            # Для upscale моделей промпт не нужен (они только увеличивают разрешение)
            # Передаём None вместо пустой строки, чтобы промпт не добавлялся в input_data
            is_upscale = "upscale" in model_name.lower() or "topaz" in model_name.lower() or "recraft" in model_name.lower()
            use_prompt = test_prompt if not is_upscale else None
            use_negative_prompt = negative_prompt if not is_upscale else None
            
            # Для Topaz upscale_factor обязателен - убеждаемся что он установлен
            if "topaz" in use_model_name.lower() and not use_upscale_factor:
                use_upscale_factor = "2"  # Дефолт для Topaz
                logger.warning(f"KIE: use_upscale_factor был None для Topaz, установил дефолт 2x")
            
            logger.info(f"KIE: финальные параметры для {use_model_name}: upscale_factor={use_upscale_factor}, prompt={'есть' if use_prompt else 'нет'}, image_input={'есть' if use_image_input else 'нет'}")
            
            # Запускаем задачу через KIE API с reference image (если поддерживается)
            kie_result = await kie_run_task_and_wait(
                api_key=settings.kie_api_key,
                model=use_model_name,
                prompt=use_prompt,
                image_input=use_image_input,
                aspect_ratio=use_aspect_ratio,
                resolution=use_resolution,
                quality=use_quality,
                negative_prompt=use_negative_prompt,
                output_format="png",
                upscale_factor=use_upscale_factor,
                max_seconds=settings.kie_max_seconds,
                poll_seconds=3.0,
            )
            
            if not kie_result.image_url:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message_id,
                    text="❌ Генератор вернул задачу без изображения. Попробуй ещё раз.",
                )
                return
            
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text="Скачиваю результат…",
            )
            
            # Скачиваем изображение
            image_bytes = await asyncio.to_thread(
                kie_download_image_bytes,
                kie_result.image_url,
                timeout_s=30.0,
            )
            
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text="Отправляю в Telegram…",
            )
            
            # Отправляем результат
            import io
            bio = io.BytesIO(image_bytes)
            # Очищаем имя модели для имени файла
            safe_model_name = model_name.replace("/", "_").replace(":", "_")
            bio.name = f"{settings.app_name}_kie_{safe_model_name}.png"
            
            # Определяем режим для caption
            if "upscale" in model_name.lower() or "topaz" in model_name.lower() or "recraft" in model_name.lower():
                mode_text = "upscale"
            elif "ideogram/character" in model_name.lower():
                mode_text = "character generation"
            elif "image-to-image" in model_name.lower() or "edit" in model_name.lower() or "remix" in model_name.lower():
                mode_text = "image-to-image"
            else:
                mode_text = "text-to-image"
            await _safe_send_document(
                bot=bot,
                chat_id=chat_id,
                document=bio,
                caption=f"{model_label} Готово ({mode_text})",
            )
            # Логируем успешную генерацию для аналитики
            try:
                store.log_event(user_id, "generation", {"mode": mode_text, "model": model_name, "provider": "kie"})
            except Exception:
                pass
            
            await bot.delete_message(chat_id=chat_id, message_id=status_message_id)
            
        except KieError as e:
            error_msg = str(e)
            logger.error(f"KIE ошибка: {error_msg}")
            
            # Специальная обработка NSFW ошибки
            if "nsfw" in error_msg.lower():
                user_msg = (
                    "❌ Запрос заблокирован как небезопасный контент (NSFW).\n\n"
                    "Возможные причины:\n"
                    "• Загруженное фото содержит контент, который модель считает небезопасным\n"
                    "• Промпт содержит слова, которые вызывают срабатывание фильтра\n\n"
                    "Попробуй:\n"
                    "• Загрузить другое фото (портрет в обычной одежде)\n"
                    "• Использовать другую модель"
                )
            else:
                user_msg = USER_FRIENDLY_ERROR
            
            await bot.edit_message_text(
                chat_id=query.message.chat_id,
                message_id=status_msg.message_id,
                text=user_msg,
            )
        except Exception as e:
            logger.error("Неожиданная ошибка в KIE тесте: %s", e, exc_info=True)
            await bot.edit_message_text(
                chat_id=query.message.chat_id,
                message_id=status_msg.message_id,
                text=USER_FRIENDLY_ERROR,
            )
        finally:
            gen_lock.release()
    
    context.application.create_task(runner())


async def handle_astria_generate_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик для генерации через Astria (после создания LoRA/FaceID)"""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    
    # Используем тот же обработчик, что и для стилей, но с фиксированным style_id="test"
    # Это запустит генерацию через Astria с единым промптом
    context.user_data["pl_last_style_id"] = "test"
    
    # Для Astria генерации фото не обязательно - используется LoRA/FaceID из базы
    # Но если есть фото - используем их, если нет - работаем только с LoRA/FaceID
    photo_file_ids = list(context.user_data.get(USERDATA_PHOTO_FILE_IDS, []))
    
    # Если нет фото, но есть LoRA/FaceID - всё равно можно генерировать
    # Проверяем наличие LoRA/FaceID в базе
    user_id = int(query.from_user.id) if query.from_user else 0
    if not photo_file_ids and user_id:
        from prismalab.storage import PrismaLabStore
        store = PrismaLabStore()
        user_profile = store.get_user(user_id)
        if not user_profile or (not user_profile.astria_lora_tune_id and not user_profile.astria_tune_id):
            await query.edit_message_text(
                "❌ Для генерации нужно:\n"
                "• Либо отправить фото\n"
                "• Либо создать LoRA (10 фото) или FaceID (1 фото)"
            )
            return
    
    settings = load_settings()
    ps = _get_prompt_strength(settings, context)
    uid = int(query.from_user.id) if query.from_user else 0
    gender = context.user_data.get(USERDATA_SUBJECT_GENDER)
    use_personal_requested = _is_personal_enabled(context)
    
    # Промпт про смеющуюся девушку
    test_prompt = (
        "same reference female character, candid laugh, head turned left, eyes squinting, looking away from camera, "
        "tropical greenery background, golden hour, realistic photo, 35mm, shallow depth of field. "
        "identity preserved, natural expression change, no face morphing, no distortion, correct anatomy, no extra fingers, no text"
    )
    
    status_msg = await query.message.reply_text("Запускаю генерацию…")
    
    await _run_style_job(
        bot=context.bot,
        chat_id=query.message.chat_id,
        photo_file_ids=photo_file_ids,
        style_id="test",
        settings=settings,
        status_message_id=status_msg.message_id,
        prompt_strength=ps,
        user_id=uid,
        subject_gender=gender,
        use_personal_requested=use_personal_requested,
        test_prompt=test_prompt,
        context=context,
    )


async def handle_style_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()

    data = query.data or ""
    if not data.startswith("pl_style:"):
        return
    style_id = data.split(":", 1)[1]
    
    # Сохраняем последний выбранный стиль для использования в KIE
    context.user_data["pl_last_style_id"] = style_id

    photo_file_ids = list(context.user_data.get(USERDATA_PHOTO_FILE_IDS, []))
    if not photo_file_ids:
        await query.edit_message_text("Сначала отправь фото, потом выбирай стиль.")
        return

    uid = int(query.from_user.id) if query.from_user else 0

    # Для кнопки "Тест" - через Astria
    if style_id == "test":
        gen_lock = await _acquire_user_generation_lock(uid)
        if gen_lock is None:
            await query.edit_message_text("Пожалуйста, немного подождите. Ещё обрабатываю прошлый запрос")
            return
        status_msg = await query.message.reply_text("Принял. Запускаю обработку…")
        settings = load_settings()
        ps = _get_prompt_strength(settings, context)
        gender = context.user_data.get(USERDATA_SUBJECT_GENDER)
        use_personal_requested = _is_personal_enabled(context)
        test_prompt = "Этот молодой человек стоит в темном костюме возле небосркеба из зеркального стекла. Рядом стоит черный автомобиль. На заднем плане длинная улица уходящая вдаль. Солнечная погода."
        context.user_data[USERDATA_USE_PERSONAL] = False

        async def runner():
            try:
                await _run_style_job(
                    bot=context.bot,
                    chat_id=query.message.chat_id,
                    photo_file_ids=photo_file_ids,
                    style_id=style_id,
                    settings=settings,
                    status_message_id=status_msg.message_id,
                    prompt_strength=ps,
                    user_id=uid,
                    subject_gender=gender,
                    use_personal_requested=use_personal_requested,
                    test_prompt=test_prompt,
                    context=context,
                )
            finally:
                gen_lock.release()

        context.application.create_task(runner())
        return
    
    # Для остальных стилей - через Astria
    # KIE запускается отдельной кнопкой и использует сохранённый стиль
    gen_lock = await _acquire_user_generation_lock(uid)
    if gen_lock is None:
        await query.edit_message_text("Пожалуйста, немного подождите. Ещё обрабатываю прошлый запрос")
        return

    status_msg = await query.message.reply_text("Принял. Запускаю обработку…")
    settings = load_settings()
    ps = _get_prompt_strength(settings, context)
    gender = context.user_data.get(USERDATA_SUBJECT_GENDER)
    use_personal_requested = _is_personal_enabled(context)

    async def runner():
        try:
            await _run_style_job(
                bot=context.bot,
                chat_id=query.message.chat_id,
                photo_file_ids=photo_file_ids,
                style_id=style_id,
                settings=settings,
                status_message_id=status_msg.message_id,
                prompt_strength=ps,
                user_id=uid,
                subject_gender=gender,
                use_personal_requested=use_personal_requested,
                test_prompt=None,
                context=context,
            )
        finally:
            gen_lock.release()

    context.application.create_task(runner())


def main() -> None:
    settings = load_settings()
    _guard_dev_only_flags()
    if not settings.bot_token:
        raise ValueError("PRISMALAB_BOT_TOKEN (или BOT_TOKEN) не найден")

    # Увеличенные таймауты: при нестабильной сети/VPN бот падал с ConnectTimeout при get_me()
    application = (
        Application.builder()
        .token(settings.bot_token)
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .get_updates_connect_timeout(30.0)
        .get_updates_read_timeout(60.0)
        .build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("menu", menu_command))
    application.add_handler(CommandHandler("profile", profile_command))
    application.add_handler(CommandHandler("newpersona", newpersona_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("getfileid", getfileid_command))
    application.add_handler(CallbackQueryHandler(handle_getfileid_album_callback, pattern="^pl_getfileid_album:"))
    application.add_handler(CommandHandler("tips", tips_command))
    application.add_handler(CallbackQueryHandler(handle_start_fast_callback, pattern="^pl_start_fast$"))
    application.add_handler(CallbackQueryHandler(handle_start_persona_callback, pattern="^pl_start_persona$"))
    application.add_handler(CallbackQueryHandler(handle_persona_create_callback, pattern="^pl_persona_create$"))
    application.add_handler(CallbackQueryHandler(handle_persona_show_credits_out_callback, pattern="^pl_persona_show_credits_out$"))
    application.add_handler(CallbackQueryHandler(handle_persona_topup_callback, pattern="^pl_persona_topup$"))
    application.add_handler(CallbackQueryHandler(handle_persona_topup_buy_callback, pattern="^pl_persona_topup_buy:"))
    application.add_handler(CallbackQueryHandler(handle_persona_topup_confirm_callback, pattern="^pl_persona_topup_confirm:"))
    application.add_handler(CallbackQueryHandler(handle_persona_recreate_callback, pattern="^pl_persona_recreate$"))
    application.add_handler(CallbackQueryHandler(handle_persona_recreate_confirm_callback, pattern="^pl_persona_recreate_confirm$"))
    application.add_handler(CallbackQueryHandler(handle_persona_recreate_cancel_callback, pattern="^pl_persona_recreate_cancel$"))
    application.add_handler(CallbackQueryHandler(handle_persona_gender_callback, pattern="^pl_persona_gender:"))
    application.add_handler(CallbackQueryHandler(handle_persona_buy_callback, pattern="^pl_persona_buy:"))
    application.add_handler(CallbackQueryHandler(handle_persona_confirm_pay_callback, pattern="^pl_persona_confirm_pay:"))
    application.add_handler(CallbackQueryHandler(handle_persona_got_it_callback, pattern="^pl_persona_got_it$"))
    application.add_handler(CallbackQueryHandler(handle_persona_reset_photos_callback, pattern="^pl_persona_reset_photos$"))
    application.add_handler(CallbackQueryHandler(handle_persona_pack_reset_photos_callback, pattern="^pl_persona_pack_reset_photos$"))
    application.add_handler(CallbackQueryHandler(handle_miniapp_pack_upload_callback, pattern="^pl_pack_upload:"))
    application.add_handler(CallbackQueryHandler(handle_persona_check_status_callback, pattern="^pl_persona_check_status$"))
    application.add_handler(CallbackQueryHandler(handle_persona_page_callback, pattern="^pl_persona_page:"))
    application.add_handler(CallbackQueryHandler(handle_persona_style_callback, pattern="^pl_persona_style:"))
    application.add_handler(CallbackQueryHandler(handle_persona_packs_callback, pattern="^pl_persona_packs$"))
    application.add_handler(CallbackQueryHandler(handle_persona_pack_buy_callback, pattern="^pl_persona_pack_buy:"))
    application.add_handler(CallbackQueryHandler(handle_persona_back_callback, pattern="^pl_persona_back$"))
    application.add_handler(CallbackQueryHandler(handle_start_tariffs_callback, pattern="^pl_start_tariffs$"))
    application.add_handler(CallbackQueryHandler(handle_start_examples_callback, pattern="^pl_start_examples$"))
    application.add_handler(CallbackQueryHandler(handle_examples_show_albums_callback, pattern="^pl_examples_show_albums$"))
    application.add_handler(CallbackQueryHandler(handle_examples_page_callback, pattern="^pl_examples_page:"))
    application.add_handler(CallbackQueryHandler(handle_start_faq_callback, pattern="^pl_start_faq$"))
    application.add_handler(CallbackQueryHandler(handle_help_callback, pattern="^pl_help$"))
    application.add_handler(CallbackQueryHandler(handle_profile_callback, pattern="^pl_profile$"))
    application.add_handler(CallbackQueryHandler(handle_profile_toggle_gender_callback, pattern="^pl_profile_toggle_gender$"))
    application.add_handler(CallbackQueryHandler(handle_profile_fast_tariffs_callback, pattern="^pl_profile_fast_tariffs$"))
    application.add_handler(CallbackQueryHandler(handle_fast_gender_callback, pattern="^pl_fast_gender:"))
    application.add_handler(CallbackQueryHandler(handle_fast_page_callback, pattern="^pl_fast_page:"))
    application.add_handler(CallbackQueryHandler(handle_fast_back_callback, pattern="^pl_fast_back$"))
    application.add_handler(CallbackQueryHandler(handle_fast_show_tariffs_callback, pattern="^pl_fast_show_tariffs$"))
    application.add_handler(CallbackQueryHandler(handle_fast_buy_callback, pattern="^pl_fast_buy:"))
    application.add_handler(PreCheckoutQueryHandler(handle_pre_checkout))
    application.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, handle_successful_payment))
    application.add_handler(CallbackQueryHandler(handle_fast_upload_photo_callback, pattern="^pl_fast_upload_photo$"))
    application.add_handler(CallbackQueryHandler(handle_fast_change_style_callback, pattern="^pl_fast_change_style$"))
    application.add_handler(CallbackQueryHandler(handle_fast_show_ready_callback, pattern="^pl_fast_show_ready$"))
    application.add_handler(CallbackQueryHandler(handle_fast_style_callback, pattern="^pl_fast_style:"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    application.add_handler(CallbackQueryHandler(handle_prompt_strength_callback, pattern="^pl_ps:"))
    application.add_handler(CallbackQueryHandler(handle_gender_callback, pattern="^pl_gender:"))
    application.add_handler(CallbackQueryHandler(handle_personal_toggle_callback, pattern="^pl_personal:"))
    # Обработчик handle_nano_banana_multi_callback удалён - больше не используется
    application.add_handler(CallbackQueryHandler(handle_reset_callback, pattern="^pl_reset$"))
    application.add_handler(CallbackQueryHandler(handle_style_callback, pattern="^pl_style:"))
    application.add_handler(CallbackQueryHandler(handle_astria_generate_callback, pattern="^pl_astria_generate$"))
    application.add_handler(CallbackQueryHandler(handle_kie_test_callback, pattern="^pl_kie_test"))

    async def post_init(app: Application) -> None:
        from prismalab.payment import run_webhook_server

        # Расширенный пул потоков для asyncio.to_thread (Astria, KIE, загрузки)
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=32, thread_name_prefix="prismalab")
        loop.set_default_executor(executor)

        run_webhook_server(app.bot, store, application=app)

        # Дневной отчёт в 21:00 по Москве (UTC+3)
        async def daily_report_job(context: ContextTypes.DEFAULT_TYPE) -> None:
            await alert_daily_report(store)

        from datetime import time as dt_time
        job_queue = app.job_queue
        if job_queue:
            # 21:00 MSK = 18:00 UTC
            job_queue.run_daily(daily_report_job, time=dt_time(hour=18, minute=0, second=0))
            logger.info("Дневной отчёт запланирован на 21:00 MSK")

        default_commands = [
            BotCommand("menu", "🏠 Главное меню"),
            BotCommand("profile", "👤 Профиль"),
            BotCommand("newpersona", "✨ Создать новую Персону"),
            BotCommand("help", "❓ Помощь"),
        ]
        await app.bot.set_my_commands(default_commands, scope=BotCommandScopeDefault())
        if OWNER_ID:
            owner_commands = default_commands + [
                BotCommand("tips", "📋 Шпаргалка"),
                BotCommand("getfileid", "🖼 Добавить фото в примеры"),
            ]
            await app.bot.set_my_commands(owner_commands, scope=BotCommandScopeChat(chat_id=OWNER_ID))

    async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Глобальный обработчик ошибок: логируем и показываем пользователю дружественное сообщение."""
        # Игнорируем BadRequest (например, "Message is not modified" при двойном клике)
        if isinstance(context.error, BadRequest):
            logger.debug("BadRequest проигнорирован: %s", context.error)
            return
        logger.error("Необработанное исключение:", exc_info=context.error)
        if isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text(USER_FRIENDLY_ERROR)
            except Exception:
                pass

    application.add_error_handler(_error_handler)
    application.post_init = post_init
    logger.info("%s запущен", settings.app_name)
    if use_telegram_payments():
        logger.info("Оплата: Telegram Payments (инвойс в боте)")
    elif use_yookassa():
        logger.info("ЮKassa: включена, оплата по ссылке + webhook")
    else:
        logger.warning("Оплата не настроена. Режим симуляции — кредиты без оплаты")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
