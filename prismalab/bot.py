#!/usr/bin/env python3
"""
PrismaLab — Telegram-бот стилизации фото (Astria, KIE).

Запуск (из корня репозитория):
  python3 -m prismalab.bot
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

from telegram import (
    BotCommand,
    BotCommandScopeChat,
    BotCommandScopeDefault,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PreCheckoutQueryHandler,
    filters,
)

from prismalab.alerts import alert_daily_report, alert_generation_error
from prismalab.astria_client import (
    AstriaError,
)
from prismalab.astria_client import (
    download_first_image_bytes as astria_download_first_image_bytes,
)
from prismalab.astria_client import (
    run_prompt_and_wait as astria_run_prompt_and_wait,
)
from prismalab.kie_client import (
    KieError,
)
from prismalab.kie_client import (
    download_image_bytes as kie_download_image_bytes,
)
from prismalab.kie_client import (
    run_task_and_wait as kie_run_task_and_wait,
)
from prismalab.kie_client import (
    upload_file_base64 as kie_upload_file_base64,
)
from prismalab.payment import (
    use_telegram_payments,
    use_yookassa,
)
from prismalab.persona_prompts import PERSONA_STYLE_PROMPTS
from prismalab.settings import load_settings  # сначала загружаем .env
from prismalab.storage import PrismaLabStore
from prismalab.styles import get_style

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("prismalab")
# Важно: httpx на INFO логирует URL Telegram API, где есть токен бота.
logging.getLogger("httpx").setLevel(logging.WARNING)


# Активные pack-polling run_id: блокируем fallback, пока жив основной polling.
_pack_polling_active: set[str] = set()
# run_id, которые сейчас обрабатывает основной flow (включая обучение tune).
# Recovery проверяет этот set и не вмешивается.
_pack_processing_active: set[str] = set()


# --- Конфигурация, константы и feature flags импортируются из config.py ---
from prismalab.config import (  # noqa: E402
    ALLOWED_USERS,
    OWNER_ID,
    USER_FRIENDLY_ERROR,
    USERDATA_ASTRIA_FACEID_FILE_IDS,
    USERDATA_ASTRIA_LORA_FILE_IDS,
    USERDATA_MODE,
    USERDATA_NANO_BANANA_FILE_IDS,
    USERDATA_PHOTO_FILE_IDS,
    USERDATA_PROFILE_DELETE_JOB,
    USERDATA_PROMPT_STRENGTH,
    USERDATA_SUBJECT_GENDER,
    USERDATA_USE_PERSONAL,
    _guard_dev_only_flags,
)


# --- Офферы паков импортируются из pack_offers.py ---

from prismalab.messages import (  # noqa: E402
    PERSONA_CREDITS_OUT_MESSAGE,
    _fast_credits_word,
    _format_balance_express,
    _format_balance_persona,
    _generations_count_fast,
)

from prismalab.keyboards import (  # noqa: E402
    _persona_app_keyboard,
    _persona_credits_out_keyboard,
    _persona_recreate_confirm_keyboard,
    _start_keyboard,
)

from prismalab.telegram_utils import (  # noqa: E402
    _acquire_user_generation_lock,
    _safe_edit_status,
    _safe_get_file_bytes,
    _safe_send_document,
)

from prismalab.image_utils import (  # noqa: E402
    _postprocess_output,
)

store: PrismaLabStore | None = None


def _get_store() -> PrismaLabStore:
    """Lazy init store — не создаёт подключение при импорте модуля."""
    global store
    if store is None:
        store = PrismaLabStore()
        store.init_admin_tables()
    return store


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


# --- Персона ---

PERSONA_RECREATE_TARIFF_MESSAGE = """<b>Выберите тариф для новой Персоны</b>

Вы загрузите <b>10 фото</b>, мы обучим новую модель и начислим кредиты.

• Персона + 5 кредитов – 299 ₽
• Персона + 20 кредитов – 599 ₽
• Персона + 40 кредитов – 999 ₽"""


PERSONA_PACKS_MESSAGE = """<b>Готовые фотосеты</b>

Фотосет = фиксированная цена за серию готовых кадров.
После оплаты запускаем генерацию автоматически и присылаем весь результат в чат."""


PERSONA_ERROR_MESSAGE = "Произошла ошибка, кредит не списали. Попробуйте ещё раз"
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


    # Lock освобождается в caller через gen_lock.release()


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
        try:
            store.log_event(user_id, "reset")
        except Exception:
            pass
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
    skip_post_message: bool = False,
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
                    extra["reply_markup"] = _persona_app_keyboard()
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
            logger.info("[ASTRIA] LoRA/FaceID найден, фото не требуется.")
        if not refs and needs_photo:
            err_msg = PERSONA_ERROR_MESSAGE if is_persona_style else "Не нашёл фото для обработки."
            extra = {}
            if is_persona_style and context:
                extra["reply_markup"] = _persona_app_keyboard()
            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=err_msg, **extra)
            return

        # Только Astria (Replicate удалён)
        if not settings.astria_api_key:
            err_msg = PERSONA_ERROR_MESSAGE if is_persona_style else USER_FRIENDLY_ERROR
            extra = {}
            if is_persona_style and context:
                extra["reply_markup"] = _persona_app_keyboard()
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
        logger.info("[ASTRIA Generate] base_model будет Flux1.dev (1504944) для LoRA" if use_lora else "[ASTRIA Generate] base_model будет Realistic Vision (690204) для FaceID")

        if not active_tune_id:
            err_msg = PERSONA_ERROR_MESSAGE if is_persona_style else (
                "❌ Для генерации нужно сначала создать FaceID или LoRA tune.\n\n"
                "• 📸 FaceID (1 фото) - быстро, для одной сцены\n"
                "• 🎯 LoRA (10 фото) - качественно, для множества сцен"
            )
            extra = {}
            if is_persona_style and context:
                extra["reply_markup"] = _persona_app_keyboard()
            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=err_msg, **extra)
            return
        if use_lora:
            # LoRA генерация: используем <lora:{tune_id}:strength> в промпте
            # Inference на базовой модели Flux1.dev (1504944)
            lora_weight = 1.1  # Вес LoRA для сходства (1.0–1.2)

            # Промпт для Astria LoRA (lora_prompt_override — для Персоны)
            # Токен должен совпадать с name при обучении: person (бот) или woman/man (пак)
            stored_class = getattr(user_profile, "persona_lora_class_name", None) if user_profile else None
            lora_class = stored_class or "person"
            ohwx_token = f"ohwx {lora_class}"
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
            aspect_ratio="3:4",
            max_seconds=settings.astria_max_seconds,
            poll_seconds=6.0,
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
            if not skip_post_message:
                if credits <= 0:
                    profile = store.get_user(user_id)
                    text, reply_markup = _persona_credits_out_content(profile)
                else:
                    text = f"<b>Готово!</b>\n\nМожете вернуться в приложение ✨<b>Персона</b> и попробовать новые стили\n\n{_format_balance_persona(credits)}"
                    reply_markup = _persona_app_keyboard()
                await bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=reply_markup,
                    parse_mode="HTML",
                )
        else:
            profile = store.get_user(user_id)
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message_id,
                text="Готово. Хочешь другой стиль?",
                reply_markup=_start_keyboard(profile),
            )
        return

    except AstriaError as e:
        logger.warning("Astria error: %s", e)
        gen_type = "persona" if is_persona_style else "express"
        await alert_generation_error(user_id, str(e), gen_type)
        err_msg = PERSONA_ERROR_MESSAGE if is_persona_style else USER_FRIENDLY_ERROR
        if is_persona_style and context:
            await _safe_edit_status(
                bot, chat_id, status_message_id, text=err_msg,
                reply_markup=_persona_app_keyboard(),
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
                    await _safe_edit_status(
                        bot, chat_id, status_message_id, text=err_text,
                        reply_markup=_persona_app_keyboard(),
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
            status_text += "⏱ Это может занять 30-60 секунд."
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
                logger.info("KIE: используем Imagen4 Ultra (text-to-image only)")
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
                    logger.info("KIE: text-to-image не поддерживает image_urls, переключаюсь на edit версию")
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
                logger.warning("KIE: use_upscale_factor был None для Topaz, установил дефолт 2x")

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


async def handle_web_app_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик данных из Mini App (tg.sendData)."""
    user_id = update.effective_user.id
    if ALLOWED_USERS and user_id not in ALLOWED_USERS:
        return
    data_str = update.effective_message.web_app_data.data
    try:
        data = json.loads(data_str)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Invalid web_app_data from user %s: %s", user_id, data_str[:200])
        return

    action = data.get("action", "")
    logger.info("web_app_data from user %s: action=%s", user_id, action)

    if action == "persona_style_selected":
        style_id = data.get("style_id")
        style_slug = data.get("style_slug", "")
        style_title = data.get("style_title", "")

        store.log_event(user_id, "persona_style_selected", {
            "style_id": style_id,
            "style_slug": style_slug,
            "style_title": style_title,
        })

        # Проверяем, есть ли у юзера Персона
        profile = store.get_user(user_id)
        has_persona = bool(
            getattr(profile, "astria_lora_tune_id", None)
            or getattr(profile, "astria_lora_pack_tune_id", None)
        )

        if has_persona:
            await update.effective_message.reply_text(
                f"Вы выбрали стиль: *{style_title}*\n\n"
                "У вас уже есть Персона. Генерация фото в выбранном стиле будет доступна скоро!",
                parse_mode="Markdown",
            )
        else:
            # Сохраняем выбранный стиль в user_data для дальнейшего флоу
            context.user_data["pending_persona_style_id"] = style_id
            context.user_data["pending_persona_style_slug"] = style_slug
            context.user_data["pending_persona_style_title"] = style_title

            await update.effective_message.reply_text(
                f"Вы выбрали стиль: *{style_title}*\n\n"
                "Для генерации фото в этом стиле нужно сначала создать Персону. "
                "Нажмите /newpersona, чтобы начать.",
                parse_mode="Markdown",
            )

    elif action == "buy_credits":
        await update.effective_message.reply_text(
            "Используйте /menu для покупки кредитов."
        )


# --- Хэндлеры из handlers/ (импорт внизу файла, чтобы избежать circular imports) ---
from prismalab.handlers.navigation import (  # noqa: E402
    getfileid_command,
    handle_examples_page_callback,
    handle_examples_show_albums_callback,
    handle_fast_back_callback,
    handle_getfileid_album_callback,
    handle_help_callback,
    handle_profile_callback,
    handle_profile_fast_tariffs_callback,
    handle_profile_toggle_gender_callback,
    handle_start_examples_callback,
    handle_start_faq_callback,
    handle_start_fast_callback,
    handle_start_persona_callback,
    handle_start_tariffs_callback,
    help_command,
    menu_command,
    profile_command,
    start_command,
    tips_command,
)

from prismalab.handlers.fast_photo import (  # noqa: E402
    handle_fast_buy_callback,
    handle_fast_change_style_callback,
    handle_fast_gender_callback,
    handle_fast_page_callback,
    handle_fast_show_ready_callback,
    handle_fast_show_tariffs_callback,
    handle_fast_style_callback,
    handle_fast_upload_photo_callback,
    handle_pre_checkout,
    handle_successful_payment,
)

from prismalab.handlers.persona import (  # noqa: E402
    handle_persona_back_callback,
    handle_persona_buy_callback,
    handle_persona_check_status_callback,
    handle_persona_confirm_pay_callback,
    handle_persona_create_callback,
    handle_persona_gender_callback,
    handle_persona_got_it_callback,
    handle_persona_pack_reset_photos_callback,
    handle_persona_page_callback,
    handle_persona_recreate_callback,
    handle_persona_recreate_cancel_callback,
    handle_persona_recreate_confirm_callback,
    handle_persona_reset_photos_callback,
    handle_persona_show_credits_out_callback,
    handle_persona_style_callback,
    handle_persona_topup_buy_callback,
    handle_persona_topup_callback,
    handle_persona_topup_confirm_callback,
    handle_miniapp_pack_upload_callback,
    newpersona_command,
)

from prismalab.handlers.packs import (  # noqa: E402
    _recover_pending_pack_runs,
    handle_persona_pack_buy_callback,
    handle_persona_pack_retry_callback,
    handle_persona_packs_callback,
)

from prismalab.handlers.photos import (  # noqa: E402
    handle_document,
    handle_photo,
    handle_text,
)


def main() -> None:
    _get_store()  # Инициализируем store при запуске бота, не при импорте
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
    application.add_handler(CallbackQueryHandler(handle_persona_pack_retry_callback, pattern="^pl_persona_pack_retry:"))
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
    application.add_handler(MessageHandler(filters.StatusUpdate.WEB_APP_DATA, handle_web_app_data))
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

        bot_me = await app.bot.get_me()
        run_webhook_server(app.bot, store, application=app, bot_username=bot_me.username)

        # Дневной отчёт в 21:00 по Москве (UTC+3)
        async def daily_report_job(context: ContextTypes.DEFAULT_TYPE) -> None:
            await alert_daily_report(store)

        from datetime import time as dt_time
        job_queue = app.job_queue
        if job_queue:
            # 21:00 MSK = 18:00 UTC
            job_queue.run_daily(daily_report_job, time=dt_time(hour=18, minute=0, second=0))
            logger.info("Дневной отчёт запланирован на 21:00 MSK")
            # Восстановление прерванных pack runs каждые 5 мин (бот рестарт во время обучения pack tune)
            job_queue.run_repeating(_recover_pending_pack_runs, interval=300, first=60)
            logger.info("Pack recovery job: каждые 5 мин")

        default_commands = [
            BotCommand("menu", "🏠 Главное меню"),
            BotCommand("profile", "👤 Профиль"),
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
