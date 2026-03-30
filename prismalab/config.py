"""Конфигурация PrismaLab: константы, env-переменные, feature flags.

Единый модуль для всей конфигурации — не разбросана по файлам.
"""
from __future__ import annotations

import os

# --- Владелец и доступ ---

OWNER_ID = int(os.getenv("PRISMALAB_OWNER_ID") or "0")

MINIAPP_URL = os.getenv("MINIAPP_URL", "")

_allowed_users_str = os.getenv("ALLOWED_USERS", "")
ALLOWED_USERS: set[int] = (
    set(int(x.strip()) for x in _allowed_users_str.split(",") if x.strip().isdigit())
    if _allowed_users_str
    else set()
)

# --- Лимиты ---

MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024  # 15 МБ

# --- Сообщение об ошибке (единое, без технических деталей) ---

USER_FRIENDLY_ERROR = "Произошла ошибка. Кредит не списали. Попробуйте ещё раз."

# --- USERDATA ключи (context.user_data) ---

USERDATA_PHOTO_FILE_IDS = "prismalab_photo_file_ids"
USERDATA_ASTRIA_FACEID_FILE_IDS = "prismalab_astria_faceid_file_ids"
USERDATA_ASTRIA_LORA_FILE_IDS = "prismalab_astria_lora_file_ids"
USERDATA_NANO_BANANA_FILE_IDS = "prismalab_nano_banana_file_ids"
USERDATA_MODE = "prismalab_mode"  # normal | fast | persona | persona_pack_upload | astria_faceid | astria_lora
USERDATA_JOB_LOCK = "prismalab_job_lock"  # deprecated
USERDATA_PROMPT_STRENGTH = "prismalab_prompt_strength"
USERDATA_USE_PERSONAL = "prismalab_use_personal"
USERDATA_SUBJECT_GENDER = "prismalab_subject_gender"  # male | female | None
USERDATA_PERSONA_WAITING_UPLOAD = "prismalab_persona_waiting_upload"  # bool, ждём 10 фото
USERDATA_PERSONA_PHOTOS = "prismalab_persona_photos"  # list of file_id для 10 фото Персоны
USERDATA_PERSONA_CREDITS = "prismalab_persona_credits"  # 5, 10 или 20
USERDATA_PERSONA_TRAINING_STATUS = "prismalab_persona_training"  # "training" | "done" | "error"
USERDATA_FAST_SELECTED_STYLE = "prismalab_fast_selected_style"  # style_id когда ждём фото
USERDATA_FAST_CUSTOM_PROMPT = "prismalab_fast_custom_prompt"
USERDATA_FAST_LAST_MSG_ID = "prismalab_fast_last_msg_id"
USERDATA_FAST_STYLE_MSG_ID = "prismalab_fast_style_msg_id"
USERDATA_FAST_PERSONA_MSG_ID = "prismalab_fast_persona_msg_id"
USERDATA_FAST_STYLE_PAGE = "prismalab_fast_style_page"
USERDATA_PERSONA_UPLOAD_MSG_IDS = "prismalab_persona_upload_msg_ids"
USERDATA_PERSONA_STYLE_MSG_ID = "prismalab_persona_style_msg_id"
USERDATA_PERSONA_SELECTED_STYLE = "prismalab_persona_selected_style"
USERDATA_PERSONA_STYLE_PAGE = "prismalab_persona_style_page"
USERDATA_PERSONA_RECREATING = "prismalab_persona_recreating"
USERDATA_PERSONA_PACK_WAITING_UPLOAD = "prismalab_persona_pack_waiting_upload"
USERDATA_PERSONA_PACK_PHOTOS = "prismalab_persona_pack_photos"
USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS = "prismalab_persona_pack_upload_msg_ids"
USERDATA_PERSONA_PACK_IN_PROGRESS = "prismalab_persona_pack_in_progress"
USERDATA_PERSONA_PACK_GIFT_APPLIED = "prismalab_persona_pack_gift_applied"
USERDATA_PROFILE_DELETE_JOB = "prismalab_profile_delete_job"
USERDATA_GETFILEID_EXPECTING_PHOTO = "prismalab_getfileid_expecting_photo"
USERDATA_EXAMPLES_MEDIA_IDS = "prismalab_examples_media_ids"
USERDATA_EXAMPLES_NAV_MSG_ID = "prismalab_examples_nav_msg_id"
USERDATA_EXAMPLES_PAGE = "prismalab_examples_page"
USERDATA_EXAMPLES_INTRO_MSG_ID = "prismalab_examples_intro_msg_id"
USERDATA_PERSONA_SELECTED_PACK_ID = "prismalab_persona_selected_pack_id"
USERDATA_PERSONA_TRAINING_MSG_ID = "prismalab_persona_training_msg_id"


# --- Feature flags ---

def _is_dev_runtime() -> bool:
    """Жёстко считаем dev только при TABLE_PREFIX=dev_*"""
    prefix = (os.getenv("TABLE_PREFIX") or "").strip().lower()
    return prefix.startswith("dev_")


def miniapp_v2_enabled() -> bool:
    """Feature flag: V2 Mini App UI (два раздела: Экспресс + Фотосеты).
    Default: off (0). Включается MINIAPP_V2=1."""
    raw = (os.getenv("MINIAPP_V2") or "").strip().lower()
    return raw in {"1", "true", "yes"}


def express_via_miniapp() -> bool:
    """Кнопка «Экспресс-фото» в боте ведёт в Mini App (вместо inline-кнопок).
    Включается когда MINIAPP_V2=1 и MINIAPP_URL задан."""
    return miniapp_v2_enabled() and bool(MINIAPP_URL)


def express_filters_v3() -> bool:
    """Feature flag: V3 фильтры экспресс (категории/теги/выбор провайдера).
    Default: off (0). Включается EXPRESS_FILTERS_V3=1."""
    raw = (os.getenv("EXPRESS_FILTERS_V3") or "").strip().lower()
    return raw in ("1", "true", "yes")


def packs_use_credits() -> bool:
    """Feature flag: паки покупаются за persona_credits вместо ₽.
    Default: off (0). Включается PACKS_USE_CREDITS=1."""
    raw = (os.getenv("PACKS_USE_CREDITS") or "").strip().lower()
    return raw in {"1", "true", "yes"}


def _dev_skip_pack_payment() -> bool:
    """Dev-флаг: тест паков без оплаты. В проде выключен."""
    raw = (os.getenv("PRISMALAB_DEV_SKIP_PACK_PAYMENT") or "").strip().lower()
    return _is_dev_runtime() and raw in {"1", "true", "yes", "y"}


def _dev_pack_train_from_images() -> bool:
    """Dev-флаг: паки через загрузку 10 фото (без tune_ids)."""
    raw = (os.getenv("PRISMALAB_DEV_PACKS_TRAIN_FROM_IMAGES") or "").strip().lower()
    return _is_dev_runtime() and raw in {"1", "true", "yes", "y"}


def _use_unified_pack_persona_flow() -> bool:
    """Unified flow: если Персоны нет, ведём через persona-flow → автозапуск фотосета."""
    raw = (os.getenv("PRISMALAB_UNIFIED_PACK_PERSONA_FLOW") or "1").strip().lower()
    return raw not in {"0", "false", "no", "n", "off"}


def _guard_dev_only_flags() -> None:
    """Fail-fast: dev-флаги на проде → не стартуем."""
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
