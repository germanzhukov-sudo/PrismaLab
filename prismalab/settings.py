import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv

# Загружаем .env из текущей директории и из корня проекта (рядом с prismalab/)
load_dotenv()
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")


@dataclass(frozen=True)
class PrismaLabSettings:
    app_name: str
    bot_token: str
    prompt_strength: float
    # Astria
    astria_api_key: str
    astria_tune_id: str
    astria_cfg_scale: float
    astria_steps: int
    astria_denoising_strength: float
    astria_super_resolution: bool
    astria_hires_fix: bool
    astria_face_correct: bool
    astria_face_swap: bool
    astria_max_seconds: int
    # KIE API (тестирование моделей)
    kie_api_key: str
    kie_max_seconds: int  # Таймаут ожидания результата KIE (Seedream и др.)


def load_settings() -> PrismaLabSettings:
    bot_token = (os.getenv("PRISMALAB_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
    prompt_strength_raw = (os.getenv("PRISMALAB_PROMPT_STRENGTH") or "0.7").strip()
    try:
        prompt_strength = float(prompt_strength_raw)
    except Exception as e:
        raise ValueError("PRISMALAB_PROMPT_STRENGTH должен быть числом") from e
    if not (0.0 <= prompt_strength <= 1.0):
        raise ValueError("PRISMALAB_PROMPT_STRENGTH должен быть в диапазоне 0..1")

    # Astria
    astria_api_key = (os.getenv("PRISMALAB_ASTRIA_API_KEY") or os.getenv("ASTRIA_API_KEY") or "").strip()
    astria_tune_id = (os.getenv("PRISMALAB_ASTRIA_TUNE_ID") or "").strip()
    cfg_raw = (os.getenv("PRISMALAB_ASTRIA_CFG_SCALE") or "4.5").strip()
    try:
        astria_cfg_scale = float(cfg_raw)
    except Exception as e:
        raise ValueError("PRISMALAB_ASTRIA_CFG_SCALE должен быть числом") from e
    steps_raw = (os.getenv("PRISMALAB_ASTRIA_STEPS") or "25").strip()
    try:
        astria_steps = int(steps_raw)
    except Exception as e:
        raise ValueError("PRISMALAB_ASTRIA_STEPS должен быть числом") from e
    den_raw = (os.getenv("PRISMALAB_ASTRIA_DENOISING_STRENGTH") or "0.35").strip()
    try:
        astria_denoising_strength = float(den_raw)
    except Exception as e:
        raise ValueError("PRISMALAB_ASTRIA_DENOISING_STRENGTH должен быть числом") from e
    astria_denoising_strength = max(0.0, min(1.0, astria_denoising_strength))
    astria_super_resolution = (os.getenv("PRISMALAB_ASTRIA_SUPER_RESOLUTION") or "true").strip().lower() in {"1", "true", "yes", "y"}
    astria_hires_fix = (os.getenv("PRISMALAB_ASTRIA_HIRES_FIX") or "true").strip().lower() in {"1", "true", "yes", "y"}
    astria_face_correct = (os.getenv("PRISMALAB_ASTRIA_FACE_CORRECT") or "true").strip().lower() in {"1", "true", "yes", "y"}
    astria_face_swap = (os.getenv("PRISMALAB_ASTRIA_FACE_SWAP") or "false").strip().lower() in {"1", "true", "yes", "y"}
    astria_max_raw = (os.getenv("PRISMALAB_ASTRIA_MAX_SECONDS") or "420").strip()
    try:
        astria_max_seconds = int(astria_max_raw)
    except Exception as e:
        raise ValueError("PRISMALAB_ASTRIA_MAX_SECONDS должен быть числом") from e

    return PrismaLabSettings(
        app_name=(os.getenv("PRISMALAB_APP_NAME") or "PrismaLab").strip(),
        bot_token=bot_token,
        prompt_strength=prompt_strength,
        astria_api_key=astria_api_key,
        astria_tune_id=astria_tune_id,
        astria_cfg_scale=astria_cfg_scale,
        astria_steps=astria_steps,
        astria_denoising_strength=astria_denoising_strength,
        astria_super_resolution=astria_super_resolution,
        astria_hires_fix=astria_hires_fix,
        astria_face_correct=astria_face_correct,
        astria_face_swap=astria_face_swap,
        astria_max_seconds=astria_max_seconds,
        kie_api_key=(os.getenv("PRISMALAB_KIE_API_KEY") or os.getenv("KIE_API_KEY") or "").strip(),
        kie_max_seconds=_parse_kie_max_seconds(),
    )


def _parse_kie_max_seconds() -> int:
    raw = (os.getenv("PRISMALAB_KIE_MAX_SECONDS") or "600").strip()
    try:
        v = int(raw)
    except ValueError:
        return 600
    return max(120, min(1200, v))  # 2–20 мин
