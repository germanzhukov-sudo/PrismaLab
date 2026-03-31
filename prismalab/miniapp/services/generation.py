"""Сервис генерации экспресс-фото.

Единая точка выбора провайдера (seedream / nano-banana-pro) и параметров.
Без Request/Response — чистая бизнес-логика.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import secrets
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("prismalab.miniapp.services.generation")


# ── Результат генерации ──────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Результат генерации фото."""
    data_url: str          # base64 data URL (data:image/{jpeg|png};base64,...)
    provider: str          # seedream / nano-banana-pro
    style_slug: str
    raw_bytes: bytes = b""  # исходные байты для upload/send
    mime_type: str = "image/jpeg"
    file_ext: str = "jpg"


# ── Подготовка параметров для KIE API ────────────────────────────────

# Дефолтные параметры по провайдерам
_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "seedream": {
        "model": "seedream/4.5-edit",
        "aspect_ratio": "1:1",
        "quality": "basic",
        "resolution": None,
        "output_format": "jpg",
    },
    "nano-banana-pro": {
        "model": "nano-banana-pro",
        "aspect_ratio": "1:1",
        "quality": None,
        "resolution": "2K",
        "output_format": "png",
    },
}


# ── Custom generation capabilities ──────────────────────────────────

CUSTOM_CAPABILITIES: dict[str, Any] = {
    "providers": {
        "seedream": {
            "max_photos": 14,
            "models": {"text_only": "seedream/4.5", "with_refs": "seedream/4.5-edit"},
        },
        "nano-banana-pro": {
            "max_photos": 8,
            "models": {"text_only": "nano-banana-pro", "with_refs": "nano-banana-pro"},
        },
    },
    "max_prompt_length": 2000,
    "allowed_mime": ["image/jpeg", "image/png", "image/webp"],
    "max_file_size_mb": 15,
}


def build_generation_kwargs(
    *,
    provider: str,
    prompt: str,
    negative_prompt: str = "",
    image_url: str | None = None,
    image_urls: list[str] | None = None,
    model_override: str | None = None,
    model_params_json: str = "",
    api_key: str,
    max_seconds: int = 300,
    poll_seconds: float = 3.0,
) -> dict[str, Any]:
    """Собирает kwargs для kie_client.run_task_and_wait().

    image_url — single image (express backward compat).
    image_urls — multi-image (custom generation).
    model_override — explicit model (custom: text-only vs with-refs).
    """
    defaults = _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["seedream"]).copy()

    # Override из model_params (JSON строка из БД)
    overrides: dict[str, Any] = {}
    if model_params_json:
        try:
            overrides = json.loads(model_params_json)
            if not isinstance(overrides, dict):
                overrides = {}
        except (json.JSONDecodeError, TypeError):
            overrides = {}

    # image_input: multi-image > single image > None
    if image_urls:
        img_input = image_urls
    elif image_url:
        img_input = [image_url]
    else:
        img_input = None

    # model: explicit override > model_params > provider default
    model = model_override or overrides.get("model", defaults["model"])

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "model": model,
        "prompt": prompt,
        "image_input": img_input,
        "aspect_ratio": overrides.get("aspect_ratio", defaults["aspect_ratio"]),
        "output_format": overrides.get("output_format", defaults["output_format"]),
        "max_seconds": max_seconds,
        "poll_seconds": poll_seconds,
    }

    # quality — только для seedream
    quality = overrides.get("quality", defaults.get("quality"))
    if quality:
        kwargs["quality"] = quality

    # resolution — только для nano-banana-pro
    resolution = overrides.get("resolution", defaults.get("resolution"))
    if resolution:
        kwargs["resolution"] = resolution

    # negative_prompt
    neg = overrides.get("negative_prompt", negative_prompt)
    if neg:
        kwargs["negative_prompt"] = neg

    return kwargs


# ── Подготовка фото ──────────────────────────────────────────────────

def prepare_photo(photo_bytes: bytes, *, max_side: int = 1024, jpeg_quality: int = 92) -> bytes:
    """Resize + конвертация фото в JPEG. Без I/O — чистая трансформация."""
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(photo_bytes))
    img = ImageOps.exif_transpose(img) or img

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()


# ── Основной flow генерации ──────────────────────────────────────────

async def run_generation(
    *,
    photo_bytes: bytes,
    style_slug: str,
    prompt: str,
    negative_prompt: str = "",
    provider: str = "seedream",
    model_params_json: str = "",
    api_key: str,
    max_seconds: int = 300,
) -> GenerationResult:
    """Запускает генерацию фото через KIE.

    1. Подготавливает фото (resize, JPEG)
    2. Загружает в KIE
    3. Генерирует через выбранного провайдера
    4. Скачивает результат
    5. Возвращает GenerationResult с base64 data URL

    Raises:
        RuntimeError: при ошибке генерации
    """
    from prismalab.kie_client import (
        download_image_bytes as kie_download,
        run_task_and_wait as kie_run,
        upload_file_base64 as kie_upload,
    )

    # 1. Подготовка фото
    prepared = await asyncio.to_thread(prepare_photo, photo_bytes)

    # 2. Загрузка в KIE
    random_id = secrets.token_hex(8)
    uploaded_url = await asyncio.to_thread(
        kie_upload,
        api_key=api_key,
        image_bytes=prepared,
        file_name=f"miniapp_{random_id}.jpg",
    )

    # 3. Генерация
    kwargs = build_generation_kwargs(
        provider=provider,
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_url=uploaded_url,
        model_params_json=model_params_json,
        api_key=api_key,
        max_seconds=max_seconds,
    )
    kie_result = await kie_run(**kwargs)

    if not kie_result.image_url:
        raise RuntimeError(f"KIE returned no image URL (provider={provider})")

    # 4. Скачивание результата
    result_bytes = await asyncio.to_thread(kie_download, kie_result.image_url)

    # 5. Base64 data URL (MIME по output_format)
    out_fmt = str(kwargs.get("output_format", "jpg")).lower()
    mime = "image/png" if out_fmt == "png" else "image/jpeg"
    file_ext = "png" if out_fmt == "png" else "jpg"
    result_b64 = base64.b64encode(result_bytes).decode()
    data_url = f"data:{mime};base64,{result_b64}"

    logger.info("Generation done: provider=%s style=%s", provider, style_slug)
    return GenerationResult(
        data_url=data_url,
        provider=provider,
        style_slug=style_slug,
        raw_bytes=result_bytes,
        mime_type=mime,
        file_ext=file_ext,
    )


# ── Custom prompt generation ────────────────────────────────────────

async def run_custom_generation(
    *,
    prompt: str,
    photo_bytes_list: list[bytes],
    provider: str = "seedream",
    api_key: str,
    max_seconds: int = 300,
) -> GenerationResult:
    """Генерация по свободному промпту с 0..N фото-референсами.

    Model selection:
    - text-only (no photos) → seedream/4.5 или nano-banana-pro
    - with-refs (photos) → seedream/4.5-edit или nano-banana-pro
    """
    from prismalab.kie_client import (
        download_image_bytes as kie_download,
        run_task_and_wait as kie_run,
        upload_file_base64 as kie_upload,
    )

    has_photos = len(photo_bytes_list) > 0

    # Model selection by mode
    caps = CUSTOM_CAPABILITIES["providers"].get(provider, CUSTOM_CAPABILITIES["providers"]["seedream"])
    model = caps["models"]["with_refs"] if has_photos else caps["models"]["text_only"]

    # Upload photos if any
    uploaded_urls: list[str] = []
    if has_photos:
        for i, photo_bytes in enumerate(photo_bytes_list):
            prepared = await asyncio.to_thread(prepare_photo, photo_bytes)
            random_id = secrets.token_hex(8)
            url = await asyncio.to_thread(
                kie_upload,
                api_key=api_key,
                image_bytes=prepared,
                file_name=f"custom_{random_id}_{i}.jpg",
            )
            uploaded_urls.append(url)

    # Build kwargs
    kwargs = build_generation_kwargs(
        provider=provider,
        prompt=prompt,
        image_urls=uploaded_urls or None,
        model_override=model,
        api_key=api_key,
        max_seconds=max_seconds,
    )
    kie_result = await kie_run(**kwargs)

    if not kie_result.image_url:
        raise RuntimeError(f"KIE returned no image URL (provider={provider}, custom)")

    # Download result
    result_bytes = await asyncio.to_thread(kie_download, kie_result.image_url)

    out_fmt = str(kwargs.get("output_format", "jpg")).lower()
    mime = "image/png" if out_fmt == "png" else "image/jpeg"
    file_ext = "png" if out_fmt == "png" else "jpg"
    result_b64 = base64.b64encode(result_bytes).decode()
    data_url = f"data:{mime};base64,{result_b64}"

    logger.info("Custom generation done: provider=%s photos=%d", provider, len(photo_bytes_list))
    return GenerationResult(
        data_url=data_url,
        provider=provider,
        style_slug="__custom__",
        raw_bytes=result_bytes,
        mime_type=mime,
        file_ext=file_ext,
    )
