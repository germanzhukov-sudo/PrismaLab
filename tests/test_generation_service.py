"""Тесты сервиса генерации (miniapp/services/generation.py)."""
from __future__ import annotations

import json

from prismalab.miniapp.services.generation import (
    build_generation_kwargs,
    prepare_photo,
)


# ── build_generation_kwargs ──────────────────────────────────────────


def test_seedream_defaults():
    """Seedream: дефолтные параметры."""
    kwargs = build_generation_kwargs(
        provider="seedream",
        prompt="test prompt",
        image_url="https://example.com/img.jpg",
        api_key="test-key",
    )
    assert kwargs["model"] == "seedream/4.5-edit"
    assert kwargs["prompt"] == "test prompt"
    assert kwargs["image_input"] == ["https://example.com/img.jpg"]
    assert kwargs["aspect_ratio"] == "1:1"
    assert kwargs["quality"] == "basic"
    assert kwargs["output_format"] == "jpg"
    assert kwargs["api_key"] == "test-key"
    # seedream не использует resolution
    assert "resolution" not in kwargs


def test_nano_banana_pro_defaults():
    """Nano Banana Pro: дефолтные параметры."""
    kwargs = build_generation_kwargs(
        provider="nano-banana-pro",
        prompt="test prompt",
        image_url="https://example.com/img.jpg",
        api_key="test-key",
    )
    assert kwargs["model"] == "nano-banana-pro"
    assert kwargs["resolution"] == "2K"
    assert kwargs["output_format"] == "png"
    # nano-banana-pro не использует quality
    assert "quality" not in kwargs


def test_model_params_override():
    """model_params JSON переопределяет дефолты."""
    overrides = json.dumps({
        "aspect_ratio": "16:9",
        "quality": "high",
        "model": "seedream/4.5-edit-custom",
    })
    kwargs = build_generation_kwargs(
        provider="seedream",
        prompt="test",
        api_key="k",
        model_params_json=overrides,
    )
    assert kwargs["aspect_ratio"] == "16:9"
    assert kwargs["quality"] == "high"
    assert kwargs["model"] == "seedream/4.5-edit-custom"


def test_model_params_override_nano():
    """model_params override для nano-banana-pro."""
    overrides = json.dumps({"resolution": "4K"})
    kwargs = build_generation_kwargs(
        provider="nano-banana-pro",
        prompt="test",
        api_key="k",
        model_params_json=overrides,
    )
    assert kwargs["resolution"] == "4K"


def test_negative_prompt_from_arg():
    """negative_prompt передаётся из аргумента."""
    kwargs = build_generation_kwargs(
        provider="seedream",
        prompt="test",
        negative_prompt="bad quality, blurry",
        api_key="k",
    )
    assert kwargs["negative_prompt"] == "bad quality, blurry"


def test_negative_prompt_from_model_params():
    """negative_prompt из model_params приоритетнее аргумента."""
    overrides = json.dumps({"negative_prompt": "override negative"})
    kwargs = build_generation_kwargs(
        provider="seedream",
        prompt="test",
        negative_prompt="original negative",
        api_key="k",
        model_params_json=overrides,
    )
    assert kwargs["negative_prompt"] == "override negative"


def test_no_image_url():
    """Без image_url → image_input=None."""
    kwargs = build_generation_kwargs(
        provider="seedream",
        prompt="test",
        image_url=None,
        api_key="k",
    )
    assert kwargs["image_input"] is None


def test_unknown_provider_falls_back_to_seedream():
    """Неизвестный провайдер → seedream дефолты."""
    kwargs = build_generation_kwargs(
        provider="unknown-model",
        prompt="test",
        api_key="k",
    )
    assert kwargs["model"] == "seedream/4.5-edit"
    assert kwargs["quality"] == "basic"


def test_invalid_model_params_json():
    """Невалидный JSON в model_params — игнорируется."""
    kwargs = build_generation_kwargs(
        provider="seedream",
        prompt="test",
        api_key="k",
        model_params_json="not valid json{",
    )
    # Должен вернуть дефолты без ошибки
    assert kwargs["model"] == "seedream/4.5-edit"


def test_max_seconds_and_poll():
    """max_seconds и poll_seconds передаются."""
    kwargs = build_generation_kwargs(
        provider="seedream",
        prompt="test",
        api_key="k",
        max_seconds=600,
        poll_seconds=5.0,
    )
    assert kwargs["max_seconds"] == 600
    assert kwargs["poll_seconds"] == 5.0


def test_seedream_output_format_jpg():
    """Seedream → output_format=jpg."""
    kwargs = build_generation_kwargs(provider="seedream", prompt="t", api_key="k")
    assert kwargs["output_format"] == "jpg"


def test_nano_banana_pro_output_format_png():
    """Nano Banana Pro → output_format=png."""
    kwargs = build_generation_kwargs(provider="nano-banana-pro", prompt="t", api_key="k")
    assert kwargs["output_format"] == "png"


# ── prepare_photo ────────────────────────────────────────────────────


def test_prepare_photo_small_image():
    """Маленькое фото не ресайзится, конвертируется в JPEG."""
    from PIL import Image

    img = Image.new("RGB", (100, 100), (255, 0, 0))
    buf = __import__("io").BytesIO()
    img.save(buf, format="PNG")

    result = prepare_photo(buf.getvalue())
    assert len(result) > 0
    # Проверяем что это JPEG (starts with FF D8)
    assert result[:2] == b"\xff\xd8"


def test_prepare_photo_large_image_resized():
    """Большое фото ресайзится до max_side."""
    from PIL import Image

    img = Image.new("RGB", (3000, 2000), (0, 255, 0))
    buf = __import__("io").BytesIO()
    img.save(buf, format="PNG")

    result = prepare_photo(buf.getvalue(), max_side=1024)
    # Проверяем что результат — валидный JPEG
    result_img = Image.open(__import__("io").BytesIO(result))
    assert max(result_img.size) <= 1024


def test_prepare_photo_rgba_converted():
    """RGBA фото конвертируется в RGB."""
    from PIL import Image

    img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    buf = __import__("io").BytesIO()
    img.save(buf, format="PNG")

    result = prepare_photo(buf.getvalue())
    result_img = Image.open(__import__("io").BytesIO(result))
    assert result_img.mode == "RGB"
