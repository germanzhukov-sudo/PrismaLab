"""Обработка изображений: resize, crop, padding, postprocess."""

from __future__ import annotations

import io

from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# Округление размеров до кратности 64
# ---------------------------------------------------------------------------

def _round_to_64(x: float) -> int:
    return max(64, int(round(x / 64.0)) * 64)


def _ceil_to_64(x: int) -> int:
    # Всегда округляем ВВЕРХ, чтобы не ужимать и не терять детали лица.
    return max(64, ((max(1, int(x)) + 63) // 64) * 64)


# ---------------------------------------------------------------------------
# Подготовка фото для моделей
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Постобработка
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Хелперы aspect ratio и промптов
# ---------------------------------------------------------------------------

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
