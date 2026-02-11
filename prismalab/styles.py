from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StylePreset:
    id: str
    title: str
    prompt: str
    negative_prompt: str = ""


# 10 базовых стилей (можно править под твою модель/вкусы)
STYLES: list[StylePreset] = [
    StylePreset(
        id="anime",
        title="Аниме",
        prompt="a portrait photo in anime style, clean lines, vibrant colors",
        negative_prompt="blurry, lowres, bad anatomy, bad hands, deformed, ugly",
    ),
    StylePreset(
        id="cyberpunk",
        title="Киберпанк",
        prompt="a portrait photo, cyberpunk style, neon lights, cinematic",
        negative_prompt="blurry, lowres, washed out, bad anatomy, deformed",
    ),
    StylePreset(
        id="watercolor",
        title="Акварель",
        prompt="a portrait painted in watercolor, soft brush strokes, paper texture",
        negative_prompt="photorealistic, harsh outlines, noisy, lowres",
    ),
    StylePreset(
        id="oil_painting",
        title="Масло (импасто)",
        prompt="an oil painting portrait, impasto, thick paint, dramatic lighting",
        negative_prompt="blurry, lowres, plastic, flat shading",
    ),
    StylePreset(
        id="noir",
        title="Нуар (Ч/Б)",
        prompt="a black and white film noir portrait, dramatic lighting, film grain",
        negative_prompt="color, low contrast, blurry, lowres",
    ),
    StylePreset(
        id="comic",
        title="Комикс",
        prompt="a comic book portrait, bold ink lines, halftone shading",
        negative_prompt="photorealistic, blurry, lowres",
    ),
    StylePreset(
        id="pixel_art",
        title="Пиксель-арт",
        prompt="a pixel art portrait, 16-bit, limited color palette",
        negative_prompt="blurry, smooth, photorealistic",
    ),
    StylePreset(
        id="claymation",
        title="Пластилин",
        prompt="a claymation portrait, plasticine, studio lighting",
        negative_prompt="blurry, lowres, bad anatomy",
    ),
    StylePreset(
        id="3d_pixar",
        title="3D (Pixar-like)",
        prompt="a 3D animated portrait, pixar-like style, soft lighting",
        negative_prompt="blurry, lowres, uncanny, bad anatomy",
    ),
    StylePreset(
        id="vintage_film",
        title="Винтаж (плёнка)",
        # 70–90s film look: тёплый тон, лёгкое зерно, слегка выцветшие цвета, window light, dust + light leaks.
        # (Промпт пишем по-английски: большинство SD/SDXL/InstantID лучше реагируют именно так.)
        prompt=(
            "1970s-1990s vintage analog film photo, kodak film, warm tone, soft window light, "
            "subtle fine film grain, slightly faded colors, natural skin texture, "
            "film dust, light leaks, professional film photography, sharp focus, high detail"
        ),
        negative_prompt=(
            "cartoon, illustration, anime, overprocessed, plastic skin, strong blur, out of focus, "
            "distorted face, deformed, bad anatomy, extra fingers, wrong eyes, "
            "text, watermark, logo"
        ),
    ),
    StylePreset(
        id="nyc_70s",
        title="NYC 70s (сцена)",
        prompt=(
            "Cinematic vintage 35mm film photo, 1970s New York City street scene at night after rain. "
            "A single person standing near a classic 1970s American car parked on the curb. "
            "Wet asphalt, warm street lamps, soft reflections on the road, subtle fog in the distance. "
            "Brick buildings with fire escapes, vintage shop signs, blurred taxi lights, distant skyline. "
            "Kodak film look: natural skin texture, soft warm tones, slightly faded colors, authentic film grain, "
            "dust, tiny scratches, subtle light leaks. "
            "Candid documentary photo, realistic, photorealistic, high detail. "
            "85mm lens, shallow depth of field, cinematic lighting. "
            "Keep the same person and recognizable facial features."
        ),
        negative_prompt=(
            "blurry, lowres, bad quality, low quality, jpeg artifacts, out of focus, "
            "deformed face, face distortion, changed facial features, different person, "
            "asymmetrical eyes, duplicated face, two faces, multiple people, "
            "plastic skin, doll face, over-smooth skin, "
            "cartoon, anime, illustration, text, watermark, logo"
        ),
    ),
    StylePreset(
        id="restaurant",
        title="Ресторан",
        prompt=(
            "Cinematic photorealistic full-body shot of a woman sitting at a table in a luxurious fine-dining restaurant. "
            "Crystal chandeliers, elegant table settings, people in the background (softly blurred). "
            "She is sitting on a beautiful chair, legs crossed. "
            "One hand adjusts her hair near her ear, the other hand holds a glass with a white cocktail. "
            "Outfit: elegant white satin long-sleeve off-shoulder blouse and a voluminous white satin short skirt (tasteful, not revealing). "
            "Large gold earrings, perfect evening makeup. "
            "She is looking slightly to the side. "
            "Classy fashion editorial, fully clothed, no cleavage, non-sexual. "
            "Realistic body proportions, natural skin texture, cinematic lighting, shallow depth of field, high detail."
        ),
        negative_prompt=(
            "nsfw, nude, naked, lingerie, underwear, cleavage, nipples, see-through, erotic, sexual, "
            "cartoon, anime, illustration, plastic skin, doll face, over-smoothed skin, "
            "blurry, lowres, low quality, out of focus, "
            "face changed, different person, altered facial features, face distortion, "
            "bad anatomy, deformed, extra fingers, missing fingers, extra limbs, "
            "duplicate face, two faces, multiple people in foreground, "
            "text, watermark, logo"
        ),
    ),
    StylePreset(
        id="restaurant_safe",
        title="Ресторан (safe)",
        prompt=(
            "Cinematic photorealistic full-body shot of a woman sitting at a table in a luxurious fine-dining restaurant. "
            "Crystal chandeliers, elegant table settings, people in the background (softly blurred). "
            "She is sitting on a beautiful chair, legs crossed, relaxed natural pose. "
            "One hand adjusts her hair near her ear, the other hand holds a glass with a white cocktail. "
            "Outfit: elegant white long-sleeve satin blouse (modest) and a white knee-length satin skirt. "
            "Large gold earrings, tasteful evening makeup. "
            "She is looking slightly to the side. "
            "Classy fashion editorial, fully clothed, non-sexual. "
            "Realistic body proportions, natural skin texture, cinematic lighting, shallow depth of field, high detail."
        ),
        negative_prompt=(
            "nsfw, nude, naked, lingerie, underwear, cleavage, nipples, see-through, erotic, sexual, "
            "cartoon, anime, illustration, plastic skin, doll face, over-smoothed skin, "
            "blurry, lowres, low quality, out of focus, "
            "face changed, different person, altered facial features, face distortion, "
            "bad anatomy, deformed, extra fingers, missing fingers, extra limbs, "
            "duplicate face, two faces, multiple people in foreground, "
            "text, watermark, logo"
        ),
    ),
]


def get_style(style_id: str) -> StylePreset | None:
    for s in STYLES:
        if s.id == style_id:
            return s
    return None

