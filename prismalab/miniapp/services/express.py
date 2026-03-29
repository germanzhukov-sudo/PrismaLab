"""Сервис каталога экспресс-стилей.

Единая точка получения стилей/тем для Mini App.
Источник: БД (express_styles) с fallback на захардкоженные списки.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("prismalab.miniapp.services.express")


# ── Результаты ────────────────────────────────────────────────────────

@dataclass
class StyleInfo:
    """Карточка стиля для API."""
    slug: str
    title: str
    emoji: str
    theme: str
    image_url: str

    def to_api_dict(self) -> dict[str, str]:
        """Формат для JSON-ответа (backward compat: id/label)."""
        return {
            "id": self.slug,
            "label": self.title,
            "emoji": self.emoji,
            "theme": self.theme,
            "image_url": self.image_url,
        }


@dataclass
class ResolvedStyle:
    """Все данные стиля, нужные для генерации."""
    slug: str
    title: str
    prompt: str
    negative_prompt: str
    provider: str
    model_params_json: str


# ── Hardcoded fallback (пока в БД нет стилей) ────────────────────────

_FALLBACK_STYLES_MALE: list[dict[str, str]] = [
    {"slug": "night_bar", "title": "Ночной бар", "emoji": "🍸", "theme": "lifestyle"},
    {"slug": "suit_window", "title": "В костюме у окна", "emoji": "🪟", "theme": "business"},
    {"slug": "park_walk", "title": "Прогулка в парке", "emoji": "🌳", "theme": "outdoor"},
    {"slug": "morning_coffee", "title": "Утренний кофе", "emoji": "☕", "theme": "lifestyle"},
    {"slug": "forest_portrait", "title": "Лесной портрет", "emoji": "🌲", "theme": "outdoor"},
    {"slug": "night_club", "title": "Ночной клуб", "emoji": "🎶", "theme": "lifestyle"},
    {"slug": "artist_workshop", "title": "Мастерская художника", "emoji": "🎨", "theme": "creative"},
    {"slug": "sunset_silhouette", "title": "Силуэт на закате", "emoji": "🌅", "theme": "outdoor"},
    {"slug": "biker", "title": "Байкер", "emoji": "🏍", "theme": "lifestyle"},
    {"slug": "pilot", "title": "Пилот", "emoji": "✈️", "theme": "creative"},
]

_FALLBACK_STYLES_FEMALE: list[dict[str, str]] = [
    {"slug": "wedding", "title": "Свадебный образ", "emoji": "💍", "theme": "glamour"},
    {"slug": "wet_window", "title": "Мокрое окно", "emoji": "🌧", "theme": "mood"},
    {"slug": "evening_glamour", "title": "Вечерний гламур", "emoji": "✨", "theme": "glamour"},
    {"slug": "neon_cyberpunk", "title": "Неоновый киберпанк", "emoji": "🌃", "theme": "creative"},
    {"slug": "dramatic_light", "title": "Драматический свет", "emoji": "💡", "theme": "mood"},
    {"slug": "city_noir", "title": "Городской нуар", "emoji": "🌑", "theme": "mood"},
    {"slug": "studio_smoke", "title": "Студийный дым", "emoji": "💨", "theme": "mood"},
    {"slug": "bw_reflection", "title": "Чёрно-белая рефлексия", "emoji": "🖤", "theme": "mood"},
    {"slug": "ballroom", "title": "Бальный зал", "emoji": "👑", "theme": "glamour"},
    {"slug": "greek_queen", "title": "Греческая королева", "emoji": "🏛", "theme": "creative"},
    {"slug": "wet_shirt", "title": "Мокрая рубашка", "emoji": "💧", "theme": "mood"},
    {"slug": "cleopatra", "title": "Клеопатра", "emoji": "🐍", "theme": "creative"},
    {"slug": "old_money", "title": "Old money", "emoji": "💎", "theme": "glamour"},
    {"slug": "lavender_beauty", "title": "Лавандовое бьюти", "emoji": "💜", "theme": "beauty"},
    {"slug": "silver_illusion", "title": "Серебряная иллюзия", "emoji": "🪞", "theme": "beauty"},
    {"slug": "white_purity", "title": "Белоснежная чистота", "emoji": "🤍", "theme": "beauty"},
    {"slug": "burgundy_velvet", "title": "Бордовый бархат", "emoji": "🍷", "theme": "beauty"},
    {"slug": "grey_cashmere", "title": "Серый кашемир", "emoji": "🧣", "theme": "beauty"},
    {"slug": "black_mesh", "title": "Чёрная сетка", "emoji": "🖤", "theme": "beauty"},
    {"slug": "lavender_silk", "title": "Лавандовый шёлк", "emoji": "💜", "theme": "beauty"},
    {"slug": "silk_lingerie_hotel", "title": "Шёлковое бельё в отеле", "emoji": "🏨", "theme": "lifestyle"},
    {"slug": "bath_petals", "title": "Ванна с лепестками", "emoji": "🛁", "theme": "lifestyle"},
    {"slug": "champagne_balcony", "title": "Шампанское на балконе", "emoji": "🥂", "theme": "lifestyle"},
    {"slug": "rainy_window", "title": "Дождливое окно", "emoji": "🌧", "theme": "mood"},
    {"slug": "coffee_hotel", "title": "Кофе в отеле", "emoji": "☕", "theme": "lifestyle"},
    {"slug": "jazz_bar", "title": "Джазовый бар", "emoji": "🎷", "theme": "lifestyle"},
    {"slug": "picnic_blanket", "title": "Пикник на пледе", "emoji": "🧺", "theme": "outdoor"},
    {"slug": "art_studio", "title": "Художественная студия", "emoji": "🎨", "theme": "creative"},
    {"slug": "winter_fireplace", "title": "Уют зимнего камина", "emoji": "🔥", "theme": "lifestyle"},
]


def _fallback_to_style_info(raw: dict[str, str]) -> StyleInfo:
    return StyleInfo(
        slug=raw["slug"],
        title=raw["title"],
        emoji=raw.get("emoji", ""),
        theme=raw.get("theme", ""),
        image_url="",
    )


def _db_row_to_style_info(row: dict[str, Any]) -> StyleInfo:
    return StyleInfo(
        slug=row["slug"],
        title=row["title"],
        emoji=row.get("emoji", ""),
        theme=row.get("theme", ""),
        image_url=row.get("image_url", ""),
    )


# ── Публичные функции ─────────────────────────────────────────────────

def get_styles(store: Any, *, gender: str = "female", theme: str | None = None) -> list[StyleInfo]:
    """Каталог стилей: БД → fallback на hardcoded.

    Args:
        store: PrismaLabStore instance
        gender: "male" / "female"
        theme: фильтр по теме (None = все)

    Returns:
        Список StyleInfo, отсортированный по sort_order.
    """
    # Fallback только если в БД вообще нет стилей для этого gender
    # (а не "есть, но все отключены / не совпала тема")
    all_db_styles = store.get_express_styles(gender=gender, active_only=False)
    db_has_styles = len(all_db_styles) > 0

    if db_has_styles:
        # БД управляет каталогом — берём только active
        db_active = store.get_express_styles(gender=gender, active_only=True)
        styles = [_db_row_to_style_info(row) for row in db_active]
    else:
        # БД пустая — fallback на hardcoded
        fallback = _FALLBACK_STYLES_FEMALE if gender == "female" else _FALLBACK_STYLES_MALE
        styles = [_fallback_to_style_info(row) for row in fallback]
        logger.debug("Using fallback styles for gender=%s (DB empty)", gender)

    if theme:
        styles = [s for s in styles if s.theme == theme]

    return styles


def get_themes(store: Any, *, gender: str | None = None) -> list[str]:
    """Список уникальных тем: БД → fallback на hardcoded.

    Returns:
        Отсортированный список тем.
    """
    # Проверяем наличие стилей в БД (любых, не только active)
    check_gender = gender if gender else None
    all_db_styles = store.get_express_styles(gender=check_gender, active_only=False)
    db_has_styles = len(all_db_styles) > 0

    if db_has_styles:
        db_themes = store.get_express_themes(gender=gender, active_only=True)
        return sorted(db_themes)

    # Fallback: БД пустая
    fallback = _FALLBACK_STYLES_FEMALE + _FALLBACK_STYLES_MALE
    if gender:
        fallback = _FALLBACK_STYLES_FEMALE if gender == "female" else _FALLBACK_STYLES_MALE
    themes = sorted(set(s["theme"] for s in fallback if s.get("theme")))
    logger.debug("Using fallback themes (DB empty)")
    return themes


def resolve_style(store: Any, slug: str) -> ResolvedStyle | None:
    """Резолвит стиль по slug: БД → fallback на PERSONA_STYLE_PROMPTS.

    Returns:
        ResolvedStyle с prompt/provider/negative_prompt/model_params,
        или None если стиль не найден нигде.
    """
    # 1. Ищем в БД (только активные — админ может выключить стиль)
    row = store.get_express_style_by_slug(slug)
    if row:
        if not row.get("is_active", 1):
            return None  # стиль выключен админом
        return ResolvedStyle(
            slug=row["slug"],
            title=row["title"],
            prompt=row.get("prompt", ""),
            negative_prompt=row.get("negative_prompt", ""),
            provider=row.get("provider", "seedream") or "seedream",
            model_params_json=row.get("model_params", ""),
        )

    # 2. Fallback: PERSONA_STYLE_PROMPTS
    from prismalab.persona_prompts import PERSONA_STYLE_PROMPTS

    prompt = PERSONA_STYLE_PROMPTS.get(slug, "")
    if prompt:
        return ResolvedStyle(
            slug=slug,
            title=slug,  # нет title в fallback
            prompt=prompt,
            negative_prompt="",
            provider="seedream",
            model_params_json="",
        )

    # 3. Ищем в hardcoded списках (для title)
    for s in _FALLBACK_STYLES_MALE + _FALLBACK_STYLES_FEMALE:
        if s["slug"] == slug:
            return ResolvedStyle(
                slug=slug,
                title=s["title"],
                prompt="",  # промпт не найден — routes.py сгенерирует fallback
                negative_prompt="",
                provider="seedream",
                model_params_json="",
            )

    return None
