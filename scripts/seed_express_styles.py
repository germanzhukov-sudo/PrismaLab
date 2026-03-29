#!/usr/bin/env python3
"""Идемпотентный сидер экспресс-стилей в БД (upsert по slug).

Источники данных:
  - keyboards.py → FAST_STYLES_MALE / FAST_STYLES_FEMALE (slug, title)
  - persona_prompts.py → PERSONA_STYLE_PROMPTS (prompt по slug)

Запуск:
  python3 scripts/seed_express_styles.py          # dev (TABLE_PREFIX из .env)
  TABLE_PREFIX= python3 scripts/seed_express_styles.py  # prod
"""
from __future__ import annotations

import os
import sys

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prismalab.persona_prompts import PERSONA_STYLE_PROMPTS

# ── Данные из keyboards.py ──────────────────────────────────────────
# (title, slug, emoji, theme)

FAST_STYLES_MALE: list[tuple[str, str, str, str]] = [
    ("Ночной бар", "night_bar", "🍸", "lifestyle"),
    ("В костюме у окна", "suit_window", "🪟", "business"),
    ("Прогулка в парке", "park_walk", "🌳", "outdoor"),
    ("Утренний кофе", "morning_coffee", "☕", "lifestyle"),
    ("Лесной портрет", "forest_portrait", "🌲", "outdoor"),
    ("Ночной клуб", "night_club", "🎶", "lifestyle"),
    ("Мастерская художника", "artist_workshop", "🎨", "creative"),
    ("Силуэт на закате", "sunset_silhouette", "🌅", "outdoor"),
    ("Байкер", "biker", "🏍", "lifestyle"),
    ("Пилот", "pilot", "✈️", "creative"),
]

FAST_STYLES_FEMALE: list[tuple[str, str, str, str]] = [
    ("Свадебный образ", "wedding", "💍", "glamour"),
    ("Мокрое окно", "wet_window", "🌧", "mood"),
    ("Вечерний гламур", "evening_glamour", "✨", "glamour"),
    ("Неоновый киберпанк", "neon_cyberpunk", "🌃", "creative"),
    ("Драматический свет", "dramatic_light", "💡", "mood"),
    ("Городской нуар", "city_noir", "🌑", "mood"),
    ("Студийный дым", "studio_smoke", "💨", "mood"),
    ("Чёрно-белая рефлексия", "bw_reflection", "🖤", "mood"),
    ("Бальный зал", "ballroom", "👑", "glamour"),
    ("Греческая королева", "greek_queen", "🏛", "creative"),
    ("Мокрая рубашка", "wet_shirt", "💧", "mood"),
    ("Клеопатра", "cleopatra", "🐍", "creative"),
    ("Old money", "old_money", "💎", "glamour"),
    ("Лавандовое бьюти", "lavender_beauty", "💜", "beauty"),
    ("Серебряная иллюзия", "silver_illusion", "🪞", "beauty"),
    ("Белоснежная чистота", "white_purity", "🤍", "beauty"),
    ("Бордовый бархат", "burgundy_velvet", "🍷", "beauty"),
    ("Серый кашемир", "grey_cashmere", "🧣", "beauty"),
    ("Чёрная сетка", "black_mesh", "🖤", "beauty"),
    ("Лавандовый шёлк", "lavender_silk", "💜", "beauty"),
    ("Шёлковое бельё в отеле", "silk_lingerie_hotel", "🏨", "lifestyle"),
    ("Ванна с лепестками", "bath_petals", "🛁", "lifestyle"),
    ("Шампанское на балконе", "champagne_balcony", "🥂", "lifestyle"),
    ("Дождливое окно", "rainy_window", "🌧", "mood"),
    ("Кофе в отеле", "coffee_hotel", "☕", "lifestyle"),
    ("Джазовый бар", "jazz_bar", "🎷", "lifestyle"),
    ("Пикник на пледе", "picnic_blanket", "🧺", "outdoor"),
    ("Художественная студия", "art_studio", "🎨", "creative"),
    ("Уют зимнего камина", "winter_fireplace", "🔥", "lifestyle"),
]


def main() -> None:
    from prismalab.storage import PrismaLabStore

    store = PrismaLabStore()
    store.init_admin_tables()

    count = 0

    for order, (title, slug, emoji, theme) in enumerate(FAST_STYLES_MALE, start=1):
        prompt = PERSONA_STYLE_PROMPTS.get(slug, "")
        store.upsert_express_style(
            slug=slug, title=title, emoji=emoji, theme=theme,
            gender="male", prompt=prompt, negative_prompt="",
            provider="seedream", image_url="", model_params="",
            sort_order=order,
        )
        count += 1

    for order, (title, slug, emoji, theme) in enumerate(FAST_STYLES_FEMALE, start=1):
        prompt = PERSONA_STYLE_PROMPTS.get(slug, "")
        store.upsert_express_style(
            slug=slug, title=title, emoji=emoji, theme=theme,
            gender="female", prompt=prompt, negative_prompt="",
            provider="seedream", image_url="", model_params="",
            sort_order=order,
        )
        count += 1

    print(f"Seeded {count} express styles (upsert by slug)")

    # Статистика
    males = store.get_express_styles(gender="male")
    females = store.get_express_styles(gender="female")
    themes = store.get_express_themes()
    print(f"   Male: {len(males)}, Female: {len(females)}, Total: {len(males) + len(females)}")
    print(f"   Themes: {themes}")


if __name__ == "__main__":
    main()
