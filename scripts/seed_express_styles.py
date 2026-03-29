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

FAST_STYLES_MALE: list[tuple[str, str, str]] = [
    # (title, slug, emoji)
    ("Ночной бар", "night_bar", "🍸"),
    ("В костюме у окна", "suit_window", "🪟"),
    ("Прогулка в парке", "park_walk", "🌳"),
    ("Утренний кофе", "morning_coffee", "☕"),
    ("Лесной портрет", "forest_portrait", "🌲"),
    ("Ночной клуб", "night_club", "🎶"),
    ("Мастерская художника", "artist_workshop", "🎨"),
    ("Силуэт на закате", "sunset_silhouette", "🌅"),
    ("Байкер", "biker", "🏍"),
    ("Пилот", "pilot", "✈️"),
]

FAST_STYLES_FEMALE: list[tuple[str, str, str]] = [
    ("Свадебный образ", "wedding", "💍"),
    ("Мокрое окно", "wet_window", "🌧"),
    ("Вечерний гламур", "evening_glamour", "✨"),
    ("Неоновый киберпанк", "neon_cyberpunk", "🌃"),
    ("Драматический свет", "dramatic_light", "💡"),
    ("Городской нуар", "city_noir", "🌑"),
    ("Студийный дым", "studio_smoke", "💨"),
    ("Чёрно-белая рефлексия", "bw_reflection", "🖤"),
    ("Бальный зал", "ballroom", "👑"),
    ("Греческая королева", "greek_queen", "🏛"),
    ("Мокрая рубашка", "wet_shirt", "💧"),
    ("Клеопатра", "cleopatra", "🐍"),
    ("Old money", "old_money", "💎"),
    ("Лавандовое бьюти", "lavender_beauty", "💜"),
    ("Серебряная иллюзия", "silver_illusion", "🪞"),
    ("Белоснежная чистота", "white_purity", "🤍"),
    ("Бордовый бархат", "burgundy_velvet", "🍷"),
    ("Серый кашемир", "grey_cashmere", "🧣"),
    ("Чёрная сетка", "black_mesh", "🖤"),
    ("Лавандовый шёлк", "lavender_silk", "💜"),
    ("Шёлковое бельё в отеле", "silk_lingerie_hotel", "🏨"),
    ("Ванна с лепестками", "bath_petals", "🛁"),
    ("Шампанское на балконе", "champagne_balcony", "🥂"),
    ("Дождливое окно", "rainy_window", "🌧"),
    ("Кофе в отеле", "coffee_hotel", "☕"),
    ("Джазовый бар", "jazz_bar", "🎷"),
    ("Пикник на пледе", "picnic_blanket", "🧺"),
    ("Художественная студия", "art_studio", "🎨"),
    ("Уют зимнего камина", "winter_fireplace", "🔥"),
]


def main() -> None:
    from prismalab.storage import PrismaLabStore

    store = PrismaLabStore()
    store.init_admin_tables()

    count = 0

    for order, (title, slug, emoji) in enumerate(FAST_STYLES_MALE, start=1):
        prompt = PERSONA_STYLE_PROMPTS.get(slug, "")
        store.upsert_express_style(
            slug=slug, title=title, emoji=emoji,
            gender="male", prompt=prompt,
            model="seedream", sort_order=order,
        )
        count += 1

    for order, (title, slug, emoji) in enumerate(FAST_STYLES_FEMALE, start=1):
        prompt = PERSONA_STYLE_PROMPTS.get(slug, "")
        store.upsert_express_style(
            slug=slug, title=title, emoji=emoji,
            gender="female", prompt=prompt,
            model="seedream", sort_order=order,
        )
        count += 1

    print(f"✅ Seeded {count} express styles (upsert by slug)")

    # Показать статистику
    males = store.get_express_styles(gender="male")
    females = store.get_express_styles(gender="female")
    print(f"   Male: {len(males)}, Female: {len(females)}, Total: {len(males) + len(females)}")


if __name__ == "__main__":
    main()
