"""Офферы паков: загрузка из env + дефолтные.

Единственный источник дефолтных паков. Используется в:
- handlers/packs.py (бот-flow покупки/генерации)
- payment.py (вебхук обработка)
- miniapp/services/photosets.py (Mini App API)
- admin/app.py (админка: себестоимость, ценообразование)
- keyboards.py (клавиатуры бота)

Не дублировать этот список нигде!
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("prismalab")

# Канонический список паков-дефолтов.
# Используется как fallback, если env PRISMALAB_ASTRIA_PACK_OFFERS не задан.
# Цены (price_rub) — дефолтные, могут быть переопределены через админку (tariffs service).
_DEFAULT_PACK_OFFERS: list[dict[str, Any]] = [
    {"id": 4345, "title": "8 марта", "price_rub": 599, "expected_images": 20, "class_name": "woman", "category": "female"},
    {"id": 4344, "title": "Алиса в стране чудес", "price_rub": 599, "expected_images": 16, "class_name": "woman", "category": "female"},
    {"id": 248, "title": "Собачий арт", "price_rub": 499, "expected_images": 16, "class_name": "dog", "category": "animals"},
    {"id": 682, "title": "Котомагия", "price_rub": 799, "expected_images": 43, "class_name": "cat", "category": "animals"},
    {"id": 593, "title": "Детский хэллоуин", "price_rub": 499, "expected_images": 19, "class_name": "boy", "category": "child"},
    {"id": 859, "title": "Детская праздничная коллекция", "price_rub": 799, "expected_images": 40, "class_name": "girl", "category": "child"},
    {"id": 2152, "title": "Скандинавская мягкость", "price_rub": 799, "expected_images": 44, "class_name": "girl", "category": "child"},
    {"id": 2501, "title": "Нежная съёмка для новорождённых", "price_rub": 1499, "expected_images": 80, "class_name": "girl", "category": "child"},
]


def _pack_offers() -> list[dict[str, Any]]:
    """Конфиг паков: env PRISMALAB_ASTRIA_PACK_OFFERS + _DEFAULT_PACK_OFFERS."""
    seen_ids: set[int] = set()
    offers: list[dict[str, Any]] = []

    raw = (os.getenv("PRISMALAB_ASTRIA_PACK_OFFERS") or "").strip()
    if raw:
        try:
            items = json.loads(raw)
            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    try:
                        pack_id = int(it.get("id"))
                        title = str(it.get("title") or f"Фотосет #{pack_id}")
                        price_rub = float(it.get("price_rub"))
                        expected_images = int(it.get("expected_images") or 0)
                        class_name_raw = str(it.get("class_name") or "").strip().lower()
                        class_name = class_name_raw if class_name_raw in {"man", "woman", "boy", "girl", "dog", "cat"} else ""
                        category = str(it.get("category") or "").strip().lower()
                        if category not in ("female", "male", "child", "animals"):
                            category = "female"
                        credit_cost = int(it.get("credit_cost") or expected_images)
                        seen_ids.add(pack_id)
                        offers.append({
                            "id": pack_id,
                            "title": title,
                            "price_rub": max(1.0, price_rub),
                            "expected_images": max(0, expected_images),
                            "credit_cost": max(0, credit_cost),
                            "class_name": class_name,
                            "category": category,
                        })
                    except Exception:
                        continue
        except Exception:
            logger.warning("PRISMALAB_ASTRIA_PACK_OFFERS: невалидный JSON")

    for p in _DEFAULT_PACK_OFFERS:
        if p["id"] not in seen_ids:
            offers.append(dict(p))
            seen_ids.add(p["id"])

    return offers


def _find_pack_offer(pack_id: int) -> dict[str, Any] | None:
    """Найти оффер пака по ID."""
    for offer in _pack_offers():
        if int(offer.get("id") or 0) == int(pack_id):
            return offer
    return None
