"""Офферы паков: загрузка из env + дефолтные.

Отдельный модуль чтобы разорвать круговой импорт payment.py → bot.py.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("prismalab")

# Паки, которые всегда в списке (Mini App + бот)
_DEFAULT_PACK_OFFERS: list[dict[str, Any]] = [
    {"id": 4345, "title": "8 марта", "price_rub": 319.0, "expected_images": 20, "class_name": "woman"},
    {"id": 4344, "title": "Алиса в стране чудес", "price_rub": 319.0, "expected_images": 16, "class_name": "woman"},
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
                        seen_ids.add(pack_id)
                        offers.append({
                            "id": pack_id,
                            "title": title,
                            "price_rub": max(1.0, price_rub),
                            "expected_images": max(0, expected_images),
                            "class_name": class_name,
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
