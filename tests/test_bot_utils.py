"""Тесты чистых функций из bot.py."""
from __future__ import annotations

import os

os.environ.pop("DATABASE_URL", None)
os.environ["PRISMALAB_DB_PATH"] = ":memory:"


def test_find_pack_offer_found():
    """_find_pack_offer находит пак по ID."""
    from prismalab.bot import _find_pack_offer
    offer = _find_pack_offer(4345)
    assert offer is not None
    assert offer["id"] == 4345
    assert offer["title"] == "8 марта"


def test_find_pack_offer_not_found():
    """_find_pack_offer возвращает None для несуществующего ID."""
    from prismalab.bot import _find_pack_offer
    offer = _find_pack_offer(999999)
    assert offer is None


def test_pack_offers_returns_list():
    """_pack_offers возвращает непустой список."""
    from prismalab.bot import _pack_offers
    offers = _pack_offers()
    assert isinstance(offers, list)
    assert len(offers) > 0


def test_pack_offers_have_required_fields():
    """Каждый оффер имеет обязательные поля."""
    from prismalab.bot import _pack_offers
    for offer in _pack_offers():
        assert "id" in offer
        assert "title" in offer
        assert "price_rub" in offer
        assert "expected_images" in offer


def test_format_balance_persona():
    """_format_balance_persona возвращает строку с числом."""
    from prismalab.bot import _format_balance_persona
    text = _format_balance_persona(20)
    assert isinstance(text, str)
    assert "20" in text
