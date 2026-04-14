"""Тесты сервиса фотосетов (miniapp/services/photosets.py)."""
from __future__ import annotations

from prismalab.miniapp.services.photosets import (
    extract_pack_cost_info,
    load_pack_offers,
    resolve_pack_class_key,
    resolve_pack_cost_data,
    resolve_pack_expected_images,
)


# ── load_pack_offers ──────────────────────────────────────────────────


def test_load_pack_offers_defaults():
    """Без env → только DEFAULT_PACKS."""
    offers = load_pack_offers()
    assert len(offers) >= 2
    ids = [o["id"] for o in offers]
    assert 4345 in ids
    assert 4344 in ids


def test_load_pack_offers_all_have_required_fields():
    """Все offers имеют обязательные поля."""
    offers = load_pack_offers()
    for o in offers:
        assert "id" in o
        assert "title" in o
        assert "price_rub" in o
        assert "expected_images" in o
        assert "class_name" in o
        assert "category" in o


# ── resolve_pack_class_key ────────────────────────────────────────────


def test_resolve_class_key_female():
    assert resolve_pack_class_key({"category": "female"}) == "woman"


def test_resolve_class_key_male():
    assert resolve_pack_class_key({"class_name": "male"}) == "man"


def test_resolve_class_key_woman():
    assert resolve_pack_class_key({"class_name": "woman"}) == "woman"


def test_resolve_class_key_child():
    assert resolve_pack_class_key({"category": "child", "class_name": "girl"}) == "girl"


def test_resolve_class_key_empty():
    assert resolve_pack_class_key({}) == "woman"


# ── extract_pack_cost_info ────────────────────────────────────────────


def test_extract_cost_info_cost():
    assert extract_pack_cost_info({"cost": "500"}) == ("cost", "500")


def test_extract_cost_info_cost_mc():
    assert extract_pack_cost_info({"cost_mc": "abc123"}) == ("cost_mc", "abc123")


def test_extract_cost_info_priority():
    """cost приоритетнее cost_mc."""
    assert extract_pack_cost_info({"cost": "100", "cost_mc": "abc"}) == ("cost", "100")


def test_extract_cost_info_not_dict():
    assert extract_pack_cost_info("not a dict") == ("", "")


def test_extract_cost_info_empty():
    assert extract_pack_cost_info({}) == ("", "")


# ── resolve_pack_expected_images ──────────────────────────────────────


def test_expected_images_from_astria_class():
    """num_images_by_class[woman] → точное значение."""
    offer = {"class_name": "woman", "category": "female", "expected_images": 20}
    pack_data = {"num_images_by_class": {"woman": 16}}
    assert resolve_pack_expected_images(offer, pack_data) == 16


def test_expected_images_alias_female():
    """Если woman нет, но есть female → берём female."""
    offer = {"class_name": "woman", "category": "female", "expected_images": 20}
    pack_data = {"num_images_by_class": {"female": 12}}
    assert resolve_pack_expected_images(offer, pack_data) == 12


def test_expected_images_fallback_config():
    """Нет num_images_by_class → берём из config."""
    offer = {"category": "female", "expected_images": 25}
    pack_data = {}
    assert resolve_pack_expected_images(offer, pack_data) == 25


def test_expected_images_fallback_default():
    """Нет ни Astria ни config → 20."""
    offer = {"category": "female"}
    pack_data = {}
    assert resolve_pack_expected_images(offer, pack_data) == 20


def test_expected_images_default_num_images():
    """default_num_images когда класс не совпадает."""
    offer = {"class_name": "unicorn", "category": "other"}
    pack_data = {"num_images_by_class": {}, "default_num_images": 30}
    assert resolve_pack_expected_images(offer, pack_data) == 30


# ── resolve_pack_cost_data ────────────────────────────────────────────


def test_cost_data_from_class():
    offer = {"category": "female"}
    pack_data = {"cost_by_class": {"woman": {"field": "cost_mc", "value": "abc123"}}}
    assert resolve_pack_cost_data(offer, pack_data) == ("cost_mc", "abc123")


def test_cost_data_fallback_default():
    offer = {"category": "female"}
    pack_data = {
        "cost_by_class": {},
        "default_cost": {"field": "cost", "value": "500"},
    }
    assert resolve_pack_cost_data(offer, pack_data) == ("cost", "500")


def test_cost_data_empty():
    assert resolve_pack_cost_data({}, {}) == ("", "")
