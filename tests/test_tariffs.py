"""Tests for prismalab.tariffs — unified pricing service."""
import pytest
from prismalab.storage import PrismaLabStore
from prismalab.tariffs import (
    _ALL_DEFAULTS,
    _DEFAULT_FAST,
    _DEFAULT_PERSONA_CREATE,
    _DEFAULT_PERSONA_TOPUP,
    get_all_tariffs,
    get_pack_sell_price,
    get_price,
    get_tariff_prices,
    get_valid_credits,
    reset_pack_sell_price,
    set_pack_sell_price,
    set_tariff_prices,
)


# Uses `store` fixture from conftest.py (fresh temp SQLite DB per test)


# ===== Fallback to defaults =====


def test_tariff_fallback_to_default_fast(store):
    """Without DB overrides, fast tariffs return defaults."""
    prices = get_tariff_prices(store, "fast")
    assert prices == _DEFAULT_FAST


def test_tariff_fallback_to_default_persona_create(store):
    prices = get_tariff_prices(store, "persona_create")
    assert prices == _DEFAULT_PERSONA_CREATE


def test_tariff_fallback_to_default_persona_topup(store):
    prices = get_tariff_prices(store, "persona_topup")
    assert prices == _DEFAULT_PERSONA_TOPUP


def test_tariff_unknown_product_type(store):
    """Unknown product type returns empty dict."""
    prices = get_tariff_prices(store, "nonexistent")
    assert prices == {}


# ===== DB overrides =====


def test_tariff_db_overrides_default(store):
    """DB price overrides the default for a specific tariff."""
    # Default fast/5 = 199
    assert get_price(store, "fast", 5) == 199
    # Override to 249
    set_tariff_prices(store, "fast", {5: 249})
    assert get_price(store, "fast", 5) == 249
    # Other tariffs unchanged
    assert get_price(store, "fast", 10) == 299


def test_tariff_db_overrides_persona_create(store):
    set_tariff_prices(store, "persona_create", {20: 699})
    assert get_price(store, "persona_create", 20) == 699
    # 5 and 40 unchanged
    assert get_price(store, "persona_create", 5) == 299
    assert get_price(store, "persona_create", 40) == 999


def test_tariff_db_overrides_multiple(store):
    """Can override multiple tariffs at once."""
    set_tariff_prices(store, "persona_topup", {10: 199, 20: 399, 30: 599})
    prices = get_tariff_prices(store, "persona_topup")
    assert prices == {10: 199, 20: 399, 30: 599}


# ===== get_price =====


def test_get_price_fast(store):
    price = get_price(store, "fast", 5)
    assert price == 199


def test_get_price_persona_create(store):
    price = get_price(store, "persona_create", 20)
    assert price == 599


def test_get_price_persona_topup(store):
    price = get_price(store, "persona_topup", 30)
    assert price == 629


def test_get_price_nonexistent_credits(store):
    price = get_price(store, "fast", 999)
    assert price is None


# ===== get_valid_credits =====


def test_valid_credits_fast(store):
    assert get_valid_credits(store, "fast") == [5, 10, 30]


def test_valid_credits_persona_create(store):
    assert get_valid_credits(store, "persona_create") == [5, 20, 40]


def test_valid_credits_persona_topup(store):
    assert get_valid_credits(store, "persona_topup") == [10, 20, 30]


# ===== get_all_tariffs =====


def test_get_all_tariffs(store):
    tariffs = get_all_tariffs(store)
    assert "fast" in tariffs
    assert "persona_create" in tariffs
    assert "persona_topup" in tariffs

    # Check structure
    for item in tariffs["fast"]:
        assert "credits" in item
        assert "price" in item

    # Check sorted by credits
    fast_credits = [x["credits"] for x in tariffs["fast"]]
    assert fast_credits == sorted(fast_credits)


def test_all_tariffs_fast_included(store):
    """PRICES_FAST is covered by tariffs service (blocker fix)."""
    tariffs = get_all_tariffs(store)
    fast_map = {x["credits"]: x["price"] for x in tariffs["fast"]}
    assert fast_map == {5: 199, 10: 299, 30: 699}


def test_all_tariffs_reflect_overrides(store):
    """get_all_tariffs returns DB-overridden prices."""
    set_tariff_prices(store, "fast", {5: 249})
    tariffs = get_all_tariffs(store)
    fast_map = {x["credits"]: x["price"] for x in tariffs["fast"]}
    assert fast_map[5] == 249
    assert fast_map[10] == 299  # unchanged


# ===== Pack sell prices =====


def test_pack_sell_price_fallback(store):
    """Without DB override, returns default price from pack_offers."""
    price = get_pack_sell_price(store, 4345, 319.0)
    assert price == 319.0


def test_pack_sell_price_override(store):
    """DB override changes the sell price."""
    set_pack_sell_price(store, 4345, 699.0)
    price = get_pack_sell_price(store, 4345, 319.0)
    assert price == 699.0


def test_pack_sell_price_reset(store):
    """Resetting override returns to default."""
    set_pack_sell_price(store, 4345, 699.0)
    assert get_pack_sell_price(store, 4345, 319.0) == 699.0
    reset_pack_sell_price(store, 4345)
    assert get_pack_sell_price(store, 4345, 319.0) == 319.0


def test_pack_sell_price_different_default(store):
    price = get_pack_sell_price(store, 9999, 499.0)
    assert price == 499.0


# ===== UI price == payment price consistency =====


def test_ui_price_equals_payment_price(store):
    """The price shown in UI (get_price) must match what _amount_rub returns."""
    from prismalab.payment import _amount_rub
    for product_type in ("fast", "persona_create", "persona_topup"):
        for credits in get_valid_credits(store, product_type):
            ui_price = get_price(store, product_type, credits)
            payment_price = _amount_rub(store, product_type, credits)
            assert float(ui_price) == payment_price, \
                f"Mismatch for {product_type}/{credits}: UI={ui_price}, payment={payment_price}"


def test_ui_price_equals_payment_after_override(store):
    """After DB override, UI and payment prices still match."""
    from prismalab.payment import _amount_rub
    set_tariff_prices(store, "fast", {5: 249})
    assert get_price(store, "fast", 5) == 249
    assert _amount_rub(store, "fast", 5) == 249.0


# ===== _amount_rub integration =====


def test_amount_rub_uses_tariffs(store):
    """_amount_rub reads from tariffs service, not hardcoded constants."""
    from prismalab.payment import _amount_rub
    assert _amount_rub(store, "fast", 5) == 199.0
    assert _amount_rub(store, "persona_create", 20) == 599.0
    assert _amount_rub(store, "persona_topup", 10) == 229.0


def test_amount_rub_unknown_credits(store):
    """_amount_rub with unknown credits returns fallback 10."""
    from prismalab.payment import _amount_rub
    assert _amount_rub(store, "fast", 999) == 10.0


# ===== Backward compatibility =====


def test_no_hardcoded_prices_in_payment():
    """No PRICES_* constants should exist in payment module."""
    import prismalab.payment as pm
    assert not hasattr(pm, "PRICES_FAST")
    assert not hasattr(pm, "PRICES_PERSONA_CREATE")
    assert not hasattr(pm, "PRICES_PERSONA_TOPUP")


# ===== Single source of pack offers =====


def test_pack_offers_single_source():
    """All modules import from pack_offers.py, not their own defaults."""
    from prismalab.pack_offers import _DEFAULT_PACK_OFFERS
    from prismalab.miniapp.services.photosets import DEFAULT_PACKS
    # photosets.DEFAULT_PACKS should be the same object as pack_offers._DEFAULT_PACK_OFFERS
    assert DEFAULT_PACKS is _DEFAULT_PACK_OFFERS


def test_load_pack_offers_delegates_to_pack_offers():
    """photosets.load_pack_offers() delegates to pack_offers._pack_offers()."""
    from prismalab.miniapp.services.photosets import load_pack_offers
    from prismalab.pack_offers import _pack_offers
    photosets_offers = load_pack_offers()
    canonical_offers = _pack_offers()
    # Same IDs in same order
    assert [o["id"] for o in photosets_offers] == [o["id"] for o in canonical_offers]
    # Same prices
    assert [o["price_rub"] for o in photosets_offers] == [o["price_rub"] for o in canonical_offers]


def test_pack_list_price_with_override(store):
    """Pack list API returns override price, not default (P1 fix)."""
    from prismalab.miniapp.services.photosets import load_pack_offers

    offers = load_pack_offers()
    if not offers:
        return  # no packs to test
    pack = offers[0]
    pack_id = pack["id"]
    default_price = pack["price_rub"]

    # Before override: get_pack_sell_price returns default
    assert get_pack_sell_price(store, pack_id, default_price) == default_price

    # Set override
    set_pack_sell_price(store, pack_id, 777.0)

    # After override: tariffs service returns override
    assert get_pack_sell_price(store, pack_id, default_price) == 777.0

    # Clean up
    reset_pack_sell_price(store, pack_id)
