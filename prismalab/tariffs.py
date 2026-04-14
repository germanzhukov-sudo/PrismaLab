"""
Единый сервис цен продажи.

Единственная точка доступа ко всем ценам: экспресс, персона (создание/пополнение), паки.
Хранит дефолты, читает override из admin_settings, используется в payment.py, routes.py, handlers/.

Инварианты:
1. Это единственный источник цен продажи (fast/persona/packs).
2. Никаких хардкодов цен в payment.py, routes.py, handlers/*.
3. Для паков: admin_settings pack_price_* -> env/default из offer.
4. Обратная совместимость: если в БД нет ключей, всё работает на дефолтах.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prismalab.storage import Storage

logger = logging.getLogger("prismalab.tariffs")

# --- Дефолтные цены (credits -> rub) ---

_DEFAULT_FAST: dict[int, int] = {5: 199, 10: 299, 30: 699}
_DEFAULT_PERSONA_CREATE: dict[int, int] = {5: 299, 20: 599, 40: 999}
_DEFAULT_PERSONA_TOPUP: dict[int, int] = {10: 229, 20: 439, 30: 629}

_ALL_DEFAULTS: dict[str, dict[int, int]] = {
    "fast": _DEFAULT_FAST,
    "persona_create": _DEFAULT_PERSONA_CREATE,
    "persona_topup": _DEFAULT_PERSONA_TOPUP,
}

# Префикс ключей в admin_settings: tariff_fast_5=199, tariff_persona_create_20=599
_KEY_PREFIX = "tariff_"
# Префикс для pack sell price: pack_price_4345=599
_PACK_PRICE_PREFIX = "pack_price_"


def _tariff_key(product_type: str, credits: int) -> str:
    """Ключ в admin_settings для тарифа."""
    return f"{_KEY_PREFIX}{product_type}_{credits}"


def _parse_tariff_key(key: str) -> tuple[str, int] | None:
    """Парсит ключ admin_settings -> (product_type, credits) или None."""
    if not key.startswith(_KEY_PREFIX):
        return None
    rest = key[len(_KEY_PREFIX):]
    # rest = "fast_5" или "persona_create_20"
    # Находим последний _ для отделения credits
    idx = rest.rfind("_")
    if idx <= 0:
        return None
    product_type = rest[:idx]
    try:
        credits = int(rest[idx + 1:])
    except ValueError:
        return None
    if product_type not in _ALL_DEFAULTS:
        return None
    return product_type, credits


# ===== Credit-based tariffs =====


def get_tariff_prices(store: Storage, product_type: str) -> dict[int, int]:
    """Цены для product_type из БД, fallback на дефолт.

    product_type: 'fast' | 'persona_create' | 'persona_topup'
    Returns: {credits: price_rub}
    """
    defaults = dict(_ALL_DEFAULTS.get(product_type, {}))
    if not defaults:
        logger.warning("Unknown product_type: %s", product_type)
        return {}

    prefix = f"{_KEY_PREFIX}{product_type}_"
    try:
        overrides = store.get_admin_settings_by_prefix(prefix)
    except Exception as e:
        logger.warning("Failed to read tariff overrides for %s: %s", product_type, e)
        return defaults

    for key, value in overrides.items():
        parsed = _parse_tariff_key(key)
        if parsed and parsed[0] == product_type:
            try:
                defaults[parsed[1]] = int(float(value))
            except (ValueError, TypeError):
                pass

    return defaults


def set_tariff_prices(store: Storage, product_type: str, prices: dict[int, int]) -> None:
    """Сохраняет цены тарифа в admin_settings."""
    if product_type not in _ALL_DEFAULTS:
        raise ValueError(f"Unknown product_type: {product_type}")

    settings = {}
    for credits, price in prices.items():
        settings[_tariff_key(product_type, credits)] = str(int(price))
    store.set_admin_settings_bulk(settings)


def get_all_tariffs(store: Storage) -> dict[str, list[dict]]:
    """Все тарифы для API. Returns: {product_type: [{credits, price}, ...]}."""
    result = {}
    for product_type in _ALL_DEFAULTS:
        prices = get_tariff_prices(store, product_type)
        result[product_type] = sorted(
            [{"credits": c, "price": p} for c, p in prices.items()],
            key=lambda x: x["credits"],
        )
    return result


def get_price(store: Storage, product_type: str, credits: int) -> int | None:
    """Цена конкретного тарифа. Returns None если не найден."""
    prices = get_tariff_prices(store, product_type)
    return prices.get(credits)


def get_valid_credits(store: Storage, product_type: str) -> list[int]:
    """Список допустимых значений credits для product_type."""
    return sorted(get_tariff_prices(store, product_type).keys())


# ===== Pack sell prices =====


def get_pack_sell_price(store: Storage, pack_id: int, default_price: float) -> float:
    """Цена пака в рублях. DB override -> default_price из pack_offers."""
    key = f"{_PACK_PRICE_PREFIX}{pack_id}"
    try:
        overrides = store.get_admin_settings_by_prefix(key)
        if key in overrides:
            return float(overrides[key])
    except Exception as e:
        logger.warning("Failed to read pack price for %s: %s", pack_id, e)
    return float(default_price)


def set_pack_sell_price(store: Storage, pack_id: int, price_rub: float) -> None:
    """Сохраняет цену пака."""
    store.set_admin_setting(f"{_PACK_PRICE_PREFIX}{pack_id}", str(price_rub))


def reset_pack_sell_price(store: Storage, pack_id: int) -> None:
    """Сбрасывает override цены пака (вернётся к дефолту из env/pack_offers)."""
    store.delete_admin_setting(f"{_PACK_PRICE_PREFIX}{pack_id}")


def get_all_product_types() -> list[str]:
    """Список всех кредитных product_type."""
    return list(_ALL_DEFAULTS.keys())


def get_default_credits(product_type: str) -> list[int]:
    """Список дефолтных credits для product_type (для построения формы в админке)."""
    return sorted(_ALL_DEFAULTS.get(product_type, {}).keys())


def get_pack_price_overrides(store: Storage) -> dict[int, float]:
    """Все override цен паков из БД. Returns: {pack_id: price_rub}."""
    raw = store.get_admin_settings_by_prefix(_PACK_PRICE_PREFIX)
    result = {}
    for key, value in raw.items():
        try:
            pack_id = int(key[len(_PACK_PRICE_PREFIX):])
            result[pack_id] = float(value)
        except (ValueError, TypeError):
            pass
    return result
