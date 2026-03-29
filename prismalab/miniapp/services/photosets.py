"""Сервис фотосетов (паков Astria).

Бизнес-логика: загрузка офферов, Astria API, кеширование,
разрешение num_images / cost по классу.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time as _time
from typing import Any

from .generation import run_generation

logger = logging.getLogger("prismalab.miniapp.services.photosets")


# ── Константы ──────────────────────────────────────────────────────────

DEFAULT_PACKS: list[dict] = [
    {"id": 4345, "title": "8 марта", "price_rub": 319, "expected_images": 20, "class_name": "woman", "category": "female"},
    {"id": 4344, "title": "Алиса в стране чудес", "price_rub": 319, "expected_images": 16, "class_name": "woman", "category": "female"},
]

PACK_ID_CATEGORIES: dict[int, str] = {
    4345: "female",
    4344: "female",
}

PACK_COVER_OVERRIDES: dict[int, str] = {
    236: "https://mp.astria.ai/hga2j0ptkyn1unwm1naek1ayf6k0",
    623: "https://mp.astria.ai/asxb0jc3qiwkbknr7m9a71rgnowv",
    3576: "https://mp.astria.ai/svn1catp91nxirtwot2nmt6mx5op",
}

PACK_EXAMPLES_OVERRIDES: dict[int, list[str]] = {
    236: [
        "https://mp.astria.ai/hga2j0ptkyn1unwm1naek1ayf6k0",
        "https://mp.astria.ai/9vtitu8pl1v1bqw4uibj4n3hz80a",
        "https://mp.astria.ai/f9d14hlp4w11agywmkc4zh772oqc",
        "https://mp.astria.ai/ohwdtcya4n7od7ye34v7449y4un3",
        "https://mp.astria.ai/v8irl07y2307mxrr6wy12dktv5ya",
        "https://mp.astria.ai/3f18jbt6pp7aqo50inredyb5ohw8",
        "https://mp.astria.ai/ufknksimiibkj3r28to9ws6nkv3e",
        "https://mp.astria.ai/ir4987ij5qtczduqilk385sdapbc",
        "https://mp.astria.ai/yeai5u3hb6icn80lti24ujpdv753",
        "https://mp.astria.ai/x8vqlw5dxxlm92tvv3ppb4ts9jgk",
    ],
    623: [
        "https://mp.astria.ai/asxb0jc3qiwkbknr7m9a71rgnowv",
        "https://mp.astria.ai/gydbv0tbodkbozvxld2r8a9hv10d",
        "https://mp.astria.ai/a237lzgc7q46wlin8tfqoh8oey62",
        "https://mp.astria.ai/jb8mlgus777inyeo694dtu2giwo0",
        "https://mp.astria.ai/caoms98qwq06jpkp7f291hixd446",
        "https://mp.astria.ai/yaouuw16b8qqkx17jf3yof80odlj",
        "https://mp.astria.ai/bfayi5os3t7ihvq8hoinwd58f31v",
        "https://mp.astria.ai/zj7z475q89486l36mnl4n79tfgjz",
        "https://mp.astria.ai/cypp0uhf3s73lnq9oc25dpi97dd9",
        "https://mp.astria.ai/d5q470ovx5mk6bezhy9i33nq96wt",
        "https://mp.astria.ai/muvbot2a30wcdf3upbg4n6hdw78n",
        "https://mp.astria.ai/qgsy24f2an4jfwjpk4se926elbfv",
        "https://mp.astria.ai/8uxtoxuzoaqmwh48pu10t5ykvigd",
        "https://mp.astria.ai/ct2r8w6hli41i1flwwwh3lx2ivqb",
        "https://mp.astria.ai/tsddiam7re9bva0dru6e479ds4eb",
    ],
    3576: [
        "https://mp.astria.ai/svn1catp91nxirtwot2nmt6mx5op",
        "https://mp.astria.ai/3u5q2m35e0ebrv6r3l4bmitd23t6",
        "https://mp.astria.ai/z3lksic1hmuf9d9xpjiiaknsoy1e",
        "https://mp.astria.ai/kz8ezioaa4ddvruw54chewmyfom7",
        "https://mp.astria.ai/v5uxczakvu6kpwr2z62ipcmhebw6",
        "https://mp.astria.ai/6zqv2zqayvlt8snkie5ly1ts0v8t",
        "https://mp.astria.ai/iorwtqivrbeb0tl336tri24ww1rx",
        "https://mp.astria.ai/53mp9ecmrgy2utanvqji5gmzn2qh",
        "https://mp.astria.ai/084a4ct2g04qpolgdm9avlivxngw",
        "https://mp.astria.ai/ttlxmq3kbnndgs7waonagco5v29z",
        "https://mp.astria.ai/y90vdyg26srup67vrq1638hhayi7",
        "https://mp.astria.ai/8dpii3pjvx2gc76zdk8xv7ja27a8",
        "https://mp.astria.ai/x39iywqtemkjyzol49pg004nc1hr",
        "https://mp.astria.ai/52ni4iklqvurqztu0rysn1mys4xg",
        "https://mp.astria.ai/thxpjl9r6hfvrnawcap0d0joytlv",
    ],
}

_PACKS_FETCH_CONCURRENCY = 4
_PACK_CACHE_TTL = 3600
_GALLERY_CACHE_TTL = 300


# ── Кеши ───────────────────────────────────────────────────────────────

_pack_cache: dict[tuple[int, str], dict] = {}
_gallery_cache: dict[str, Any] = {"packs": {}, "ts": 0.0}


def clear_caches() -> None:
    """Очистка кешей (для тестов)."""
    _pack_cache.clear()
    _gallery_cache.clear()
    _gallery_cache["packs"] = {}
    _gallery_cache["ts"] = 0.0


# ── Офферы ─────────────────────────────────────────────────────────────

def load_pack_offers() -> list[dict]:
    """Парсит PRISMALAB_ASTRIA_PACK_OFFERS из env + добавляет DEFAULT_PACKS."""
    seen_ids: set[int] = set()
    result: list[dict] = []

    raw = os.getenv("PRISMALAB_ASTRIA_PACK_OFFERS", "")
    if raw:
        try:
            offers = json.loads(raw)
            if isinstance(offers, list):
                for o in offers:
                    if not isinstance(o, dict):
                        continue
                    pack_id = int(o.get("id") or 0)
                    if not pack_id:
                        continue
                    seen_ids.add(pack_id)
                    category = PACK_ID_CATEGORIES.get(pack_id)
                    if not category:
                        category = str(o.get("category") or "").strip().lower()
                    if category not in ("female", "child", "animals"):
                        category = "female"
                    result.append({
                        "id": pack_id,
                        "title": str(o.get("title") or f"Pack #{pack_id}"),
                        "price_rub": int(o.get("price_rub") or 0),
                        "expected_images": int(o.get("expected_images") or 20),
                        "class_name": str(o.get("class_name") or "woman"),
                        "category": category,
                    })
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("Ошибка парсинга PRISMALAB_ASTRIA_PACK_OFFERS: %s", e)

    for p in DEFAULT_PACKS:
        if p["id"] not in seen_ids:
            result.append(dict(p))
            seen_ids.add(p["id"])

    return result


# ── Resolve helpers ────────────────────────────────────────────────────

def resolve_pack_class_key(offer: dict) -> str:
    """Определяет Astria class_name по offer (woman/man/girl/boy/dog/cat)."""
    category = str(offer.get("category") or "").strip().lower()
    if category == "female":
        return "woman"
    class_name = str(offer.get("class_name") or "").strip().lower()
    aliases = {"female": "woman", "male": "man"}
    resolved = aliases.get(class_name, class_name)
    return resolved or "woman"


def extract_pack_cost_info(class_cost: Any) -> tuple[str, str]:
    """Извлекает (field, value) из cost-объекта Astria."""
    if not isinstance(class_cost, dict):
        return "", ""
    for key in ("cost", "cost_mc", "price", "amount"):
        value = class_cost.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if not value_str:
            continue
        return str(key), value_str
    return "", ""


def resolve_pack_expected_images(offer: dict, pack_data: dict, *, pack_id: int | None = None) -> int:
    """Точное кол-во фото: Astria costs[class].num_images → config → 20."""
    try:
        configured_expected = int(offer.get("expected_images") or 0)
    except Exception:
        configured_expected = 0

    class_key = resolve_pack_class_key(offer)

    by_class = pack_data.get("num_images_by_class")
    if isinstance(by_class, dict):
        variants = [class_key]
        if class_key == "woman":
            variants.extend(["female", "person"])
        elif class_key == "man":
            variants.extend(["male", "person"])
        elif class_key in ("girl", "boy"):
            variants.extend(["child", "person", "woman", "man", "female", "male"])
        elif class_key in ("dog", "cat"):
            variants.extend(["person", "woman", "man", "female", "male"])
        elif class_key in {"female", "male"}:
            variants.append("person")
        else:
            variants.extend(["person", "woman", "man", "female", "male"])
        for key in variants:
            value = by_class.get(key)
            if isinstance(value, int) and value > 0:
                logger.debug("pack %s expected_images=%s (Astria class=%s)", pack_id, value, key)
                return value
        default_num_images = pack_data.get("default_num_images")
        if isinstance(default_num_images, int) and default_num_images > 0:
            logger.debug("pack %s expected_images=%s (Astria default)", pack_id, default_num_images)
            return default_num_images
        for v in by_class.values():
            if isinstance(v, int) and v > 0:
                logger.debug("pack %s expected_images=%s (Astria fallback)", pack_id, v)
                return v

    if configured_expected > 0:
        logger.debug("pack %s expected_images=%s (config)", pack_id, configured_expected)
        return configured_expected
    return 20


def resolve_pack_cost_data(offer: dict, pack_data: dict) -> tuple[str, str]:
    """Извлекает (cost_field, cost_value) для оплаты пака."""
    class_key = resolve_pack_class_key(offer)
    by_class = pack_data.get("cost_by_class")
    if isinstance(by_class, dict):
        variants = [class_key]
        if class_key == "woman":
            variants.extend(["female", "person"])
        elif class_key == "man":
            variants.extend(["male", "person"])
        elif class_key in {"female", "male"}:
            variants.append("person")
        else:
            variants.extend(["person", "woman", "man", "female", "male"])
        for key in variants:
            value = by_class.get(key)
            if isinstance(value, dict):
                field = str(value.get("field") or "").strip()
                cost_value = str(value.get("value") or "").strip()
                if field and cost_value:
                    return field, cost_value
        default_cost = pack_data.get("default_cost")
        if isinstance(default_cost, dict):
            field = str(default_cost.get("field") or "").strip()
            cost_value = str(default_cost.get("value") or "").strip()
            if field and cost_value:
                return field, cost_value
    return "", ""


# ── Astria API ─────────────────────────────────────────────────────────

def pack_data_from_raw(pack_raw: dict[str, Any], *, include_examples: bool, filter_class: str = "") -> dict[str, Any]:
    """Парсит raw-ответ Astria в стандартный pack_data dict."""
    cover_url = pack_raw.get("cover_url") or ""
    num_images_by_class: dict[str, int] = {}
    cost_by_class: dict[str, dict[str, str]] = {}
    costs = pack_raw.get("costs")
    if isinstance(costs, dict):
        for class_name, class_cost in costs.items():
            if not isinstance(class_cost, dict):
                continue
            cls = str(class_name).strip().lower()
            if not cls:
                continue
            try:
                num_images = int(class_cost.get("num_images") or 0)
            except Exception:
                num_images = 0
            if num_images > 0:
                num_images_by_class[cls] = num_images
            cost_field, cost_value = extract_pack_cost_info(class_cost)
            if cost_field and cost_value:
                cost_by_class[cls] = {"field": cost_field, "value": cost_value}

    default_num_images = 0
    unique_num_images = sorted(set(v for v in num_images_by_class.values() if isinstance(v, int) and v > 0))
    if len(unique_num_images) == 1:
        default_num_images = int(unique_num_images[0])
    if default_num_images <= 0:
        try:
            top_level_num = int(pack_raw.get("num_images") or 0)
            if top_level_num > 0:
                default_num_images = top_level_num
        except Exception:
            pass

    default_cost: dict[str, str] = {}
    unique_costs = sorted(set((v.get("field"), v.get("value")) for v in cost_by_class.values() if isinstance(v, dict)))
    if len(unique_costs) == 1:
        field, value = unique_costs[0]
        if field and value:
            default_cost = {"field": str(field), "value": str(value)}

    examples: list[str] = []
    if include_examples:
        prompts_per_class = pack_raw.get("prompts_per_class")
        if isinstance(prompts_per_class, dict):
            fc = filter_class.strip().lower()
            allowed_classes = set()
            if fc:
                allowed_classes.add(fc)
                aliases = {"woman": "female", "female": "woman", "man": "male", "male": "man"}
                if fc in aliases:
                    allowed_classes.add(aliases[fc])

            for _class_name, prompts_list in prompts_per_class.items():
                cls = str(_class_name).strip().lower()
                if allowed_classes and cls not in allowed_classes:
                    continue
                if not isinstance(prompts_list, list):
                    continue
                for prompt_obj in prompts_list:
                    if not isinstance(prompt_obj, dict):
                        continue
                    imgs = prompt_obj.get("images")
                    if isinstance(imgs, list):
                        for img in imgs:
                            if isinstance(img, str) and img.startswith("http"):
                                examples.append(img)
                    elif isinstance(imgs, str) and imgs.startswith("http"):
                        examples.append(imgs)

    return {
        "cover_url": cover_url,
        "examples": examples,
        "num_images_by_class": num_images_by_class,
        "cost_by_class": cost_by_class,
        "default_num_images": default_num_images,
        "default_cost": default_cost,
    }


def _empty_pack_data() -> dict[str, Any]:
    return {
        "cover_url": "",
        "examples": [],
        "num_images_by_class": {},
        "cost_by_class": {},
        "default_num_images": 0,
        "default_cost": {},
    }


async def fetch_pack_data(pack_id: int, *, astria_api_key: str, use_cache: bool = True, filter_class: str = "") -> dict:
    """Загружает данные пака из Astria API и кеширует на 1 час."""
    cache_key = (pack_id, (filter_class or "").strip().lower())
    if use_cache:
        cached = _pack_cache.get(cache_key)
        if cached and (_time.time() - cached["ts"]) < _PACK_CACHE_TTL:
            return cached["data"]
    if not astria_api_key:
        return _empty_pack_data()
    try:
        from prismalab.astria_client import _get_pack, _timeout_s
        pack_raw = await asyncio.to_thread(
            _get_pack,
            api_key=astria_api_key,
            pack_id=pack_id,
            timeout_s=_timeout_s(30.0),
        )
        cover_url = pack_raw.get("cover_url") or ""
        num_images_by_class: dict[str, int] = {}
        cost_by_class: dict[str, dict[str, str]] = {}
        costs = pack_raw.get("costs")
        if isinstance(costs, dict):
            for class_name, class_cost in costs.items():
                if not isinstance(class_cost, dict):
                    continue
                cls = str(class_name).strip().lower()
                if not cls:
                    continue
                try:
                    num_images = int(class_cost.get("num_images") or 0)
                except Exception:
                    num_images = 0
                if num_images > 0:
                    num_images_by_class[cls] = num_images
                cost_field, cost_value = extract_pack_cost_info(class_cost)
                if cost_field and cost_value:
                    cost_by_class[cls] = {"field": cost_field, "value": cost_value}
        default_num_images = 0
        unique_num_images = sorted(set(v for v in num_images_by_class.values() if isinstance(v, int) and v > 0))
        if len(unique_num_images) == 1:
            default_num_images = int(unique_num_images[0])
        if default_num_images <= 0:
            try:
                top_level_num = int(pack_raw.get("num_images") or 0)
                if top_level_num > 0:
                    default_num_images = top_level_num
            except Exception:
                pass
        default_cost: dict[str, str] = {}
        unique_costs = sorted(set((v.get("field"), v.get("value")) for v in cost_by_class.values() if isinstance(v, dict)))
        if len(unique_costs) == 1:
            field, value = unique_costs[0]
            if field and value:
                default_cost = {"field": str(field), "value": str(value)}
        examples: list[str] = []
        prompts_per_class = pack_raw.get("prompts_per_class")
        if isinstance(prompts_per_class, dict):
            fc = filter_class.strip().lower()
            allowed_classes = set()
            if fc:
                allowed_classes.add(fc)
                aliases = {"woman": "female", "female": "woman", "man": "male", "male": "man"}
                if fc in aliases:
                    allowed_classes.add(aliases[fc])
            for class_name, prompts_list in prompts_per_class.items():
                cls = str(class_name).strip().lower()
                if allowed_classes and cls not in allowed_classes:
                    continue
                if not isinstance(prompts_list, list):
                    continue
                for prompt_obj in prompts_list:
                    if not isinstance(prompt_obj, dict):
                        continue
                    imgs = prompt_obj.get("images")
                    if isinstance(imgs, list):
                        for img in imgs:
                            if isinstance(img, str) and img.startswith("http"):
                                examples.append(img)
                    elif isinstance(imgs, str) and imgs.startswith("http"):
                        examples.append(imgs)
        if filter_class.strip() and examples:
            cover_url = examples[0]
        if pack_id in PACK_COVER_OVERRIDES:
            cover_url = PACK_COVER_OVERRIDES[pack_id]
        if pack_id in PACK_EXAMPLES_OVERRIDES:
            examples = PACK_EXAMPLES_OVERRIDES[pack_id]
        data = {
            "cover_url": cover_url,
            "examples": examples,
            "num_images_by_class": num_images_by_class,
            "cost_by_class": cost_by_class,
            "default_num_images": default_num_images,
            "default_cost": default_cost,
        }
        _pack_cache[cache_key] = {"data": data, "ts": _time.time()}
        return data
    except Exception as e:
        logger.warning("Ошибка загрузки пака %s из Astria: %s", pack_id, e)
        return _empty_pack_data()


async def fetch_gallery_pack_index(*, astria_api_key: str, use_cache: bool = True) -> dict[int, dict[str, Any]]:
    """Загружает индекс gallery packs из Astria."""
    if use_cache and (_time.time() - float(_gallery_cache.get("ts") or 0.0)) < _GALLERY_CACHE_TTL:
        packs = _gallery_cache.get("packs")
        if isinstance(packs, dict):
            return packs
    if not astria_api_key:
        return {}
    try:
        from prismalab.astria_client import _get_gallery_packs, _timeout_s
        gallery_raw = await asyncio.to_thread(
            _get_gallery_packs,
            api_key=astria_api_key,
            public=True,
            listed=True,
            timeout_s=_timeout_s(8.0),
        )
        index: dict[int, dict[str, Any]] = {}
        if isinstance(gallery_raw, list):
            for item in gallery_raw:
                if not isinstance(item, dict):
                    continue
                try:
                    pack_id = int(item.get("id") or 0)
                except Exception:
                    pack_id = 0
                if pack_id <= 0:
                    continue
                index[pack_id] = pack_data_from_raw(item, include_examples=False)
        _gallery_cache["packs"] = index
        _gallery_cache["ts"] = _time.time()
        return index
    except Exception as e:
        logger.warning("Ошибка загрузки gallery packs из Astria: %s", e)
        return {}


async def get_packs_list(*, astria_api_key: str) -> list[dict]:
    """Полный список паков для API: offers + Astria data."""
    offers = load_pack_offers()
    if not offers:
        return []

    gallery_index = await fetch_gallery_pack_index(astria_api_key=astria_api_key, use_cache=True)
    if not gallery_index:
        try:
            gallery_index = await asyncio.wait_for(
                fetch_gallery_pack_index(astria_api_key=astria_api_key, use_cache=False),
                timeout=2.5,
            )
        except Exception:
            gallery_index = {}

    need_detail = [int(o["id"]) for o in offers]
    sem = asyncio.Semaphore(_PACKS_FETCH_CONCURRENCY)
    _offer_class = {int(o["id"]): str(o.get("class_name") or "").lower() for o in offers}

    async def _fetch_detail(pid: int) -> tuple[int, dict]:
        async with sem:
            data = await fetch_pack_data(pid, astria_api_key=astria_api_key, filter_class=_offer_class.get(pid, ""))
            return pid, data

    if need_detail:
        details = await asyncio.gather(*[_fetch_detail(pid) for pid in need_detail], return_exceptions=True)
        for d in details:
            if isinstance(d, Exception):
                continue
            pack_id, data = d
            if isinstance(data, dict) and pack_id:
                gallery_index[pack_id] = {**(gallery_index.get(pack_id) or {}), **data}

    result = []
    for offer in offers:
        pack_id = int(offer["id"])
        pack_data = gallery_index.get(pack_id)
        if not pack_data:
            offer_class = str(offer.get("class_name") or "").strip().lower()
            cache_key = (pack_id, offer_class)
            cached = _pack_cache.get(cache_key)
            if isinstance(cached, dict):
                pack_data = cached.get("data")
        if not isinstance(pack_data, dict):
            pack_data = _empty_pack_data()
        expected_images = resolve_pack_expected_images(offer, pack_data, pack_id=pack_id)
        result.append({
            "id": offer["id"],
            "title": offer["title"],
            "price_rub": offer["price_rub"],
            "expected_images": expected_images,
            "cover_url": pack_data.get("cover_url", ""),
            "category": offer.get("category", "female"),
        })
    return result


async def get_pack_detail(pack_id: int, *, astria_api_key: str) -> dict | None:
    """Детали пака с галереей. None если пак не найден."""
    offers = load_pack_offers()
    offer = next((o for o in offers if o["id"] == pack_id), None)
    if not offer:
        return None
    pack_data = await fetch_pack_data(
        pack_id, astria_api_key=astria_api_key,
        filter_class=str(offer.get("class_name") or ""),
    )
    expected_images = resolve_pack_expected_images(offer, pack_data, pack_id=pack_id)
    return {
        "id": offer["id"],
        "title": offer["title"],
        "price_rub": offer["price_rub"],
        "expected_images": expected_images,
        "cover_url": pack_data["cover_url"],
        "examples": pack_data["examples"],
    }


def get_pack_buy_data(pack_id: int) -> dict | None:
    """Возвращает offer + class_key для покупки. None если пак не найден."""
    offers = load_pack_offers()
    offer = next((o for o in offers if o["id"] == pack_id), None)
    if not offer:
        return None
    return {
        "offer": offer,
        "class_key": resolve_pack_class_key(offer),
    }


# ── Генерация фотосета-стиля (4 фото) ─────────────────────────────────

# In-memory idempotency + user locks (для dev; в проде можно Redis)
_photoset_requests: dict[tuple[int, str], dict] = {}  # (user_id, request_id) → result dict
_photoset_request_ttl = 3600  # 1 час
_user_locks: dict[int, asyncio.Lock] = {}  # user_id → asyncio.Lock


def _cleanup_old_requests() -> None:
    """Удаляет протухшие записи."""
    now = _time.time()
    expired = [key for key, data in _photoset_requests.items()
               if now - data.get("ts", 0) > _photoset_request_ttl]
    for key in expired:
        del _photoset_requests[key]


def check_photoset_idempotency(user_id: int, request_id: str) -> dict | None:
    """Проверяет есть ли результат для данного (user_id, request_id). None если нет."""
    _cleanup_old_requests()
    return _photoset_requests.get((user_id, request_id))


def get_user_lock(user_id: int) -> asyncio.Lock:
    """Возвращает asyncio.Lock для данного user_id (lazy create)."""
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    return _user_locks[user_id]


async def run_style_photoset_generation(
    *,
    user_id: int,
    style_id: int,
    style_row: dict,
    photo_bytes: bytes,
    request_id: str,
    store: Any,
    api_key: str,
    max_seconds: int = 300,
) -> dict:
    """Оркестрирует генерацию фотосета-стиля (4 фото).

    1. Резервирует credit_cost кредитов
    2. Запускает 4 генерации последовательно
    3. При частичном успехе — возвращает кредиты за фейлы
    4. Логирует результат

    Returns:
        dict с контрактом: status, images, requested_count, success_count,
        credits_spent, credits_refunded, credits_balance, request_id
    """
    credit_cost = int(style_row.get("credit_cost", 4) or 4)
    prompt = style_row.get("prompt", "")
    negative_prompt = style_row.get("negative_prompt", "")
    provider = style_row.get("provider", "seedream") or "seedream"
    model_params_json = style_row.get("model_params", "")
    style_slug = style_row.get("slug", str(style_id))
    requested_count = 4

    start_ts = _time.time()

    # 1. Резервируем кредиты атомарно
    reserved = store.reserve_persona_credits(user_id, credit_cost)
    if not reserved:
        profile = store.get_user(user_id)
        result = {
            "status": "error",
            "error": "no_credits",
            "images": [],
            "requested_count": requested_count,
            "success_count": 0,
            "credits_spent": 0,
            "credits_refunded": 0,
            "credits_balance": profile.persona_credits_remaining,
            "request_id": request_id,
            "ts": _time.time(),
        }
        _photoset_requests[(user_id, request_id)] = result
        return result

    # 2. Запускаем 4 генерации последовательно
    images: list[str] = []
    errors: list[str] = []

    for i in range(requested_count):
        try:
            gen_result = await run_generation(
                photo_bytes=photo_bytes,
                style_slug=f"{style_slug}_{i+1}",
                prompt=prompt,
                negative_prompt=negative_prompt,
                provider=provider,
                model_params_json=model_params_json,
                api_key=api_key,
                max_seconds=max_seconds,
            )
            images.append(gen_result.data_url)
        except Exception as e:
            logger.warning(
                "Photoset generation %d/%d failed: user=%s style=%s err=%s",
                i + 1, requested_count, user_id, style_id, e,
            )
            errors.append(str(e))

    success_count = len(images)
    duration_ms = int((_time.time() - start_ts) * 1000)

    # 3. Пропорциональный возврат при частичном успехе
    if success_count == 0:
        # Полный фейл — возврат всех кредитов
        credits_spent = 0
        credits_refunded = credit_cost
        store.refund_persona_credits(user_id, credit_cost)
        status = "error"
    elif success_count < requested_count:
        # Частичный успех — пропорциональный возврат
        credits_spent = (credit_cost * success_count) // requested_count
        credits_refunded = credit_cost - credits_spent
        if credits_refunded > 0:
            store.refund_persona_credits(user_id, credits_refunded)
        status = "partial"
    else:
        # Полный успех
        credits_spent = credit_cost
        credits_refunded = 0
        status = "done"

    # Получаем актуальный баланс
    profile = store.get_user(user_id)
    credits_balance = profile.persona_credits_remaining

    # 4. Логирование
    store.log_event(user_id, "photoset_generation", {
        "style_id": style_id,
        "style_slug": style_slug,
        "provider": provider,
        "requested_count": requested_count,
        "success_count": success_count,
        "credits_spent": credits_spent,
        "credits_refunded": credits_refunded,
        "duration_ms": duration_ms,
        "request_id": request_id,
        "source": "miniapp",
    })

    logger.info(
        "Photoset generation: user=%s style=%s request_id=%s "
        "success=%d/%d spent=%d refund=%d duration=%dms provider=%s",
        user_id, style_id, request_id,
        success_count, requested_count,
        credits_spent, credits_refunded, duration_ms, provider,
    )

    result = {
        "status": status,
        "images": images,
        "requested_count": requested_count,
        "success_count": success_count,
        "credits_spent": credits_spent,
        "credits_refunded": credits_refunded,
        "credits_balance": credits_balance,
        "request_id": request_id,
        "ts": _time.time(),
    }
    if errors:
        result["errors"] = errors

    _photoset_requests[(user_id, request_id)] = result
    return result
