"""Starlette-приложение для Telegram Mini App PrismaLab."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from .auth import validate_init_data

logger = logging.getLogger("prismalab.miniapp")

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

BOT_TOKEN = os.getenv("PRISMALAB_BOT_TOKEN", "")
ASTRIA_API_KEY = os.getenv("PRISMALAB_ASTRIA_API_KEY", "")
MINIAPP_URL = os.getenv("MINIAPP_URL", "")
OWNER_ID = int(os.getenv("PRISMALAB_OWNER_ID") or "0")

# Стили (дублируем из bot.py, чтобы не тянуть весь бот)
FAST_STYLES_MALE = [
    {"id": "night_bar", "label": "Ночной бар", "emoji": "🍸"},
    {"id": "suit_window", "label": "В костюме у окна", "emoji": "🪟"},
    {"id": "park_walk", "label": "Прогулка в парке", "emoji": "🌳"},
    {"id": "morning_coffee", "label": "Утренний кофе", "emoji": "☕"},
    {"id": "forest_portrait", "label": "Лесной портрет", "emoji": "🌲"},
    {"id": "night_club", "label": "Ночной клуб", "emoji": "🎶"},
    {"id": "artist_workshop", "label": "Мастерская художника", "emoji": "🎨"},
    {"id": "sunset_silhouette", "label": "Силуэт на закате", "emoji": "🌅"},
    {"id": "biker", "label": "Байкер", "emoji": "🏍"},
    {"id": "pilot", "label": "Пилот", "emoji": "✈️"},
]

FAST_STYLES_FEMALE = [
    {"id": "wedding", "label": "Свадебный образ", "emoji": "💍"},
    {"id": "wet_window", "label": "Мокрое окно", "emoji": "🌧"},
    {"id": "evening_glamour", "label": "Вечерний гламур", "emoji": "✨"},
    {"id": "neon_cyberpunk", "label": "Неоновый киберпанк", "emoji": "🌃"},
    {"id": "dramatic_light", "label": "Драматический свет", "emoji": "💡"},
    {"id": "city_noir", "label": "Городской нуар", "emoji": "🌑"},
    {"id": "studio_smoke", "label": "Студийный дым", "emoji": "💨"},
    {"id": "bw_reflection", "label": "Чёрно-белая рефлексия", "emoji": "🖤"},
    {"id": "ballroom", "label": "Бальный зал", "emoji": "👑"},
    {"id": "greek_queen", "label": "Греческая королева", "emoji": "🏛"},
    {"id": "wet_shirt", "label": "Мокрая рубашка", "emoji": "💧"},
    {"id": "cleopatra", "label": "Клеопатра", "emoji": "🐍"},
    {"id": "old_money", "label": "Old money", "emoji": "💎"},
    {"id": "lavender_beauty", "label": "Лавандовое бьюти", "emoji": "💜"},
    {"id": "silver_illusion", "label": "Серебряная иллюзия", "emoji": "🪞"},
    {"id": "white_purity", "label": "Белоснежная чистота", "emoji": "🤍"},
    {"id": "burgundy_velvet", "label": "Бордовый бархат", "emoji": "🍷"},
    {"id": "grey_cashmere", "label": "Серый кашемир", "emoji": "🧣"},
    {"id": "black_mesh", "label": "Чёрная сетка", "emoji": "🖤"},
    {"id": "lavender_silk", "label": "Лавандовый шёлк", "emoji": "💜"},
    {"id": "silk_lingerie_hotel", "label": "Шёлковое бельё в отеле", "emoji": "🏨"},
    {"id": "bath_petals", "label": "Ванна с лепестками", "emoji": "🛁"},
    {"id": "champagne_balcony", "label": "Шампанское на балконе", "emoji": "🥂"},
    {"id": "rainy_window", "label": "Дождливое окно", "emoji": "🌧"},
    {"id": "coffee_hotel", "label": "Кофе в отеле", "emoji": "☕"},
    {"id": "jazz_bar", "label": "Джазовый бар", "emoji": "🎷"},
    {"id": "picnic_blanket", "label": "Пикник на пледе", "emoji": "🧺"},
    {"id": "art_studio", "label": "Художественная студия", "emoji": "🎨"},
    {"id": "winter_fireplace", "label": "Уют зимнего камина", "emoji": "🔥"},
]

# In-memory хранилище задач генерации (task_id → status/result)
_generation_tasks: dict[str, dict] = {}

# Ссылка на store (устанавливается при создании приложения)
_store = None


def get_store():
    global _store
    if _store is None:
        from prismalab.storage import PrismaLabStore
        _store = PrismaLabStore()
    return _store


def set_store(store):
    global _store
    _store = store


def _get_user_from_request(request: Request) -> dict | None:
    """Извлекает и валидирует initData из заголовка или query."""
    init_data = request.headers.get("X-Telegram-Init-Data", "")
    if not init_data:
        init_data = request.query_params.get("init_data", "")
    if not init_data:
        return None
    return validate_init_data(init_data, BOT_TOKEN)


# ========== Страницы ==========

async def app_page(request: Request):
    """Главная страница Mini App."""
    return templates.TemplateResponse("app.html", {"request": request})


# ========== API ==========

async def api_auth(request: Request):
    """Валидация initData и возврат профиля пользователя."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    init_data = body.get("init_data", "")
    user = validate_init_data(init_data, BOT_TOKEN)
    if not user:
        return JSONResponse({"error": "Invalid init data"}, status_code=401)

    store = get_store()
    profile = store.get_user(user["user_id"])

    user_id = user["user_id"]
    return JSONResponse({
        "user_id": user_id,
        "first_name": user["first_name"],
        "credits": {
            "fast": profile.paid_generations_remaining,
            "free_used": profile.free_generation_used,
        },
        "gender": profile.subject_gender,
        "packs_enabled": True,
        "has_persona": bool(getattr(profile, "astria_lora_tune_id", None)),
        "persona_credits": getattr(profile, "persona_credits_remaining", 0) or 0,
    })


async def api_profile(request: Request):
    """Обновление профиля (пол). Сохраняет в БД как в основном боте."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    init_data = body.get("init_data", "")
    user = validate_init_data(init_data, BOT_TOKEN)
    if not user:
        return JSONResponse({"error": "Invalid init data"}, status_code=401)

    gender = body.get("gender")
    if gender not in ("male", "female"):
        return JSONResponse({"error": "Invalid gender"}, status_code=400)

    store = get_store()
    store.set_subject_gender(user["user_id"], gender)
    return JSONResponse({"ok": True, "gender": gender})


async def api_styles(request: Request):
    """Список стилей по полу."""
    gender = request.query_params.get("gender", "female")
    styles = FAST_STYLES_FEMALE if gender == "female" else FAST_STYLES_MALE
    return JSONResponse({"styles": styles, "gender": gender})


async def api_generate(request: Request):
    """Запуск генерации фото."""
    # Читаем multipart form
    form = await request.form()
    init_data = form.get("init_data", "")

    user = validate_init_data(str(init_data), BOT_TOKEN)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    user_id = user["user_id"]
    style_id = form.get("style_id", "")
    photo = form.get("photo")

    if not style_id:
        return JSONResponse({"error": "No style selected"}, status_code=400)
    if not photo:
        return JSONResponse({"error": "No photo uploaded"}, status_code=400)

    # Проверяем кредиты
    store = get_store()
    profile = store.get_user(user_id)

    has_free = not profile.free_generation_used
    has_paid = profile.paid_generations_remaining > 0

    if not has_free and not has_paid:
        return JSONResponse({"error": "no_credits", "message": "Нет кредитов"}, status_code=402)

    # Читаем фото
    photo_bytes = await photo.read()
    if len(photo_bytes) > 15 * 1024 * 1024:
        return JSONResponse({"error": "Photo too large (max 15MB)"}, status_code=413)

    # Создаём задачу генерации
    task_id = str(uuid.uuid4())[:8]
    _generation_tasks[task_id] = {
        "status": "processing",
        "user_id": user_id,
        "style_id": style_id,
        "result_url": None,
        "error": None,
    }

    # Запускаем генерацию в фоне
    asyncio.get_event_loop().create_task(
        _run_generation(task_id, user_id, style_id, photo_bytes, has_free, profile)
    )

    return JSONResponse({"task_id": task_id, "status": "processing"})


async def _run_generation(task_id: str, user_id: int, style_id: str, photo_bytes: bytes, use_free: bool, profile):
    """Фоновая генерация через KIE."""
    import secrets
    from PIL import Image, ImageOps
    from prismalab.kie_client import (
        upload_file_base64 as kie_upload_file_base64,
        run_task_and_wait as kie_run_task_and_wait,
        download_image_bytes as kie_download_image_bytes,
    )
    from prismalab.persona_prompts import PERSONA_STYLE_PROMPTS
    from prismalab.settings import load_settings

    settings = load_settings()
    store = get_store()

    try:
        # Подготавливаем фото (resize до 1024)
        img = Image.open(io.BytesIO(photo_bytes))
        img = ImageOps.exif_transpose(img) or img
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        max_side = 1024
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        prepared_bytes = buf.getvalue()

        random_id = secrets.token_hex(8)

        # Загружаем в KIE
        uploaded_url = await asyncio.to_thread(
            kie_upload_file_base64,
            api_key=settings.kie_api_key,
            image_bytes=prepared_bytes,
            file_name=f"miniapp_{random_id}.jpg",
        )

        # Определяем промпт и пол
        gender = profile.subject_gender or "female"

        # Ищем промпт для стиля
        style_label = style_id
        all_styles = FAST_STYLES_FEMALE if gender == "female" else FAST_STYLES_MALE
        for s in all_styles:
            if s["id"] == style_id:
                style_label = s["label"]
                break

        # Получаем промпт из PERSONA_STYLE_PROMPTS (dict[str, str])
        prompt = PERSONA_STYLE_PROMPTS.get(style_id, "")
        if not prompt:
            prompt = f"Professional photo portrait, {style_label} style, high quality, detailed"

        # Запускаем генерацию через KIE (Seedream 4.5-edit)
        kie_result = await kie_run_task_and_wait(
            api_key=settings.kie_api_key,
            model="seedream/4.5-edit",
            prompt=prompt,
            image_input=[uploaded_url],
            aspect_ratio="1:1",
            quality="basic",
            output_format="jpg",
            max_seconds=settings.kie_max_seconds,
            poll_seconds=3.0,
        )

        if not kie_result.image_url:
            raise RuntimeError("KIE returned no image URL")

        # Скачиваем результат
        result_bytes = await asyncio.to_thread(kie_download_image_bytes, kie_result.image_url)

        # Сохраняем результат как base64 data URL
        result_b64 = base64.b64encode(result_bytes).decode()
        data_url = f"data:image/jpeg;base64,{result_b64}"

        # Списываем кредит
        if use_free:
            store.spend_free_generation(user_id)
        else:
            store.set_paid_generations_remaining(user_id, profile.paid_generations_remaining - 1)

        # Логируем
        store.log_event(user_id, "generation", {
            "mode": "fast",
            "style": style_id,
            "provider": "kie",
            "source": "miniapp",
        })

        _generation_tasks[task_id] = {
            "status": "done",
            "user_id": user_id,
            "style_id": style_id,
            "result_url": data_url,
            "error": None,
        }
        logger.info("Mini App generation done: user=%s style=%s task=%s", user_id, style_id, task_id)

    except Exception as e:
        logger.exception("Mini App generation error: user=%s task=%s: %s", user_id, task_id, e)
        _generation_tasks[task_id] = {
            "status": "error",
            "user_id": user_id,
            "style_id": style_id,
            "result_url": None,
            "error": str(e),
        }


async def api_status(request: Request):
    """Проверка статуса генерации."""
    task_id = request.path_params.get("task_id", "")
    task = _generation_tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    response = {"task_id": task_id, "status": task["status"]}
    if task["status"] == "done":
        response["result_url"] = task["result_url"]
    elif task["status"] == "error":
        response["error"] = task["error"]

    return JSONResponse(response)


# ========== Паки Astria ==========

# Паки, которые всегда в списке (даже если нет в env)
DEFAULT_PACKS: list[dict] = [
    {"id": 248, "title": "Dog Art", "price_rub": 990, "expected_images": 16, "class_name": "dog", "category": "animals"},
    {"id": 682, "title": "Cat Meowgic", "price_rub": 990, "expected_images": 43, "class_name": "cat", "category": "animals"},
    {"id": 593, "title": "Kids Halloween", "price_rub": 790, "expected_images": 19, "class_name": "boy", "category": "child"},
    {"id": 859, "title": "Kids Holiday", "price_rub": 790, "expected_images": 40, "class_name": "girl", "category": "child"},
    {"id": 2152, "title": "Nordic Girl", "price_rub": 790, "expected_images": 44, "class_name": "girl", "category": "child"},
    {"id": 2501, "title": "Newborn Dreams", "price_rub": 790, "expected_images": 80, "class_name": "girl", "category": "child"},
]

# Маппинг pack_id → category (если не задан в env)
PACK_ID_CATEGORIES: dict[int, str] = {
    248: "animals",
    682: "animals",
    593: "child",
    859: "child",
    2152: "child",
    2501: "child",
}


def _load_pack_offers() -> list[dict]:
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
                    # Маппинг всегда приоритетнее env — чтобы детские/животные не попадали в женские
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


# Кеш: pack_id → {"data": {...}, "ts": float}
_pack_cache: dict[int, dict] = {}
_PACK_CACHE_TTL = 3600  # 1 час


async def _fetch_pack_data(pack_id: int) -> dict:
    """Загружает данные пака из Astria API и кеширует на 1 час."""
    import time as _time
    cached = _pack_cache.get(pack_id)
    if cached and (_time.time() - cached["ts"]) < _PACK_CACHE_TTL:
        return cached["data"]
    if not ASTRIA_API_KEY:
        return {"cover_url": "", "examples": []}
    try:
        from prismalab.astria_client import _get_pack, _timeout_s
        pack_raw = await asyncio.to_thread(
            _get_pack,
            api_key=ASTRIA_API_KEY,
            pack_id=pack_id,
            timeout_s=_timeout_s(30.0),
        )
        cover_url = pack_raw.get("cover_url") or ""
        # Примеры: prompts_per_class → для нашего class_name → images
        examples: list[str] = []
        prompts_per_class = pack_raw.get("prompts_per_class")
        if isinstance(prompts_per_class, dict):
            for class_name, prompts_list in prompts_per_class.items():
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
        data = {"cover_url": cover_url, "examples": examples}
        _pack_cache[pack_id] = {"data": data, "ts": __import__('time').time()}
        return data
    except Exception as e:
        logger.warning("Ошибка загрузки пака %s из Astria: %s", pack_id, e)
        return {"cover_url": "", "examples": []}


async def api_packs(request: Request):
    """Список доступных паков с обложками. Только для owner."""
    user = _get_user_from_request(request)
    if not user or not OWNER_ID or user["user_id"] != OWNER_ID:
        return JSONResponse({"packs": []})
    offers = _load_pack_offers()
    if not offers:
        return JSONResponse({"packs": []})
    # Параллельная загрузка всех паков из Astria API
    pack_datas = await asyncio.gather(*[_fetch_pack_data(o["id"]) for o in offers])
    result = []
    for offer, pack_data in zip(offers, pack_datas):
        result.append({
            "id": offer["id"],
            "title": offer["title"],
            "price_rub": offer["price_rub"],
            "expected_images": offer["expected_images"],
            "cover_url": pack_data["cover_url"],
            "category": offer.get("category", "female"),
        })
    return JSONResponse({"packs": result})


async def api_pack_detail(request: Request):
    """Детали пака с галереей примеров. Только для owner."""
    user = _get_user_from_request(request)
    if not user or not OWNER_ID or user["user_id"] != OWNER_ID:
        return JSONResponse({"error": "Forbidden"}, status_code=403)
    pack_id_str = request.path_params.get("pack_id", "")
    try:
        pack_id = int(pack_id_str)
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid pack_id"}, status_code=400)
    offers = _load_pack_offers()
    offer = next((o for o in offers if o["id"] == pack_id), None)
    if not offer:
        return JSONResponse({"error": "Pack not found"}, status_code=404)
    pack_data = await _fetch_pack_data(pack_id)
    return JSONResponse({
        "id": offer["id"],
        "title": offer["title"],
        "price_rub": offer["price_rub"],
        "expected_images": offer["expected_images"],
        "cover_url": pack_data["cover_url"],
        "examples": pack_data["examples"],
    })


async def api_pack_buy(request: Request):
    """Создание платежа за пак через ЮKassa. Только для owner."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    init_data = body.get("init_data", "")
    user = validate_init_data(init_data, BOT_TOKEN)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not OWNER_ID or user["user_id"] != OWNER_ID:
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    pack_id_str = request.path_params.get("pack_id", "")
    try:
        pack_id = int(pack_id_str)
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid pack_id"}, status_code=400)

    offers = _load_pack_offers()
    offer = next((o for o in offers if o["id"] == pack_id), None)
    if not offer:
        return JSONResponse({"error": "Pack not found"}, status_code=404)

    user_id = user["user_id"]
    price_rub = offer["price_rub"]

    from prismalab.payment import create_payment, apply_test_amount

    amount = apply_test_amount(float(price_rub))
    return_url = MINIAPP_URL.rstrip("/") + f"?pack_paid={pack_id}" if MINIAPP_URL else ""
    if not MINIAPP_URL:
        logger.warning("MINIAPP_URL не задан — после оплаты пака пользователь не вернётся в Mini App")

    url, payment_id_or_err = create_payment(
        amount_rub=amount,
        description=f"PrismaLab — {offer['title']}",
        metadata={
            "user_id": str(user_id),
            "chat_id": str(user_id),
            "product_type": "persona_pack",
            "credits": str(offer["expected_images"]),
            "pack_id": str(pack_id),
        },
        return_url=return_url,
    )

    if not url:
        logger.error("Ошибка создания платежа пака %s: %s", pack_id, payment_id_or_err)
        return JSONResponse({"error": "Payment creation failed"}, status_code=500)

    # Запускаем поллинг платежа
    store = get_store()
    from prismalab.payment import poll_payment_status
    # Нужен bot для отправки сообщения после оплаты — получаем из контекста
    # В Mini App нет прямого доступа к bot; поллинг запустится в payment.py через webhook
    # Поэтому просто возвращаем ссылку, webhook обработает оплату

    return JSONResponse({"payment_url": url, "payment_id": payment_id_or_err})


# ========== Роуты ==========

routes = [
    Route("/app", app_page, methods=["GET"]),
    Route("/app/", app_page, methods=["GET"]),
    Route("/app/api/auth", api_auth, methods=["POST"]),
    Route("/app/api/profile", api_profile, methods=["POST"]),
    Route("/app/api/styles", api_styles, methods=["GET"]),
    Route("/app/api/generate", api_generate, methods=["POST"]),
    Route("/app/api/status/{task_id}", api_status, methods=["GET"]),
    Route("/app/api/packs", api_packs, methods=["GET"]),
    Route("/app/api/packs/{pack_id:int}", api_pack_detail, methods=["GET"]),
    Route("/app/api/packs/{pack_id:int}/buy", api_pack_buy, methods=["POST"]),
    Mount("/app/static", StaticFiles(directory=str(BASE_DIR / "static")), name="miniapp_static"),
]


def create_miniapp(store=None):
    """Создаёт Starlette-приложение Mini App."""
    if store:
        set_store(store)
    app = Starlette(routes=routes)
    return app
