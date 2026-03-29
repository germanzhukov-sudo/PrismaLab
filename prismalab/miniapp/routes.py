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
from starlette.responses import JSONResponse
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
# Ссылки на bot/application (нужны для payment polling после оплаты)
_bot = None
_application = None
_bot_username: str = ""


def get_store():
    global _store
    if _store is None:
        from prismalab.storage import PrismaLabStore
        _store = PrismaLabStore()
    return _store


def set_store(store):
    global _store
    _store = store


def get_bot():
    return _bot


def set_bot(bot):
    global _bot
    _bot = bot


def get_application():
    return _application


def set_application(application):
    global _application
    _application = application


def get_bot_username() -> str:
    return _bot_username


def set_bot_username(username: str):
    global _bot_username
    _bot_username = username


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
        "has_persona": bool(
            getattr(profile, "astria_lora_tune_id", None)
            or getattr(profile, "astria_lora_pack_tune_id", None)
        ),
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
        download_image_bytes as kie_download_image_bytes,
    )
    from prismalab.kie_client import (
        run_task_and_wait as kie_run_task_and_wait,
    )
    from prismalab.kie_client import (
        upload_file_base64 as kie_upload_file_base64,
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

# Паки, которые всегда в списке (даже если нет в env). Child и animals — не трогать порядок.
DEFAULT_PACKS: list[dict] = [
    {"id": 4345, "title": "8 марта", "price_rub": 319, "expected_images": 20, "class_name": "woman", "category": "female"},
    {"id": 4344, "title": "Алиса в стране чудес", "price_rub": 319, "expected_images": 16, "class_name": "woman", "category": "female"},
]

# Маппинг pack_id → category (если не задан в env)
PACK_ID_CATEGORIES: dict[int, str] = {
    4345: "female",
    4344: "female",
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


# Ручной override обложек для паков, где Astria отдаёт мужчину в cover
PACK_COVER_OVERRIDES: dict[int, str] = {
    236: "https://mp.astria.ai/hga2j0ptkyn1unwm1naek1ayf6k0",  # Игра престолов
    623: "https://mp.astria.ai/asxb0jc3qiwkbknr7m9a71rgnowv",  # Красная дорожка
    3576: "https://mp.astria.ai/svn1catp91nxirtwot2nmt6mx5op",  # Блеск и бизнес
}

# Ручной override примеров — только женские фото в ленте пака
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

# Кеш: pack_id → {"data": {...}, "ts": float}
_pack_cache: dict[tuple[int, str], dict] = {}
_PACK_CACHE_TTL = 3600  # 1 час
_PACKS_FETCH_CONCURRENCY = 4
_gallery_cache: dict[str, Any] = {"packs": {}, "ts": 0.0}
_GALLERY_CACHE_TTL = 300  # 5 минут


def _resolve_pack_class_key(offer: dict) -> str:
    category = str(offer.get("category") or "").strip().lower()
    if category == "female":
        return "woman"
    class_name = str(offer.get("class_name") or "").strip().lower()
    aliases = {
        "female": "woman",
        "male": "man",
    }
    resolved = aliases.get(class_name, class_name)
    return resolved or "woman"


def _extract_pack_cost_info(class_cost: Any) -> tuple[str, str]:
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


def _resolve_pack_expected_images(offer: dict, pack_data: dict, *, pack_id: int | None = None) -> int:
    """Точное количество фото для пака: сначала из Astria costs[class_name].num_images, потом fallback в конфиг."""
    try:
        configured_expected = int(offer.get("expected_images") or 0)
    except Exception:
        configured_expected = 0

    class_key = _resolve_pack_class_key(offer)

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
        # Fallback: если Astria вернул данные, но класс не совпал — берём любое значение из num_images_by_class
        for v in by_class.values():
            if isinstance(v, int) and v > 0:
                logger.debug("pack %s expected_images=%s (Astria fallback, class_key=%s not in %s)", pack_id, v, class_key, list(by_class.keys()))
                return v

    if configured_expected > 0:
        logger.debug("pack %s expected_images=%s (config)", pack_id, configured_expected)
        return configured_expected
    return 20


def _resolve_pack_cost_data(offer: dict, pack_data: dict) -> tuple[str, str]:
    class_key = _resolve_pack_class_key(offer)
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


async def _fetch_pack_data(pack_id: int, *, use_cache: bool = True, filter_class: str = "") -> dict:
    """Загружает данные пака из Astria API и кеширует на 1 час. Ключ кэша: (pack_id, filter_class)."""
    import time as _time
    cache_key = (pack_id, (filter_class or "").strip().lower())
    if use_cache:
        cached = _pack_cache.get(cache_key)
        if cached and (_time.time() - cached["ts"]) < _PACK_CACHE_TTL:
            return cached["data"]
    if not ASTRIA_API_KEY:
        return {
            "cover_url": "",
            "examples": [],
            "num_images_by_class": {},
            "cost_by_class": {},
            "default_num_images": 0,
            "default_cost": {},
        }
    try:
        from prismalab.astria_client import _get_pack, _timeout_s
        pack_raw = await asyncio.to_thread(
            _get_pack,
            api_key=ASTRIA_API_KEY,
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
                cost_field, cost_value = _extract_pack_cost_info(class_cost)
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
        # Примеры: prompts_per_class → фильтруем по class_name
        examples: list[str] = []
        prompts_per_class = pack_raw.get("prompts_per_class")
        if isinstance(prompts_per_class, dict):
            # Определяем какие классы показывать
            fc = filter_class.strip().lower()
            allowed_classes = set()
            if fc:
                allowed_classes.add(fc)
                # Алиасы: woman↔female, man↔male
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
        # Для паков с filter_class используем первый пример как cover — чтобы не показывать мужчину в женских паках
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
        _pack_cache[cache_key] = {"data": data, "ts": __import__('time').time()}
        return data
    except Exception as e:
        logger.warning("Ошибка загрузки пака %s из Astria: %s", pack_id, e)
        return {
            "cover_url": "",
            "examples": [],
            "num_images_by_class": {},
            "cost_by_class": {},
            "default_num_images": 0,
            "default_cost": {},
        }


def _pack_data_from_raw(pack_raw: dict[str, Any], *, include_examples: bool, filter_class: str = "") -> dict[str, Any]:
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
            cost_field, cost_value = _extract_pack_cost_info(class_cost)
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
            # Фильтруем по class_name
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


async def _fetch_gallery_pack_index(*, use_cache: bool = True) -> dict[int, dict[str, Any]]:
    import time as _time
    if use_cache and (_time.time() - float(_gallery_cache.get("ts") or 0.0)) < _GALLERY_CACHE_TTL:
        packs = _gallery_cache.get("packs")
        if isinstance(packs, dict):
            return packs
    if not ASTRIA_API_KEY:
        return {}
    try:
        from prismalab.astria_client import _get_gallery_packs, _timeout_s
        gallery_raw = await asyncio.to_thread(
            _get_gallery_packs,
            api_key=ASTRIA_API_KEY,
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
                index[pack_id] = _pack_data_from_raw(item, include_examples=False)
        _gallery_cache["packs"] = index
        _gallery_cache["ts"] = _time.time()
        return index
    except Exception as e:
        logger.warning("Ошибка загрузки gallery packs из Astria: %s", e)
        return {}


async def api_packs(request: Request):
    """Список доступных паков с обложками."""
    user = _get_user_from_request(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    offers = _load_pack_offers()
    if not offers:
        return JSONResponse({"packs": []})
    # Gallery даёт cover_url, но без num_images (cost_mc_hash вместо costs).
    # Всегда догружаем GET /p/:id для каждого пака — гарантированно получаем num_images из Astria.
    gallery_index = await _fetch_gallery_pack_index(use_cache=True)
    if not gallery_index:
        try:
            gallery_index = await asyncio.wait_for(_fetch_gallery_pack_index(use_cache=False), timeout=2.5)
        except Exception:
            gallery_index = {}

    # Все паки — доп. запрос GET /p/:id для точного num_images
    need_detail = [int(o["id"]) for o in offers]

    # Параллельно догружаем детали паков
    sem = asyncio.Semaphore(_PACKS_FETCH_CONCURRENCY)

    # Маппинг pack_id → class_name из offer
    _offer_class = {int(o["id"]): str(o.get("class_name") or "").lower() for o in offers}

    async def _fetch_detail(pid: int) -> tuple[int, dict]:
        async with sem:
            data = await _fetch_pack_data(pid, filter_class=_offer_class.get(pid, ""))
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
            pack_data = {
                "cover_url": "",
                "examples": [],
                "num_images_by_class": {},
                "cost_by_class": {},
                "default_num_images": 0,
                "default_cost": {},
            }
        expected_images = _resolve_pack_expected_images(offer, pack_data, pack_id=pack_id)
        result.append({
            "id": offer["id"],
            "title": offer["title"],
            "price_rub": offer["price_rub"],
            "expected_images": expected_images,
            "cover_url": pack_data.get("cover_url", ""),
            "category": offer.get("category", "female"),
        })
    return JSONResponse({"packs": result})


async def api_pack_detail(request: Request):
    """Детали пака с галереей примеров."""
    user = _get_user_from_request(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    pack_id_str = request.path_params.get("pack_id", "")
    try:
        pack_id = int(pack_id_str)
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid pack_id"}, status_code=400)
    offers = _load_pack_offers()
    offer = next((o for o in offers if o["id"] == pack_id), None)
    if not offer:
        return JSONResponse({"error": "Pack not found"}, status_code=404)
    # В карточке тоже используем кеш ради скорости.
    pack_data = await _fetch_pack_data(pack_id, filter_class=str(offer.get("class_name") or ""))
    expected_images = _resolve_pack_expected_images(offer, pack_data, pack_id=pack_id)
    return JSONResponse({
        "id": offer["id"],
        "title": offer["title"],
        "price_rub": offer["price_rub"],
        "expected_images": expected_images,
        "cover_url": pack_data["cover_url"],
        "examples": pack_data["examples"],
    })


async def api_pack_buy(request: Request):
    """Создание платежа за пак через ЮKassa."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    init_data = body.get("init_data", "")
    user = validate_init_data(init_data, BOT_TOKEN)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    pack_id_str = request.path_params.get("pack_id", "")
    try:
        pack_id = int(pack_id_str)
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid pack_id"}, status_code=400)

    offers = _load_pack_offers()
    offer = next((o for o in offers if o["id"] == pack_id), None)
    if not offer:
        return JSONResponse({"error": "Pack not found"}, status_code=404)
    # На покупке берём live-данные без кеша: точные цифры в момент оплаты.
    pack_data = await _fetch_pack_data(pack_id, use_cache=False)
    expected_images = _resolve_pack_expected_images(offer, pack_data, pack_id=pack_id)
    pack_cost_field, pack_cost_value = _resolve_pack_cost_data(offer, pack_data)

    user_id = user["user_id"]
    price_rub = offer["price_rub"]

    from prismalab.payment import apply_test_amount, create_payment

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
            "credits": str(expected_images),
            "pack_id": str(pack_id),
            "pack_title": str(offer.get("title") or "")[:100],
            "pack_class": _resolve_pack_class_key(offer)[:24],
            "pack_num_images": str(expected_images),
            "pack_cost_field": pack_cost_field[:24],
            "pack_cost_value": pack_cost_value[:64],
        },
        return_url=return_url,
    )

    if not url:
        logger.error("Ошибка создания платежа пака %s: %s", pack_id, payment_id_or_err)
        try:
            from prismalab.alerts import alert_payment_error
            asyncio.get_event_loop().create_task(
                alert_payment_error(user_id, "persona_pack", str(payment_id_or_err or "payment creation failed"))
            )
        except Exception:
            pass
        return JSONResponse({"error": "Payment creation failed"}, status_code=500)

    # Запускаем поллинг платежа
    bot = get_bot()
    application = get_application()
    if bot and application:
        from prismalab.payment import poll_payment_status
        asyncio.get_event_loop().create_task(poll_payment_status(
            payment_id=payment_id_or_err,
            bot=bot,
            store=get_store(),
            user_id=user_id,
            chat_id=user_id,
            credits=expected_images,
            product_type="persona_pack",
            amount_rub=amount,
            application=application,
        ))
    else:
        logger.warning("pack_buy: bot/application not set — payment polling skipped, webhook will handle")

    return JSONResponse({"payment_url": url, "payment_id": payment_id_or_err})


# ========== Стили Персоны ==========

async def api_persona_styles(request: Request):
    """Каталог стилей персоны (для Mini App)."""
    gender = request.query_params.get("gender", "")
    store = get_store()
    styles = store.get_persona_styles(active_only=True, gender=gender if gender else None)
    result = []
    for s in styles:
        result.append({
            "id": s["id"],
            "slug": s["slug"],
            "title": s["title"],
            "description": s.get("description") or "",
            "gender": s["gender"],
            "image_url": s.get("image_url") or "",
        })
    return JSONResponse({"styles": result})


async def api_persona_style_detail(request: Request):
    """Детали одного стиля персоны."""
    style_id = int(request.path_params["style_id"])
    store = get_store()
    s = store.get_persona_style(style_id)
    if not s:
        return JSONResponse({"error": "Style not found"}, status_code=404)
    return JSONResponse({
        "id": s["id"],
        "slug": s["slug"],
        "title": s["title"],
        "description": s.get("description") or "",
        "gender": s["gender"],
        "image_url": s.get("image_url") or "",
    })


# ========== Персона: покупка, докуп, генерация ==========


async def api_persona_buy(request: Request):
    """Покупка персоны из Mini App: создаёт платёж ЮKassa и возвращает URL для оплаты."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    init_data = body.get("init_data", "")
    user = validate_init_data(init_data, BOT_TOKEN)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        credits = int(body.get("credits", 0))
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid credits"}, status_code=400)

    from prismalab.payment import (
        PRICES_PERSONA_CREATE,
        apply_test_amount,
        create_payment,
        poll_payment_status,
    )

    if credits not in PRICES_PERSONA_CREATE:
        return JSONResponse(
            {"error": f"Invalid credits value: {credits}. Allowed: {list(PRICES_PERSONA_CREATE.keys())}"},
            status_code=400,
        )

    user_id = user["user_id"]
    price_rub = PRICES_PERSONA_CREATE[credits]
    amount = apply_test_amount(float(price_rub))

    from prismalab.payment import YOOKASSA_RETURN_URL

    url, payment_id_or_err = create_payment(
        amount_rub=amount,
        description=f"PrismaLab — Создание персоны ({credits} фото)",
        metadata={
            "user_id": str(user_id),
            "chat_id": str(user_id),
            "product_type": "persona_create",
            "credits": str(credits),
        },
        return_url=YOOKASSA_RETURN_URL,
    )

    if not url:
        logger.error("Ошибка создания платежа persona_create: user=%s err=%s", user_id, payment_id_or_err)
        try:
            from prismalab.alerts import alert_payment_error
            asyncio.get_event_loop().create_task(
                alert_payment_error(user_id, "persona_create", str(payment_id_or_err or "payment creation failed"))
            )
        except Exception:
            pass
        return JSONResponse({"error": "Payment creation failed"}, status_code=500)

    # Запускаем поллинг платежа (нужны bot и application)
    bot = get_bot()
    application = get_application()
    if bot and application:
        asyncio.get_event_loop().create_task(poll_payment_status(
            payment_id=payment_id_or_err,
            bot=bot,
            store=get_store(),
            user_id=user_id,
            chat_id=user_id,
            credits=credits,
            product_type="persona_create",
            amount_rub=amount,
            application=application,
        ))
    else:
        logger.warning("persona_buy: bot/application not set — payment polling skipped, webhook will handle")

    return JSONResponse({"payment_url": url, "payment_id": payment_id_or_err})


async def api_persona_topup(request: Request):
    """Докуп кредитов персоны из Mini App."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    init_data = body.get("init_data", "")
    user = validate_init_data(init_data, BOT_TOKEN)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        credits = int(body.get("credits", 0))
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid credits"}, status_code=400)

    from prismalab.payment import (
        PRICES_PERSONA_TOPUP,
        apply_test_amount,
        create_payment,
        poll_payment_status,
    )

    if credits not in PRICES_PERSONA_TOPUP:
        return JSONResponse(
            {"error": f"Invalid credits value: {credits}. Allowed: {list(PRICES_PERSONA_TOPUP.keys())}"},
            status_code=400,
        )

    user_id = user["user_id"]
    price_rub = PRICES_PERSONA_TOPUP[credits]
    amount = apply_test_amount(float(price_rub))

    from prismalab.payment import YOOKASSA_RETURN_URL

    url, payment_id_or_err = create_payment(
        amount_rub=amount,
        description=f"PrismaLab — Докуп кредитов персоны ({credits} фото)",
        metadata={
            "user_id": str(user_id),
            "chat_id": str(user_id),
            "product_type": "persona_topup",
            "credits": str(credits),
        },
        return_url=YOOKASSA_RETURN_URL,
    )

    if not url:
        logger.error("Ошибка создания платежа persona_topup: user=%s err=%s", user_id, payment_id_or_err)
        try:
            from prismalab.alerts import alert_payment_error
            asyncio.get_event_loop().create_task(
                alert_payment_error(user_id, "persona_topup", str(payment_id_or_err or "payment creation failed"))
            )
        except Exception:
            pass
        return JSONResponse({"error": "Payment creation failed"}, status_code=500)

    bot = get_bot()
    application = get_application()
    if bot and application:
        asyncio.get_event_loop().create_task(poll_payment_status(
            payment_id=payment_id_or_err,
            bot=bot,
            store=get_store(),
            user_id=user_id,
            chat_id=user_id,
            credits=credits,
            product_type="persona_topup",
            amount_rub=amount,
            application=application,
        ))
    else:
        logger.warning("persona_topup: bot/application not set — payment polling skipped, webhook will handle")

    return JSONResponse({"payment_url": url, "payment_id": payment_id_or_err})


async def api_persona_generate(request: Request):
    """Сохраняет батч стилей для генерации и возвращает deeplink в бот."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    init_data = body.get("init_data", "")
    user = validate_init_data(init_data, BOT_TOKEN)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    user_id = user["user_id"]
    styles = body.get("styles", [])
    if not styles or not isinstance(styles, list):
        return JSONResponse({"error": "No styles selected"}, status_code=400)

    store = get_store()
    profile = store.get_user(user_id)
    has_persona = bool(getattr(profile, "astria_lora_tune_id", None))
    if not has_persona:
        return JSONResponse({"error": "No persona"}, status_code=400)

    credits = getattr(profile, "persona_credits_remaining", 0) or 0
    if credits <= 0:
        return JSONResponse({"error": "No credits"}, status_code=402)

    # Ограничиваем батч кредитами и обогащаем промптами из БД
    batch = styles[:credits]
    all_db_styles = store.get_persona_styles(active_only=False)
    db_by_slug = {s["slug"]: s for s in all_db_styles}
    for item in batch:
        db_style = db_by_slug.get(item.get("slug", ""))
        if db_style and db_style.get("prompt"):
            item["prompt"] = db_style["prompt"]

    store.set_pending_persona_batch(user_id, json.dumps(batch))
    logger.info("Persona batch saved for user %s: %d styles", user_id, len(batch))

    # Deeplink в бот
    bot_username = get_bot_username()
    bot_link = f"https://t.me/{bot_username}?start=persona_batch" if bot_username else ""
    return JSONResponse({"ok": True, "count": len(batch), "bot_link": bot_link})


# ========== Аналитика ==========

ALLOWED_TRACK_EVENTS = {
    # Общие
    "miniapp_open",
    "miniapp_gender_select",
    # Навигация
    "nav_persona",
    "nav_fast",
    "nav_packs",
    "nav_profile",
    # Fast photo
    "fast_style_select",
    "fast_upload",
    "fast_generate_start",
    "fast_generate_done",
    "fast_download",
    "fast_try_another",
    # Persona
    "persona_style_filter",
    "persona_style_view",
    "persona_style_select",
    "persona_buy_init",
    "persona_buy_confirm",
    "persona_topup_init",
    "persona_topup_confirm",
    "persona_generate_batch",
    # Packs
    "pack_category_select",
    "pack_detail_view",
    "pack_buy",
}


async def api_track(request: Request):
    """Логирование аналитических событий из Mini App."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    init_data = body.get("init_data", "")
    user = validate_init_data(init_data, BOT_TOKEN)
    if not user:
        return JSONResponse({"error": "Invalid init data"}, status_code=401)

    event = body.get("event", "")
    if event not in ALLOWED_TRACK_EVENTS:
        return JSONResponse({"error": "Unknown event"}, status_code=400)

    event_data = body.get("data") or {}
    if not isinstance(event_data, dict):
        event_data = {}
    event_data["source"] = "miniapp"

    store = get_store()
    store.log_event(user["user_id"], event, event_data)
    return JSONResponse({"ok": True})


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
    Route("/app/api/persona-styles", api_persona_styles, methods=["GET"]),
    Route("/app/api/persona-styles/{style_id:int}", api_persona_style_detail, methods=["GET"]),
    Route("/app/api/persona/buy", api_persona_buy, methods=["POST"]),
    Route("/app/api/persona/topup", api_persona_topup, methods=["POST"]),
    Route("/app/api/persona/generate", api_persona_generate, methods=["POST"]),
    Route("/app/api/track", api_track, methods=["POST"]),
    Mount("/app/static", StaticFiles(directory=str(BASE_DIR / "static")), name="miniapp_static"),
]


def create_miniapp(store=None):
    """Создаёт Starlette-приложение Mini App."""
    if store:
        set_store(store)
    app = Starlette(routes=routes)
    return app
