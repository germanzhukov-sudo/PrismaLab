"""Starlette-приложение для Telegram Mini App PrismaLab."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path

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

MINIAPP_V2 = os.getenv("MINIAPP_V2", "") == "1"


async def app_page(request: Request):
    """Главная страница Mini App. MINIAPP_V2=1 → V2 UI."""
    from prismalab.config import custom_request_v1, express_filters_v3
    template = "app_v2.html" if MINIAPP_V2 else "app.html"
    return templates.TemplateResponse(template, {
        "request": request,
        "express_v3": express_filters_v3(),
        "custom_request_v1": custom_request_v1(),
    })


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
    try:
        profile = store.get_user(user["user_id"])
    except Exception as e:
        logger.warning("Mini App auth DB error user=%s: %s", user.get("user_id"), e)
        return JSONResponse({"error": "Temporary backend error"}, status_code=503)

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
    from .services.express import get_styles

    gender = request.query_params.get("gender", "female")
    theme = request.query_params.get("theme")  # опциональный фильтр
    store = get_store()
    styles = get_styles(store, gender=gender, theme=theme or None)
    return JSONResponse({
        "styles": [s.to_api_dict() for s in styles],
        "gender": gender,
    })


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


async def _run_generation(
    task_id: str,
    user_id: int,
    style_id: str,
    photo_bytes: bytes,
    use_free: bool,
    profile,
    provider_override: str | None = None,
):
    """Фоновая генерация через KIE — тонкий слой над services."""
    from prismalab.settings import load_settings

    from .services.express import resolve_style
    from .services.generation import run_generation

    settings = load_settings()
    store = get_store()

    try:
        # Резолвим стиль: БД → PERSONA_STYLE_PROMPTS → hardcoded
        resolved = resolve_style(store, style_id)

        if resolved and resolved.prompt:
            prompt = resolved.prompt
            provider = resolved.provider
            negative_prompt = resolved.negative_prompt
            model_params_json = resolved.model_params_json
        else:
            # Fallback промпт по title
            title = resolved.title if resolved else style_id
            prompt = f"Professional photo portrait, {title} style, high quality, detailed"
            provider = resolved.provider if resolved else "seedream"
            negative_prompt = resolved.negative_prompt if resolved else ""
            model_params_json = resolved.model_params_json if resolved else ""

        # V3: пользователь может выбрать провайдер
        if provider_override and provider_override in ("seedream", "nano-banana-pro"):
            provider = provider_override
            # Сбрасываем model_params при смене провайдера (дефолты провайдера)
            if resolved and provider != resolved.provider:
                model_params_json = ""

        # Вызываем сервис генерации
        result = await run_generation(
            photo_bytes=photo_bytes,
            style_slug=style_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            provider=provider,
            model_params_json=model_params_json,
            api_key=settings.kie_api_key,
            max_seconds=settings.kie_max_seconds,
        )

        # Списываем кредит
        if use_free:
            store.spend_free_generation(user_id)
        else:
            store.set_paid_generations_remaining(user_id, profile.paid_generations_remaining - 1)

        # Логируем
        store.log_event(user_id, "generation", {
            "mode": "fast",
            "style": style_id,
            "provider": provider,
            "source": "miniapp",
        })

        # Phase 4: upload → save history → TG send (best-effort)
        style_title = resolved.title if resolved else style_id
        image_url = None
        tg_sent = False
        mime_type = result.mime_type or "image/jpeg"
        file_ext = result.file_ext or "jpg"

        # 1. Upload в Supabase Storage
        if result.raw_bytes:
            try:
                from prismalab.supabase_storage import upload_generation
                image_url = await asyncio.to_thread(
                    upload_generation,
                    result.raw_bytes,
                    user_id,
                    file_ext,
                    mime_type,
                )
            except Exception as upload_err:
                logger.warning("Upload to storage failed (task=%s): %s", task_id, upload_err)

        # 2. Save в generation_history
        try:
            store.save_generation_history(
                user_id=user_id,
                mode="express",
                style_slug=style_id,
                style_title=style_title,
                provider=provider,
                image_url=image_url,
            )
        except Exception as hist_err:
            logger.warning("Save generation history failed (task=%s): %s", task_id, hist_err)

        # 3. TG send_document (best-effort)
        if result.raw_bytes and _bot:
            try:
                import io
                doc = io.BytesIO(result.raw_bytes)
                doc.name = f"{style_title}_{task_id}.{file_ext}"
                await _bot.send_document(
                    chat_id=user_id,
                    document=doc,
                    caption=f"✨ {style_title}",
                    read_timeout=60,
                    write_timeout=60,
                    connect_timeout=30,
                )
                tg_sent = True
            except Exception as tg_err:
                logger.warning("TG send_document failed (task=%s, user=%s): %s", task_id, user_id, tg_err)

        _generation_tasks[task_id] = {
            "status": "done",
            "user_id": user_id,
            "style_id": style_id,
            "result_url": result.data_url,
            "image_url": image_url,
            "tg_sent": tg_sent,
            "error": None,
        }
        logger.info("Mini App generation done: user=%s style=%s provider=%s task=%s",
                     user_id, style_id, provider, task_id)

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
        response["image_url"] = task.get("image_url")
        response["tg_sent"] = bool(task.get("tg_sent"))
    elif task["status"] == "error":
        response["error"] = task["error"]

    return JSONResponse(response)


# ========== Паки Astria ==========




async def api_packs(request: Request):
    """Список доступных паков с обложками."""
    from .services.photosets import get_packs_list

    user = _get_user_from_request(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    packs = await get_packs_list(astria_api_key=ASTRIA_API_KEY)
    return JSONResponse({"packs": packs})


async def api_pack_detail(request: Request):
    """Детали пака с галереей примеров."""
    from .services.photosets import get_pack_detail

    user = _get_user_from_request(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    pack_id_str = request.path_params.get("pack_id", "")
    try:
        pack_id = int(pack_id_str)
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid pack_id"}, status_code=400)
    detail = await get_pack_detail(pack_id, astria_api_key=ASTRIA_API_KEY)
    if not detail:
        return JSONResponse({"error": "Pack not found"}, status_code=404)
    return JSONResponse(detail)


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

    from .services.photosets import (
        fetch_pack_data,
        get_pack_buy_data,
        resolve_pack_class_key,
        resolve_pack_cost_data,
        resolve_pack_expected_images,
    )

    buy_data = get_pack_buy_data(pack_id)
    if not buy_data:
        return JSONResponse({"error": "Pack not found"}, status_code=404)
    offer = buy_data["offer"]
    # На покупке берём live-данные без кеша: точные цифры в момент оплаты.
    pack_data = await fetch_pack_data(pack_id, astria_api_key=ASTRIA_API_KEY, use_cache=False)
    expected_images = resolve_pack_expected_images(offer, pack_data, pack_id=pack_id)
    pack_cost_field, pack_cost_value = resolve_pack_cost_data(offer, pack_data)

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
            "pack_class": resolve_pack_class_key(offer)[:24],
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
            "credit_cost": int(s.get("credit_cost", 4) or 4),
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
    # V2 Express
    "v2_express_theme_select",
    "v2_express_style_select",
    "v2_express_upload",
    "v2_express_generate_start",
    "v2_express_generate_done",
    "v2_express_download",
    # V2 Photosets
    "v2_photoset_view",
    "v2_photoset_detail",
    "v2_photoset_generate_start",
    "v2_photoset_generate_done",
    "v2_photoset_buy",
    # V2 Navigation
    "v2_nav_express",
    "v2_nav_photosets",
    "v2_nav_profile",
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


# ========== API V2 — Express & Photosets ==========


async def api_v2_express_themes(request: Request):
    """V2: Список тем экспресс-стилей."""
    from .services.express import get_themes

    user = _get_user_from_request(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    gender = request.query_params.get("gender", "")
    store = get_store()
    themes = get_themes(store, gender=gender or None)
    return JSONResponse({"themes": themes, "gender": gender})


async def api_v2_express_styles(request: Request):
    """V2: Каталог экспресс-стилей (из БД с fallback)."""
    from .services.express import get_styles

    user = _get_user_from_request(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    gender = request.query_params.get("gender", "female")
    theme = request.query_params.get("theme", "")
    store = get_store()
    styles = get_styles(store, gender=gender, theme=theme or None)
    return JSONResponse({
        "styles": [s.to_api_dict() for s in styles],
        "gender": gender,
        "theme": theme,
    })


async def api_v2_express_generate(request: Request):
    """V2: Генерация экспресс-фото (поддержка провайдеров из БД)."""
    from .services.express import resolve_style

    form = await request.form()
    init_data = form.get("init_data", "")

    user = validate_init_data(str(init_data), BOT_TOKEN)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    user_id = user["user_id"]
    style_slug = str(form.get("style_id", "") or form.get("style_slug", "")).strip()
    photo = form.get("photo")

    if not style_slug:
        return JSONResponse({"error": "No style selected"}, status_code=400)
    if not photo:
        return JSONResponse({"error": "No photo uploaded"}, status_code=400)

    # Проверяем что стиль существует и активен
    store = get_store()
    resolved = resolve_style(store, style_slug)
    if not resolved:
        return JSONResponse({"error": "Style not found or inactive"}, status_code=404)

    # Проверяем кредиты
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
        "style_id": style_slug,
        "result_url": None,
        "error": None,
    }

    asyncio.get_event_loop().create_task(
        _run_generation(task_id, user_id, style_slug, photo_bytes, has_free, profile)
    )

    return JSONResponse({
        "task_id": task_id,
        "status": "processing",
        "provider": resolved.provider,
    })


async def api_v2_photosets(request: Request):
    """V2: Список фотосетов (паков). Включает packs_use_credits flag."""
    from prismalab.config import packs_use_credits
    from .services.photosets import get_packs_list

    user = _get_user_from_request(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    packs = await get_packs_list(astria_api_key=ASTRIA_API_KEY)

    # Опциональный фильтр по category
    category = request.query_params.get("category", "")
    if category:
        packs = [p for p in packs if p.get("category") == category]

    return JSONResponse({
        "packs": packs,
        "packs_use_credits": packs_use_credits(),
    })


async def api_v2_pack_buy_credits(request: Request):
    """V2: Покупка пака за persona_credits (если PACKS_USE_CREDITS=1)."""
    from prismalab.config import packs_use_credits
    from .services.photosets import get_pack_buy_data

    if not packs_use_credits():
        return JSONResponse({"error": "Credit-based pack purchase is not enabled"}, status_code=403)

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

    buy_data = get_pack_buy_data(pack_id)
    if not buy_data:
        return JSONResponse({"error": "Pack not found"}, status_code=404)
    offer = buy_data["offer"]
    credit_cost = int(offer.get("credit_cost", offer.get("expected_images", 20)))

    user_id = user["user_id"]
    store = get_store()
    profile = store.get_user(user_id)

    if profile.persona_credits_remaining < credit_cost:
        return JSONResponse({
            "error": "no_credits",
            "message": f"Нужно {credit_cost} кредитов, есть {profile.persona_credits_remaining}",
            "credits_balance": profile.persona_credits_remaining,
            "credits_required": credit_cost,
        }, status_code=402)

    # Атомарное списание
    reserved = store.reserve_persona_credits(user_id, credit_cost)
    if not reserved:
        return JSONResponse({"error": "no_credits", "message": "Не удалось списать кредиты"}, status_code=402)

    # Логируем
    store.log_event(user_id, "pack_buy_credits", {
        "pack_id": pack_id,
        "pack_title": offer.get("title", ""),
        "credit_cost": credit_cost,
        "source": "miniapp",
    })

    # Запускаем генерацию пака — через бот (как при обычной покупке)
    bot = get_bot()
    application = get_application()
    if bot and application:
        try:
            from prismalab.handlers.packs import _run_persona_pack_generation
            import asyncio
            asyncio.get_event_loop().create_task(
                _run_persona_pack_generation(
                    bot=bot, store=store, application=application,
                    user_id=user_id, chat_id=user_id,
                    pack_id=str(pack_id),
                    pack_title=str(offer.get("title", "")),
                    pack_class=buy_data["class_key"],
                    pack_num_images=str(offer.get("expected_images", 20)),
                )
            )
        except Exception as e:
            logger.warning("pack_buy_credits: failed to start generation: %s", e)

    updated_profile = store.get_user(user_id)
    return JSONResponse({
        "ok": True,
        "pack_id": pack_id,
        "credits_spent": credit_cost,
        "credits_balance": updated_profile.persona_credits_remaining,
    })


async def api_v2_photoset_generate(request: Request):
    """V2: Генерация фотосета-стиля (4 фото с биллингом).

    POST /app/api/v2/photosets/{kind}/{id}/generate
    kind=style → persona_style, kind=pack → 501 (пока)
    """
    from .services.photosets import (
        _photoset_requests,
        check_photoset_idempotency,
        get_user_lock,
        run_style_photoset_generation,
    )

    kind = request.path_params.get("kind", "")
    item_id_str = request.path_params.get("id", "")

    if kind == "pack":
        return JSONResponse({"error": "Pack generation via credits not yet implemented"}, status_code=501)
    if kind != "style":
        return JSONResponse({"error": "Invalid kind, expected 'style' or 'pack'"}, status_code=400)

    # Читаем form (multipart: init_data, photo, request_id)
    form = await request.form()
    init_data = form.get("init_data", "")

    user = validate_init_data(str(init_data), BOT_TOKEN)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    user_id = user["user_id"]

    try:
        item_id = int(item_id_str)
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid style id"}, status_code=400)

    request_id = str(form.get("request_id", "")).strip()
    if not request_id:
        return JSONResponse({"error": "request_id required"}, status_code=400)

    # Идемпотентность: ключ = (user_id, request_id)
    existing = check_photoset_idempotency(user_id, request_id)
    if existing:
        return JSONResponse(existing)

    # User-level lock: asyncio.Lock per user — атомарная проверка
    lock = get_user_lock(user_id)
    if lock.locked():
        return JSONResponse(
            {"error": "Generation already in progress", "status": "processing"},
            status_code=409,
        )

    # Валидируем стиль
    store = get_store()
    style_row = store.get_persona_style(item_id)
    if not style_row:
        return JSONResponse({"error": "Style not found"}, status_code=404)
    if not style_row.get("is_active", 1):
        return JSONResponse({"error": "Style is inactive"}, status_code=404)

    # Читаем фото
    photo = form.get("photo")
    if not photo:
        return JSONResponse({"error": "No photo uploaded"}, status_code=400)

    photo_bytes = await photo.read()
    if len(photo_bytes) > 15 * 1024 * 1024:
        return JSONResponse({"error": "Photo too large (max 15MB)"}, status_code=413)

    # Проверяем кредиты (предварительно, reserve будет в сервисе)
    credit_cost = int(style_row.get("credit_cost", 4) or 4)
    profile = store.get_user(user_id)
    if profile.persona_credits_remaining < credit_cost:
        return JSONResponse({
            "error": "no_credits",
            "message": f"Нужно {credit_cost} кредитов, есть {profile.persona_credits_remaining}",
            "credits_balance": profile.persona_credits_remaining,
            "credits_required": credit_cost,
        }, status_code=402)

    # Регистрируем request_id как processing
    idem_key = (user_id, request_id)
    _photoset_requests[idem_key] = {
        "status": "processing",
        "images": [],
        "requested_count": 4,
        "success_count": 0,
        "credits_spent": 0,
        "credits_refunded": 0,
        "credits_balance": profile.persona_credits_remaining,
        "request_id": request_id,
        "ts": __import__("time").time(),
    }

    from prismalab.settings import load_settings
    settings = load_settings()

    async with lock:
        try:
            result = await run_style_photoset_generation(
                user_id=user_id,
                style_id=item_id,
                style_row=style_row,
                photo_bytes=photo_bytes,
                request_id=request_id,
                store=store,
                api_key=settings.kie_api_key,
                max_seconds=settings.kie_max_seconds,
            )
        except Exception as e:
            logger.exception("Photoset generation error: user=%s request_id=%s: %s", user_id, request_id, e)
            result = {
                "status": "error",
                "error": str(e),
                "images": [],
                "requested_count": 4,
                "success_count": 0,
                "credits_spent": 0,
                "credits_refunded": 0,
                "credits_balance": store.get_user(user_id).persona_credits_remaining,
                "request_id": request_id,
                "ts": __import__("time").time(),
            }
            _photoset_requests[idem_key] = result

    return JSONResponse(result)


# ========== V3: Express Catalog + Generate ==========

async def api_v3_express_catalog(request: Request):
    """V3: Единый каталог — категории, теги, стили с фильтрацией.

    GET /app/api/v3/express/catalog
    Params:
      - category (slug, опционально): фильтр по категории
      - tags (comma-separated slugs, опционально): фильтр по тегам
    Response: {categories, tags, styles, last_provider}
    """
    from .services.express import get_categories, get_styles_filtered, get_tags_for_category

    user = _get_user_from_request(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    store = get_store()
    user_id = user["user_id"]

    # Категории (всегда все активные)
    categories = get_categories(store)

    # Текущий выбранный фильтр
    cat_slug = request.query_params.get("category", "").strip()
    tag_slugs_raw = request.query_params.get("tags", "").strip()
    tag_slugs = [t.strip() for t in tag_slugs_raw.split(",") if t.strip()] if tag_slugs_raw else []

    # Теги для выбранной категории
    tags = []
    if cat_slug and cat_slug != "all":
        cat_row = store.get_express_category_by_slug(cat_slug)
        if cat_row:
            tags = get_tags_for_category(store, cat_row["id"])

    # Стили
    cat_filter = [cat_slug] if cat_slug and cat_slug != "all" else None
    styles = get_styles_filtered(store, category_slugs=cat_filter, tag_slugs=tag_slugs or None)

    # Кредиты + последний провайдер
    profile = store.get_user(user_id)
    has_free = not profile.free_generation_used
    fast_credits = profile.paid_generations_remaining
    last_provider = store.get_user_last_express_provider(user_id)

    return JSONResponse({
        "categories": [c.to_api_dict() for c in categories],
        "tags": [t.to_api_dict() for t in tags],
        "styles": [s.to_api_dict() for s in styles],
        "credits": {
            "fast": fast_credits,
            "free_used": profile.free_generation_used,
            "balance": fast_credits if has_free is False else fast_credits + 1,
        },
        "last_provider": last_provider,
    })


async def api_v3_express_generate(request: Request):
    """V3: Генерация с выбором провайдера пользователем.

    POST /app/api/v3/express/generate
    FormData: init_data, style_id, photo, provider (seedream|nano-banana-pro)
    """
    from .services.express import resolve_style

    form = await request.form()
    init_data = form.get("init_data", "")

    user = validate_init_data(str(init_data), BOT_TOKEN)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    user_id = user["user_id"]
    style_slug = str(form.get("style_id", "") or form.get("style_slug", "")).strip()
    photo = form.get("photo")
    provider_choice = str(form.get("provider", "")).strip()

    if not style_slug:
        return JSONResponse({"error": "No style selected"}, status_code=400)
    if not photo:
        return JSONResponse({"error": "No photo uploaded"}, status_code=400)

    store = get_store()
    resolved = resolve_style(store, style_slug)
    if not resolved:
        return JSONResponse({"error": "Style not found or inactive"}, status_code=404)

    # Кредиты
    profile = store.get_user(user_id)
    has_free = not profile.free_generation_used
    has_paid = profile.paid_generations_remaining > 0
    if not has_free and not has_paid:
        return JSONResponse({"error": "no_credits", "message": "Нет кредитов"}, status_code=402)

    # Фото
    photo_bytes = await photo.read()
    if len(photo_bytes) > 15 * 1024 * 1024:
        return JSONResponse({"error": "Photo too large (max 15MB)"}, status_code=413)

    # Провайдер: выбор пользователя → запомнить
    provider_override = provider_choice if provider_choice in ("seedream", "nano-banana-pro") else None
    actual_provider = provider_override or resolved.provider
    if provider_override:
        store.set_user_last_express_provider(user_id, provider_override)

    # Задача
    task_id = str(uuid.uuid4())[:8]
    _generation_tasks[task_id] = {
        "status": "processing",
        "user_id": user_id,
        "style_id": style_slug,
        "result_url": None,
        "error": None,
    }

    asyncio.get_event_loop().create_task(
        _run_generation(task_id, user_id, style_slug, photo_bytes, has_free, profile,
                        provider_override=provider_override)
    )

    return JSONResponse({
        "task_id": task_id,
        "status": "processing",
        "provider": actual_provider,
    })


async def api_v3_history(request: Request):
    """V3: История генераций пользователя.

    GET /app/api/v3/history?mode=express&limit=20&offset=0
    """
    user = _get_user_from_request(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    store = get_store()
    user_id = user["user_id"]

    mode = request.query_params.get("mode", "").strip() or None
    if mode and mode not in ("express", "photoset"):
        mode = None

    try:
        limit = int(request.query_params.get("limit", "20"))
    except (ValueError, TypeError):
        limit = 20
    try:
        offset = int(request.query_params.get("offset", "0"))
    except (ValueError, TypeError):
        offset = 0

    rows = store.get_generation_history(user_id, mode=mode, limit=limit, offset=offset)
    total = store.get_generation_history_total(user_id, mode=mode)

    items = []
    for r in rows:
        items.append({
            "id": r.get("id"),
            "mode": r.get("mode", "express"),
            "style_slug": r.get("style_slug"),
            "style_title": r.get("style_title"),
            "provider": r.get("provider"),
            "image_url": r.get("image_url"),
            "created_at": str(r.get("created_at", "")),
        })

    return JSONResponse({"items": items, "total": total, "limit": limit, "offset": offset})


# ========== Custom Prompt Generation ==========

async def api_v3_custom_capabilities(request: Request):
    """V3: Capabilities for custom prompt generation."""
    from .services.generation import CUSTOM_CAPABILITIES
    return JSONResponse(CUSTOM_CAPABILITIES)


async def api_v3_custom_generate(request: Request):
    """V3: Generate from free-form prompt with optional photo references.

    POST /app/api/v3/custom/generate
    FormData: init_data, prompt, provider, request_id, photos (repeated key)
    """
    import uuid as _uuid
    from PIL import Image as _PILImage

    from .services.generation import CUSTOM_CAPABILITIES, run_custom_generation
    from .services.locks import acquire_generation_lock, release_generation_lock

    form = await request.form()
    init_data = form.get("init_data", "")

    user = validate_init_data(str(init_data), BOT_TOKEN)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    user_id = user["user_id"]
    prompt = str(form.get("prompt", "")).strip()
    provider_choice = str(form.get("provider", "")).strip()
    request_id = str(form.get("request_id", "")).strip()

    # P3 fix: request_id is required
    if not request_id:
        return JSONResponse({"error": "request_id is required"}, status_code=400)
    try:
        _uuid.UUID(request_id)
    except ValueError:
        return JSONResponse({"error": "Invalid request_id (must be UUID)"}, status_code=400)

    # Validate prompt
    max_prompt = CUSTOM_CAPABILITIES["max_prompt_length"]
    if not prompt:
        return JSONResponse({"error": "Prompt is required"}, status_code=400)
    if len(prompt) > max_prompt:
        return JSONResponse({"error": f"Prompt too long (max {max_prompt} chars)"}, status_code=400)

    # Validate provider
    if provider_choice not in ("seedream", "nano-banana-pro"):
        provider_choice = "seedream"

    # P4 fix: cleanup BEFORE idempotency check
    store = get_store()
    store.cleanup_old_requests(max_age_hours=1)

    # Idempotency check
    existing_task = store.check_request_id(user_id, request_id)
    if existing_task:
        return JSONResponse({"task_id": existing_task, "status": "processing", "provider": provider_choice, "idempotent": True})

    # Read photos (repeated key "photos")
    _ALLOWED_PILLOW_FORMATS = {"JPEG", "PNG", "WEBP"}
    photos_raw = form.getlist("photos")
    photo_bytes_list: list[bytes] = []
    max_photos = CUSTOM_CAPABILITIES["providers"][provider_choice]["max_photos"]
    max_size = CUSTOM_CAPABILITIES["max_file_size_mb"] * 1024 * 1024
    allowed_mime = set(CUSTOM_CAPABILITIES["allowed_mime"])

    for i, photo in enumerate(photos_raw):
        if not hasattr(photo, "read"):
            continue
        if len(photo_bytes_list) >= max_photos:
            return JSONResponse({"error": f"Too many photos (max {max_photos} for {provider_choice})"}, status_code=400)
        photo_data = await photo.read()
        if len(photo_data) > max_size:
            return JSONResponse({"error": f"Photo {i+1} too large (max {CUSTOM_CAPABILITIES['max_file_size_mb']}MB)"}, status_code=413)
        # MIME check: content_type header
        ct = getattr(photo, "content_type", "") or ""
        if ct and ct not in allowed_mime:
            return JSONResponse({"error": f"Photo {i+1}: unsupported format ({ct})"}, status_code=400)
        # P2 fix: Pillow verify + format whitelist (catches GIF/BMP even with empty content_type)
        try:
            import io as _io
            img = _PILImage.open(_io.BytesIO(photo_data))
            fmt = (img.format or "").upper()
            img.verify()
            if fmt not in _ALLOWED_PILLOW_FORMATS:
                return JSONResponse({"error": f"Photo {i+1}: unsupported format ({fmt})"}, status_code=400)
        except Exception:
            return JSONResponse({"error": f"Photo {i+1}: invalid or corrupted image"}, status_code=400)
        photo_bytes_list.append(photo_data)

    # Total payload check (60MB)
    total_size = sum(len(b) for b in photo_bytes_list)
    if total_size > 60 * 1024 * 1024:
        return JSONResponse({"error": "Total upload size too large (max 60MB)"}, status_code=413)

    # User lock
    lock = await acquire_generation_lock(user_id)
    if lock is None:
        return JSONResponse({"error": "Generation already in progress"}, status_code=409)

    # P1 fix: everything after lock acquire wrapped in try/finally
    try:
        # Credits check
        profile = store.get_user(user_id)
        has_free = not profile.free_generation_used
        has_paid = profile.paid_generations_remaining > 0
        if not has_free and not has_paid:
            return JSONResponse({"error": "no_credits", "message": "Нет кредитов"}, status_code=402)

        # Create task
        task_id = str(_uuid.uuid4())[:8]
        _generation_tasks[task_id] = {
            "status": "processing",
            "user_id": user_id,
            "style_id": "__custom__",
            "result_url": None,
            "error": None,
        }

        # Save request_id atomically (ON CONFLICT safe)
        saved_task = store.save_request_id(request_id, user_id, task_id)
        if saved_task != task_id:
            # Race: another worker already created this request
            del _generation_tasks[task_id]
            return JSONResponse({"task_id": saved_task, "status": "processing", "provider": provider_choice, "idempotent": True})

        # Background generation (lock released inside _run_custom_generation finally)
        asyncio.get_event_loop().create_task(
            _run_custom_generation(
                task_id, user_id, prompt, photo_bytes_list,
                provider_choice, has_free, profile, lock, request_id,
            )
        )
        lock = None  # ownership transferred to background task
    finally:
        # Release lock if background task was NOT started
        if lock is not None:
            release_generation_lock(user_id, lock)

    return JSONResponse({
        "task_id": task_id,
        "status": "processing",
        "provider": provider_choice,
    })


async def _run_custom_generation(
    task_id: str,
    user_id: int,
    prompt: str,
    photo_bytes_list: list[bytes],
    provider: str,
    use_free: bool,
    profile,
    lock,
    request_id: str | None,
):
    """Background custom generation with guaranteed lock release."""
    from prismalab.settings import load_settings

    from .services.generation import run_custom_generation
    from .services.locks import release_generation_lock

    settings = load_settings()
    store = get_store()

    try:
        result = await run_custom_generation(
            prompt=prompt,
            photo_bytes_list=photo_bytes_list,
            provider=provider,
            api_key=settings.kie_api_key,
            max_seconds=settings.kie_max_seconds,
        )

        # Deduct credit
        if use_free:
            store.spend_free_generation(user_id)
        else:
            store.set_paid_generations_remaining(user_id, profile.paid_generations_remaining - 1)

        # Log event
        store.log_event(user_id, "generation", {
            "mode": "custom",
            "provider": provider,
            "refs_count": len(photo_bytes_list),
            "source": "miniapp",
        })

        # Upload to Supabase Storage
        image_url = None
        if result.raw_bytes:
            try:
                from prismalab.supabase_storage import upload_generation
                image_url = await asyncio.to_thread(
                    upload_generation, result.raw_bytes, user_id, result.file_ext, result.mime_type,
                )
            except Exception as e:
                logger.warning("Custom upload failed (task=%s): %s", task_id, e)

        # Save to history
        prompt_preview = prompt.strip()[:100]
        try:
            store.save_generation_history(
                user_id=user_id,
                mode="custom",
                style_slug="__custom__",
                style_title=prompt_preview,
                provider=provider,
                image_url=image_url,
                prompt_preview=prompt_preview,
                refs_count=len(photo_bytes_list),
                request_id=request_id,
            )
        except Exception as e:
            logger.warning("Save custom history failed (task=%s): %s", task_id, e)

        # TG send_document (best-effort)
        if result.raw_bytes and _bot:
            try:
                import io
                doc = io.BytesIO(result.raw_bytes)
                doc.name = f"custom_{task_id}.{result.file_ext}"
                await _bot.send_document(
                    chat_id=user_id, document=doc,
                    caption=f"✨ {prompt_preview}",
                    read_timeout=60, write_timeout=60, connect_timeout=30,
                )
            except Exception as e:
                logger.warning("TG send custom failed (task=%s): %s", task_id, e)

        _generation_tasks[task_id] = {
            "status": "done",
            "user_id": user_id,
            "style_id": "__custom__",
            "result_url": result.data_url,
            "image_url": image_url,
            "error": None,
        }
        logger.info("Custom generation done: user=%s provider=%s photos=%d task=%s",
                     user_id, provider, len(photo_bytes_list), task_id)

    except Exception as e:
        logger.exception("Custom generation error: user=%s task=%s: %s", user_id, task_id, e)
        _generation_tasks[task_id] = {
            "status": "error",
            "user_id": user_id,
            "style_id": "__custom__",
            "result_url": None,
            "error": str(e),
        }
    finally:
        release_generation_lock(user_id, lock)


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
    # V2 endpoints
    Route("/app/api/v2/express-themes", api_v2_express_themes, methods=["GET"]),
    Route("/app/api/v2/express-styles", api_v2_express_styles, methods=["GET"]),
    Route("/app/api/v2/express-generate", api_v2_express_generate, methods=["POST"]),
    Route("/app/api/v2/photosets", api_v2_photosets, methods=["GET"]),
    Route("/app/api/v2/photosets/{kind}/{id:int}/generate", api_v2_photoset_generate, methods=["POST"]),
    Route("/app/api/v2/packs/{pack_id:int}/buy-credits", api_v2_pack_buy_credits, methods=["POST"]),
    # V3 endpoints
    Route("/app/api/v3/express/catalog", api_v3_express_catalog, methods=["GET"]),
    Route("/app/api/v3/express/generate", api_v3_express_generate, methods=["POST"]),
    Route("/app/api/v3/history", api_v3_history, methods=["GET"]),
    Route("/app/api/v3/custom/capabilities", api_v3_custom_capabilities, methods=["GET"]),
    Route("/app/api/v3/custom/generate", api_v3_custom_generate, methods=["POST"]),
    Mount("/app/static", StaticFiles(directory=str(BASE_DIR / "static")), name="miniapp_static"),
]


def create_miniapp(store=None):
    """Создаёт Starlette-приложение Mini App."""
    if store:
        set_store(store)
    app = Starlette(routes=routes)
    return app
