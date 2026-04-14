"""FastAPI приложение для админки PrismaLab."""
from __future__ import annotations

import logging
import os
import re
import time as _time

logger = logging.getLogger("prismalab.admin")
from datetime import datetime, timedelta, timezone
from pathlib import Path

_TRANSLIT = {"а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","ё":"yo","ж":"zh","з":"z","и":"i","й":"y","к":"k","л":"l","м":"m","н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u","ф":"f","х":"kh","ц":"ts","ч":"ch","ш":"sh","щ":"sch","ъ":"","ы":"y","ь":"","э":"e","ю":"yu","я":"ya"}


def _slugify(title: str, fallback_prefix: str = "item") -> str:
    """Транслитерация кириллицы + slug. 'Вечерний гламур' → 'vecherniy_glamur'."""
    slug = "".join(_TRANSLIT.get(c, c) for c in title.lower())
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug or f"{fallback_prefix}_{int(_time.time())}"

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from .constants import ADMIN_BASE

# Получаем префикс таблиц из storage
TABLE_PREFIX = os.getenv("TABLE_PREFIX", "")
from .auth import (
    create_session,
    delete_session,
    get_current_admin,
    require_auth,
    verify_password,
)

# Пути к шаблонам и статике
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.globals["admin_base"] = ADMIN_BASE


def _format_event_data(value):
    """Форматирует event_data для отображения (JSONB или JSON-строка)."""
    import json
    if value is None:
        return "—"
    if isinstance(value, dict):
        # PostgreSQL JSONB — уже dict
        parts = []
        if value.get("style"):
            parts.append(f"стиль: {value['style']}")
        if value.get("mode"):
            parts.append(f"режим: {value['mode']}")
        if value.get("provider"):
            parts.append(f"провайдер: {value['provider']}")
        if value.get("type"):
            parts.append(f"тип: {value['type']}")
        if value.get("delta"):
            parts.append(f"изменение: {value['delta']:+d}")
        if value.get("admin"):
            parts.append(f"админ: {value['admin']}")
        return ", ".join(parts) if parts else str(value)
    if isinstance(value, str):
        # SQLite — JSON-строка
        try:
            data = json.loads(value)
            return _format_event_data(data)
        except Exception:
            return value
    return str(value)


def _format_datetime(value):
    """Форматирует datetime для отображения."""
    if value is None:
        return "—"
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M")
    # Строка — обрезаем до 16 символов
    return str(value)[:16] if value else "—"


# Регистрируем фильтры в Jinja
templates.env.filters["format_event"] = _format_event_data
templates.env.filters["fmt_dt"] = _format_datetime

# Глобальная ссылка на store (устанавливается при создании приложения)
_store = None


def get_store():
    """Получить хранилище."""
    global _store
    if _store is None:
        from prismalab.storage import PrismaLabStore
        _store = PrismaLabStore()
        _store.init_admin_tables()
    return _store


def set_store(store):
    """Установить хранилище (для интеграции с ботом)."""
    global _store
    _store = store
    _store.init_admin_tables()


# Глобальная ссылка на Telegram Bot (устанавливается при создании приложения)
_bot = None


def get_bot():
    """Получить Telegram Bot."""
    global _bot
    if _bot is None:
        # Фоллбэк: создать Bot из токена
        try:
            from telegram import Bot
            token = os.getenv("PRISMALAB_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
            if token:
                _bot = Bot(token=token)
        except Exception:
            pass
    return _bot


def set_bot(bot):
    """Установить Telegram Bot (для интеграции с ботом)."""
    global _bot
    _bot = bot


# ========== Страницы ==========

async def login_page(request: Request):
    """Страница входа."""
    admin = get_current_admin(request)
    if admin:
        return RedirectResponse(url=f"{ADMIN_BASE}/", status_code=302)

    error = request.query_params.get("error")
    return templates.TemplateResponse("login.html", {"request": request, "error": error})


async def login_post(request: Request):
    """Обработка входа."""
    form = await request.form()
    username = form.get("username", "").strip()
    password = form.get("password", "")

    store = get_store()
    admin = store.get_admin_by_username(username)

    if not admin or not verify_password(password, admin["password_hash"]):
        return RedirectResponse(url=f"{ADMIN_BASE}/login?error=1", status_code=302)

    token = create_session(admin["id"], admin["username"], admin.get("display_name"))
    response = RedirectResponse(url=f"{ADMIN_BASE}/", status_code=302)
    response.set_cookie("admin_session", token, httponly=True, max_age=86400 * 7)
    return response


async def logout(request: Request):
    """Выход."""
    token = request.cookies.get("admin_session")
    if token:
        delete_session(token)
    response = RedirectResponse(url=f"{ADMIN_BASE}/login", status_code=302)
    response.delete_cookie("admin_session")
    return response


@require_auth
async def dashboard(request: Request):
    """Главная страница — дашборд."""
    period = request.query_params.get("period", "week")

    # Определяем даты (МСК = UTC+3)
    _MSK = timezone(timedelta(hours=3))
    today = datetime.now(_MSK).date()
    if period == "today":
        date_from = today.isoformat()
        date_to = today.isoformat()
    elif period == "yesterday":
        yesterday = today - timedelta(days=1)
        date_from = yesterday.isoformat()
        date_to = yesterday.isoformat()
    elif period == "week":
        date_from = (today - timedelta(days=7)).isoformat()
        date_to = today.isoformat()
    elif period == "month":
        date_from = (today - timedelta(days=30)).isoformat()
        date_to = today.isoformat()
    else:
        date_from = request.query_params.get("date_from", (today - timedelta(days=7)).isoformat())
        date_to = request.query_params.get("date_to", today.isoformat())

    store = get_store()

    # Кэшируем настройки один раз — чтобы all_time и dashboard_stats не дёргали БД повторно
    _cost_settings = store.get_cost_settings()
    _pack_costs_map = store.get_pack_costs_map()

    all_time = store.get_all_time_stats(_cost_settings=_cost_settings, _pack_costs_map=_pack_costs_map)
    stats = store.get_dashboard_stats(date_from, date_to, _cost_settings=_cost_settings, _pack_costs_map=_pack_costs_map)
    chart_data = store.get_chart_data(days=30)
    hourly_data = store.get_hourly_activity(date_from, date_to)
    pack_stats = store.get_pack_stats(date_from, date_to)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "admin": request.state.admin,
        "all_time": all_time,
        "stats": stats,
        "chart_data": chart_data,
        "hourly_data": hourly_data,
        "pack_stats": pack_stats,
        "period": period,
        "date_from": date_from,
        "date_to": date_to,
    })


@require_auth
async def users_list(request: Request):
    """Список пользователей."""
    page = int(request.query_params.get("page", 1))
    search = request.query_params.get("search", "").strip()
    limit = 50
    offset = (page - 1) * limit

    store = get_store()
    users = store.get_users_paginated(limit, offset, search if search else None)
    total = store.get_users_count(search if search else None)
    total_pages = (total + limit - 1) // limit

    return templates.TemplateResponse("users.html", {
        "request": request,
        "admin": request.state.admin,
        "users": users,
        "page": page,
        "total_pages": total_pages,
        "total": total,
        "search": search,
    })


@require_auth
async def user_detail(request: Request):
    """Детали пользователя."""
    user_id = int(request.path_params["user_id"])

    store = get_store()
    profile = store.get_user(user_id)
    history = store.get_user_history(user_id)

    return templates.TemplateResponse("user_detail.html", {
        "request": request,
        "admin": request.state.admin,
        "profile": profile,
        "history": history,
    })


@require_auth
async def user_credits_post(request: Request):
    """Начисление/списание кредитов."""
    user_id = int(request.path_params["user_id"])
    form = await request.form()
    credit_type = form.get("credit_type", "fast")
    delta = int(form.get("delta", 0))

    store = get_store()
    store.adjust_user_credits(user_id, credit_type, delta)

    # Логируем событие
    store.log_event(user_id, "admin_credits", {
        "type": credit_type,
        "delta": delta,
        "admin": request.state.admin["username"],
    })

    return RedirectResponse(url=f"{ADMIN_BASE}/users/{user_id}", status_code=302)


@require_auth
async def payments_list(request: Request):
    """Список платежей."""
    page = int(request.query_params.get("page", 1))
    product_type = request.query_params.get("product_type", "")
    date_from = request.query_params.get("date_from", "")
    date_to = request.query_params.get("date_to", "")
    limit = 50
    offset = (page - 1) * limit

    store = get_store()
    payments = store.get_payments_paginated(
        limit, offset,
        product_type=product_type if product_type else None,
        date_from=date_from if date_from else None,
        date_to=date_to if date_to else None,
    )
    total = store.get_payments_count(
        product_type=product_type if product_type else None,
        date_from=date_from if date_from else None,
        date_to=date_to if date_to else None,
    )
    total_pages = (total + limit - 1) // limit

    return templates.TemplateResponse("payments.html", {
        "request": request,
        "admin": request.state.admin,
        "payments": payments,
        "page": page,
        "total_pages": total_pages,
        "total": total,
        "product_type": product_type,
        "date_from": date_from,
        "date_to": date_to,
    })


def _do_csv_export(export_type: str, date_from: str, date_to: str) -> str:
    """Синхронная функция экспорта CSV."""
    import csv
    import io
    import logging
    import os

    import psycopg2
    from psycopg2.extras import RealDictCursor
    logger = logging.getLogger("prismalab.admin")
    logger.info("CSV export started: type=%s", export_type)

    output = io.StringIO()
    writer = csv.writer(output)

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        writer.writerow(["Error: DATABASE_URL not set"])
        return output.getvalue()

    if "sslmode=" not in db_url:
        db_url += ("&" if "?" in db_url else "?") + "sslmode=require"

    logger.info("Connecting to DB...")
    conn = psycopg2.connect(db_url, connect_timeout=10)
    logger.info("Connected!")
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            payments_table = f"public.{TABLE_PREFIX}payments"
            users_table = f"public.{TABLE_PREFIX}users"
            if export_type == "payments":
                if date_from and date_to:
                    cur.execute(
                        f"SELECT * FROM {payments_table} WHERE created_at >= %s AND created_at <= %s ORDER BY created_at DESC",
                        (date_from, date_to + " 23:59:59")
                    )
                else:
                    # Без дат — последние 10000 записей
                    cur.execute(f"SELECT * FROM {payments_table} ORDER BY created_at DESC LIMIT 10000")
                rows = cur.fetchall()
                writer.writerow(["ID", "User ID", "Payment ID", "Method", "Type", "Credits", "Amount RUB", "Created At"])
                for p in rows:
                    writer.writerow([p.get("id"), p.get("user_id"), p.get("payment_id"), p.get("payment_method"), p.get("product_type"), p.get("credits"), p.get("amount_rub"), p.get("created_at")])
            elif export_type == "users":
                # Все пользователи (до 50000)
                cur.execute(f"SELECT * FROM {users_table} ORDER BY updated_at DESC LIMIT 50000")
                rows = cur.fetchall()
                writer.writerow(["User ID", "Fast Credits", "Persona Credits", "Gender", "Updated At"])
                for u in rows:
                    writer.writerow([u.get("user_id"), u.get("paid_generations_remaining"), u.get("persona_credits_remaining"), u.get("subject_gender"), u.get("updated_at")])
    finally:
        conn.close()
        logger.info("CSV export done")

    return output.getvalue()


@require_auth
async def export_csv(request: Request):
    """Экспорт данных в CSV."""
    from starlette.responses import Response

    export_type = request.query_params.get("type", "payments")
    date_from = request.query_params.get("date_from", "")
    date_to = request.query_params.get("date_to", "")

    # Синхронно — для простоты
    csv_data = _do_csv_export(export_type, date_from, date_to)

    filename = f"prismalab_{export_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ========== API ==========

@require_auth
async def api_stats(request: Request):
    """API: статистика."""
    period = request.query_params.get("period", "week")
    _MSK = timezone(timedelta(hours=3))
    today = datetime.now(_MSK).date()

    if period == "today":
        date_from = today.isoformat()
        date_to = today.isoformat()
    elif period == "week":
        date_from = (today - timedelta(days=7)).isoformat()
        date_to = today.isoformat()
    elif period == "month":
        date_from = (today - timedelta(days=30)).isoformat()
        date_to = today.isoformat()
    else:
        date_from = request.query_params.get("date_from")
        date_to = request.query_params.get("date_to")

    store = get_store()
    stats = store.get_dashboard_stats(date_from, date_to)
    return JSONResponse(stats)


@require_auth
async def api_chart_data(request: Request):
    """API: данные для графиков."""
    days = int(request.query_params.get("days", 30))
    store = get_store()
    data = store.get_chart_data(days)
    return JSONResponse(data)


@require_auth
async def settings_page(request: Request):
    """Страница настроек себестоимости."""
    store = get_store()
    settings = store.get_cost_settings()
    saved = request.query_params.get("saved") == "1"

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "admin": request.state.admin,
        "settings": settings,
        "saved": saved,
    })


@require_auth
async def settings_post(request: Request):
    """Сохранение настроек."""
    store = get_store()
    form = await request.form()

    settings = {
        "cost_persona_create": float(form.get("cost_persona_create", 1.5)),
        "cost_fast_photo": float(form.get("cost_fast_photo", 0.035)),
        "cost_nano_banana": float(form.get("cost_nano_banana", 0.035)),
        "cost_persona_photo": float(form.get("cost_persona_photo", 0.03)),
        "usd_rub": float(form.get("usd_rub", 90.0)),
    }
    store.set_cost_settings(settings)

    return RedirectResponse(url=f"{ADMIN_BASE}/settings?saved=1", status_code=303)


# ===== Pricing (sell prices) =====


@require_auth
async def pricing_page(request: Request):
    """Страница управления ценами продажи."""
    store = get_store()
    saved = request.query_params.get("saved") == "1"

    from prismalab.tariffs import get_all_tariffs, get_pack_price_overrides

    tariffs = get_all_tariffs(store)

    # Pack offers with override info
    offers = _load_pack_offers()
    pack_overrides = get_pack_price_overrides(store)

    packs = []
    for offer in offers:
        pid = int(offer.get("id", 0))
        override_val = pack_overrides.get(pid)
        packs.append({
            "id": pid,
            "title": offer.get("title", f"Pack {pid}"),
            "class_name": offer.get("class_name", ""),
            "expected_images": offer.get("expected_images", 0),
            "default_price": float(offer.get("price_rub", 0)),
            "override_price": override_val,
        })

    discount_badge = store.get_admin_setting("photosets_discount_badge") or ""

    return templates.TemplateResponse("pricing.html", {
        "request": request,
        "admin": request.state.admin,
        "tariffs": tariffs,
        "packs": packs,
        "saved": saved,
        "discount_badge": discount_badge,
    })


@require_auth
async def pricing_post(request: Request):
    """Сохранение цен продажи."""
    store = get_store()
    form = await request.form()

    from prismalab.tariffs import (
        get_all_product_types,
        get_default_credits,
        set_tariff_prices,
        set_pack_sell_price,
        reset_pack_sell_price,
    )

    # Credit-based tariffs
    for product_type in get_all_product_types():
        prices = {}
        for credits in get_default_credits(product_type):
            field_name = f"{product_type}_{credits}"
            val = form.get(field_name)
            if val:
                try:
                    prices[credits] = int(float(val))
                except (ValueError, TypeError):
                    pass
        if prices:
            set_tariff_prices(store, product_type, prices)

    # Discount badge
    badge_val = (form.get("photosets_discount_badge") or "").strip()
    if badge_val:
        store.set_admin_setting("photosets_discount_badge", badge_val)
    else:
        store.delete_admin_setting("photosets_discount_badge")

    # Pack sell prices — пустое поле = сброс override к дефолту
    offers = _load_pack_offers()
    for offer in offers:
        pid = int(offer.get("id", 0))
        val = form.get(f"pack_price_{pid}")
        if val and val.strip():
            try:
                set_pack_sell_price(store, pid, float(val))
            except (ValueError, TypeError):
                pass
        else:
            # Пустое/отсутствующее поле — сбрасываем override, вернётся к дефолту
            reset_pack_sell_price(store, pid)

    return RedirectResponse(url=f"{ADMIN_BASE}/pricing?saved=1", status_code=303)


def _load_pack_offers() -> list[dict]:
    """Загрузка офферов паков — единый источник из pack_offers.py."""
    from prismalab.pack_offers import _pack_offers
    return _pack_offers()


@require_auth
async def photosets_unified_page(request: Request):
    """Unified страница Фотосеты: табы Паки + Стили."""
    store = get_store()
    tab = request.query_params.get("tab", "packs")
    category_filter = request.query_params.get("category", "")
    saved = request.query_params.get("saved") == "1"

    cost_settings = store.get_cost_settings()
    usd_rub = cost_settings["usd_rub"]

    if tab == "styles":
        styles = store.get_persona_styles(active_only=False, gender=category_filter or None)
        style_stats = store.get_persona_style_stats()
        for s in styles:
            sid = int(s.get("id", 0))
            st = style_stats.get(sid, {})
            s["generations"] = st.get("generations", 0)
            s["cost_usd"] = float(s.get("cost_usd", 0) or 0)
        return templates.TemplateResponse("photosets.html", {
            "request": request,
            "admin": request.state.admin,
            "tab": "styles",
            "styles": styles,
            "category_filter": category_filter,
            "usd_rub": usd_rub,
            "saved": saved,
            "admin_base": ADMIN_BASE,
        })
    else:
        offers = _load_pack_offers()
        saved_costs = {r["pack_id"]: r for r in store.get_pack_costs()}
        pack_stats_all = store.get_pack_stats()
        stats_map = {int(s["pack_id"]): s for s in pack_stats_all if s["pack_id"] and str(s["pack_id"]).isdigit()}

        packs = []
        for offer in offers:
            pid = int(offer.get("id", 0))
            cat = offer.get("category", "")
            if category_filter and cat != category_filter:
                continue
            cost_usd = float(saved_costs.get(pid, {}).get("cost_usd", 0))
            saved_cc = saved_costs.get(pid, {}).get("credit_cost")
            credit_cost = int(saved_cc) if saved_cc is not None else int(offer.get("credit_cost", offer.get("expected_images", 0)))
            price_rub = float(offer.get("price_rub", 0))
            generations = int(stats_map.get(pid, {}).get("generations", 0))
            total_images = int(stats_map.get(pid, {}).get("total_images", 0))
            cost_rub = cost_usd * usd_rub
            margin_rub = price_rub - cost_rub if cost_usd > 0 else 0
            margin_pct = round((margin_rub / price_rub) * 100, 1) if price_rub > 0 and cost_usd > 0 else 0
            packs.append({
                "id": pid, "title": offer.get("title", f"Pack {pid}"),
                "price_rub": price_rub, "class_name": offer.get("class_name", ""),
                "expected_images": offer.get("expected_images", 0),
                "credit_cost": credit_cost,
                "cost_usd": cost_usd, "cost_rub": round(cost_rub, 2),
                "margin_rub": round(margin_rub, 2), "margin_pct": margin_pct,
                "generations": generations, "total_images": total_images,
            })
        return templates.TemplateResponse("photosets.html", {
            "request": request,
            "admin": request.state.admin,
            "tab": "packs",
            "packs": packs,
            "category_filter": category_filter,
            "usd_rub": usd_rub,
            "saved": saved,
            "admin_base": ADMIN_BASE,
        })


@require_auth
async def photosets_unified_post(request: Request):
    """Сохранение себестоимости паков или стилей."""
    store = get_store()
    form = await request.form()
    save_type = form.get("save_type", "packs")

    if save_type == "styles":
        items = []
        for key, val in form.multi_items():
            if key.startswith("style_cost_") and val.strip():
                style_id = int(key.replace("style_cost_", ""))
                items.append({"style_id": style_id, "cost_usd": float(val)})
        store.set_persona_style_costs_bulk(items)
        return RedirectResponse(url=f"{ADMIN_BASE}/photosets?tab=styles&saved=1", status_code=303)
    else:
        offers = _load_pack_offers()
        items = []
        for offer in offers:
            pid = int(offer.get("id", 0))
            cost_usd = float(form.get(f"cost_{pid}", 0))
            cc_raw = form.get(f"credit_cost_{pid}", "")
            cc_val = int(cc_raw) if cc_raw.strip() else None
            items.append({"pack_id": pid, "pack_title": offer.get("title", ""), "cost_usd": cost_usd, "credit_cost": cc_val})
        store.set_pack_costs_bulk(items)
        return RedirectResponse(url=f"{ADMIN_BASE}/photosets?tab=packs&saved=1", status_code=303)


@require_auth
async def pack_costs_redirect(request: Request):
    """Legacy redirect /admin/pack-costs → /admin/photosets?tab=packs."""
    return RedirectResponse(url=f"{ADMIN_BASE}/photosets?tab=packs", status_code=303)


@require_auth
async def persona_styles_list_redirect(request: Request):
    """Legacy redirect /admin/persona-styles → /admin/photosets?tab=styles."""
    return RedirectResponse(url=f"{ADMIN_BASE}/photosets?tab=styles", status_code=303)






# ========== Рассылка ==========

@require_auth
async def broadcast_page(request: Request):
    """Страница рассылки."""
    store = get_store()
    counts = store.get_broadcast_counts()

    sent = request.query_params.get("sent")
    failed = request.query_params.get("failed")
    total = request.query_params.get("total")

    return templates.TemplateResponse("broadcast.html", {
        "request": request,
        "admin_base": ADMIN_BASE,
        "admin": request.state.admin,
        "counts": counts,
        "miniapp_url": os.getenv("MINIAPP_URL", ""),
        "sent": int(sent) if sent else None,
        "failed": int(failed) if failed else None,
        "total": int(total) if total else None,
    })


@require_auth
async def broadcast_post(request: Request):
    """Отправка рассылки."""
    import asyncio
    import io

    token = (os.getenv("PRISMALAB_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")).strip()
    if not token:
        return PlainTextResponse("Bot not available. Check PRISMALAB_BOT_TOKEN.", status_code=500)

    store = get_store()
    form = await request.form()

    # Параметры
    text = str(form.get("text", "")).strip()
    filter_type = str(form.get("filter_type", "all"))
    specific_ids_raw = str(form.get("specific_ids", "")).strip()
    photo_file = form.get("photo")  # UploadFile or None

    # Кнопка
    add_button = form.get("add_button") == "1"
    btn_text = str(form.get("btn_text", "")).strip()
    btn_type = str(form.get("btn_type", "webapp"))
    btn_value = str(form.get("btn_value", "")).strip()

    logger.info("Broadcast form: add_button=%r, btn_text=%r, btn_type=%r, btn_value=%r",
                add_button, btn_text, btn_type, btn_value)

    if not text:
        return RedirectResponse(url=f"{ADMIN_BASE}/broadcast?error=no_text", status_code=303)

    # Получить список user_id
    user_ids_list = None
    if filter_type == "specific":
        # Парсим ID: разделители — запятая, пробел, новая строка
        import re
        raw_ids = re.split(r"[\s,]+", specific_ids_raw)
        user_ids_list = []
        for rid in raw_ids:
            rid = rid.strip()
            if rid.isdigit():
                user_ids_list.append(int(rid))
        if not user_ids_list:
            return RedirectResponse(url=f"{ADMIN_BASE}/broadcast?error=no_ids", status_code=303)

    target_ids = store.get_user_ids_for_broadcast(filter_type, user_ids_list)
    if not target_ids:
        return RedirectResponse(url=f"{ADMIN_BASE}/broadcast?error=no_users", status_code=303)

    # Подготовить фото
    photo_bytes = None
    if photo_file and hasattr(photo_file, "read"):
        content = await photo_file.read()
        if content and len(content) > 0:
            photo_bytes = io.BytesIO(content)

    # Подготовить клавиатуру
    keyboard = None
    if add_button and btn_text and btn_value:
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
            if btn_type == "webapp":
                keyboard = InlineKeyboardMarkup([[InlineKeyboardButton(btn_text, web_app=WebAppInfo(url=btn_value))]])
            elif btn_type == "url":
                keyboard = InlineKeyboardMarkup([[InlineKeyboardButton(btn_text, url=btn_value)]])
            else:
                keyboard = InlineKeyboardMarkup([[InlineKeyboardButton(btn_text, callback_data=btn_value)]])
            logger.info("Broadcast keyboard created: type=%s, text=%r", btn_type, btn_text)
        except Exception as e:
            logger.warning("Broadcast keyboard error: %s", e)
    else:
        logger.info("Broadcast: no keyboard (add_button=%r, btn_text=%r, btn_value=%r)", add_button, btn_text, btn_value)

    # Отправка
    sent = 0
    failed = 0

    # Важно: создаём отдельный Bot с ограниченным connection pool.
    # Админка работает в отдельном треде/event loop, поэтому нельзя
    # использовать бота из основного loop — это ломает httpx-соединения.
    # Задержка 0.15с между сообщениями снижает нагрузку на Telegram API
    # и предотвращает "bound to a different event loop" у основного бота.
    from telegram import Bot
    from telegram.request import HTTPXRequest

    broadcast_request = HTTPXRequest(
        connection_pool_size=2,
        connect_timeout=10.0,
        read_timeout=15.0,
        write_timeout=15.0,
    )
    async with Bot(token=token, request=broadcast_request) as bot:
        for uid in target_ids:
            try:
                if photo_bytes:
                    photo_bytes.seek(0)
                    await bot.send_photo(
                        chat_id=uid,
                        photo=photo_bytes,
                        caption=text,
                        reply_markup=keyboard,
                        parse_mode="HTML",
                    )
                else:
                    await bot.send_message(
                        chat_id=uid,
                        text=text,
                        reply_markup=keyboard,
                        parse_mode="HTML",
                    )
                sent += 1
            except Exception as e:
                failed += 1
                logger.warning("Broadcast to %s failed: %s", uid, e)
            await asyncio.sleep(0.15)

    logger.info("Broadcast done: sent=%d, failed=%d, total=%d, filter=%s", sent, failed, len(target_ids), filter_type)

    return RedirectResponse(
        url=f"{ADMIN_BASE}/broadcast?sent={sent}&failed={failed}&total={len(target_ids)}",
        status_code=303,
    )


# ========== Роуты ==========
# Полные пути /admin/... — aiohttp передаёт path как есть
async def _debug_path(request: Request):
    """Отладка: какой path видит Starlette (удалить после починки)."""
    path = request.scope.get("path", "?")
    return PlainTextResponse(f"path={path!r}\nroot_path={request.scope.get('root_path', '')!r}")


@require_auth
async def analytics_page(request: Request):
    """Страница аналитики: воронки конверсии и популярность."""
    period = request.query_params.get("period", "week")
    _MSK = timezone(timedelta(hours=3))
    today = datetime.now(_MSK).date()
    if period == "today":
        date_from = today.isoformat()
        date_to = today.isoformat()
    elif period == "yesterday":
        yesterday = today - timedelta(days=1)
        date_from = yesterday.isoformat()
        date_to = yesterday.isoformat()
    elif period == "week":
        date_from = (today - timedelta(days=7)).isoformat()
        date_to = today.isoformat()
    elif period == "month":
        date_from = (today - timedelta(days=30)).isoformat()
        date_to = today.isoformat()
    else:
        date_from = request.query_params.get("date_from", (today - timedelta(days=7)).isoformat())
        date_to = request.query_params.get("date_to", today.isoformat())

    store = get_store()
    funnels = store.get_funnel_data(date_from, date_to)
    popularity = store.get_popularity_data(date_from, date_to)

    return templates.TemplateResponse("analytics.html", {
        "request": request,
        "admin": request.state.admin,
        "funnels": funnels,
        "popularity": popularity,
        "period": period,
        "date_from": date_from,
        "date_to": date_to,
    })


# ========== Стили Персоны ==========

@require_auth
async def persona_styles_page(request: Request):
    """Список стилей персоны."""
    store = get_store()
    styles = store.get_persona_styles()
    saved = request.query_params.get("saved") == "1"
    deleted = request.query_params.get("deleted") == "1"

    return templates.TemplateResponse("persona_styles.html", {
        "request": request,
        "admin": request.state.admin,
        "styles": styles,
        "saved": saved,
        "deleted": deleted,
    })


@require_auth
async def persona_style_form(request: Request):
    """Форма создания/редактирования стиля."""
    style_id = request.path_params.get("style_id")
    store = get_store()
    style = None
    previews: list[str] = []
    if style_id:
        style = store.get_persona_style(int(style_id))
        if not style:
            return RedirectResponse(url=f"{ADMIN_BASE}/persona-styles", status_code=302)
        previews = store.get_style_previews(int(style_id))

    return templates.TemplateResponse("persona_style_form.html", {
        "request": request,
        "admin": request.state.admin,
        "style": style,
        "previews": previews,
    })


@require_auth
async def persona_style_save(request: Request):
    """Сохранение стиля (создание или обновление)."""
    store = get_store()
    form = await request.form()

    style_id = form.get("style_id", "").strip()
    title = form.get("title", "").strip()
    description = form.get("description", "").strip()
    prompt = form.get("prompt", "").strip()
    gender = form.get("gender", "female")
    sort_order = int(form.get("sort_order", 0) or 0)
    is_active = form.get("is_active") == "1"

    # Auto-generate slug from title
    import re
    import time as _time
    _translit = {"а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","ё":"yo","ж":"zh","з":"z","и":"i","й":"y","к":"k","л":"l","м":"m","н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u","ф":"f","х":"kh","ц":"ts","ч":"ch","ш":"sh","щ":"sch","ъ":"","ы":"y","ь":"","э":"e","ю":"yu","я":"ya"}
    slug = "".join(_translit.get(c, c) for c in title.lower())
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_") or f"style_{int(_time.time())}"

    credit_cost = int(form.get("credit_cost", 4) or 4)

    # Обработка загрузки превью (до 4 файлов)
    from prismalab.supabase_storage import upload_image, delete_image
    existing_previews_raw = form.get("existing_previews", "").strip()
    existing_previews = [u for u in existing_previews_raw.split("\n") if u.strip()] if existing_previews_raw else []
    # Какие превью удалить (unchecked в форме)
    keep_indices = set()
    for key in form.keys():
        if key.startswith("keep_preview_"):
            try:
                keep_indices.add(int(key.replace("keep_preview_", "")))
            except ValueError:
                pass
    # Если форма edit и были превью — оставляем только отмеченные
    preview_urls: list[str] = []
    removed_urls: list[str] = []
    if style_id and existing_previews:
        for i, url in enumerate(existing_previews):
            if i in keep_indices:
                preview_urls.append(url)
            else:
                removed_urls.append(url)
    # Загрузить новые файлы (до 4 - len(preview_urls))
    image_files = form.getlist("images")
    for image_file in image_files:
        if len(preview_urls) >= 4:
            break
        if image_file and hasattr(image_file, "read"):
            file_bytes = await image_file.read()
            if file_bytes and len(file_bytes) > 0:
                content_type = getattr(image_file, "content_type", "image/jpeg") or "image/jpeg"
                filename = getattr(image_file, "filename", "style.jpg") or "style.jpg"
                new_url = upload_image(file_bytes, filename, content_type)
                if new_url:
                    preview_urls.append(new_url)
    # Удалить убранные файлы из Supabase
    for url in removed_urls:
        delete_image(url)
    # image_url = первое превью (для обратной совместимости)
    image_url = preview_urls[0] if preview_urls else ""

    if style_id:
        # Обновление — сдвинуть остальные если порядок занят
        store._shift_persona_style_sort_order(sort_order, exclude_id=int(style_id))
        store.update_persona_style(
            int(style_id),
            slug=slug, title=title, description=description,
            prompt=prompt, gender=gender, image_url=image_url,
            sort_order=sort_order, is_active=is_active,
            credit_cost=credit_cost,
        )
        store.set_style_previews(int(style_id), preview_urls)
    else:
        # Создание
        if not title:
            return RedirectResponse(url=f"{ADMIN_BASE}/persona-styles/new", status_code=302)
        store._shift_persona_style_sort_order(sort_order)
        new_id = store.create_persona_style(
            slug=slug, title=title, description=description,
            prompt=prompt, gender=gender, image_url=image_url,
            sort_order=sort_order, credit_cost=credit_cost,
        )
        if new_id and preview_urls:
            store.set_style_previews(new_id, preview_urls)

    return RedirectResponse(url=f"{ADMIN_BASE}/persona-styles?saved=1", status_code=303)


@require_auth
async def persona_style_move(request: Request):
    """Переместить стиль вверх/вниз (swap с соседом)."""
    style_id = int(request.path_params["style_id"])
    direction = request.path_params.get("direction", "")  # up или down
    store = get_store()

    styles = store.get_persona_styles()  # уже отсортированы по sort_order, id
    idx = next((i for i, s in enumerate(styles) if s["id"] == style_id), None)
    if idx is not None:
        if direction == "up" and idx > 0:
            store.swap_persona_style_order(styles[idx]["id"], styles[idx - 1]["id"])
        elif direction == "down" and idx < len(styles) - 1:
            store.swap_persona_style_order(styles[idx]["id"], styles[idx + 1]["id"])

    return RedirectResponse(url=f"{ADMIN_BASE}/persona-styles", status_code=303)


@require_auth
async def persona_style_delete(request: Request):
    """Удаление стиля."""
    style_id = int(request.path_params["style_id"])
    store = get_store()

    # Удалить все превью из Supabase Storage + legacy image_url (сирота)
    from prismalab.supabase_storage import delete_image
    preview_urls = set(store.get_style_previews(style_id))
    for url in preview_urls:
        delete_image(url)
    style = store.get_persona_style(style_id)
    if style:
        legacy_url = (style.get("image_url") or "").strip()
        if legacy_url and legacy_url not in preview_urls:
            delete_image(legacy_url)

    store.delete_persona_style(style_id)
    return RedirectResponse(url=f"{ADMIN_BASE}/persona-styles?deleted=1", status_code=303)


# ========== Экспресс-стили ==========

@require_auth
async def express_styles_page(request: Request):
    """Список экспресс-стилей с фильтрами."""
    store = get_store()
    filter_gender = request.query_params.get("gender", "").strip()
    filter_provider = request.query_params.get("provider", "").strip()
    saved = request.query_params.get("saved") == "1"
    deleted = request.query_params.get("deleted") == "1"

    styles = store.get_express_styles(
        gender=filter_gender or None,
        active_only=False,
    )

    if filter_provider:
        styles = [s for s in styles if (s.get("provider") or "seedream") == filter_provider]

    # Batch-загрузка категорий для всех стилей (без N+1)
    style_categories_map = store.get_all_style_categories_map()
    style_tags_map = store.get_all_style_tags_map()

    return templates.TemplateResponse("express_styles.html", {
        "request": request,
        "admin": request.state.admin,
        "active_tab": "styles",
        "styles": styles,
        "style_categories_map": style_categories_map,
        "style_tags_map": style_tags_map,
        "filter_gender": filter_gender,
        "filter_provider": filter_provider,
        "saved": saved,
        "deleted": deleted,
    })


@require_auth
async def express_style_form(request: Request):
    """Форма создания/редактирования экспресс-стиля."""
    style_id = request.path_params.get("style_id")
    store = get_store()
    style = None
    selected_cat_ids: set[int] = set()
    selected_tag_ids: set[int] = set()
    if style_id:
        style = store.get_express_style(int(style_id))
        if not style:
            return RedirectResponse(url=f"{ADMIN_BASE}/express-styles", status_code=302)
        selected_cat_ids = {c["id"] for c in store.get_style_categories(int(style_id))}
        selected_tag_ids = {t["id"] for t in store.get_style_tags(int(style_id))}

    all_categories = store.get_express_categories(active_only=False)
    all_tags = store.get_express_tags(active_only=False)
    # Маппинг category_id → [tag_id, ...] для JS-фильтрации
    import json
    cat_tags_map = {}
    for cat in all_categories:
        cat_tags_map[cat["id"]] = [t["id"] for t in store.get_category_tags(cat["id"])]

    return templates.TemplateResponse("express_style_form.html", {
        "request": request,
        "admin": request.state.admin,
        "style": style,
        "all_categories": all_categories,
        "all_tags": all_tags,
        "selected_cat_ids": selected_cat_ids,
        "selected_tag_ids": selected_tag_ids,
        "cat_tags_map_json": json.dumps(cat_tags_map),
    })


@require_auth
async def express_style_save(request: Request):
    """Сохранение экспресс-стиля (создание или обновление)."""
    store = get_store()
    form = await request.form()

    style_id = form.get("style_id", "").strip()
    title = form.get("title", "").strip()
    emoji = form.get("emoji", "").strip()
    prompt = form.get("prompt", "").strip()
    negative_prompt = form.get("negative_prompt", "").strip()
    gender = form.get("gender", "female")
    theme = "general"  # compat-only field: theme replaced by categories in V3
    provider = form.get("provider", "seedream")
    model_params = form.get("model_params", "").strip()
    sort_order = int(form.get("sort_order", 0) or 0)
    is_active = form.get("is_active") == "1"
    category_ids = [int(x) for x in form.getlist("category_ids")]
    tag_ids = [int(x) for x in form.getlist("tag_ids")]

    slug = _slugify(title, "express")

    # Обработка загрузки картинки
    image_url = form.get("existing_image_url", "").strip()
    image_file = form.get("image")
    if image_file and hasattr(image_file, "read"):
        file_bytes = await image_file.read()
        if file_bytes and len(file_bytes) > 0:
            from prismalab.supabase_storage import upload_image
            content_type = getattr(image_file, "content_type", "image/jpeg") or "image/jpeg"
            filename = getattr(image_file, "filename", "style.jpg") or "style.jpg"
            new_url = upload_image(file_bytes, filename, content_type)
            if new_url:
                if image_url:
                    from prismalab.supabase_storage import delete_image
                    delete_image(image_url)
                image_url = new_url

    if style_id:
        existing = store.get_express_style(int(style_id))
        old_sort = existing.get("sort_order", 0) if existing else 0
        old_slug = existing.get("slug", "") if existing else ""
        if sort_order != old_sort:
            store.move_express_style_to_order(int(style_id), sort_order)
        # При edit сохраняем старый slug если title не менялся (стабильный id в API)
        if old_slug and existing and existing.get("title") == title:
            slug = old_slug
        else:
            # Title изменился → проверяем уникальность нового slug
            base_slug = slug
            counter = 1
            conflict = store.get_express_style_by_slug(slug)
            while conflict and conflict.get("id") != int(style_id):
                slug = f"{base_slug}_{counter}"
                counter += 1
                conflict = store.get_express_style_by_slug(slug)
        store.update_express_style(
            int(style_id),
            slug=slug, title=title, emoji=emoji,
            prompt=prompt, negative_prompt=negative_prompt,
            gender=gender, theme=theme, provider=provider,
            model_params=model_params, image_url=image_url,
            sort_order=sort_order, is_active=is_active,
        )
        store._renumber_express_styles()
        store.set_style_categories(int(style_id), category_ids)
        # Серверная валидация: только теги, разрешённые выбранными категориями
        if category_ids:
            allowed = store.get_allowed_tag_ids_for_categories(category_ids)
            tag_ids = [t for t in tag_ids if t in allowed]
        store.set_style_tags(int(style_id), tag_ids)
    else:
        if not title:
            return RedirectResponse(url=f"{ADMIN_BASE}/express-styles/new", status_code=302)
        # Уникальный slug — добавляем суффикс при коллизии
        base_slug = slug
        counter = 1
        while store.get_express_style_by_slug(slug):
            slug = f"{base_slug}_{counter}"
            counter += 1
        store._shift_express_style_sort_order(sort_order)
        new_id = store.create_express_style(
            slug=slug, title=title, emoji=emoji,
            prompt=prompt, negative_prompt=negative_prompt,
            gender=gender, theme=theme, provider=provider,
            model_params=model_params, image_url=image_url,
            sort_order=sort_order,
        )
        store._renumber_express_styles()
        if new_id:
            store.set_style_categories(new_id, category_ids)
            if category_ids:
                allowed = store.get_allowed_tag_ids_for_categories(category_ids)
                tag_ids = [t for t in tag_ids if t in allowed]
            store.set_style_tags(new_id, tag_ids)

    return RedirectResponse(url=f"{ADMIN_BASE}/express-styles?saved=1", status_code=303)


@require_auth
async def express_style_move(request: Request):
    """Переместить экспресс-стиль вверх/вниз."""
    style_id = int(request.path_params["style_id"])
    direction = request.path_params.get("direction", "")
    store = get_store()

    styles = store.get_express_styles()
    idx = next((i for i, s in enumerate(styles) if s["id"] == style_id), None)
    if idx is not None:
        if direction == "up" and idx > 0:
            store.swap_express_style_order(styles[idx]["id"], styles[idx - 1]["id"])
        elif direction == "down" and idx < len(styles) - 1:
            store.swap_express_style_order(styles[idx]["id"], styles[idx + 1]["id"])

    return RedirectResponse(url=f"{ADMIN_BASE}/express-styles", status_code=303)


@require_auth
async def express_style_delete(request: Request):
    """Удаление экспресс-стиля."""
    style_id = int(request.path_params["style_id"])
    store = get_store()

    style = store.get_express_style(style_id)
    if style and style.get("image_url"):
        from prismalab.supabase_storage import delete_image
        delete_image(style["image_url"])

    store.delete_express_style(style_id)
    return RedirectResponse(url=f"{ADMIN_BASE}/express-styles?deleted=1", status_code=303)


# ========== Express Categories CRUD ==========

@require_auth
async def express_categories_page(request: Request):
    """Список категорий."""
    store = get_store()
    saved = request.query_params.get("saved") == "1"
    deleted = request.query_params.get("deleted") == "1"
    categories = store.get_express_categories(active_only=False)
    # Обогащаем: теги и кол-во стилей (batch, без N+1)
    cat_style_counts = store.get_category_style_counts()
    for cat in categories:
        cat["_tags"] = store.get_category_tags(cat["id"])
        cat["_style_count"] = cat_style_counts.get(cat["id"], 0)
    return templates.TemplateResponse("express_categories.html", {
        "request": request, "admin": request.state.admin,
        "active_tab": "categories",
        "categories": categories, "saved": saved, "deleted": deleted,
        "admin_base": ADMIN_BASE,
    })


@require_auth
async def express_category_form(request: Request):
    """Форма создания/редактирования категории."""
    store = get_store()
    category_id = request.path_params.get("category_id")
    category = store.get_express_category(category_id) if category_id else None
    if category_id and not category:
        return RedirectResponse(url=f"{ADMIN_BASE}/express-categories", status_code=303)

    all_tags = store.get_express_tags(active_only=False)
    selected_tag_ids = set()
    if category:
        selected_tag_ids = {t["id"] for t in store.get_category_tags(category["id"])}

    return templates.TemplateResponse("express_category_form.html", {
        "request": request, "admin": request.state.admin,
        "category": category, "all_tags": all_tags,
        "selected_tag_ids": selected_tag_ids,
        "admin_base": ADMIN_BASE,
    })


@require_auth
async def express_category_save(request: Request):
    """Сохранение категории."""
    store = get_store()
    form = await request.form()
    category_id = form.get("category_id")
    title = form.get("title", "").strip()
    slug = form.get("slug", "").strip() or _slugify(title, "cat")
    sort_order = int(form.get("sort_order", 0) or 0)
    is_active = form.get("is_active") == "1"
    tag_ids = [int(x) for x in form.getlist("tag_ids")]

    if category_id:
        cid = int(category_id)
        existing = store.get_express_category(cid)
        # Сохраняем slug если title не менялся
        if existing and existing.get("title") == title:
            slug = existing["slug"]
        else:
            base_slug = slug
            counter = 1
            conflict = store.get_express_category_by_slug(slug)
            while conflict and conflict.get("id") != cid:
                slug = f"{base_slug}_{counter}"
                counter += 1
                conflict = store.get_express_category_by_slug(slug)
        store.update_express_category(cid, slug=slug, title=title,
                                       sort_order=sort_order, is_active=is_active)
        store.set_category_tags(cid, tag_ids)
    else:
        # Уникальный slug
        base_slug = slug
        counter = 1
        while store.get_express_category_by_slug(slug):
            slug = f"{base_slug}_{counter}"
            counter += 1
        new_id = store.create_express_category(slug, title, sort_order, is_active)
        if new_id and tag_ids:
            store.set_category_tags(new_id, tag_ids)

    return RedirectResponse(url=f"{ADMIN_BASE}/express-categories?saved=1", status_code=303)


@require_auth
async def express_category_delete(request: Request):
    """Удаление категории."""
    category_id = int(request.path_params["category_id"])
    store = get_store()
    store.delete_express_category(category_id)
    return RedirectResponse(url=f"{ADMIN_BASE}/express-categories?deleted=1", status_code=303)


# ========== Express Tags CRUD ==========

@require_auth
async def express_tags_page(request: Request):
    """Список тегов."""
    store = get_store()
    saved = request.query_params.get("saved") == "1"
    deleted = request.query_params.get("deleted") == "1"
    tags = store.get_express_tags(active_only=False)
    # Обогащаем: категории и кол-во стилей (batch, без N+1)
    all_categories = store.get_express_categories(active_only=False)
    tag_style_counts = store.get_tag_style_counts()
    # Собираем category→tags маппинг за один проход
    cat_tag_sets: dict[int, set[int]] = {}
    for cat in all_categories:
        cat_tag_sets[cat["id"]] = {t["id"] for t in store.get_category_tags(cat["id"])}
    for tag in tags:
        tag["_categories"] = [c for c in all_categories if tag["id"] in cat_tag_sets.get(c["id"], set())]
        tag["_style_count"] = tag_style_counts.get(tag["id"], 0)
    return templates.TemplateResponse("express_tags.html", {
        "request": request, "admin": request.state.admin,
        "active_tab": "tags",
        "tags": tags, "saved": saved, "deleted": deleted,
        "admin_base": ADMIN_BASE,
    })


@require_auth
async def express_tag_form(request: Request):
    """Форма создания/редактирования тега."""
    store = get_store()
    tag_id = request.path_params.get("tag_id")
    tag = store.get_express_tag(tag_id) if tag_id else None
    if tag_id and not tag:
        return RedirectResponse(url=f"{ADMIN_BASE}/express-tags", status_code=303)

    return templates.TemplateResponse("express_tag_form.html", {
        "request": request, "admin": request.state.admin,
        "tag": tag, "admin_base": ADMIN_BASE,
    })


@require_auth
async def express_tag_save(request: Request):
    """Сохранение тега."""
    store = get_store()
    form = await request.form()
    tag_id = form.get("tag_id")
    title = form.get("title", "").strip()
    slug = form.get("slug", "").strip() or _slugify(title, "tag")
    sort_order = int(form.get("sort_order", 0) or 0)
    is_active = form.get("is_active") == "1"

    if tag_id:
        tid = int(tag_id)
        existing = store.get_express_tag(tid)
        if existing and existing.get("title") == title:
            slug = existing["slug"]
        else:
            base_slug = slug
            counter = 1
            conflict = store.get_express_tag_by_slug(slug)
            while conflict and conflict.get("id") != tid:
                slug = f"{base_slug}_{counter}"
                counter += 1
                conflict = store.get_express_tag_by_slug(slug)
        store.update_express_tag(tid, slug=slug, title=title,
                                  sort_order=sort_order, is_active=is_active)
    else:
        base_slug = slug
        counter = 1
        while store.get_express_tag_by_slug(slug):
            slug = f"{base_slug}_{counter}"
            counter += 1
        store.create_express_tag(slug, title, sort_order, is_active)

    return RedirectResponse(url=f"{ADMIN_BASE}/express-tags?saved=1", status_code=303)


@require_auth
async def express_tag_delete(request: Request):
    """Удаление тега."""
    tag_id = int(request.path_params["tag_id"])
    store = get_store()
    store.delete_express_tag(tag_id)
    return RedirectResponse(url=f"{ADMIN_BASE}/express-tags?deleted=1", status_code=303)


routes = [
    Route("/admin", dashboard, methods=["GET"]),  # без слэша — тот же дашборд
    Route("/admin/", dashboard, methods=["GET"]),
    Route("/admin/debug", _debug_path, methods=["GET"]),
    Route("/admin/login", login_page, methods=["GET"]),
    Route("/admin/login", login_post, methods=["POST"]),
    Route("/admin/logout", logout, methods=["GET", "POST"]),
    Route("/admin/users", users_list, methods=["GET"]),
    Route("/admin/users/{user_id:int}", user_detail, methods=["GET"]),
    Route("/admin/users/{user_id:int}/credits", user_credits_post, methods=["POST"]),
    Route("/admin/payments", payments_list, methods=["GET"]),
    Route("/admin/analytics", analytics_page, methods=["GET"]),
    Route("/admin/export", export_csv, methods=["GET"]),
    Route("/admin/photosets", photosets_unified_page, methods=["GET"]),
    Route("/admin/photosets", photosets_unified_post, methods=["POST"]),
    Route("/admin/pack-costs", pack_costs_redirect, methods=["GET"]),
    Route("/admin/persona-styles", persona_styles_list_redirect, methods=["GET"]),
    Route("/admin/broadcast", broadcast_page, methods=["GET"]),
    Route("/admin/broadcast", broadcast_post, methods=["POST"]),
    Route("/admin/pricing", pricing_page, methods=["GET"]),
    Route("/admin/pricing", pricing_post, methods=["POST"]),
    Route("/admin/settings", settings_page, methods=["GET"]),
    Route("/admin/settings", settings_post, methods=["POST"]),
    Route("/admin/persona-styles/new", persona_style_form, methods=["GET"]),
    Route("/admin/persona-styles/{style_id:int}/edit", persona_style_form, methods=["GET"]),
    Route("/admin/persona-styles/save", persona_style_save, methods=["POST"]),
    Route("/admin/persona-styles/{style_id:int}/move/{direction}", persona_style_move, methods=["POST"]),
    Route("/admin/persona-styles/{style_id:int}/delete", persona_style_delete, methods=["POST"]),
    Route("/admin/express-styles", express_styles_page, methods=["GET"]),
    Route("/admin/express-styles/new", express_style_form, methods=["GET"]),
    Route("/admin/express-styles/{style_id:int}/edit", express_style_form, methods=["GET"]),
    Route("/admin/express-styles/save", express_style_save, methods=["POST"]),
    Route("/admin/express-styles/{style_id:int}/move/{direction}", express_style_move, methods=["POST"]),
    Route("/admin/express-styles/{style_id:int}/delete", express_style_delete, methods=["POST"]),
    Route("/admin/express-categories", express_categories_page, methods=["GET"]),
    Route("/admin/express-categories/new", express_category_form, methods=["GET"]),
    Route("/admin/express-categories/{category_id:int}/edit", express_category_form, methods=["GET"]),
    Route("/admin/express-categories/save", express_category_save, methods=["POST"]),
    Route("/admin/express-categories/{category_id:int}/delete", express_category_delete, methods=["POST"]),
    Route("/admin/express-tags", express_tags_page, methods=["GET"]),
    Route("/admin/express-tags/new", express_tag_form, methods=["GET"]),
    Route("/admin/express-tags/{tag_id:int}/edit", express_tag_form, methods=["GET"]),
    Route("/admin/express-tags/save", express_tag_save, methods=["POST"]),
    Route("/admin/express-tags/{tag_id:int}/delete", express_tag_delete, methods=["POST"]),
    Route("/admin/api/stats", api_stats, methods=["GET"]),
    Route("/admin/api/chart", api_chart_data, methods=["GET"]),
    Mount("/admin/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static"),
]


def create_admin_app(store=None, bot=None):
    """Создаёт приложение админки."""
    if store:
        set_store(store)
    if bot:
        set_bot(bot)

    middleware = [
        Middleware(SessionMiddleware, secret_key=os.getenv("ADMIN_SECRET_KEY", "prismalab-admin-secret-change-me")),
    ]

    app = Starlette(routes=routes, middleware=middleware)
    return app


# Для запуска отдельно: python -m prismalab.admin.app
if __name__ == "__main__":
    import uvicorn
    app = create_admin_app()
    uvicorn.run(app, host="0.0.0.0", port=8081)
