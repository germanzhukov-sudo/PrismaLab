"""FastAPI приложение для админки PrismaLab."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from .constants import ADMIN_BASE
from .auth import (
    create_session,
    delete_session,
    get_current_admin,
    hash_password,
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

    # Определяем даты
    today = datetime.now().date()
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
    stats = store.get_dashboard_stats(date_from, date_to)
    chart_data = store.get_chart_data(days=30)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "admin": request.state.admin,
        "stats": stats,
        "chart_data": chart_data,
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
    import os
    import logging
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
            if export_type == "payments":
                if date_from and date_to:
                    cur.execute(
                        "SELECT * FROM public.payments WHERE created_at >= %s AND created_at <= %s ORDER BY created_at DESC",
                        (date_from, date_to + " 23:59:59")
                    )
                else:
                    # Без дат — последние 10000 записей
                    cur.execute("SELECT * FROM public.payments ORDER BY created_at DESC LIMIT 10000")
                rows = cur.fetchall()
                writer.writerow(["ID", "User ID", "Payment ID", "Method", "Type", "Credits", "Amount RUB", "Created At"])
                for p in rows:
                    writer.writerow([p.get("id"), p.get("user_id"), p.get("payment_id"), p.get("payment_method"), p.get("product_type"), p.get("credits"), p.get("amount_rub"), p.get("created_at")])
            elif export_type == "users":
                # Все пользователи (до 50000)
                cur.execute("SELECT * FROM public.users ORDER BY updated_at DESC LIMIT 50000")
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
    today = datetime.now().date()

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


# ========== Роуты ==========
# Полные пути /admin/... — aiohttp передаёт path как есть
async def _debug_path(request: Request):
    """Отладка: какой path видит Starlette (удалить после починки)."""
    path = request.scope.get("path", "?")
    return PlainTextResponse(f"path={path!r}\nroot_path={request.scope.get('root_path', '')!r}")


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
    Route("/admin/export", export_csv, methods=["GET"]),
    Route("/admin/api/stats", api_stats, methods=["GET"]),
    Route("/admin/api/chart", api_chart_data, methods=["GET"]),
    Mount("/admin/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static"),
]


def create_admin_app(store=None):
    """Создаёт приложение админки."""
    if store:
        set_store(store)

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
