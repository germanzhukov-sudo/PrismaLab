"""Авторизация для админки."""
from __future__ import annotations

import hashlib
import secrets
from functools import wraps
from typing import Callable

from starlette.requests import Request
from starlette.responses import RedirectResponse

from .constants import ADMIN_BASE

# Простые сессии в памяти (для прода лучше Redis)
_sessions: dict[str, dict] = {}


def hash_password(password: str) -> str:
    """Хеширует пароль с солью."""
    salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
    return f"{salt}:{hashed.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    """Проверяет пароль."""
    try:
        salt, hashed = password_hash.split(":")
        new_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return new_hash.hex() == hashed
    except Exception:
        return False


def create_session(admin_id: int, username: str, display_name: str | None) -> str:
    """Создаёт сессию и возвращает токен."""
    token = secrets.token_urlsafe(32)
    _sessions[token] = {
        "admin_id": admin_id,
        "username": username,
        "display_name": display_name or username,
    }
    return token


def get_session(token: str) -> dict | None:
    """Получает данные сессии по токену."""
    return _sessions.get(token)


def delete_session(token: str) -> None:
    """Удаляет сессию."""
    _sessions.pop(token, None)


def get_current_admin(request: Request) -> dict | None:
    """Получает текущего админа из cookie."""
    token = request.cookies.get("admin_session")
    if not token:
        return None
    return get_session(token)


def require_auth(func: Callable):
    """Декоратор для проверки авторизации."""
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        admin = get_current_admin(request)
        if not admin:
            return RedirectResponse(url=f"{ADMIN_BASE}/login", status_code=302)
        request.state.admin = admin
        return await func(request, *args, **kwargs)
    return wrapper
