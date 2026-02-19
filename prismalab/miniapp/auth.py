"""Валидация Telegram Mini App initData."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
from typing import Any


def validate_init_data(init_data_str: str, bot_token: str, max_age: int = 86400) -> dict | None:
    """
    Проверяет подпись initData от Telegram Mini App.

    Возвращает dict с данными пользователя при успехе, None при ошибке.
    https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app
    """
    if not init_data_str or not bot_token:
        return None

    # Парсим query string
    parsed = urllib.parse.parse_qs(init_data_str, keep_blank_values=True)

    received_hash = parsed.get("hash", [None])[0]
    if not received_hash:
        return None

    # Собираем data_check_string: все пары кроме hash, сортированные, через \n
    pairs = []
    for key, values in parsed.items():
        if key == "hash":
            continue
        pairs.append(f"{key}={values[0]}")

    data_check_string = "\n".join(sorted(pairs))

    # HMAC-SHA256: secret_key = HMAC("WebAppData", bot_token)
    secret_key = hmac.new(
        b"WebAppData",
        bot_token.encode("utf-8"),
        hashlib.sha256,
    ).digest()

    calculated_hash = hmac.new(
        secret_key,
        data_check_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(calculated_hash, received_hash):
        return None

    # Проверяем auth_date (не старше max_age секунд)
    auth_date_str = parsed.get("auth_date", [None])[0]
    if auth_date_str:
        try:
            auth_date = int(auth_date_str)
            if time.time() - auth_date > max_age:
                return None
        except ValueError:
            return None

    # Парсим user
    user_str = parsed.get("user", [None])[0]
    if not user_str:
        return None

    try:
        user = json.loads(user_str)
    except (json.JSONDecodeError, TypeError):
        return None

    return {
        "user_id": user.get("id"),
        "first_name": user.get("first_name", ""),
        "last_name": user.get("last_name", ""),
        "username": user.get("username", ""),
        "language_code": user.get("language_code", "ru"),
        "auth_date": auth_date_str,
    }
