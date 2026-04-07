"""Тесты payment.py — чистые функции без импорта bot.py."""
from __future__ import annotations

import os

# Гарантируем что бот не импортируется
os.environ.pop("DATABASE_URL", None)
os.environ.setdefault("PRISMALAB_ASTRIA_PACK_CALLBACK_SECRET", "test_secret_12345")


def test_pack_callback_token():
    """HMAC-токен создаётся и проверяется корректно."""
    from prismalab.payment import _make_pack_callback_token, _verify_pack_callback_token
    token = _make_pack_callback_token(123, 456, 789, "run_001")
    assert isinstance(token, str)
    assert len(token) > 0
    assert _verify_pack_callback_token("123", "456", "789", "run_001", token) is True


def test_pack_callback_token_wrong():
    """Неверный токен отклоняется."""
    from prismalab.payment import _verify_pack_callback_token
    assert _verify_pack_callback_token("123", "456", "789", "run_001", "wrong_token") is False


def test_pack_callback_token_tampered():
    """Изменённые параметры отклоняются."""
    from prismalab.payment import _make_pack_callback_token, _verify_pack_callback_token
    token = _make_pack_callback_token(123, 456, 789, "run_001")
    # Подменяем user_id
    assert _verify_pack_callback_token("999", "456", "789", "run_001", token) is False


def test_build_pack_callback_url():
    """callback URL содержит все параметры."""
    os.environ["PRISMALAB_WEBHOOK_BASE_URL"] = "https://example.com"
    from prismalab.payment import build_pack_callback_url
    url = build_pack_callback_url(123, 456, 789, "run_001")
    assert "user_id=123" in url
    assert "chat_id=456" in url
    assert "pack_id=789" in url
    assert "run_id=run_001" in url
    assert "token=" in url


def test_amount_rub():
    """_amount_rub возвращает числовое значение через tariffs service."""
    from prismalab.payment import _amount_rub
    from prismalab.storage import PrismaLabStore
    store = PrismaLabStore()
    amount = _amount_rub(store, "fast", 5)
    assert isinstance(amount, (int, float))
    assert amount > 0


def test_use_yookassa_config():
    """use_yookassa зависит от env переменных."""
    from prismalab.payment import use_yookassa
    result = use_yookassa()
    assert isinstance(result, bool)
