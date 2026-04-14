"""Тесты атомарного контракта кредитов для persona batch generation (Task 5).

Ключевой инвариант: `initial_credits - final_credits == expected_debit` —
проверяем финансовую согласованность при параллельных сценариях.

Проверяем:
- Atomic claim helper не позволяет двум параллельным потокам захватить один батч
- API cleanup preservers legacy batches (возвращает 409)
- Refund на `not generated`
- Нет double-debit при повторном клике generate
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pytest

os.environ.pop("DATABASE_URL", None)
os.environ["TABLE_PREFIX"] = ""

for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("prismalab"):
        del sys.modules[mod_name]


@pytest.fixture
def store(tmp_path):
    os.environ.pop("DATABASE_URL", None)
    if "prismalab.storage" in sys.modules:
        del sys.modules["prismalab.storage"]
    from prismalab.storage import PrismaLabStore
    db_file = str(tmp_path / "test_race.db")
    s = PrismaLabStore(db_path=db_file)
    s.init_admin_tables()
    return s


FAKE_USER = {"user_id": 12345, "first_name": "Test", "username": "test"}


@pytest.fixture
def app(store):
    from prismalab.miniapp.routes import create_miniapp, set_store
    set_store(store)
    return create_miniapp(store=store)


@pytest.fixture
def client(app):
    from starlette.testclient import TestClient
    return TestClient(app)


def _auth_headers():
    return {"X-Telegram-Init-Data": "fake_init_data"}


def _setup_user_with_persona(store, *, credits=20):
    """Создаёт юзера с persona (tune_id) + кредиты."""
    store.get_user(12345)
    store.set_persona_credits(12345, credits)
    with store._connect() as conn:
        store._execute(
            conn,
            f"UPDATE {store._users_table} SET astria_lora_tune_id = %s WHERE user_id = %s",
            ("fake_tune_123", 12345),
        )


def _setup_styles(store, *, count=2, credit_cost=4):
    """Создаёт N persona styles с заданной ценой."""
    slugs = []
    for i in range(count):
        slug = f"test_style_{i}"
        store.create_persona_style(
            slug=slug,
            title=f"Test Style {i}",
            prompt=f"test prompt {i}",
            gender="female",
            credit_cost=credit_cost,
        )
        slugs.append(slug)
    return slugs


# ── 1. claim_and_clear helper: atomicity ─────────────────────────────────


def test_claim_and_clear_returns_old_value(store):
    """claim_and_clear возвращает старое значение и очищает поле."""
    _setup_user_with_persona(store, credits=10)
    store.set_pending_persona_batch(12345, '[{"slug": "foo"}]')

    # Первый claim получает батч
    result1 = store.claim_and_clear_pending_persona_batch(12345)
    assert result1 == '[{"slug": "foo"}]'

    # Второй claim уже получает None (поле очищено)
    result2 = store.claim_and_clear_pending_persona_batch(12345)
    assert result2 is None

    # В БД поле действительно NULL
    assert store.get_pending_persona_batch(12345) is None


def test_claim_and_clear_no_pending_returns_none(store):
    """claim_and_clear без pending возвращает None без ошибок."""
    _setup_user_with_persona(store, credits=10)
    result = store.claim_and_clear_pending_persona_batch(12345)
    assert result is None


# ── 2. API cleanup vs legacy pending batch ───────────────────────────────


@patch("prismalab.miniapp.routes.get_bot_username", return_value="test_bot")
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_api_cleanup_preserves_legacy_batch(mock_auth, mock_bot, client, store):
    """Если pending — LEGACY (без pre_reserved), API возвращает 409 и батч восстановлен."""
    _setup_user_with_persona(store, credits=10)
    _setup_styles(store, count=2)

    # Ставим legacy pending (без pre_reserved)
    legacy_batch = json.dumps([
        {"slug": "old_style", "title": "Old", "credit_cost": 4, "prompt": "old"}
    ])
    store.set_pending_persona_batch(12345, legacy_batch)
    initial_credits = store.get_user(12345).persona_credits_remaining

    # Пытаемся запустить новый generate
    resp = client.post(
        "/app/api/persona/generate",
        json={"init_data": "fake", "styles": [{"slug": "test_style_0"}]},
    )

    # Ожидаем 409 Conflict
    assert resp.status_code == 409, f"Expected 409, got {resp.status_code}: {resp.text}"

    # Баланс не изменился
    final_credits = store.get_user(12345).persona_credits_remaining
    assert final_credits == initial_credits, (
        f"Legacy cleanup changed balance: {initial_credits} -> {final_credits}"
    )

    # Legacy pending восстановлен
    restored = store.get_pending_persona_batch(12345)
    assert restored == legacy_batch, "Legacy batch should be restored unchanged"


# ── 3. API cleanup: pre_reserved retry → refund + new reserve ────────────


@patch("prismalab.miniapp.routes.get_bot_username", return_value="test_bot")
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_idempotent_retry_no_double_debit(mock_auth, mock_bot, client, store):
    """Повторный generate без ухода в бота: первый refunded, второй активен.

    Инвариант: `final_credits == initial - cost_of_last_request` (не сумма обоих).
    """
    _setup_user_with_persona(store, credits=20)
    _setup_styles(store, count=2, credit_cost=4)
    initial = store.get_user(12345).persona_credits_remaining
    assert initial == 20

    # Первый reserve: 1 стиль = 4 кредита
    resp1 = client.post(
        "/app/api/persona/generate",
        json={"init_data": "fake", "styles": [{"slug": "test_style_0"}]},
    )
    assert resp1.status_code == 200, resp1.text
    mid = store.get_user(12345).persona_credits_remaining
    assert mid == 16, f"After first reserve: {initial} - 4 = 16, got {mid}"

    # Второй reserve БЕЗ ухода в бота: 2 стиля = 8 кредитов
    resp2 = client.post(
        "/app/api/persona/generate",
        json={"init_data": "fake", "styles": [
            {"slug": "test_style_0"},
            {"slug": "test_style_1"},
        ]},
    )
    assert resp2.status_code == 200, resp2.text
    final = store.get_user(12345).persona_credits_remaining

    # Инвариант: первый рефанднут, осталось 12 = 20 - 8
    assert final == 12, (
        f"Expected final=12 (20 initial - 8 last request), got {final}. "
        f"Double-debit detected!" if final < 12 else
        f"Refund didn't happen!"
    )

    # В response вернулся правильный credits_balance
    assert resp2.json().get("credits_balance") == 12


# ── 4. API: save_batch fails → rollback ─────────────────────────────────


@patch("prismalab.miniapp.routes.get_bot_username", return_value="test_bot")
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_save_batch_fail_refunds(mock_auth, mock_bot, client, store):
    """Если set_pending_persona_batch упал — кредиты возвращены."""
    _setup_user_with_persona(store, credits=10)
    _setup_styles(store, count=1, credit_cost=4)
    initial = store.get_user(12345).persona_credits_remaining

    # Подменяем set_pending_persona_batch чтобы упал
    orig = store.set_pending_persona_batch

    def failing_save(*a, **kw):
        raise RuntimeError("simulated DB failure")

    store.set_pending_persona_batch = failing_save
    try:
        resp = client.post(
            "/app/api/persona/generate",
            json={"init_data": "fake", "styles": [{"slug": "test_style_0"}]},
        )
    finally:
        store.set_pending_persona_batch = orig

    # Ожидаем 500
    assert resp.status_code == 500, f"Expected 500, got {resp.status_code}: {resp.text}"

    # Баланс не изменился (rollback сработал)
    final = store.get_user(12345).persona_credits_remaining
    assert final == initial, (
        f"Rollback failed: initial={initial}, final={final}. "
        f"Credits should be refunded on save_batch failure!"
    )


# ── 5. API success: pre_reserved flag сохраняется в батче ───────────────


@patch("prismalab.miniapp.routes.get_bot_username", return_value="test_bot")
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_successful_generate_saves_pre_reserved_flag(mock_auth, mock_bot, client, store):
    """После успешного /api/persona/generate в pending батч каждый item имеет pre_reserved=True."""
    _setup_user_with_persona(store, credits=20)
    _setup_styles(store, count=2, credit_cost=4)

    resp = client.post(
        "/app/api/persona/generate",
        json={"init_data": "fake", "styles": [
            {"slug": "test_style_0"},
            {"slug": "test_style_1"},
        ]},
    )
    assert resp.status_code == 200

    pending = store.get_pending_persona_batch(12345)
    assert pending is not None
    items = json.loads(pending)
    assert len(items) == 2
    assert all(item.get("pre_reserved") is True for item in items)

    # credits_balance в ответе корректный
    data = resp.json()
    assert data["credits_balance"] == 12  # 20 - (4 * 2)


# ── 6. Concurrent claim: только один поток получает batch ───────────────


def test_parallel_claim_only_one_wins(store):
    """Два параллельных claim'а: один получает батч, второй — None.

    Эмулирует race между api_persona_generate cleanup и navigation handler.
    """
    import threading

    _setup_user_with_persona(store, credits=10)
    batch_json = json.dumps([{"slug": "race_style", "credit_cost": 4, "pre_reserved": True}])
    store.set_pending_persona_batch(12345, batch_json)

    results: list = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        result = store.claim_and_clear_pending_persona_batch(12345)
        results.append(result)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Ровно один получил, второй — None
    got_batches = [r for r in results if r is not None]
    got_none = [r for r in results if r is None]
    assert len(got_batches) == 1, f"Expected exactly 1 winner, got {len(got_batches)}: {results}"
    assert len(got_none) == 1, f"Expected exactly 1 loser, got {len(got_none)}"
    assert got_batches[0] == batch_json

    # В БД pending очищен
    assert store.get_pending_persona_batch(12345) is None
