"""Тесты V2 стили фотосетов: Astria batch flow (через бот), deprecated KIE endpoint."""
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
    db_file = str(tmp_path / "test_photoset.db")
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


def _setup_style_and_credits(store, *, credits=10, credit_cost=4, is_active=True):
    """Создаёт persona_style + даёт кредиты пользователю."""
    style_id = store.create_persona_style(
        slug="test_photoset_style",
        title="Test Photoset",
        prompt="IDENTICAL FACE test prompt",
        gender="female",
        credit_cost=credit_cost,
    )
    if not is_active:
        store.update_persona_style(style_id, is_active=0)
    store.get_user(12345)
    store.set_persona_credits(12345, credits)
    return style_id


# ── Unified DTO: style содержит slug ──────────────────────────────────


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_photosets_style_has_slug(mock_auth, client, store):
    """Style в /app/api/v2/photosets содержит slug."""
    store.create_persona_style(
        slug="slug_test", title="Slug Test", gender="female", credit_cost=4,
    )
    resp = client.get("/app/api/v2/photosets", headers=_auth_headers())
    assert resp.status_code == 200
    photosets = resp.json()["photosets"]

    style_item = next((p for p in photosets if p["type"] == "style"), None)
    assert style_item is not None
    assert "slug" in style_item
    assert style_item["slug"] == "slug_test"


# ── 410 Gone: style generate deprecated ─────────────────────────────


def test_v2_photoset_style_generate_410(client):
    """POST /app/api/v2/photosets/style/{id}/generate → 410 Gone."""
    resp = client.post("/app/api/v2/photosets/style/1/generate")
    assert resp.status_code == 410
    data = resp.json()
    assert data["error"] == "deprecated"
    assert "persona/generate" in data["message"]


def test_v2_photoset_pack_generate_501(client):
    """POST /app/api/v2/photosets/pack/{id}/generate → 501."""
    resp = client.post("/app/api/v2/photosets/pack/1/generate")
    assert resp.status_code == 501


# ── /app/api/persona/generate: batch flow ────────────────────────────


@patch("prismalab.miniapp.routes.get_bot_username", return_value="test_bot")
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_persona_generate_v2_batch(mock_auth, mock_bot, client, store):
    """/app/api/persona/generate: batch с одним стилем → ok + bot_link."""
    store.create_persona_style(
        slug="batch_style", title="Batch", prompt="test", gender="female", credit_cost=4,
    )
    store.get_user(12345)
    store.set_persona_credits(12345, 10)
    # Даём юзеру tune_id чтобы has_persona = True
    with store._connect() as conn:
        store._execute(conn,
            f"UPDATE {store._users_table} SET astria_lora_tune_id = %s WHERE user_id = %s",
            (999, 12345),
        )

    resp = client.post("/app/api/persona/generate", json={
        "init_data": "fake",
        "styles": [{"slug": "batch_style"}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["count"] == 1
    assert "bot_link" in data
    assert "test_bot" in data["bot_link"]
    assert "persona_batch" in data["bot_link"]


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_persona_generate_credits_cutoff(mock_auth, client, store):
    """Batch cutoff: если кредитов меньше credit_cost → стиль обрезается."""
    store.create_persona_style(
        slug="expensive", title="Expensive", prompt="test", gender="female", credit_cost=8,
    )
    store.get_user(12345)
    store.set_persona_credits(12345, 5)
    with store._connect() as conn:
        store._execute(conn,
            f"UPDATE {store._users_table} SET astria_lora_tune_id = %s WHERE user_id = %s",
            (999, 12345),
        )

    resp = client.post("/app/api/persona/generate", json={
        "init_data": "fake",
        "styles": [{"slug": "expensive"}],
    })
    # credit_cost=8 > credits=5 → batch пустой → 400
    assert resp.status_code == 400
    assert "No valid styles" in resp.json()["error"]


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_persona_generate_no_credits(mock_auth, client, store):
    """0 кредитов → 402."""
    store.get_user(12345)
    store.set_persona_credits(12345, 0)
    with store._connect() as conn:
        store._execute(conn,
            f"UPDATE {store._users_table} SET astria_lora_tune_id = %s WHERE user_id = %s",
            (999, 12345),
        )

    resp = client.post("/app/api/persona/generate", json={
        "init_data": "fake",
        "styles": [{"slug": "any"}],
    })
    assert resp.status_code == 402


# ── has_persona gate: оба tune ID ────────────────────────────────────


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_persona_generate_no_persona(mock_auth, client, store):
    """Нет персоны (ни одного tune_id) → 400."""
    store.get_user(12345)
    store.set_persona_credits(12345, 10)

    resp = client.post("/app/api/persona/generate", json={
        "init_data": "fake",
        "styles": [{"slug": "any"}],
    })
    assert resp.status_code == 400
    assert resp.json()["error"] == "No persona"


@patch("prismalab.miniapp.routes.get_bot_username", return_value="test_bot")
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_persona_generate_pack_tune_id_accepted(mock_auth, mock_bot, client, store):
    """has_persona учитывает astria_lora_pack_tune_id (не только astria_lora_tune_id)."""
    store.create_persona_style(
        slug="pack_tune_test", title="Test", prompt="test", gender="female", credit_cost=4,
    )
    store.get_user(12345)
    store.set_persona_credits(12345, 10)
    # Только pack tune_id, без основного
    with store._connect() as conn:
        store._execute(conn,
            f"UPDATE {store._users_table} SET astria_lora_pack_tune_id = %s WHERE user_id = %s",
            (888, 12345),
        )

    resp = client.post("/app/api/persona/generate", json={
        "init_data": "fake",
        "styles": [{"slug": "pack_tune_test"}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True


# ── P1: валидация styles (injection / unknown slug) ──────────────────


def _setup_user_with_persona(store, user_id=12345, credits=10):
    """Создаёт юзера с tune_id и кредитами."""
    store.get_user(user_id)
    store.set_persona_credits(user_id, credits)
    with store._connect() as conn:
        store._execute(conn,
            f"UPDATE {store._users_table} SET astria_lora_tune_id = %s WHERE user_id = %s",
            (999, user_id),
        )


@patch("prismalab.miniapp.routes.get_bot_username", return_value="test_bot")
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_persona_generate_unknown_slug_400(mock_auth, mock_bot, client, store):
    """Неизвестный slug → пустой batch → 400."""
    _setup_user_with_persona(store)

    resp = client.post("/app/api/persona/generate", json={
        "init_data": "fake",
        "styles": [{"slug": "nonexistent_style"}],
    })
    assert resp.status_code == 400
    assert "No valid styles" in resp.json()["error"]


@patch("prismalab.miniapp.routes.get_bot_username", return_value="test_bot")
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_persona_generate_ignores_client_prompt(mock_auth, mock_bot, client, store):
    """Клиентский prompt игнорируется — batch содержит prompt из каталога."""
    store.create_persona_style(
        slug="safe_style", title="Safe", prompt="CATALOG PROMPT", gender="female", credit_cost=4,
    )
    _setup_user_with_persona(store)

    resp = client.post("/app/api/persona/generate", json={
        "init_data": "fake",
        "styles": [{"slug": "safe_style", "prompt": "INJECTED CUSTOM PROMPT"}],
    })
    assert resp.status_code == 200
    assert resp.json()["count"] == 1

    # Проверяем что в pending_persona_batch сохранился каталожный prompt
    batch_json = store.get_pending_persona_batch(12345)
    assert batch_json is not None
    batch = json.loads(batch_json)
    assert batch[0]["prompt"] == "CATALOG PROMPT"
    assert "INJECTED" not in json.dumps(batch)


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_persona_generate_invalid_styles_format(mock_auth, client, store):
    """styles: ['bad'] (не dict) → 400."""
    _setup_user_with_persona(store)

    resp = client.post("/app/api/persona/generate", json={
        "init_data": "fake",
        "styles": ["bad_string"],
    })
    assert resp.status_code == 400
    assert resp.json()["error"] == "Invalid styles format"


@patch("prismalab.miniapp.routes.get_bot_username", return_value="test_bot")
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_persona_generate_inactive_style_400(mock_auth, mock_bot, client, store):
    """Inactive стиль пропускается (active_only=True) → пустой batch → 400."""
    sid = store.create_persona_style(
        slug="inactive_style", title="Inactive", prompt="test", gender="female", credit_cost=4,
    )
    store.update_persona_style(sid, is_active=0)
    _setup_user_with_persona(store)

    resp = client.post("/app/api/persona/generate", json={
        "init_data": "fake",
        "styles": [{"slug": "inactive_style"}],
    })
    assert resp.status_code == 400
    assert "No valid styles" in resp.json()["error"]


# ── Storage: reserve + refund (всё ещё актуальны) ────────────────────


def test_reserve_credits_success(store):
    """reserve_persona_credits: достаточно кредитов → True."""
    store.get_user(99)
    store.set_persona_credits(99, 10)
    assert store.reserve_persona_credits(99, 4) is True
    profile = store.get_user(99)
    assert profile.persona_credits_remaining == 6


def test_reserve_credits_insufficient(store):
    """reserve_persona_credits: недостаточно → False, баланс не меняется."""
    store.get_user(99)
    store.set_persona_credits(99, 2)
    assert store.reserve_persona_credits(99, 4) is False
    profile = store.get_user(99)
    assert profile.persona_credits_remaining == 2


def test_refund_credits(store):
    """refund_persona_credits: возвращает кредиты."""
    store.get_user(99)
    store.set_persona_credits(99, 6)
    store.refund_persona_credits(99, 3)
    profile = store.get_user(99)
    assert profile.persona_credits_remaining == 9
