"""Тесты V2 генерации фотосета-стиля (4 фото + биллинг)."""
from __future__ import annotations

import os
import sys
from io import BytesIO
from unittest.mock import AsyncMock, patch

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


@pytest.fixture(autouse=True)
def _cleanup_photoset_state():
    """Очистка in-memory state между тестами."""
    yield
    from prismalab.miniapp.services.photosets import _photoset_requests, _user_locks
    _photoset_requests.clear()
    _user_locks.clear()


@pytest.fixture
def app(store):
    from prismalab.miniapp.routes import create_miniapp, set_store
    set_store(store)
    return create_miniapp(store=store)


@pytest.fixture
def client(app):
    from starlette.testclient import TestClient
    return TestClient(app)


def _fake_photo():
    """Минимальный JPEG для тестов."""
    return ("test.jpg", BytesIO(b"\xff\xd8" + b"\x00" * 100), "image/jpeg")


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
    store.get_user(12345)  # auto-create
    store.set_persona_credits(12345, credits)
    return style_id


# ── Вспомогательная mock-генерация ────────────────────────────────────

def _make_gen_mock(*, success_count=4, total=4):
    """Создаёт мок run_generation: первые success_count успешны, остальные фейл."""
    call_count = 0

    async def mock_run_generation(
        photo_bytes=b"",
        style_slug="",
        prompt="",
        negative_prompt="",
        provider="seedream",
        model_params_json="",
        api_key="",
        max_seconds=300,
        **kwargs,
    ):
        nonlocal call_count
        call_count += 1
        if call_count <= success_count:
            from prismalab.miniapp.services.generation import GenerationResult
            return GenerationResult(
                data_url=f"data:image/jpeg;base64,fake_{call_count}",
                provider=provider,
                style_slug=style_slug,
            )
        else:
            raise RuntimeError(f"Generation {call_count} failed")

    return mock_run_generation


# ── 4/4 Success ───────────────────────────────────────────────────────

@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_photoset_4_of_4_success(mock_auth, client, store, monkeypatch):
    """4/4 → status=done, 4 images, credits_spent=4, refund=0."""
    import prismalab.miniapp.services.photosets as ps_mod
    monkeypatch.setattr(ps_mod, "run_generation", _make_gen_mock(success_count=4))

    style_id = _setup_style_and_credits(store, credits=10, credit_cost=4)

    resp = client.post(
        f"/app/api/v2/photosets/style/{style_id}/generate",
        data={"init_data": "fake", "request_id": "req-4-4"},
        files={"photo": _fake_photo()},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "done"
    assert data["requested_count"] == 4
    assert data["success_count"] == 4
    assert len(data["images"]) == 4
    assert data["credits_spent"] == 4
    assert data["credits_refunded"] == 0
    assert data["credits_balance"] == 6  # 10 - 4
    assert data["request_id"] == "req-4-4"


# ── 2/4 Partial ──────────────────────────────────────────────────────

@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_photoset_2_of_4_partial(mock_auth, client, store, monkeypatch):
    """2/4 → status=partial, 2 images, refund пропорциональный."""
    import prismalab.miniapp.services.photosets as ps_mod
    monkeypatch.setattr(ps_mod, "run_generation", _make_gen_mock(success_count=2))

    style_id = _setup_style_and_credits(store, credits=10, credit_cost=4)

    resp = client.post(
        f"/app/api/v2/photosets/style/{style_id}/generate",
        data={"init_data": "fake", "request_id": "req-2-4"},
        files={"photo": _fake_photo()},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "partial"
    assert data["success_count"] == 2
    assert len(data["images"]) == 2
    # credit_cost=4, success=2/4 → spent = (4*2)//4 = 2, refund = 4-2 = 2
    assert data["credits_spent"] == 2
    assert data["credits_refunded"] == 2
    assert data["credits_balance"] == 8  # 10 - 4 + 2 = 8


# ── 0/4 Error ─────────────────────────────────────────────────────────

@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_photoset_0_of_4_error(mock_auth, client, store, monkeypatch):
    """0/4 → status=error, full refund."""
    import prismalab.miniapp.services.photosets as ps_mod
    monkeypatch.setattr(ps_mod, "run_generation", _make_gen_mock(success_count=0))

    style_id = _setup_style_and_credits(store, credits=10, credit_cost=4)

    resp = client.post(
        f"/app/api/v2/photosets/style/{style_id}/generate",
        data={"init_data": "fake", "request_id": "req-0-4"},
        files={"photo": _fake_photo()},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert data["success_count"] == 0
    assert len(data["images"]) == 0
    assert data["credits_spent"] == 0
    assert data["credits_refunded"] == 4
    assert data["credits_balance"] == 10  # полный возврат


# ── No credits → 402 ─────────────────────────────────────────────────

@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_photoset_no_credits(mock_auth, client, store):
    """Нет кредитов → 402."""
    style_id = _setup_style_and_credits(store, credits=2, credit_cost=4)

    resp = client.post(
        f"/app/api/v2/photosets/style/{style_id}/generate",
        data={"init_data": "fake", "request_id": "req-no-credits"},
        files={"photo": _fake_photo()},
    )
    assert resp.status_code == 402
    data = resp.json()
    assert data["error"] == "no_credits"
    assert data["credits_balance"] == 2
    assert data["credits_required"] == 4


# ── Inactive style → 404 ─────────────────────────────────────────────

@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_photoset_inactive_style(mock_auth, client, store):
    """Inactive стиль → 404."""
    style_id = _setup_style_and_credits(store, is_active=False)

    resp = client.post(
        f"/app/api/v2/photosets/style/{style_id}/generate",
        data={"init_data": "fake", "request_id": "req-inactive"},
        files={"photo": _fake_photo()},
    )
    assert resp.status_code == 404


# ── Idempotency: duplicate request_id ─────────────────────────────────

@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_photoset_idempotent_request(mock_auth, client, store, monkeypatch):
    """Повтор request_id → тот же результат без повторной генерации."""
    import prismalab.miniapp.services.photosets as ps_mod
    monkeypatch.setattr(ps_mod, "run_generation", _make_gen_mock(success_count=4))

    style_id = _setup_style_and_credits(store, credits=10, credit_cost=4)

    # Первый запрос
    resp1 = client.post(
        f"/app/api/v2/photosets/style/{style_id}/generate",
        data={"init_data": "fake", "request_id": "req-idem"},
        files={"photo": _fake_photo()},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["credits_balance"] == 6

    # Повтор — тот же request_id
    resp2 = client.post(
        f"/app/api/v2/photosets/style/{style_id}/generate",
        data={"init_data": "fake", "request_id": "req-idem"},
        files={"photo": _fake_photo()},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()

    # Тот же результат, баланс не изменился
    assert data2["request_id"] == "req-idem"
    assert data2["credits_spent"] == data1["credits_spent"]
    assert data2["success_count"] == data1["success_count"]

    # Кредиты списались ОДИН раз
    profile = store.get_user(12345)
    assert profile.persona_credits_remaining == 6


# ── Cross-user isolation ──────────────────────────────────────────────

FAKE_USER2 = {"user_id": 99999, "first_name": "Other", "username": "other"}


def test_photoset_cross_user_isolation(client, store, monkeypatch):
    """Два юзера с одинаковым request_id → каждый получает свой результат."""
    import prismalab.miniapp.services.photosets as ps_mod
    monkeypatch.setattr(ps_mod, "run_generation", _make_gen_mock(success_count=4))

    # Setup user1
    style_id = store.create_persona_style(
        slug="cross_test", title="Cross", prompt="test", gender="female", credit_cost=4,
    )
    store.get_user(12345)
    store.set_persona_credits(12345, 10)
    store.get_user(99999)
    store.set_persona_credits(99999, 10)

    # User1 генерирует
    with patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER):
        resp1 = client.post(
            f"/app/api/v2/photosets/style/{style_id}/generate",
            data={"init_data": "fake", "request_id": "same-req-id"},
            files={"photo": _fake_photo()},
        )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["credits_balance"] == 6  # user1: 10 - 4

    # Reset mock counter for user2
    monkeypatch.setattr(ps_mod, "run_generation", _make_gen_mock(success_count=4))

    # User2 с тем же request_id — должен получить СВОЙ результат, не кэш user1
    with patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER2):
        resp2 = client.post(
            f"/app/api/v2/photosets/style/{style_id}/generate",
            data={"init_data": "fake", "request_id": "same-req-id"},
            files={"photo": _fake_photo()},
        )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["credits_balance"] == 6  # user2: 10 - 4 (свой баланс)

    # Оба списали кредиты независимо
    assert store.get_user(12345).persona_credits_remaining == 6
    assert store.get_user(99999).persona_credits_remaining == 6


# ── Concurrent tap → 409 ─────────────────────────────────────────────

@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_photoset_concurrent_tap_409(mock_auth, client, store):
    """Если генерация уже идёт → 409."""
    import asyncio
    style_id = _setup_style_and_credits(store, credits=10, credit_cost=4)

    # Имитируем locked asyncio.Lock
    from prismalab.miniapp.services.photosets import get_user_lock
    lock = get_user_lock(12345)
    # Acquire lock без await (синхронно через internal _locked flag)
    lock._locked = True

    try:
        resp = client.post(
            f"/app/api/v2/photosets/style/{style_id}/generate",
            data={"init_data": "fake", "request_id": "req-concurrent"},
            files={"photo": _fake_photo()},
        )
        assert resp.status_code == 409
        assert "already in progress" in resp.json()["error"]
    finally:
        lock._locked = False


# ── Unauthorized → 401 ───────────────────────────────────────────────

@patch("prismalab.miniapp.routes.validate_init_data", return_value=None)
def test_photoset_unauthorized(mock_auth, client):
    resp = client.post(
        "/app/api/v2/photosets/style/1/generate",
        data={"init_data": "fake", "request_id": "req-unauth"},
        files={"photo": _fake_photo()},
    )
    assert resp.status_code == 401


# ── kind=pack → 501 ──────────────────────────────────────────────────

@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_photoset_pack_not_implemented(mock_auth, client, store):
    resp = client.post(
        "/app/api/v2/photosets/pack/123/generate",
        data={"init_data": "fake", "request_id": "req-pack"},
        files={"photo": _fake_photo()},
    )
    assert resp.status_code == 501


# ── Missing request_id → 400 ─────────────────────────────────────────

@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_photoset_missing_request_id(mock_auth, client, store):
    _setup_style_and_credits(store)

    resp = client.post(
        "/app/api/v2/photosets/style/1/generate",
        data={"init_data": "fake"},
        files={"photo": _fake_photo()},
    )
    assert resp.status_code == 400
    assert "request_id" in resp.json()["error"]


# ── Storage: reserve + refund ─────────────────────────────────────────

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
