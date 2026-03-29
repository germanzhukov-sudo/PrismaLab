"""Тесты V2 API endpoints Mini App (routes.py)."""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

# Убираем DATABASE_URL чтобы storage не подключился к проду
os.environ.pop("DATABASE_URL", None)
os.environ["TABLE_PREFIX"] = ""

for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("prismalab"):
        del sys.modules[mod_name]


@pytest.fixture
def store(tmp_path):
    """SQLite store во временном файле."""
    os.environ.pop("DATABASE_URL", None)
    if "prismalab.storage" in sys.modules:
        del sys.modules["prismalab.storage"]
    from prismalab.storage import PrismaLabStore
    db_file = str(tmp_path / "test_v2.db")
    s = PrismaLabStore(db_path=db_file)
    s.init_admin_tables()
    return s


FAKE_USER = {"user_id": 12345, "first_name": "Test", "username": "test"}


def _make_app(store):
    """Создаёт Starlette app с замоканным store и auth."""
    from prismalab.miniapp.routes import create_miniapp, set_store
    set_store(store)
    return create_miniapp(store=store)


@pytest.fixture
def app(store):
    return _make_app(store)


@pytest.fixture
def client(app):
    from starlette.testclient import TestClient
    return TestClient(app)


def _auth_headers():
    """Заголовок X-Telegram-Init-Data (будет замокан)."""
    return {"X-Telegram-Init-Data": "fake_init_data"}


# ── GET /app/api/v2/express-themes ────────────────────────────────────


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_express_themes_fallback(mock_auth, client, store):
    """Пустая БД → fallback темы."""
    resp = client.get("/app/api/v2/express-themes", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert "themes" in data
    assert len(data["themes"]) > 0


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_express_themes_from_db(mock_auth, client, store):
    """Темы из БД."""
    store.create_express_style(slug="t1", title="T1", gender="female", theme="glamour")
    store.create_express_style(slug="t2", title="T2", gender="female", theme="mood")

    resp = client.get("/app/api/v2/express-themes?gender=female", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert sorted(data["themes"]) == ["glamour", "mood"]


@patch("prismalab.miniapp.routes.validate_init_data", return_value=None)
def test_v2_express_themes_unauthorized(mock_auth, client):
    """Без auth → 401."""
    resp = client.get("/app/api/v2/express-themes")
    assert resp.status_code == 401


# ── GET /app/api/v2/express-styles ────────────────────────────────────


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_express_styles_fallback(mock_auth, client, store):
    """Пустая БД → fallback стили."""
    resp = client.get("/app/api/v2/express-styles?gender=female", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["styles"]) > 0
    # Backward compat: id/label
    style = data["styles"][0]
    assert "id" in style
    assert "label" in style


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_express_styles_from_db(mock_auth, client, store):
    """Стили из БД."""
    store.create_express_style(
        slug="db_style", title="DB Style", emoji="🔥",
        gender="male", theme="test", provider="seedream",
    )
    resp = client.get("/app/api/v2/express-styles?gender=male", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["styles"]) == 1
    assert data["styles"][0]["id"] == "db_style"


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_express_styles_filter_theme(mock_auth, client, store):
    """Фильтр по теме."""
    store.create_express_style(slug="s1", title="S1", gender="female", theme="glamour")
    store.create_express_style(slug="s2", title="S2", gender="female", theme="mood")

    resp = client.get("/app/api/v2/express-styles?gender=female&theme=glamour", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["styles"]) == 1
    assert data["styles"][0]["id"] == "s1"


@patch("prismalab.miniapp.routes.validate_init_data", return_value=None)
def test_v2_express_styles_unauthorized(mock_auth, client):
    resp = client.get("/app/api/v2/express-styles")
    assert resp.status_code == 401


# ── POST /app/api/v2/express-generate ─────────────────────────────────


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_express_generate_no_style(mock_auth, client, store):
    """Без style_id → 400."""
    from io import BytesIO
    resp = client.post(
        "/app/api/v2/express-generate",
        data={"init_data": "fake", "style_id": ""},
        files={"photo": ("test.jpg", BytesIO(b"\xff\xd8" + b"\x00" * 100), "image/jpeg")},
    )
    assert resp.status_code == 400


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_express_generate_no_photo(mock_auth, client, store):
    """Без фото → 400."""
    resp = client.post(
        "/app/api/v2/express-generate",
        data={"init_data": "fake", "style_id": "wedding"},
    )
    assert resp.status_code == 400


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_express_generate_style_not_found(mock_auth, client, store):
    """Несуществующий стиль → 404."""
    from io import BytesIO
    resp = client.post(
        "/app/api/v2/express-generate",
        data={"init_data": "fake", "style_id": "nonexistent_xyz"},
        files={"photo": ("test.jpg", BytesIO(b"\xff\xd8" + b"\x00" * 100), "image/jpeg")},
    )
    assert resp.status_code == 404


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_express_generate_inactive_style(mock_auth, client, store):
    """Inactive стиль → 404."""
    from io import BytesIO
    sid = store.create_express_style(slug="off", title="Off", gender="female", prompt="test")
    store.update_express_style(sid, is_active=0)

    resp = client.post(
        "/app/api/v2/express-generate",
        data={"init_data": "fake", "style_id": "off"},
        files={"photo": ("test.jpg", BytesIO(b"\xff\xd8" + b"\x00" * 100), "image/jpeg")},
    )
    assert resp.status_code == 404


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_express_generate_no_credits(mock_auth, client, store):
    """Нет кредитов → 402."""
    from io import BytesIO
    store.create_express_style(slug="test_style", title="Test", gender="female", prompt="test prompt")
    # Потратить free generation
    profile = store.get_user(12345)
    store.spend_free_generation(12345)

    resp = client.post(
        "/app/api/v2/express-generate",
        data={"init_data": "fake", "style_id": "test_style"},
        files={"photo": ("test.jpg", BytesIO(b"\xff\xd8" + b"\x00" * 100), "image/jpeg")},
    )
    assert resp.status_code == 402


@patch("prismalab.miniapp.routes._run_generation")
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_express_generate_success(mock_auth, mock_gen, client, store):
    """Успешный запуск → task_id + provider."""
    from io import BytesIO
    # mock _run_generation чтобы не запускать KIE
    async def noop(*args, **kwargs):
        pass
    mock_gen.side_effect = noop

    store.create_express_style(
        slug="gen_test", title="Gen Test", gender="female",
        prompt="test prompt", provider="nano-banana-pro",
    )

    resp = client.post(
        "/app/api/v2/express-generate",
        data={"init_data": "fake", "style_id": "gen_test"},
        files={"photo": ("test.jpg", BytesIO(b"\xff\xd8" + b"\x00" * 100), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert data["status"] == "processing"
    assert data["provider"] == "nano-banana-pro"


@patch("prismalab.miniapp.routes.validate_init_data", return_value=None)
def test_v2_express_generate_unauthorized(mock_auth, client):
    resp = client.post("/app/api/v2/express-generate", data={"init_data": "fake"})
    assert resp.status_code == 401


# ── GET /app/api/v2/photosets ─────────────────────────────────────────


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_photosets_list(mock_auth, client, store):
    """Список паков (из DEFAULT_PACKS без Astria API key)."""
    resp = client.get("/app/api/v2/photosets", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert "packs" in data
    # DEFAULT_PACKS имеет >= 2 пака
    assert len(data["packs"]) >= 2


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_photosets_filter_category(mock_auth, client, store):
    """Фильтр по category."""
    resp = client.get("/app/api/v2/photosets?category=female", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    for pack in data["packs"]:
        assert pack["category"] == "female"


@patch("prismalab.miniapp.routes.validate_init_data", return_value=None)
def test_v2_photosets_unauthorized(mock_auth, client):
    resp = client.get("/app/api/v2/photosets")
    assert resp.status_code == 401


# ── Old endpoints still work ──────────────────────────────────────────


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_old_styles_endpoint_still_works(mock_auth, client, store):
    """Старый /app/api/styles работает."""
    resp = client.get("/app/api/styles?gender=female")
    assert resp.status_code == 200
    data = resp.json()
    assert "styles" in data


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_old_packs_endpoint_still_works(mock_auth, client, store):
    """Старый /app/api/packs работает."""
    resp = client.get("/app/api/packs", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert "packs" in data


# ── Phase 7: Packs via Credits ──────────────────────────────────────


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_photosets_includes_credits_flag(mock_auth, client, store):
    """V2 photosets endpoint отдаёт packs_use_credits flag."""
    resp = client.get("/app/api/v2/photosets", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert "packs_use_credits" in data
    assert isinstance(data["packs_use_credits"], bool)


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_photosets_packs_have_credit_cost(mock_auth, client, store):
    """Паки в ответе содержат credit_cost."""
    resp = client.get("/app/api/v2/photosets", headers=_auth_headers())
    data = resp.json()
    for pack in data.get("packs", []):
        assert "credit_cost" in pack


@patch("prismalab.config.packs_use_credits", return_value=False)
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_pack_buy_credits_disabled(mock_auth, mock_flag, client, store):
    """Если PACKS_USE_CREDITS=0 → 403."""
    store.get_user(12345)
    store.set_persona_credits(12345, 100)
    resp = client.post(
        "/app/api/v2/packs/4345/buy-credits",
        json={"init_data": "fake"},
    )
    assert resp.status_code == 403


@patch("prismalab.config.packs_use_credits", return_value=True)
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_pack_buy_credits_success(mock_auth, mock_flag, client, store):
    """Покупка пака за кредиты: списание + ok."""
    store.get_user(12345)
    store.set_persona_credits(12345, 30)
    resp = client.post(
        "/app/api/v2/packs/4345/buy-credits",
        json={"init_data": "fake"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["credits_spent"] == 20  # 4345 has credit_cost=20
    assert data["credits_balance"] == 10  # 30 - 20

    profile = store.get_user(12345)
    assert profile.persona_credits_remaining == 10


@patch("prismalab.config.packs_use_credits", return_value=True)
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_pack_buy_credits_insufficient(mock_auth, mock_flag, client, store):
    """Покупка пака за кредиты: недостаточно → 402."""
    store.get_user(12345)
    store.set_persona_credits(12345, 5)
    resp = client.post(
        "/app/api/v2/packs/4345/buy-credits",
        json={"init_data": "fake"},
    )
    assert resp.status_code == 402
    data = resp.json()
    assert data["error"] == "no_credits"
    assert data["credits_balance"] == 5
    assert data["credits_required"] == 20

    # Кредиты не списались
    profile = store.get_user(12345)
    assert profile.persona_credits_remaining == 5


@patch("prismalab.config.packs_use_credits", return_value=True)
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_pack_buy_credits_not_found(mock_auth, mock_flag, client, store):
    """Покупка несуществующего пака → 404."""
    store.get_user(12345)
    store.set_persona_credits(12345, 100)
    resp = client.post(
        "/app/api/v2/packs/99999/buy-credits",
        json={"init_data": "fake"},
    )
    assert resp.status_code == 404
