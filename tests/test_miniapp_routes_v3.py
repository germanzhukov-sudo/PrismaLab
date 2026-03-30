"""Тесты V3 API endpoints: /app/api/v3/express/catalog и /app/api/v3/express/generate."""
from __future__ import annotations

import os
import sys
from io import BytesIO
from unittest.mock import patch

import pytest

os.environ.pop("DATABASE_URL", None)
os.environ["TABLE_PREFIX"] = ""

for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("prismalab"):
        del sys.modules[mod_name]


FAKE_USER = {"user_id": 12345, "first_name": "Test", "username": "test"}
AUTH_HEADER = {"X-Telegram-Init-Data": "fake"}


@pytest.fixture
def store(tmp_path):
    os.environ.pop("DATABASE_URL", None)
    if "prismalab.storage" in sys.modules:
        del sys.modules["prismalab.storage"]
    from prismalab.storage import PrismaLabStore
    db_file = str(tmp_path / "test_v3.db")
    s = PrismaLabStore(db_path=db_file)
    s.init_admin_tables()
    return s


@pytest.fixture
def app(store):
    from prismalab.miniapp.routes import create_miniapp, set_store
    set_store(store)
    return create_miniapp(store=store)


@pytest.fixture
def client(app):
    from starlette.testclient import TestClient
    return TestClient(app)


def _seed_style(store, slug="test_style", title="Test Style", provider="seedream", **kw):
    defaults = dict(emoji="🔥", gender="female", theme="glamour", prompt="test prompt")
    defaults.update(kw)
    return store.create_express_style(slug=slug, title=title, provider=provider, **defaults)


def _seed_catalog(store):
    """Создаёт категорию, тег, стиль и связи."""
    cat_id = store.create_express_category("lifestyle", "Lifestyle")
    tag_id = store.create_express_tag("fashion", "Fashion")
    style_id = _seed_style(store)
    store.set_style_categories(style_id, [cat_id])
    store.set_style_tags(style_id, [tag_id])
    store.set_category_tags(cat_id, [tag_id])
    return cat_id, tag_id, style_id


def _fake_photo():
    return ("photo", ("test.jpg", BytesIO(b"\xff\xd8" + b"\x00" * 100), "image/jpeg"))


# ── Catalog: auth ────────────────────────────────────────────────────

class TestV3CatalogAuth:

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=None)
    def test_catalog_401_no_auth(self, mock_auth, client):
        resp = client.get("/app/api/v3/express/catalog")
        assert resp.status_code == 401

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_catalog_200_empty(self, mock_auth, client, store):
        store.get_user(FAKE_USER["user_id"])
        resp = client.get("/app/api/v3/express/catalog", headers=AUTH_HEADER)
        assert resp.status_code == 200
        data = resp.json()
        assert "categories" in data
        assert "tags" in data
        assert "styles" in data
        assert "credits" in data
        assert "last_provider" in data


# ── Catalog: filtering ───────────────────────────────────────────────

class TestV3CatalogFiltering:

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_catalog_returns_categories_and_styles(self, mock_auth, client, store):
        store.get_user(FAKE_USER["user_id"])
        _seed_catalog(store)
        resp = client.get("/app/api/v3/express/catalog", headers=AUTH_HEADER)
        data = resp.json()
        assert len(data["categories"]) >= 1
        assert any(c["slug"] == "lifestyle" for c in data["categories"])
        assert len(data["styles"]) >= 1

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_catalog_filter_by_category(self, mock_auth, client, store):
        store.get_user(FAKE_USER["user_id"])
        _seed_catalog(store)
        # Стиль без категории — не должен попасть в фильтр
        _seed_style(store, slug="orphan", title="Orphan")

        resp = client.get("/app/api/v3/express/catalog?category=lifestyle", headers=AUTH_HEADER)
        data = resp.json()
        slugs = [s["id"] for s in data["styles"]]
        assert "test_style" in slugs
        assert "orphan" not in slugs

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_catalog_filter_by_category_returns_tags(self, mock_auth, client, store):
        store.get_user(FAKE_USER["user_id"])
        _seed_catalog(store)
        resp = client.get("/app/api/v3/express/catalog?category=lifestyle", headers=AUTH_HEADER)
        data = resp.json()
        assert len(data["tags"]) >= 1
        assert any(t["slug"] == "fashion" for t in data["tags"])

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_catalog_invalid_category_returns_all(self, mock_auth, client, store):
        store.get_user(FAKE_USER["user_id"])
        _seed_catalog(store)
        resp = client.get("/app/api/v3/express/catalog?category=nonexistent", headers=AUTH_HEADER)
        data = resp.json()
        # Невалидная категория — tags пустые, стили без фильтра по неизвестному slug
        assert data["tags"] == []


# ── Catalog: credits & provider ──────────────────────────────────────

class TestV3CatalogCredits:

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_catalog_credits_and_default_provider(self, mock_auth, client, store):
        uid = FAKE_USER["user_id"]
        store.get_user(uid)
        store.set_paid_generations_remaining(uid, 5)
        resp = client.get("/app/api/v3/express/catalog", headers=AUTH_HEADER)
        data = resp.json()
        assert data["credits"]["fast"] == 5
        assert data["last_provider"] == "seedream"  # default

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_catalog_remembers_last_provider(self, mock_auth, client, store):
        uid = FAKE_USER["user_id"]
        store.get_user(uid)
        store.set_user_last_express_provider(uid, "nano-banana-pro")
        resp = client.get("/app/api/v3/express/catalog", headers=AUTH_HEADER)
        assert resp.json()["last_provider"] == "nano-banana-pro"


# ── Generate: auth & validation ──────────────────────────────────────

class TestV3GenerateValidation:

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=None)
    def test_generate_401_no_auth(self, mock_auth, client):
        resp = client.post("/app/api/v3/express/generate", data={"init_data": "bad"})
        assert resp.status_code == 401

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_generate_400_no_style(self, mock_auth, client, store):
        store.get_user(FAKE_USER["user_id"])
        resp = client.post(
            "/app/api/v3/express/generate",
            data={"init_data": "fake", "style_id": ""},
            files=[_fake_photo()],
        )
        assert resp.status_code == 400

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_generate_400_no_photo(self, mock_auth, client, store):
        store.get_user(FAKE_USER["user_id"])
        _seed_style(store)
        resp = client.post(
            "/app/api/v3/express/generate",
            data={"init_data": "fake", "style_id": "test_style"},
        )
        assert resp.status_code == 400

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_generate_404_unknown_style(self, mock_auth, client, store):
        store.get_user(FAKE_USER["user_id"])
        resp = client.post(
            "/app/api/v3/express/generate",
            data={"init_data": "fake", "style_id": "nonexistent"},
            files=[_fake_photo()],
        )
        assert resp.status_code == 404


# ── Generate: credits ────────────────────────────────────────────────

class TestV3GenerateCredits:

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_generate_402_no_credits(self, mock_auth, client, store):
        uid = FAKE_USER["user_id"]
        store.get_user(uid)
        store.spend_free_generation(uid)
        # 0 paid credits, free used → 402
        _seed_style(store)
        resp = client.post(
            "/app/api/v3/express/generate",
            data={"init_data": "fake", "style_id": "test_style"},
            files=[_fake_photo()],
        )
        assert resp.status_code == 402
        assert resp.json()["error"] == "no_credits"


# ── Generate: success & provider override ────────────────────────────

class TestV3GenerateSuccess:

    @patch("prismalab.miniapp.routes._run_generation")
    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_generate_200_default_provider(self, mock_auth, mock_gen, client, store):
        async def noop(*a, **kw): pass
        mock_gen.side_effect = noop

        uid = FAKE_USER["user_id"]
        store.get_user(uid)
        store.set_paid_generations_remaining(uid, 3)
        _seed_style(store, provider="seedream")

        resp = client.post(
            "/app/api/v3/express/generate",
            data={"init_data": "fake", "style_id": "test_style"},
            files=[_fake_photo()],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "processing"
        assert data["provider"] == "seedream"

    @patch("prismalab.miniapp.routes._run_generation")
    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_generate_provider_override(self, mock_auth, mock_gen, client, store):
        async def noop(*a, **kw): pass
        mock_gen.side_effect = noop

        uid = FAKE_USER["user_id"]
        store.get_user(uid)
        store.set_paid_generations_remaining(uid, 3)
        _seed_style(store, provider="seedream")

        resp = client.post(
            "/app/api/v3/express/generate",
            data={"init_data": "fake", "style_id": "test_style", "provider": "nano-banana-pro"},
            files=[_fake_photo()],
        )
        assert resp.status_code == 200
        assert resp.json()["provider"] == "nano-banana-pro"
        # Проверяем что provider сохранился
        assert store.get_user_last_express_provider(uid) == "nano-banana-pro"

    @patch("prismalab.miniapp.routes._run_generation")
    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_generate_invalid_provider_ignored(self, mock_auth, mock_gen, client, store):
        async def noop(*a, **kw): pass
        mock_gen.side_effect = noop

        uid = FAKE_USER["user_id"]
        store.get_user(uid)
        store.set_paid_generations_remaining(uid, 3)
        _seed_style(store, provider="seedream")

        resp = client.post(
            "/app/api/v3/express/generate",
            data={"init_data": "fake", "style_id": "test_style", "provider": "invalid-provider"},
            files=[_fake_photo()],
        )
        assert resp.status_code == 200
        # Невалидный provider → используется дефолт стиля
        assert resp.json()["provider"] == "seedream"
