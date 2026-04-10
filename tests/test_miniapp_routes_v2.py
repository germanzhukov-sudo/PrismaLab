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
    """Unified список фотосетов + backward-compat packs."""
    resp = client.get("/app/api/v2/photosets", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert "photosets" in data
    assert "packs" in data
    # DEFAULT_PACKS имеет >= 2 пака
    assert len(data["packs"]) >= 2
    assert len(data["photosets"]) >= len(data["packs"])


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_photosets_dto_has_category(mock_auth, client, store):
    """Every photoset item has a 'category' field."""
    resp = client.get("/app/api/v2/photosets", headers=_auth_headers())
    assert resp.status_code == 200
    for item in resp.json()["photosets"]:
        assert "category" in item, f"Missing category in {item['id']}"


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
        assert "preview_urls" in pack


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_v2_photosets_unified_dto_shape(mock_auth, client, store):
    """Unified DTO стабилен: id/entity_id/type/title/credit_cost/preview_urls/num_images."""
    style_id = store.create_persona_style(
        slug="dto_style",
        title="DTO Style",
        gender="female",
        image_url="https://fallback.jpg",
        credit_cost=7,
    )
    store.set_style_previews(style_id, ["https://s1.jpg", "https://s2.jpg", "https://s3.jpg", "https://s4.jpg"])

    resp = client.get("/app/api/v2/photosets", headers=_auth_headers())
    assert resp.status_code == 200
    photosets = resp.json()["photosets"]

    style_item = next((p for p in photosets if p["type"] == "style" and p["entity_id"] == style_id), None)
    assert style_item is not None
    assert style_item["id"] == f"style:{style_id}"
    assert style_item["slug"] == "dto_style"
    assert style_item["title"] == "DTO Style"
    assert style_item["credit_cost"] == 7
    assert style_item["num_images"] == 4
    assert style_item["preview_urls"] == [
        "https://s1.jpg",
        "https://s2.jpg",
        "https://s3.jpg",
        "https://s4.jpg",
    ]

    pack_item = next((p for p in photosets if p["type"] == "pack"), None)
    assert pack_item is not None
    assert str(pack_item["id"]).startswith("pack:")
    assert isinstance(pack_item["entity_id"], int)
    assert isinstance(pack_item["title"], str)
    assert isinstance(pack_item["credit_cost"], int)
    assert isinstance(pack_item["num_images"], int)
    assert isinstance(pack_item["preview_urls"], list)


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


# ── Phase 4: Lock info in photoset DTO ──────────────────────────────


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_photoset_dto_no_persona_locked(mock_auth, client, store):
    """Без персоны все фотосеты locked с reason=need_persona."""
    style_id = store.create_persona_style(
        slug="lock_test_1", title="Lock test", gender="female",
        image_url="https://img.jpg", credit_cost=8,
    )
    resp = client.get("/app/api/v2/photosets", headers=_auth_headers())
    data = resp.json()
    for item in data["photosets"]:
        assert item["is_locked"] is True
        assert item["unlock_reason"] == "need_persona"


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_photoset_dto_zero_credits_locked(mock_auth, client, store):
    """Персона есть, 0 кредитов → стили locked, рублёвые паки unlocked."""
    store.set_astria_lora_tune(user_id=12345, tune_id="tune_123", class_name="woman")
    store.set_persona_credits(12345, 0)
    style_id = store.create_persona_style(
        slug="lock_test_2", title="Lock test 2", gender="female",
        image_url="https://img.jpg", credit_cost=8,
    )
    resp = client.get("/app/api/v2/photosets", headers=_auth_headers())
    data = resp.json()
    for item in data["photosets"]:
        if item["type"] == "style":
            assert item["is_locked"] is True
            assert item["unlock_reason"] == "need_credits"
        elif item["type"] == "pack":
            # Packs bought for ₽ (PACKS_USE_CREDITS=0) are never locked by credits
            assert item["is_locked"] is False


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_photoset_dto_locked_not_enough_credits(mock_auth, client, store):
    """Персона + credits < cost → is_locked=True."""
    store.set_astria_lora_tune(user_id=12345, tune_id="tune_123", class_name="woman")
    store.set_persona_credits(12345, 4)
    style_id = store.create_persona_style(
        slug="lock_test_3", title="Expensive style", gender="female",
        image_url="https://img.jpg", credit_cost=8,
    )
    resp = client.get("/app/api/v2/photosets", headers=_auth_headers())
    data = resp.json()
    style_item = next((p for p in data["photosets"] if p["type"] == "style" and p["entity_id"] == style_id), None)
    assert style_item is not None
    assert style_item["is_locked"] is True
    assert style_item["unlock_cost"] == 8
    assert style_item["unlock_reason"] == "need_credits"


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_photoset_dto_unlocked_enough_credits(mock_auth, client, store):
    """Персона + credits >= cost → is_locked=False."""
    store.set_astria_lora_tune(user_id=12345, tune_id="tune_123", class_name="woman")
    store.set_persona_credits(12345, 10)
    style_id = store.create_persona_style(
        slug="lock_test_4", title="Affordable style", gender="female",
        image_url="https://img.jpg", credit_cost=4,
    )
    resp = client.get("/app/api/v2/photosets", headers=_auth_headers())
    data = resp.json()
    style_item = next((p for p in data["photosets"] if p["type"] == "style" and p["entity_id"] == style_id), None)
    assert style_item is not None
    assert style_item["is_locked"] is False
    assert "unlock_cost" not in style_item


@patch("prismalab.config.packs_use_credits", return_value=False)
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_photoset_dto_pack_not_locked_when_credits_off(mock_auth, mock_flag, client, store):
    """Паки с PACKS_USE_CREDITS=0 → никогда не locked по кредитам."""
    store.set_astria_lora_tune(user_id=12345, tune_id="tune_123", class_name="woman")
    store.set_persona_credits(12345, 2)  # мало кредитов
    resp = client.get("/app/api/v2/photosets", headers=_auth_headers())
    data = resp.json()
    packs_in_dto = [p for p in data["photosets"] if p["type"] == "pack"]
    for p in packs_in_dto:
        assert p["is_locked"] is False


@patch("prismalab.config.packs_use_credits", return_value=True)
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_photoset_dto_pack_locked_when_credits_on(mock_auth, mock_flag, client, store):
    """Паки с PACKS_USE_CREDITS=1 + мало кредитов → locked."""
    store.set_astria_lora_tune(user_id=12345, tune_id="tune_123", class_name="woman")
    store.set_persona_credits(12345, 2)  # мало кредитов, паки стоят 20+
    resp = client.get("/app/api/v2/photosets", headers=_auth_headers())
    data = resp.json()
    packs_in_dto = [p for p in data["photosets"] if p["type"] == "pack"]
    # Все паки стоят >= 20 кредитов, а у юзера 2
    locked_packs = [p for p in packs_in_dto if p["is_locked"]]
    assert len(locked_packs) == len(packs_in_dto)


# ── Phase 4: Admin features — get_admin_setting, cost_usd, discount badge ──


def test_get_admin_setting(store):
    """get_admin_setting returns value for existing key, None for missing."""
    store.set_admin_setting("test_key_123", "hello")
    assert store.get_admin_setting("test_key_123") == "hello"
    assert store.get_admin_setting("nonexistent_key_xyz") is None


def test_persona_style_cost_usd(store):
    """cost_usd field is stored and retrieved."""
    style_id = store.create_persona_style(
        slug="cost_test", title="Cost Test", gender="female",
        image_url="https://img.jpg",
    )
    # Default is 0
    style = store.get_persona_style(style_id)
    assert float(style.get("cost_usd", 0) or 0) == 0.0

    # Update cost_usd
    store.update_persona_style(style_id, cost_usd=1.25)
    style = store.get_persona_style(style_id)
    assert float(style["cost_usd"]) == 1.25


def test_set_persona_style_costs_bulk(store):
    """Bulk cost update works."""
    s1 = store.create_persona_style(slug="bulk1", title="B1", gender="female", image_url="")
    s2 = store.create_persona_style(slug="bulk2", title="B2", gender="male", image_url="")
    store.set_persona_style_costs_bulk([
        {"style_id": s1, "cost_usd": 0.50},
        {"style_id": s2, "cost_usd": 1.75},
    ])
    assert float(store.get_persona_style(s1)["cost_usd"]) == 0.50
    assert float(store.get_persona_style(s2)["cost_usd"]) == 1.75


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_auth_returns_discount_badge(mock_auth, client, store):
    """Auth API returns discount_badge from admin_settings."""
    store.set_admin_setting("photosets_discount_badge", "для вас -40%")
    resp = client.post("/app/api/auth", json={"init_data": "fake"})
    assert resp.status_code == 200
    assert resp.json()["discount_badge"] == "для вас -40%"


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_auth_returns_empty_discount_badge(mock_auth, client, store):
    """Auth API returns empty discount_badge when not set."""
    resp = client.post("/app/api/auth", json={"init_data": "fake"})
    assert resp.status_code == 200
    assert resp.json()["discount_badge"] == ""


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_auth_returns_featured_custom_contract(mock_auth, client, store):
    """Auth API returns featured_custom list with correct shape."""
    resp = client.post("/app/api/auth", json={"init_data": "fake"})
    assert resp.status_code == 200
    data = resp.json()
    assert "featured_custom" in data
    featured = data["featured_custom"]
    assert isinstance(featured, list)
    assert len(featured) >= 5  # At least 5 curated examples
    for item in featured:
        assert "title" in item and isinstance(item["title"], str) and item["title"]
        assert "image_url" in item and isinstance(item["image_url"], str)
        assert item["image_url"].startswith("https://")


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_auth_returns_featured_styles_and_packs(mock_auth, client, store):
    """Auth API returns featured_styles and featured_packs fields."""
    resp = client.post("/app/api/auth", json={"init_data": "fake"})
    assert resp.status_code == 200
    data = resp.json()
    assert "featured_styles" in data
    assert "featured_packs" in data
    assert isinstance(data["featured_styles"], list)
    assert isinstance(data["featured_packs"], list)


# ── Phase 2: POST /app/api/fast/buy ─────────────────────────────


@patch("prismalab.miniapp.routes.create_payment", return_value=("https://pay.example.com", "pay_123"))
@patch("prismalab.miniapp.routes.poll_payment_status")
@patch("prismalab.miniapp.routes.get_bot", return_value=None)
@patch("prismalab.miniapp.routes.get_application", return_value=None)
@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_fast_buy_success(mock_auth, mock_app, mock_bot, mock_poll, mock_pay, client, store):
    """POST /app/api/fast/buy with valid credits returns payment_url."""
    resp = client.post("/app/api/fast/buy", json={"init_data": "fake", "credits": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert "payment_url" in data
    assert data["payment_url"] == "https://pay.example.com"


@patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
def test_fast_buy_invalid_credits(mock_auth, client, store):
    """POST /app/api/fast/buy with invalid credits returns 400."""
    resp = client.post("/app/api/fast/buy", json={"init_data": "fake", "credits": 99})
    assert resp.status_code == 400


@patch("prismalab.miniapp.routes.validate_init_data", return_value=None)
def test_fast_buy_unauthorized(mock_auth, client):
    """POST /app/api/fast/buy without auth returns 401."""
    resp = client.post("/app/api/fast/buy", json={"init_data": "bad"})
    assert resp.status_code == 401
