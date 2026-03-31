"""Tests for custom prompt generation: capabilities, validation, idempotency, lock."""
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
FAKE_USER_2 = {"user_id": 99999, "first_name": "Other", "username": "other"}
AUTH_HEADER = {"X-Telegram-Init-Data": "fake"}


@pytest.fixture
def store(tmp_path):
    os.environ.pop("DATABASE_URL", None)
    if "prismalab.storage" in sys.modules:
        del sys.modules["prismalab.storage"]
    from prismalab.storage import PrismaLabStore
    db_file = str(tmp_path / "test_custom.db")
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


def _fake_photo():
    """Minimal valid JPEG."""
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (10, 10), "red").save(buf, format="JPEG")
    return buf.getvalue()


def _fake_photo_file(name="test.jpg"):
    return ("photos", (name, BytesIO(_fake_photo()), "image/jpeg"))


# ── Capabilities ─────────────────────────────────────────────────────

class TestCustomCapabilities:

    def test_capabilities_200(self, client):
        resp = client.get("/app/api/v3/custom/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        assert "seedream" in data["providers"]
        assert "nano-banana-pro" in data["providers"]
        assert data["providers"]["seedream"]["max_photos"] == 14
        assert data["providers"]["nano-banana-pro"]["max_photos"] == 8
        assert data["max_prompt_length"] == 2000
        assert "allowed_mime" in data
        assert "max_file_size_mb" in data


# ── Auth ─────────────────────────────────────────────────────────────

class TestCustomAuth:

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=None)
    def test_generate_401(self, mock_auth, client):
        resp = client.post("/app/api/v3/custom/generate", data={"init_data": "bad"})
        assert resp.status_code == 401


# ── Validation ───────────────────────────────────────────────────────

class TestCustomValidation:

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_missing_request_id_400(self, mock_auth, client, store):
        store.get_user(FAKE_USER["user_id"])
        resp = client.post("/app/api/v3/custom/generate",
                           data={"init_data": "fake", "prompt": "test", "provider": "seedream"})
        assert resp.status_code == 400
        assert "request_id" in resp.json()["error"]

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_empty_prompt_400(self, mock_auth, client, store):
        store.get_user(FAKE_USER["user_id"])
        resp = client.post("/app/api/v3/custom/generate",
                           data={"init_data": "fake", "prompt": "",
                                 "request_id": "550e8400-e29b-41d4-a716-446655440000"})
        assert resp.status_code == 400
        assert "Prompt" in resp.json()["error"]

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_long_prompt_400(self, mock_auth, client, store):
        store.get_user(FAKE_USER["user_id"])
        resp = client.post("/app/api/v3/custom/generate",
                           data={"init_data": "fake", "prompt": "x" * 2001,
                                 "request_id": "550e8400-e29b-41d4-a716-446655440001"})
        assert resp.status_code == 400
        assert "long" in resp.json()["error"]

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_invalid_request_id_400(self, mock_auth, client, store):
        store.get_user(FAKE_USER["user_id"])
        resp = client.post("/app/api/v3/custom/generate",
                           data={"init_data": "fake", "prompt": "test", "request_id": "not-a-uuid"})
        assert resp.status_code == 400
        assert "UUID" in resp.json()["error"]

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_too_many_photos_nano_400(self, mock_auth, client, store):
        store.get_user(FAKE_USER["user_id"])
        store.set_paid_generations_remaining(FAKE_USER["user_id"], 5)
        files = [_fake_photo_file(f"photo_{i}.jpg") for i in range(9)]
        resp = client.post("/app/api/v3/custom/generate",
                           data={"init_data": "fake", "prompt": "test", "provider": "nano-banana-pro",
                                 "request_id": "550e8400-e29b-41d4-a716-446655440002"},
                           files=files)
        assert resp.status_code == 400
        assert "Too many" in resp.json()["error"]

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_gif_rejected_400(self, mock_auth, client, store):
        """GIF with empty content_type must be rejected by Pillow format check."""
        from PIL import Image
        uid = FAKE_USER["user_id"]
        store.get_user(uid)
        store.set_paid_generations_remaining(uid, 3)
        buf = BytesIO()
        Image.new("RGB", (10, 10), "blue").save(buf, format="GIF")
        files = [("photos", ("test.gif", BytesIO(buf.getvalue()), ""))]
        resp = client.post("/app/api/v3/custom/generate",
                           data={"init_data": "fake", "prompt": "test",
                                 "request_id": "550e8400-e29b-41d4-a716-446655440003"},
                           files=files)
        assert resp.status_code == 400
        assert "unsupported format" in resp.json()["error"].lower() or "GIF" in resp.json()["error"]

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_invalid_provider_defaults_to_seedream(self, mock_auth, client, store):
        """Invalid provider silently defaults to seedream."""
        uid = FAKE_USER["user_id"]
        store.get_user(uid)
        store.set_paid_generations_remaining(uid, 3)

        with patch("prismalab.miniapp.routes._run_custom_generation") as mock_gen:
            async def noop(*a, **kw): pass
            mock_gen.side_effect = noop
            resp = client.post("/app/api/v3/custom/generate",
                               data={"init_data": "fake", "prompt": "test", "provider": "bad-provider",
                                     "request_id": "550e8400-e29b-41d4-a716-446655440004"})
            assert resp.status_code == 200
            assert resp.json()["provider"] == "seedream"


# ── Credits ──────────────────────────────────────────────────────────

class TestCustomCredits:

    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_no_credits_402(self, mock_auth, client, store):
        uid = FAKE_USER["user_id"]
        store.get_user(uid)
        store.spend_free_generation(uid)
        resp = client.post("/app/api/v3/custom/generate",
                           data={"init_data": "fake", "prompt": "test", "provider": "seedream",
                                 "request_id": "550e8400-e29b-41d4-a716-446655440010"})
        assert resp.status_code == 402
        assert resp.json()["error"] == "no_credits"


# ── Success: text-only and with photos ───────────────────────────────

class TestCustomSuccess:

    @patch("prismalab.miniapp.routes._run_custom_generation")
    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_text_only_200(self, mock_auth, mock_gen, client, store):
        async def noop(*a, **kw): pass
        mock_gen.side_effect = noop
        uid = FAKE_USER["user_id"]
        store.get_user(uid)
        store.set_paid_generations_remaining(uid, 3)

        resp = client.post("/app/api/v3/custom/generate",
                           data={"init_data": "fake", "prompt": "A cat in space", "provider": "seedream",
                                 "request_id": "550e8400-e29b-41d4-a716-446655440020"})
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "processing"
        assert data["provider"] == "seedream"

    @patch("prismalab.miniapp.routes._run_custom_generation")
    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_with_photos_200(self, mock_auth, mock_gen, client, store):
        async def noop(*a, **kw): pass
        mock_gen.side_effect = noop
        uid = FAKE_USER["user_id"]
        store.get_user(uid)
        store.set_paid_generations_remaining(uid, 3)

        resp = client.post("/app/api/v3/custom/generate",
                           data={"init_data": "fake", "prompt": "Portrait in this style", "provider": "seedream",
                                 "request_id": "550e8400-e29b-41d4-a716-446655440021"},
                           files=[_fake_photo_file("ref1.jpg"), _fake_photo_file("ref2.jpg")])
        assert resp.status_code == 200
        assert "task_id" in resp.json()


# ── Idempotency ──────────────────────────────────────────────────────

class TestCustomIdempotency:

    @patch("prismalab.miniapp.routes._run_custom_generation")
    @patch("prismalab.miniapp.routes.validate_init_data", return_value=FAKE_USER)
    def test_same_request_id_returns_same_task(self, mock_auth, mock_gen, client, store):
        async def noop(*a, **kw): pass
        mock_gen.side_effect = noop
        uid = FAKE_USER["user_id"]
        store.get_user(uid)
        store.set_paid_generations_remaining(uid, 5)

        req_id = "550e8400-e29b-41d4-a716-446655440000"

        resp1 = client.post("/app/api/v3/custom/generate",
                            data={"init_data": "fake", "prompt": "test", "request_id": req_id})
        assert resp1.status_code == 200
        task_id = resp1.json()["task_id"]

        # Second request with same request_id → idempotent
        resp2 = client.post("/app/api/v3/custom/generate",
                            data={"init_data": "fake", "prompt": "test", "request_id": req_id})
        assert resp2.status_code == 200
        assert resp2.json()["task_id"] == task_id
        assert resp2.json().get("idempotent") is True

    @patch("prismalab.miniapp.routes._run_custom_generation")
    @patch("prismalab.miniapp.routes.validate_init_data")
    def test_same_request_id_different_users_not_deduplicated(self, mock_auth, mock_gen, client, store):
        """Same request_id for different users should NOT be deduplicated."""
        async def noop(*a, **kw): pass
        mock_gen.side_effect = noop

        req_id = "550e8400-e29b-41d4-a716-446655440000"

        # User 1
        mock_auth.return_value = FAKE_USER
        store.get_user(FAKE_USER["user_id"])
        store.set_paid_generations_remaining(FAKE_USER["user_id"], 5)
        resp1 = client.post("/app/api/v3/custom/generate",
                            data={"init_data": "fake", "prompt": "test", "request_id": req_id})
        task1 = resp1.json()["task_id"]

        # User 2 with same request_id
        mock_auth.return_value = FAKE_USER_2
        store.get_user(FAKE_USER_2["user_id"])
        store.set_paid_generations_remaining(FAKE_USER_2["user_id"], 5)
        resp2 = client.post("/app/api/v3/custom/generate",
                            data={"init_data": "fake", "prompt": "test", "request_id": req_id})
        task2 = resp2.json()["task_id"]

        assert task1 != task2
        assert resp2.json().get("idempotent") is not True


# ── Storage: generation_requests ─────────────────────────────────────

class TestGenerationRequests:

    def test_check_and_save(self, store):
        assert store.check_request_id(111, "req-abc") is None
        store.save_request_id("req-abc", 111, "task-xyz")
        assert store.check_request_id(111, "req-abc") == "task-xyz"

    def test_different_users_same_request_id(self, store):
        store.save_request_id("req-same", 111, "task-1")
        store.save_request_id("req-same", 222, "task-2")
        assert store.check_request_id(111, "req-same") == "task-1"
        assert store.check_request_id(222, "req-same") == "task-2"

    def test_cleanup(self, store):
        store.save_request_id("req-old", 111, "task-old")
        # Backdate the record to 2 hours ago for cleanup test
        with store._connect() as conn:
            conn.execute(
                f"UPDATE {store._generation_requests_table} SET created_at = datetime('now', '-2 hours') WHERE request_id = ?",
                ("req-old",),
            )
            conn.commit()
        store.cleanup_old_requests(max_age_hours=1)
        assert store.check_request_id(111, "req-old") is None


# ── Storage: generation_history with new fields ──────────────────────

class TestHistoryCustomFields:

    def test_save_with_custom_fields(self, store):
        store.get_user(12345)
        hid = store.save_generation_history(
            user_id=12345, mode="custom", style_slug="__custom__",
            style_title="A cat in space", provider="seedream",
            prompt_preview="A cat in space", refs_count=3, request_id="req-123",
        )
        rows = store.get_generation_history(12345, mode="custom")
        assert len(rows) == 1
        assert rows[0]["mode"] == "custom"
        assert rows[0]["prompt_preview"] == "A cat in space"
        assert rows[0]["refs_count"] == 3
        assert rows[0]["request_id"] == "req-123"

    def test_prompt_preview_sanitized(self, store):
        store.get_user(12345)
        store.save_generation_history(
            user_id=12345, mode="custom", style_slug="__custom__",
            style_title="test", provider="seedream",
            prompt_preview="<script>alert('xss')</script>" + "x" * 200,
        )
        rows = store.get_generation_history(12345)
        preview = rows[0]["prompt_preview"]
        assert "<script>" not in preview
        assert len(preview) <= 100
