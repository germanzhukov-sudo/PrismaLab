"""Тесты storage.py — критичные операции с БД."""
from __future__ import annotations


def test_get_user_auto_creates(store):
    """get_user создаёт юзера при первом обращении."""
    user = store.get_user(12345)
    assert user is not None
    assert user.user_id == 12345
    assert user.persona_credits_remaining == 0


def test_get_user_returns_same(store):
    """Повторный get_user возвращает того же юзера."""
    store.get_user(100)
    user = store.get_user(100)
    assert user.user_id == 100


def test_set_persona_credits(store):
    """set_persona_credits устанавливает кредиты."""
    store.get_user(1)
    store.set_persona_credits(1, 20)
    user = store.get_user(1)
    assert user.persona_credits_remaining == 20


def test_decrement_persona_credits(store):
    """decrement_persona_credits уменьшает на 1."""
    store.get_user(1)
    store.set_persona_credits(1, 5)
    store.decrement_persona_credits(1)
    user = store.get_user(1)
    assert user.persona_credits_remaining == 4


def test_decrement_persona_credits_not_below_zero(store):
    """Кредиты не уходят в минус."""
    store.get_user(1)
    store.set_persona_credits(1, 0)
    store.decrement_persona_credits(1)
    user = store.get_user(1)
    assert user.persona_credits_remaining >= 0


def test_spend_free_generation(store):
    """spend_free_generation помечает бесплатную генерацию использованной."""
    store.get_user(1)
    user_before = store.get_user(1)
    assert user_before.free_generation_used is False
    store.spend_free_generation(1)
    user_after = store.get_user(1)
    assert user_after.free_generation_used is True


def test_set_paid_generations(store):
    """set_paid_generations_remaining устанавливает платные генерации."""
    store.get_user(1)
    store.set_paid_generations_remaining(1, 10)
    user = store.get_user(1)
    assert user.paid_generations_remaining == 10


def test_log_payment_and_dedup(store):
    """log_payment записывает платёж, is_payment_processed дедуплицирует."""
    store.get_user(1)
    assert store.is_payment_processed("pay_001") is False
    store.log_payment(
        user_id=1,
        payment_id="pay_001",
        payment_method="yookassa",
        product_type="fast",
        credits=5,
        amount_rub=199.0,
    )
    assert store.is_payment_processed("pay_001") is True
    # Повторная запись не должна падать
    store.log_payment(
        user_id=1,
        payment_id="pay_001",
        payment_method="yookassa",
        product_type="fast",
        credits=5,
        amount_rub=199.0,
    )


def test_pending_pack_upload(store):
    """set/get/clear pending_pack_upload."""
    store.get_user(1)
    assert store.get_pending_pack_upload(1) is None
    store.set_pending_pack_upload(user_id=1, pack_id=4345)
    assert store.get_pending_pack_upload(1) == 4345
    store.clear_pending_pack_upload(1)
    assert store.get_pending_pack_upload(1) is None


def test_persona_styles_crud(store):
    """persona_styles — CRUD работает и в SQLite."""
    style_id = store.create_persona_style(
        slug="test_style",
        title="Test Style",
        description="Test",
        prompt="test prompt",
        gender="female",
        image_url="",
        sort_order=1,
    )
    assert style_id is not None

    styles = store.get_persona_styles()
    assert len(styles) >= 1
    assert styles[0]["slug"] == "test_style"

    # Read by id
    s = store.get_persona_style(style_id)
    assert s is not None
    assert s["title"] == "Test Style"

    # Read by slug
    s2 = store.get_persona_style_by_slug("test_style")
    assert s2 is not None
    assert s2["id"] == style_id

    # Update
    ok = store.update_persona_style(style_id, title="Updated Style")
    assert ok is True
    s3 = store.get_persona_style(style_id)
    assert s3["title"] == "Updated Style"

    # Delete
    ok = store.delete_persona_style(style_id)
    assert ok is True
    assert store.get_persona_style(style_id) is None


def test_persona_style_credit_cost(store):
    """credit_cost — создание, чтение, обновление."""
    style_id = store.create_persona_style(
        slug="cost_test",
        title="Cost Test",
        gender="female",
        credit_cost=6,
    )
    assert style_id is not None

    s = store.get_persona_style(style_id)
    assert s["credit_cost"] == 6

    store.update_persona_style(style_id, credit_cost=2)
    s2 = store.get_persona_style(style_id)
    assert s2["credit_cost"] == 2


def test_persona_style_credit_cost_default(store):
    """credit_cost по умолчанию = 4."""
    style_id = store.create_persona_style(
        slug="default_cost",
        title="Default Cost",
        gender="male",
    )
    s = store.get_persona_style(style_id)
    assert s["credit_cost"] == 4


def test_express_styles_crud(store):
    """express_styles — полный CRUD цикл."""
    # Create
    style_id = store.create_express_style(
        slug="night_bar",
        title="Ночной бар",
        emoji="🍸",
        gender="male",
        prompt="test prompt",
        model="seedream",
        sort_order=1,
    )
    assert style_id is not None

    # Read all
    styles = store.get_express_styles()
    assert len(styles) == 1
    assert styles[0]["slug"] == "night_bar"
    assert styles[0]["emoji"] == "🍸"
    assert styles[0]["model"] == "seedream"

    # Read by id
    s = store.get_express_style(style_id)
    assert s is not None
    assert s["title"] == "Ночной бар"

    # Read by slug
    s2 = store.get_express_style_by_slug("night_bar")
    assert s2 is not None
    assert s2["id"] == style_id

    # Update
    ok = store.update_express_style(style_id, title="Night Bar", emoji="🌙")
    assert ok is True
    s3 = store.get_express_style(style_id)
    assert s3["title"] == "Night Bar"
    assert s3["emoji"] == "🌙"

    # Delete
    ok = store.delete_express_style(style_id)
    assert ok is True
    assert store.get_express_style(style_id) is None


def test_express_styles_filter_gender(store):
    """express_styles — фильтрация по полу."""
    store.create_express_style(slug="male_1", title="Male Style", gender="male")
    store.create_express_style(slug="female_1", title="Female Style", gender="female")

    males = store.get_express_styles(gender="male")
    assert len(males) == 1
    assert males[0]["slug"] == "male_1"

    females = store.get_express_styles(gender="female")
    assert len(females) == 1
    assert females[0]["slug"] == "female_1"

    all_styles = store.get_express_styles()
    assert len(all_styles) == 2


def test_express_styles_active_filter(store):
    """express_styles — фильтрация по active_only."""
    sid = store.create_express_style(slug="active_test", title="Active Test", gender="female")
    assert len(store.get_express_styles(active_only=True)) == 1

    store.update_express_style(sid, is_active=0)
    assert len(store.get_express_styles(active_only=True)) == 0
    assert len(store.get_express_styles(active_only=False)) == 1


def test_express_styles_upsert(store):
    """upsert_express_style — insert + update по slug."""
    # Insert
    sid1 = store.upsert_express_style(slug="upsert_test", title="V1", emoji="🔥", gender="female")
    assert sid1 is not None
    s1 = store.get_express_style(sid1)
    assert s1["title"] == "V1"

    # Upsert (update existing)
    sid2 = store.upsert_express_style(slug="upsert_test", title="V2", emoji="✨", gender="female")
    assert sid2 == sid1  # same id
    s2 = store.get_express_style(sid1)
    assert s2["title"] == "V2"
    assert s2["emoji"] == "✨"

    # Only 1 row
    assert len(store.get_express_styles()) == 1


def test_log_event(store):
    """log_event не падает."""
    store.get_user(1)
    store.log_event(1, "test_event", {"key": "value"})
    # Просто проверяем что не упало


def test_subject_gender(store):
    """set_subject_gender сохраняет пол."""
    store.get_user(1)
    store.set_subject_gender(1, "male")
    user = store.get_user(1)
    assert user.subject_gender == "male"


def test_clear_user_data(store):
    """clear_user_data сбрасывает tune/model данные, но НЕ кредиты."""
    store.get_user(1)
    store.set_astria_lora_tune(user_id=1, tune_id="12345")
    store.clear_user_data(user_id=1)
    user = store.get_user(1)
    assert user.astria_lora_tune_id is None
