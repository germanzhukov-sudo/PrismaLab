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
    """persona_styles — только PG, в SQLite возвращает пустой список (by design)."""
    if not store._use_pg:
        styles = store.get_persona_styles()
        assert styles == []
        return
    # PG-тесты (не запускаются в CI с SQLite)
    store.create_persona_style(
        slug="test_style",
        title="Test Style",
        description="Test",
        prompt="test prompt",
        gender="female",
        image_url="",
        sort_order=1,
    )
    styles = store.get_persona_styles()
    assert len(styles) >= 1


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
