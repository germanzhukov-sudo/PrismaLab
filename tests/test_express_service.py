"""Тесты сервиса каталога экспресс-стилей (miniapp/services/express.py)."""
from __future__ import annotations

from prismalab.miniapp.services.express import (
    ResolvedStyle,
    StyleInfo,
    get_styles,
    get_themes,
    resolve_style,
)


# ── get_styles ────────────────────────────────────────────────────────


def test_get_styles_fallback_when_db_empty(store):
    """Пустая БД → fallback на hardcoded стили."""
    styles = get_styles(store, gender="male")
    assert len(styles) == 10  # 10 male hardcoded
    assert all(isinstance(s, StyleInfo) for s in styles)
    assert styles[0].slug == "night_bar"


def test_get_styles_fallback_female(store):
    """Fallback female стили."""
    styles = get_styles(store, gender="female")
    assert len(styles) == 29  # 29 female hardcoded
    assert styles[0].slug == "wedding"


def test_get_styles_from_db(store):
    """Стили из БД приоритетнее fallback."""
    store.create_express_style(
        slug="db_style", title="DB Style", emoji="🔥",
        theme="test", gender="male", sort_order=1,
    )
    styles = get_styles(store, gender="male")
    assert len(styles) == 1
    assert styles[0].slug == "db_style"
    assert styles[0].title == "DB Style"
    assert styles[0].emoji == "🔥"
    assert styles[0].theme == "test"


def test_get_styles_filter_by_theme(store):
    """Фильтрация по теме."""
    store.create_express_style(slug="s1", title="S1", gender="female", theme="glamour")
    store.create_express_style(slug="s2", title="S2", gender="female", theme="mood")
    store.create_express_style(slug="s3", title="S3", gender="female", theme="glamour")

    glamour = get_styles(store, gender="female", theme="glamour")
    assert len(glamour) == 2
    assert all(s.theme == "glamour" for s in glamour)


def test_get_styles_inactive_excluded(store):
    """Неактивные стили не возвращаются, active — возвращаются."""
    store.create_express_style(slug="active_one", title="Active", gender="male")
    sid2 = store.create_express_style(slug="inactive", title="X", gender="male")
    store.update_express_style(sid2, is_active=0)
    styles = get_styles(store, gender="male")
    assert len(styles) == 1
    assert styles[0].slug == "active_one"


def test_get_styles_to_api_dict(store):
    """to_api_dict формат (backward compat)."""
    store.create_express_style(
        slug="test", title="Test", emoji="🎯", theme="mood",
        gender="female", image_url="https://img.com/1.jpg",
    )
    styles = get_styles(store, gender="female")
    d = styles[0].to_api_dict()
    assert d["id"] == "test"
    assert d["label"] == "Test"
    assert d["emoji"] == "🎯"
    assert d["theme"] == "mood"
    assert d["image_url"] == "https://img.com/1.jpg"


def test_get_styles_all_inactive_returns_empty(store):
    """БД имеет стили, но все inactive → пустой список (НЕ fallback)."""
    sid = store.create_express_style(slug="off1", title="Off", gender="male")
    store.update_express_style(sid, is_active=0)
    styles = get_styles(store, gender="male")
    assert styles == []  # не fallback hardcoded!


# ── get_themes ────────────────────────────────────────────────────────


def test_get_themes_from_db(store):
    """Темы из БД."""
    store.create_express_style(slug="t1", title="T1", gender="female", theme="glamour")
    store.create_express_style(slug="t2", title="T2", gender="female", theme="mood")
    store.create_express_style(slug="t3", title="T3", gender="female", theme="glamour")

    themes = get_themes(store, gender="female")
    assert themes == ["glamour", "mood"]


def test_get_themes_fallback(store):
    """Пустая БД → fallback темы."""
    themes = get_themes(store)
    assert len(themes) > 0
    assert "lifestyle" in themes
    assert "glamour" in themes


def test_get_themes_fallback_by_gender(store):
    """Fallback темы фильтруются по полу."""
    male_themes = get_themes(store, gender="male")
    female_themes = get_themes(store, gender="female")
    # male не имеет glamour/beauty, female имеет
    assert "glamour" not in male_themes
    assert "glamour" in female_themes


# ── resolve_style ─────────────────────────────────────────────────────


def test_resolve_style_from_db(store):
    """Стиль из БД → ResolvedStyle с prompt/provider."""
    store.create_express_style(
        slug="resolve_test", title="Resolve Test", gender="female",
        prompt="beautiful portrait", negative_prompt="bad quality",
        provider="nano-banana-pro", model_params='{"resolution": "4K"}',
    )
    result = resolve_style(store, "resolve_test")
    assert result is not None
    assert isinstance(result, ResolvedStyle)
    assert result.slug == "resolve_test"
    assert result.title == "Resolve Test"
    assert result.prompt == "beautiful portrait"
    assert result.negative_prompt == "bad quality"
    assert result.provider == "nano-banana-pro"
    assert result.model_params_json == '{"resolution": "4K"}'


def test_resolve_style_inactive_returns_none(store):
    """Inactive стиль в БД → None (нельзя генерировать)."""
    sid = store.create_express_style(
        slug="disabled_style", title="Disabled", gender="female",
        prompt="should not work",
    )
    store.update_express_style(sid, is_active=0)
    result = resolve_style(store, "disabled_style")
    assert result is None


def test_resolve_style_fallback_persona_prompts(store):
    """Нет в БД → fallback на PERSONA_STYLE_PROMPTS."""
    # wedding есть в PERSONA_STYLE_PROMPTS
    result = resolve_style(store, "wedding")
    assert result is not None
    assert result.slug == "wedding"
    assert result.provider == "seedream"
    assert len(result.prompt) > 0  # есть промпт из PERSONA_STYLE_PROMPTS


def test_resolve_style_fallback_hardcoded(store):
    """Нет в БД и нет промпта → fallback с title из hardcoded."""
    # Подменяем — возьмём slug который есть в hardcoded но нет в PERSONA_STYLE_PROMPTS
    # pilot есть в hardcoded male
    result = resolve_style(store, "pilot")
    assert result is not None
    assert result.slug == "pilot"
    # Может иметь или не иметь prompt — зависит от PERSONA_STYLE_PROMPTS


def test_resolve_style_not_found(store):
    """Несуществующий стиль → None."""
    result = resolve_style(store, "completely_unknown_style_xyz")
    assert result is None


def test_resolve_style_db_priority_over_fallback(store):
    """БД имеет приоритет над PERSONA_STYLE_PROMPTS."""
    # wedding есть в PERSONA_STYLE_PROMPTS, добавим в БД с другим prompt
    store.create_express_style(
        slug="wedding", title="Custom Wedding", gender="female",
        prompt="custom wedding prompt", provider="nano-banana-pro",
    )
    result = resolve_style(store, "wedding")
    assert result is not None
    assert result.prompt == "custom wedding prompt"
    assert result.provider == "nano-banana-pro"
