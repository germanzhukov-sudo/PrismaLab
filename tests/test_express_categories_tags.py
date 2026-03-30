"""Тесты для категорий, тегов и фильтрации экспресс-стилей."""
from __future__ import annotations

import os
import sys

import pytest


# ========== Categories CRUD ==========

def test_create_and_get_category(store):
    cid = store.create_express_category("female", "Женские", sort_order=1)
    assert cid is not None
    cat = store.get_express_category(cid)
    assert cat["slug"] == "female"
    assert cat["title"] == "Женские"
    assert cat["sort_order"] == 1


def test_get_category_by_slug(store):
    store.create_express_category("male", "Мужские")
    cat = store.get_express_category_by_slug("male")
    assert cat is not None
    assert cat["title"] == "Мужские"


def test_list_categories_active_only(store):
    store.create_express_category("female", "Женские", is_active=True)
    store.create_express_category("kids", "Детские", is_active=False)
    active = store.get_express_categories(active_only=True)
    all_cats = store.get_express_categories(active_only=False)
    assert len(active) == 1
    assert len(all_cats) == 2


def test_update_category(store):
    cid = store.create_express_category("old_slug", "Old Title")
    store.update_express_category(cid, title="New Title", slug="new_slug")
    cat = store.get_express_category(cid)
    assert cat["title"] == "New Title"
    assert cat["slug"] == "new_slug"


def test_delete_category(store):
    cid = store.create_express_category("tmp", "Temp")
    store.delete_express_category(cid)
    assert store.get_express_category(cid) is None


def test_categories_sorted_by_sort_order(store):
    store.create_express_category("c", "Third", sort_order=3)
    store.create_express_category("a", "First", sort_order=1)
    store.create_express_category("b", "Second", sort_order=2)
    cats = store.get_express_categories(active_only=False)
    assert [c["slug"] for c in cats] == ["a", "b", "c"]


# ========== Tags CRUD ==========

def test_create_and_get_tag(store):
    tid = store.create_express_tag("bw", "ч/б", sort_order=1)
    assert tid is not None
    tag = store.get_express_tag(tid)
    assert tag["slug"] == "bw"
    assert tag["title"] == "ч/б"


def test_get_tag_by_slug(store):
    store.create_express_tag("sport", "Спорт")
    tag = store.get_express_tag_by_slug("sport")
    assert tag is not None
    assert tag["title"] == "Спорт"


def test_list_tags_active_only(store):
    store.create_express_tag("a", "Active", is_active=True)
    store.create_express_tag("b", "Inactive", is_active=False)
    active = store.get_express_tags(active_only=True)
    all_tags = store.get_express_tags(active_only=False)
    assert len(active) == 1
    assert len(all_tags) == 2


def test_update_tag(store):
    tid = store.create_express_tag("old", "Old")
    store.update_express_tag(tid, title="New", slug="new")
    tag = store.get_express_tag(tid)
    assert tag["title"] == "New"
    assert tag["slug"] == "new"


def test_delete_tag(store):
    tid = store.create_express_tag("tmp", "Temp")
    store.delete_express_tag(tid)
    assert store.get_express_tag(tid) is None


# ========== Junction: style ↔ categories ==========

def _create_style(store, slug, title="Test", gender="female", theme="general"):
    """Хелпер: создать стиль и вернуть id."""
    return store.create_express_style(
        slug=slug, title=title, gender=gender, theme=theme,
        prompt="test prompt", provider="seedream",
    )


def test_set_and_get_style_categories(store):
    sid = _create_style(store, "style1")
    c1 = store.create_express_category("female", "Женские")
    c2 = store.create_express_category("kids", "Детские")
    store.set_style_categories(sid, [c1, c2])
    cats = store.get_style_categories(sid)
    slugs = {c["slug"] for c in cats}
    assert slugs == {"female", "kids"}


def test_set_style_categories_replaces(store):
    sid = _create_style(store, "style1")
    c1 = store.create_express_category("female", "Женские")
    c2 = store.create_express_category("male", "Мужские")
    store.set_style_categories(sid, [c1])
    store.set_style_categories(sid, [c2])
    cats = store.get_style_categories(sid)
    assert len(cats) == 1
    assert cats[0]["slug"] == "male"


# ========== Junction: style ↔ tags ==========

def test_set_and_get_style_tags(store):
    sid = _create_style(store, "style1")
    t1 = store.create_express_tag("bw", "ч/б")
    t2 = store.create_express_tag("sport", "Спорт")
    store.set_style_tags(sid, [t1, t2])
    tags = store.get_style_tags(sid)
    slugs = {t["slug"] for t in tags}
    assert slugs == {"bw", "sport"}


def test_set_style_tags_replaces(store):
    sid = _create_style(store, "style1")
    t1 = store.create_express_tag("bw", "ч/б")
    t2 = store.create_express_tag("sport", "Спорт")
    store.set_style_tags(sid, [t1])
    store.set_style_tags(sid, [t2])
    tags = store.get_style_tags(sid)
    assert len(tags) == 1
    assert tags[0]["slug"] == "sport"


# ========== Junction: category ↔ tags ==========

def test_set_and_get_category_tags(store):
    cid = store.create_express_category("female", "Женские")
    t1 = store.create_express_tag("bw", "ч/б")
    t2 = store.create_express_tag("party", "Вечеринка")
    store.set_category_tags(cid, [t1, t2])
    tags = store.get_category_tags(cid)
    slugs = {t["slug"] for t in tags}
    assert slugs == {"bw", "party"}


def test_set_category_tags_replaces(store):
    cid = store.create_express_category("female", "Женские")
    t1 = store.create_express_tag("bw", "ч/б")
    t2 = store.create_express_tag("sport", "Спорт")
    store.set_category_tags(cid, [t1])
    store.set_category_tags(cid, [t2])
    tags = store.get_category_tags(cid)
    assert len(tags) == 1
    assert tags[0]["slug"] == "sport"


# ========== get_styles_filtered ==========

def _setup_filtered(store):
    """Создаёт тестовые данные для фильтрации.

    Категории: female, male
    Теги: bw, sport (оба разрешены для female; sport разрешён для male)
    Стили:
      - s_female_bw: female + bw
      - s_female_sport: female + sport
      - s_male_sport: male + sport
      - s_no_cat: без категории
    """
    c_female = store.create_express_category("female", "Женские")
    c_male = store.create_express_category("male", "Мужские")
    t_bw = store.create_express_tag("bw", "ч/б")
    t_sport = store.create_express_tag("sport", "Спорт")

    # Разрешённые теги
    store.set_category_tags(c_female, [t_bw, t_sport])
    store.set_category_tags(c_male, [t_sport])

    # Стили
    s1 = _create_style(store, "s_female_bw")
    store.set_style_categories(s1, [c_female])
    store.set_style_tags(s1, [t_bw])

    s2 = _create_style(store, "s_female_sport")
    store.set_style_categories(s2, [c_female])
    store.set_style_tags(s2, [t_sport])

    s3 = _create_style(store, "s_male_sport")
    store.set_style_categories(s3, [c_male])
    store.set_style_tags(s3, [t_sport])

    s4 = _create_style(store, "s_no_cat")
    # Без категорий и тегов

    return {
        "c_female": c_female, "c_male": c_male,
        "t_bw": t_bw, "t_sport": t_sport,
        "s1": s1, "s2": s2, "s3": s3, "s4": s4,
    }


def test_filtered_no_filters_returns_all(store):
    _setup_filtered(store)
    styles = store.get_styles_filtered()
    assert len(styles) == 4


def test_filtered_by_category(store):
    _setup_filtered(store)
    styles = store.get_styles_filtered(category_slugs=["female"])
    slugs = {s["slug"] for s in styles}
    assert slugs == {"s_female_bw", "s_female_sport"}


def test_filtered_by_category_male(store):
    _setup_filtered(store)
    styles = store.get_styles_filtered(category_slugs=["male"])
    slugs = {s["slug"] for s in styles}
    assert slugs == {"s_male_sport"}


def test_filtered_by_tag(store):
    """Тег без категории → фильтр только по тегу (без валидации через category_tags)."""
    _setup_filtered(store)
    styles = store.get_styles_filtered(tag_slugs=["sport"])
    slugs = {s["slug"] for s in styles}
    assert slugs == {"s_female_sport", "s_male_sport"}


def test_filtered_category_plus_tag(store):
    """female + sport → только s_female_sport (sport разрешён для female)."""
    _setup_filtered(store)
    styles = store.get_styles_filtered(category_slugs=["female"], tag_slugs=["sport"])
    slugs = {s["slug"] for s in styles}
    assert slugs == {"s_female_sport"}


def test_filtered_category_plus_invalid_tag(store):
    """male + bw → пусто (bw не разрешён для male)."""
    _setup_filtered(store)
    styles = store.get_styles_filtered(category_slugs=["male"], tag_slugs=["bw"])
    assert len(styles) == 0


def test_filtered_multiple_tags_or(store):
    """female + [bw, sport] → оба женских стиля (OR)."""
    _setup_filtered(store)
    styles = store.get_styles_filtered(category_slugs=["female"], tag_slugs=["bw", "sport"])
    slugs = {s["slug"] for s in styles}
    assert slugs == {"s_female_bw", "s_female_sport"}


def test_filtered_all_slug_ignored(store):
    """category_slugs=["all"] → как без фильтра."""
    _setup_filtered(store)
    styles = store.get_styles_filtered(category_slugs=["all"])
    assert len(styles) == 4


def test_filtered_nonexistent_slug(store):
    """Несуществующий slug → пустой результат, не ошибка."""
    _setup_filtered(store)
    styles = store.get_styles_filtered(category_slugs=["nonexistent"])
    assert len(styles) == 0


def test_filtered_inactive_style_excluded(store):
    """Неактивный стиль не попадает в результат."""
    data = _setup_filtered(store)
    store.update_express_style(data["s1"], is_active=False)
    styles = store.get_styles_filtered(category_slugs=["female"])
    slugs = {s["slug"] for s in styles}
    assert "s_female_bw" not in slugs


def test_filtered_inactive_category_excluded(store):
    """Стили неактивной категории не попадают."""
    data = _setup_filtered(store)
    store.update_express_category(data["c_female"], is_active=False)
    styles = store.get_styles_filtered(category_slugs=["female"])
    assert len(styles) == 0


# ========== Provider memory ==========

def test_set_and_get_provider(store):
    store.get_user(1)  # auto-create
    store.set_user_last_express_provider(1, "nano-banana-pro")
    assert store.get_user_last_express_provider(1) == "nano-banana-pro"


def test_provider_default_seedream(store):
    store.get_user(1)
    assert store.get_user_last_express_provider(1) == "seedream"


def test_provider_invalid_falls_back(store):
    store.get_user(1)
    store.set_user_last_express_provider(1, "invalid-provider")
    assert store.get_user_last_express_provider(1) == "seedream"


def test_provider_nonexistent_user(store):
    assert store.get_user_last_express_provider(99999) == "seedream"


def test_provider_switch_back_and_forth(store):
    store.get_user(1)
    store.set_user_last_express_provider(1, "nano-banana-pro")
    store.set_user_last_express_provider(1, "seedream")
    assert store.get_user_last_express_provider(1) == "seedream"


# ========== Delete cascade ==========

def test_delete_category_cascades_junctions(store):
    """Удаление категории каскадно удаляет junction записи."""
    cid = store.create_express_category("female", "Женские")
    tid = store.create_express_tag("bw", "ч/б")
    sid = _create_style(store, "style1")
    store.set_style_categories(sid, [cid])
    store.set_category_tags(cid, [tid])
    store.delete_express_category(cid)
    # Junction записи удалены
    assert store.get_style_categories(sid) == []
    assert store.get_category_tags(cid) == []


def test_delete_tag_cascades_junctions(store):
    """Удаление тега каскадно удаляет junction записи."""
    cid = store.create_express_category("female", "Женские")
    tid = store.create_express_tag("bw", "ч/б")
    sid = _create_style(store, "style1")
    store.set_style_tags(sid, [tid])
    store.set_category_tags(cid, [tid])
    store.delete_express_tag(tid)
    assert store.get_style_tags(sid) == []
    assert store.get_category_tags(cid) == []


# ========== Regression: SQLite + TABLE_PREFIX ==========

@pytest.fixture
def store_prefixed(tmp_path):
    """SQLite store с TABLE_PREFIX=dev_ — регресс-тест P1."""
    os.environ.pop("DATABASE_URL", None)
    os.environ["TABLE_PREFIX"] = "dev_"
    # Очистить кэш модуля чтобы TABLE_PREFIX перечитался
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("prismalab"):
            del sys.modules[mod_name]
    from prismalab.storage import PrismaLabStore
    db_file = str(tmp_path / "test_prefix.db")
    s = PrismaLabStore(db_path=db_file)
    s.init_admin_tables()
    # Восстановить TABLE_PREFIX для других тестов
    yield s
    os.environ["TABLE_PREFIX"] = ""
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("prismalab"):
            del sys.modules[mod_name]


def test_prefixed_get_user(store_prefixed):
    """get_user работает с TABLE_PREFIX=dev_ (таблица dev_users)."""
    user = store_prefixed.get_user(42)
    assert user is not None
    assert user.user_id == 42


def test_prefixed_provider_memory(store_prefixed):
    """set/get last_express_provider работает с TABLE_PREFIX=dev_."""
    store_prefixed.get_user(42)
    store_prefixed.set_user_last_express_provider(42, "nano-banana-pro")
    assert store_prefixed.get_user_last_express_provider(42) == "nano-banana-pro"


def test_prefixed_categories_crud(store_prefixed):
    """CRUD категорий работает с TABLE_PREFIX=dev_."""
    cid = store_prefixed.create_express_category("female", "Женские")
    assert cid is not None
    cat = store_prefixed.get_express_category(cid)
    assert cat["slug"] == "female"


def test_prefixed_decrement_persona_credits(store_prefixed):
    """decrement_persona_credits работает с TABLE_PREFIX=dev_."""
    store_prefixed.get_user(42)
    store_prefixed.set_persona_credits(42, 5)
    store_prefixed.decrement_persona_credits(42)
    user = store_prefixed.get_user(42)
    assert user.persona_credits_remaining == 4


def test_prefixed_log_payment(store_prefixed):
    """log_payment работает с TABLE_PREFIX=dev_."""
    store_prefixed.get_user(42)
    pid = store_prefixed.log_payment(
        user_id=42, payment_id="test_pay_1",
        payment_method="yookassa", product_type="express",
        credits=1, amount_rub=99.0,
    )
    assert pid is not None


def test_prefixed_log_event(store_prefixed):
    """log_event работает с TABLE_PREFIX=dev_."""
    store_prefixed.get_user(42)
    eid = store_prefixed.log_event(42, "generation", {"mode": "fast", "provider": "seedream"})
    assert eid is not None
