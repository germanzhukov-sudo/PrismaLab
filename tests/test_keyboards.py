"""Snapshot-тесты клавиатур — проверяем callback_data не сломаны."""
from __future__ import annotations

import os

os.environ.pop("DATABASE_URL", None)
os.environ["PRISMALAB_DB_PATH"] = ":memory:"


def _extract_callback_data(keyboard) -> list[str]:
    """Извлекает все callback_data из InlineKeyboardMarkup."""
    result = []
    for row in keyboard.inline_keyboard:
        for btn in row:
            if btn.callback_data:
                result.append(btn.callback_data)
    return result


def test_fast_tariff_keyboard_callbacks():
    """Кнопки экспресс-тарифов имеют правильные callback_data."""
    from prismalab.bot import _fast_tariff_keyboard
    kb = _fast_tariff_keyboard()
    cbs = _extract_callback_data(kb)
    assert "pl_fast_buy:5" in cbs
    assert "pl_fast_buy:10" in cbs
    assert "pl_fast_buy:30" in cbs


def test_fast_gender_keyboard_callbacks():
    """Кнопки выбора пола имеют правильные callback_data."""
    from prismalab.bot import _fast_gender_keyboard
    kb = _fast_gender_keyboard()
    cbs = _extract_callback_data(kb)
    assert "pl_fast_gender:female" in cbs
    assert "pl_fast_gender:male" in cbs


def test_persona_tariff_keyboard_callbacks():
    """Кнопки тарифов персоны имеют правильные callback_data."""
    from prismalab.bot import _persona_tariff_keyboard
    kb = _persona_tariff_keyboard()
    cbs = _extract_callback_data(kb)
    assert any("pl_persona_buy:" in cb for cb in cbs)
    assert "pl_persona_back" in cbs


def test_persona_rules_message_not_empty():
    """Текст правил персоны не пустой."""
    from prismalab.bot import PERSONA_RULES_MESSAGE
    assert isinstance(PERSONA_RULES_MESSAGE, str)
    assert len(PERSONA_RULES_MESSAGE) > 50


def test_tariffs_message_not_empty():
    """Текст тарифов не пустой."""
    from prismalab.bot import TARIFFS_MESSAGE
    assert isinstance(TARIFFS_MESSAGE, str)
    assert len(TARIFFS_MESSAGE) > 50
