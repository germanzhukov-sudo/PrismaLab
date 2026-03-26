"""Фикстуры для тестов PrismaLab.

Используем временный SQLite файл — без обращения к продовой БД.
ВАЖНО: env переменные очищаются ДО любого импорта prismalab.
"""
from __future__ import annotations

import os
import sys
import tempfile

# Убираем DATABASE_URL ДО импорта — чтобы storage.py не подключился к проду
os.environ.pop("DATABASE_URL", None)
os.environ["TABLE_PREFIX"] = ""

# Если storage уже импортирован — удаляем из кэша чтобы перечитал env
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("prismalab"):
        del sys.modules[mod_name]

import pytest


@pytest.fixture
def store(tmp_path):
    """SQLite store во временном файле. Каждый тест получает чистую БД."""
    os.environ.pop("DATABASE_URL", None)
    if "prismalab.storage" in sys.modules:
        del sys.modules["prismalab.storage"]
    from prismalab.storage import PrismaLabStore
    db_file = str(tmp_path / "test.db")
    s = PrismaLabStore(db_path=db_file)
    s.init_admin_tables()
    return s
