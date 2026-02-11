#!/usr/bin/env python3
"""Скрипт для создания админа в PrismaLab.
Можно вызвать с переменными окружения (удобно на сервере):
  CREATE_ADMIN_USERNAME=admin CREATE_ADMIN_PASSWORD=твой_пароль python create_admin.py
"""
import sys
import os

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prismalab.storage import PrismaLabStore
from prismalab.admin.auth import hash_password


def main():
    print("=== Создание админа PrismaLab ===\n")

    username = (os.getenv("CREATE_ADMIN_USERNAME") or "").strip()
    password = (os.getenv("CREATE_ADMIN_PASSWORD") or "").strip()

    if not username or not password:
        username = input("Логин: ").strip()
        if not username:
            print("Ошибка: логин не может быть пустым")
            return
        password = input("Пароль: ").strip()

    if len(password) < 4:
        print("Ошибка: пароль должен быть минимум 4 символа")
        return

    display_name = (os.getenv("CREATE_ADMIN_DISPLAY_NAME") or "").strip() or username
    if not os.getenv("CREATE_ADMIN_USERNAME"):
        display_name = input("Имя для отображения (Enter = логин): ").strip() or username

    # Создаём хранилище и таблицы
    store = PrismaLabStore()
    store.init_admin_tables()

    # Хешируем пароль
    password_hash = hash_password(password)

    # Создаём админа
    admin_id = store.create_admin(username, password_hash, display_name)

    if admin_id:
        print(f"\n✅ Админ '{username}' создан (ID: {admin_id})")
        print(f"\nТеперь можешь войти в админку: http://localhost:8080/admin/")
    else:
        print(f"\n⚠️ Админ '{username}' уже существует")


if __name__ == "__main__":
    main()
