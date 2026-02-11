#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('prismalab.db')
cursor = conn.cursor()

# Проверяем таблицы
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Таблицы в базе:", [t[0] for t in tables])

# Считаем пользователей
cursor.execute("SELECT COUNT(*) FROM users")
count = cursor.fetchone()[0]
print(f"\nВсего пользователей в базе: {count}")

# Показываем всех пользователей
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
if rows:
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]
    print("\nКолонки таблицы:", columns)
    print("\nВсе пользователи:")
    for row in rows:
        user_dict = dict(zip(columns, row))
        print(f"  User ID: {user_dict.get('user_id')}")
        print(f"    - Astria FaceID tune: {user_dict.get('astria_tune_id')}")
        print(f"    - Astria LoRA tune: {user_dict.get('astria_lora_tune_id')}")
        print(f"    - Personal model: {user_dict.get('personal_model_version')}")
        print(f"    - Updated: {user_dict.get('updated_at')}")
        print()

conn.close()
