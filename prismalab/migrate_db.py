#!/usr/bin/env python3
"""
Миграция: создаёт таблицу public.users в PostgreSQL (Supabase), если её ещё нет.
Запуск: python -m prismalab.migrate_db
Требуется DATABASE_URL в окружении.
"""
from __future__ import annotations

import os
import sys


def _pg_url_with_ssl(url: str) -> str:
    if not url or "sslmode=" in url:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}sslmode=require"


def main() -> int:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        print("DATABASE_URL не задан, миграция пропущена.", file=sys.stderr)
        return 0

    db_url = _pg_url_with_ssl(url)
    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS public.users (
                user_id BIGINT PRIMARY KEY,
                personal_model_version TEXT,
                personal_trigger_word TEXT,
                training_id TEXT,
                training_status TEXT,
                astria_tune_id TEXT,
                astria_lora_tune_id TEXT,
                free_generation_used INTEGER DEFAULT 0,
                paid_generations_remaining INTEGER DEFAULT 0,
                subject_gender TEXT,
                persona_credits_remaining INTEGER DEFAULT 0,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("Таблица public.users проверена/создана.")
        return 0
    except Exception as e:
        print(f"Ошибка миграции: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
