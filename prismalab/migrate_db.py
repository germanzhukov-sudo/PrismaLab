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
                astria_lora_tune_id_pending TEXT,
                astria_lora_pack_tune_id TEXT,
                astria_lora_pack_tune_id_pending TEXT,
                free_generation_used INTEGER DEFAULT 0,
                paid_generations_remaining INTEGER DEFAULT 0,
                subject_gender TEXT,
                persona_credits_remaining INTEGER DEFAULT 0,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        conn.commit()
        try:
            cur.execute("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS astria_lora_tune_id_pending TEXT")
            conn.commit()
        except Exception:
            conn.rollback()
        try:
            cur.execute("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS astria_lora_pack_tune_id TEXT")
            conn.commit()
        except Exception:
            conn.rollback()
        try:
            cur.execute("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS astria_lora_pack_tune_id_pending TEXT")
            conn.commit()
        except Exception:
            conn.rollback()
        try:
            cur.execute("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS pending_pack_id INTEGER")
            conn.commit()
        except Exception:
            conn.rollback()
        try:
            cur.execute("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS persona_lora_class_name TEXT")
            conn.commit()
        except Exception:
            conn.rollback()
        # Таблица для восстановления прерванных pack runs (бот рестарт во время обучения pack tune)
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.pending_pack_runs (
                    user_id BIGINT PRIMARY KEY,
                    pack_id INTEGER NOT NULL,
                    chat_id BIGINT NOT NULL,
                    run_id TEXT NOT NULL,
                    expected INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    offer_title TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            conn.commit()
            cur.execute("ALTER TABLE public.pending_pack_runs ADD COLUMN IF NOT EXISTS offer_title TEXT")
            conn.commit()
        except Exception:
            conn.rollback()
        cur.close()
        conn.close()
        print("Таблица public.users проверена/создана.")
        return 0
    except Exception as e:
        print(f"Ошибка миграции: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
