#!/usr/bin/env python3
"""Скрипт удаления пользователя из БД. Использование: python delete_user.py USER_ID"""
import os
import sys
from pathlib import Path

import dotenv
dotenv.load_dotenv(Path(__file__).resolve().parent / ".env")

USER_ID = 94258157


def _pg_url(url: str) -> str:
    if not url or "sslmode=" in url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}sslmode=require"


def main() -> None:
    user_id = int(sys.argv[1]) if len(sys.argv) > 1 else USER_ID
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        print("DATABASE_URL не задан")
        sys.exit(1)
    import psycopg2
    conn = psycopg2.connect(_pg_url(url))
    cur = conn.cursor()
    cur.execute("DELETE FROM public.payments WHERE user_id = %s", (user_id,))
    payments = cur.rowcount
    cur.execute("DELETE FROM public.user_events WHERE user_id = %s", (user_id,))
    events = cur.rowcount
    cur.execute("DELETE FROM support_reply_map WHERE user_id = %s", (user_id,))
    support = cur.rowcount
    cur.execute("DELETE FROM public.users WHERE user_id = %s", (user_id,))
    users = cur.rowcount
    conn.commit()
    conn.close()
    print(f"Удалено: users={users}, payments={payments}, events={events}, support_reply_map={support}")


if __name__ == "__main__":
    main()
