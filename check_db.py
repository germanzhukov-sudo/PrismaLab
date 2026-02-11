#!/usr/bin/env python3
"""Диагностика данных в БД для админки."""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL", "")
if "sslmode=" not in db_url:
    db_url += ("&" if "?" in db_url else "?") + "sslmode=require"

conn = psycopg2.connect(db_url)

with conn.cursor(cursor_factory=RealDictCursor) as cur:
    print("=" * 60)
    print("ДИАГНОСТИКА БАЗЫ ДАННЫХ")
    print("=" * 60)

    # 1. Users
    print("\n📊 USERS:")
    cur.execute("SELECT COUNT(*) as cnt FROM public.users")
    print(f"   Всего: {cur.fetchone()['cnt']}")

    cur.execute("SELECT COUNT(*) as cnt FROM public.users WHERE created_at IS NOT NULL")
    print(f"   С created_at: {cur.fetchone()['cnt']}")

    cur.execute("SELECT MIN(created_at) as min_dt, MAX(created_at) as max_dt FROM public.users")
    row = cur.fetchone()
    print(f"   Диапазон created_at: {row['min_dt']} - {row['max_dt']}")

    cur.execute("SELECT MIN(updated_at) as min_dt, MAX(updated_at) as max_dt FROM public.users")
    row = cur.fetchone()
    print(f"   Диапазон updated_at: {row['min_dt']} - {row['max_dt']}")

    # 2. Payments
    print("\n💳 PAYMENTS:")
    cur.execute("SELECT COUNT(*) as cnt FROM public.payments")
    print(f"   Всего: {cur.fetchone()['cnt']}")

    cur.execute("SELECT MIN(created_at) as min_dt, MAX(created_at) as max_dt FROM public.payments")
    row = cur.fetchone()
    print(f"   Диапазон: {row['min_dt']} - {row['max_dt']}")

    cur.execute("SELECT COUNT(*) as cnt, SUM(amount_rub) as total FROM public.payments WHERE created_at >= NOW() - INTERVAL '7 days'")
    row = cur.fetchone()
    print(f"   За 7 дней: {row['cnt']} платежей, {row['total']} руб")

    cur.execute("SELECT COUNT(*) as cnt, SUM(amount_rub) as total FROM public.payments WHERE created_at >= NOW() - INTERVAL '30 days'")
    row = cur.fetchone()
    print(f"   За 30 дней: {row['cnt']} платежей, {row['total']} руб")

    # Последние платежи
    print("\n   Последние 5 платежей:")
    cur.execute("SELECT created_at, amount_rub, product_type FROM public.payments ORDER BY created_at DESC LIMIT 5")
    for p in cur.fetchall():
        print(f"      {p['created_at']} | {p['amount_rub']} руб | {p['product_type']}")

    # 3. User events
    print("\n📝 USER_EVENTS:")
    cur.execute("SELECT COUNT(*) as cnt FROM public.user_events")
    print(f"   Всего: {cur.fetchone()['cnt']}")

    cur.execute("SELECT event_type, COUNT(*) as cnt FROM public.user_events GROUP BY event_type ORDER BY cnt DESC")
    for e in cur.fetchall():
        print(f"      {e['event_type']}: {e['cnt']}")

    cur.execute("SELECT MIN(created_at) as min_dt, MAX(created_at) as max_dt FROM public.user_events")
    row = cur.fetchone()
    if row['min_dt']:
        print(f"   Диапазон: {row['min_dt']} - {row['max_dt']}")
    else:
        print("   ⚠️ Таблица пустая!")

conn.close()
print("\n" + "=" * 60)
