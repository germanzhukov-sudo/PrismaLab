"""
Хранилище PrismaLab: SQLite (локально) или PostgreSQL/Supabase (DATABASE_URL).
Если задана переменная окружения DATABASE_URL — используется Supabase (PostgreSQL).
"""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("prismalab.storage")
DATABASE_URL = os.getenv("DATABASE_URL")
TABLE_PREFIX = os.getenv("TABLE_PREFIX", "")  # "" для прода, "dev_" для dev


def _pg_url_with_ssl(url: str) -> str:
    """Добавляет sslmode=require к URL Supabase, если ещё не задан."""
    if not url or "sslmode=" in url:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}sslmode=require"


class _PooledConnection:
    """Контекст-менеджер: входит — отдаёт conn, выходит — возвращает conn в пул."""

    def __init__(self, pool: Any, conn: Any) -> None:
        self._pool = pool
        self._conn = conn

    def __enter__(self) -> Any:
        return self._conn

    def __exit__(self, *args: Any) -> None:
        self._pool.putconn(self._conn)


@dataclass(frozen=True)
class UserProfile:
    user_id: int
    personal_model_version: str | None
    personal_trigger_word: str | None
    training_id: str | None
    training_status: str | None
    astria_tune_id: str | None  # FaceID tune (1 фото)
    astria_lora_tune_id: str | None  # LoRA tune (10+ фото)
    astria_lora_tune_id_pending: str | None  # LoRA tune в процессе обучения (сохраняем сразу при создании)
    persona_lora_class_name: str | None  # "person" | "woman" | "man" — trigger token для inference
    astria_lora_pack_tune_id: str | None  # LoRA tune для pack-flow (обычно woman/man)
    astria_lora_pack_tune_id_pending: str | None  # pack LoRA tune в процессе обучения
    free_generation_used: bool  # потрачена ли 1 бесплатная генерация
    paid_generations_remaining: int  # остаток купленных генераций
    persona_credits_remaining: int  # кредиты Персоны (10 или 20 после оплаты)
    subject_gender: str | None  # "male" | "female" для Быстрое фото (смена потом в профиле)


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if hasattr(row, "keys") and key not in row.keys():
        return default
    try:
        return row[key] if row is not None else default
    except (KeyError, IndexError, TypeError):
        return default


def _default_db_path() -> str:
    return os.getenv("PRISMALAB_DB_PATH") or "prismalab.db"


class PrismaLabStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = str(db_path if db_path is not None else _default_db_path())
        self._use_pg = bool(DATABASE_URL and DATABASE_URL.strip())
        self._prefix = TABLE_PREFIX
        # Имена таблиц с учётом префикса
        self._users_table = f"public.{self._prefix}users" if self._use_pg else f"{self._prefix}users"
        self._payments_table = f"public.{self._prefix}payments" if self._use_pg else f"{self._prefix}payments"
        self._user_events_table = f"public.{self._prefix}user_events" if self._use_pg else f"{self._prefix}user_events"
        self._admins_table = f"public.{self._prefix}admins" if self._use_pg else f"{self._prefix}admins"
        self._admin_settings_table = f"public.{self._prefix}admin_settings" if self._use_pg else f"{self._prefix}admin_settings"
        self._admin_pack_costs_table = f"public.{self._prefix}admin_pack_costs" if self._use_pg else f"{self._prefix}admin_pack_costs"
        self._pending_pack_runs_table = f"public.{self._prefix}pending_pack_runs" if self._use_pg else f"{self._prefix}pending_pack_runs"
        self._persona_styles_table = f"public.{self._prefix}persona_styles" if self._use_pg else f"{self._prefix}persona_styles"
        self._express_styles_table = f"public.{self._prefix}express_styles" if self._use_pg else f"{self._prefix}express_styles"
        self._express_categories_table = f"public.{self._prefix}express_categories" if self._use_pg else f"{self._prefix}express_categories"
        self._express_tags_table = f"public.{self._prefix}express_tags" if self._use_pg else f"{self._prefix}express_tags"
        self._express_style_categories_table = f"public.{self._prefix}express_style_categories" if self._use_pg else f"{self._prefix}express_style_categories"
        self._express_style_tags_table = f"public.{self._prefix}express_style_tags" if self._use_pg else f"{self._prefix}express_style_tags"
        self._express_category_tags_table = f"public.{self._prefix}express_category_tags" if self._use_pg else f"{self._prefix}express_category_tags"
        self._generation_history_table = f"public.{self._prefix}generation_history" if self._use_pg else f"{self._prefix}generation_history"
        self._generation_requests_table = f"public.{self._prefix}generation_requests" if self._use_pg else f"{self._prefix}generation_requests"
        self._persona_style_previews_table = f"public.{self._prefix}persona_style_previews" if self._use_pg else f"{self._prefix}persona_style_previews"
        self._pg_pool = None
        if self._use_pg:
            from psycopg2.pool import ThreadedConnectionPool
            db_url = _pg_url_with_ssl(DATABASE_URL)
            self._pg_pool = ThreadedConnectionPool(
                minconn=1, maxconn=50, dsn=db_url
            )
            logger.info("База данных: PostgreSQL (Supabase)")
        else:
            logger.info("База данных: SQLite (локальный файл %s)", self.db_path)
        self._init()

    def _connect_sqlite(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _connect_pg(self):
        """Возвращает контекст-менеджер: при выходе соединение возвращается в пул."""
        conn = self._pg_pool.getconn()
        return _PooledConnection(self._pg_pool, conn)

    def _connect(self):
        if self._use_pg:
            return self._connect_pg()
        return self._connect_sqlite()

    def _execute(self, conn, sql: str, params: tuple = ()):
        if self._use_pg:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
        else:
            return conn.execute(sql.replace("%s", "?"), params)

    def _init(self) -> None:
        if self._use_pg:
            self._init_pg()
        else:
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        with self._connect() as conn:
            self._execute(
                conn,
                f"""
                CREATE TABLE IF NOT EXISTS {self._users_table} (
                    user_id INTEGER PRIMARY KEY,
                    personal_model_version TEXT,
                    personal_trigger_word TEXT,
                    training_id TEXT,
                    training_status TEXT,
                    astria_tune_id TEXT,
                    astria_lora_tune_id TEXT,
                    astria_lora_pack_tune_id TEXT,
                    astria_lora_pack_tune_id_pending TEXT,
                    free_generation_used INTEGER DEFAULT 0,
                    paid_generations_remaining INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """,
            )
            for col, col_type in [
                ("astria_tune_id", "TEXT"),
                ("astria_lora_tune_id", "TEXT"),
                ("astria_lora_tune_id_pending", "TEXT"),
                ("persona_lora_class_name", "TEXT"),
                ("astria_lora_pack_tune_id", "TEXT"),
                ("astria_lora_pack_tune_id_pending", "TEXT"),
                ("free_generation_used", "INTEGER DEFAULT 0"),
                ("paid_generations_remaining", "INTEGER DEFAULT 0"),
                ("persona_credits_remaining", "INTEGER DEFAULT 0"),
                ("subject_gender", "TEXT"),
                ("pending_pack_id", "INTEGER"),
                ("pending_persona_batch", "TEXT"),
            ]:
                try:
                    self._execute(conn, f"ALTER TABLE {self._users_table} ADD COLUMN {col} {col_type}")
                except (sqlite3.OperationalError, Exception):
                    pass
            # Таблица для восстановления прерванных pack runs
            try:
                self._execute(
                    conn,
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._pending_pack_runs_table} (
                        user_id INTEGER PRIMARY KEY,
                        pack_id INTEGER NOT NULL,
                        chat_id INTEGER NOT NULL,
                        run_id TEXT NOT NULL,
                        expected INTEGER NOT NULL,
                        class_name TEXT NOT NULL,
                        offer_title TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """,
                )
            except (sqlite3.OperationalError, Exception):
                pass
            # Таблица admin_settings (для тарифов и настроек)
            try:
                self._execute(
                    conn,
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._admin_settings_table} (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """,
                )
            except (sqlite3.OperationalError, Exception):
                pass
            if not self._use_pg:
                conn.commit()

    def _init_pg(self) -> None:
        with self._connect() as conn:
            self._execute(
                conn,
                f"""
                CREATE TABLE IF NOT EXISTS {self._users_table} (
                    user_id BIGINT PRIMARY KEY,
                    personal_model_version TEXT,
                    personal_trigger_word TEXT,
                    training_id TEXT,
                    training_status TEXT,
                    astria_tune_id TEXT,
                    astria_lora_tune_id TEXT,
                    astria_lora_pack_tune_id TEXT,
                    astria_lora_pack_tune_id_pending TEXT,
                    free_generation_used INTEGER DEFAULT 0,
                    paid_generations_remaining INTEGER DEFAULT 0,
                    subject_gender TEXT,
                    persona_credits_remaining INTEGER DEFAULT 0,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
                """,
            )
            conn.commit()
            for add_col in [
                ("persona_credits_remaining", "INTEGER DEFAULT 0"),
                ("astria_lora_tune_id_pending", "TEXT"),
                ("persona_lora_class_name", "TEXT"),
                ("astria_lora_pack_tune_id", "TEXT"),
                ("astria_lora_pack_tune_id_pending", "TEXT"),
                ("pending_pack_id", "INTEGER"),
                ("pending_persona_batch", "TEXT"),
            ]:
                try:
                    self._execute(conn, f"ALTER TABLE {self._users_table} ADD COLUMN {add_col[0]} {add_col[1]}")
                    conn.commit()
                except Exception:
                    conn.rollback()
            # Таблица настроек (себестоимость, курс)
            try:
                self._execute(
                    conn,
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._admin_settings_table} (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """,
                )
                conn.commit()
            except Exception:
                conn.rollback()
            # Таблица себестоимости паков
            try:
                self._execute(
                    conn,
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._admin_pack_costs_table} (
                        pack_id INTEGER PRIMARY KEY,
                        pack_title TEXT,
                        cost_usd DECIMAL(10,4) DEFAULT 0,
                        credit_cost INTEGER DEFAULT NULL
                    )
                    """,
                )
                conn.commit()
            except Exception:
                conn.rollback()
            # Migration: add credit_cost column if missing
            try:
                self._execute(conn, f"ALTER TABLE {self._admin_pack_costs_table} ADD COLUMN IF NOT EXISTS credit_cost INTEGER DEFAULT NULL")
                conn.commit()
            except Exception:
                conn.rollback()
            logger.info("Таблица %s проверена/создана", self._users_table)

    def get_user(self, user_id: int) -> UserProfile:
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"SELECT * FROM {self._users_table} WHERE user_id = %s", (int(user_id),))
                    row = cur.fetchone()
            else:
                row = conn.execute(
                    f"SELECT * FROM {self._users_table} WHERE user_id = ?", (int(user_id),)
                ).fetchone()
            if not row:
                return UserProfile(
                    user_id=int(user_id),
                    personal_model_version=None,
                    personal_trigger_word=None,
                    training_id=None,
                    training_status=None,
                    astria_tune_id=None,
                    astria_lora_tune_id=None,
                    astria_lora_tune_id_pending=None,
                    persona_lora_class_name=None,
                    astria_lora_pack_tune_id=None,
                    astria_lora_pack_tune_id_pending=None,
                    free_generation_used=False,
                    paid_generations_remaining=0,
                    persona_credits_remaining=0,
                    subject_gender=None,
                )
            def _bool(key: str) -> bool:
                v = _row_get(row, key)
                return bool(v) if v is not None else False
            def _int(key: str) -> int:
                v = _row_get(row, key)
                return int(v) if v is not None else 0
            return UserProfile(
                user_id=int(_row_get(row, "user_id", user_id)),
                personal_model_version=_row_get(row, "personal_model_version"),
                personal_trigger_word=_row_get(row, "personal_trigger_word"),
                training_id=_row_get(row, "training_id"),
                training_status=_row_get(row, "training_status"),
                astria_tune_id=_row_get(row, "astria_tune_id"),
                astria_lora_tune_id=_row_get(row, "astria_lora_tune_id"),
                astria_lora_tune_id_pending=_row_get(row, "astria_lora_tune_id_pending"),
                persona_lora_class_name=_row_get(row, "persona_lora_class_name") or None,
                astria_lora_pack_tune_id=_row_get(row, "astria_lora_pack_tune_id"),
                astria_lora_pack_tune_id_pending=_row_get(row, "astria_lora_pack_tune_id_pending"),
                free_generation_used=_bool("free_generation_used"),
                paid_generations_remaining=_int("paid_generations_remaining"),
                persona_credits_remaining=_int("persona_credits_remaining"),
                subject_gender=_row_get(row, "subject_gender") or None,
            )

    def _run(self, sql: str, params: tuple = ()) -> None:
        with self._connect() as conn:
            self._execute(conn, sql, params)
            conn.commit()

    # ========== Настройки себестоимости ==========

    def get_cost_settings(self) -> dict:
        """Получить настройки себестоимости."""
        defaults = {
            "cost_persona_create": 1.5,
            "cost_fast_photo": 0.035,
            "cost_nano_banana": 0.035,
            "cost_persona_photo": 0.03,
            "usd_rub": 90.0,
        }
        if not self._use_pg:
            return defaults

        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"SELECT key, value FROM {self._admin_settings_table} WHERE key LIKE 'cost_%' OR key = 'usd_rub'")
                for row in cur.fetchall():
                    key = row["key"]
                    if key in defaults:
                        try:
                            defaults[key] = float(row["value"])
                        except (ValueError, TypeError):
                            pass
        return defaults

    def set_cost_settings(self, settings: dict) -> None:
        """Сохранить настройки себестоимости."""
        if not self._use_pg:
            return

        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for key, value in settings.items():
                    if key in ("cost_persona_create", "cost_fast_photo", "cost_nano_banana", "cost_persona_photo", "usd_rub"):
                        cur.execute(f"""
                            INSERT INTO {self._admin_settings_table} (key, value, updated_at)
                            VALUES (%s, %s, NOW())
                            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                        """, (key, str(value)))
                conn.commit()

    def get_admin_setting(self, key: str) -> str | None:
        """Получить одну настройку по точному ключу."""
        result = self.get_admin_settings_by_prefix(key)
        return result.get(key)

    def get_admin_settings_by_prefix(self, prefix: str) -> dict[str, str]:
        """Получить все admin_settings с данным префиксом ключа. Возвращает {key: value}."""
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"SELECT key, value FROM {self._admin_settings_table} WHERE key LIKE %s",
                        (prefix + "%",),
                    )
                    return {row["key"]: row["value"] for row in cur.fetchall()}
            else:
                cur = self._execute(conn, f"SELECT key, value FROM {self._admin_settings_table} WHERE key LIKE ?", (prefix + "%",))
                return {row[0]: row[1] for row in cur.fetchall()}

    def set_admin_setting(self, key: str, value: str) -> None:
        """Upsert одной записи в admin_settings."""
        with self._connect() as conn:
            if self._use_pg:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        INSERT INTO {self._admin_settings_table} (key, value, updated_at)
                        VALUES (%s, %s, NOW())
                        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                    """, (key, value))
            else:
                self._execute(conn, f"""
                    INSERT INTO {self._admin_settings_table} (key, value, updated_at)
                    VALUES (?, ?, datetime('now'))
                    ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = datetime('now')
                """, (key, value))
            conn.commit()

    def delete_admin_setting(self, key: str) -> None:
        """Удалить запись из admin_settings (сброс override к дефолту)."""
        with self._connect() as conn:
            if self._use_pg:
                with conn.cursor() as cur:
                    cur.execute(f"DELETE FROM {self._admin_settings_table} WHERE key = %s", (key,))
            else:
                self._execute(conn, f"DELETE FROM {self._admin_settings_table} WHERE key = ?", (key,))
            conn.commit()

    def set_admin_settings_bulk(self, settings: dict[str, str]) -> None:
        """Upsert нескольких записей в admin_settings за одну транзакцию."""
        if not settings:
            return
        with self._connect() as conn:
            if self._use_pg:
                with conn.cursor() as cur:
                    for key, value in settings.items():
                        cur.execute(f"""
                            INSERT INTO {self._admin_settings_table} (key, value, updated_at)
                            VALUES (%s, %s, NOW())
                            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                        """, (key, str(value)))
            else:
                for key, value in settings.items():
                    self._execute(conn, f"""
                        INSERT INTO {self._admin_settings_table} (key, value, updated_at)
                        VALUES (?, ?, datetime('now'))
                        ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = datetime('now')
                    """, (key, str(value)))
            conn.commit()

    @staticmethod
    def _calculate_fast_costs(
        fast_seedream: int,
        fast_nano: int,
        cost_fast_photo: float,
        cost_nano_banana: float,
        usd_rub: float,
    ) -> dict:
        """Расчёт себестоимости fast-генераций с разбивкой по провайдерам."""
        cost_seedream_rub = float(fast_seedream) * float(cost_fast_photo) * float(usd_rub)
        cost_nano_rub = float(fast_nano) * float(cost_nano_banana) * float(usd_rub)
        return {
            "fast_seedream": round(cost_seedream_rub, 2),
            "fast_nano": round(cost_nano_rub, 2),
            "fast_total": round(cost_seedream_rub + cost_nano_rub, 2),
        }

    # --- Себестоимость паков ---

    def get_pack_costs(self) -> list[dict]:
        """Получить себестоимость всех паков из admin_pack_costs."""
        if not self._use_pg:
            return []
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"SELECT pack_id, pack_title, cost_usd, credit_cost FROM {self._admin_pack_costs_table} ORDER BY pack_id")
                return [dict(row) for row in cur.fetchall()]

    def get_pack_costs_map(self) -> dict[int, float]:
        """Получить словарь {pack_id: cost_usd} для расчёта себестоимости."""
        rows = self.get_pack_costs()
        return {int(r["pack_id"]): float(r["cost_usd"] or 0) for r in rows}

    def get_pack_credit_costs_map(self) -> dict[int, int]:
        """Получить словарь {pack_id: credit_cost} — только admin overrides."""
        rows = self.get_pack_costs()
        return {int(r["pack_id"]): int(r["credit_cost"]) for r in rows if r.get("credit_cost") is not None}

    def set_pack_costs_bulk(self, items: list[dict]) -> None:
        """Массовое обновление себестоимости паков. items = [{pack_id, pack_title, cost_usd}, ...]"""
        if not self._use_pg or not items:
            return
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for item in items:
                    cc = item.get("credit_cost")
                    cc_val = int(cc) if cc is not None and str(cc).strip() else None
                    cur.execute(f"""
                        INSERT INTO {self._admin_pack_costs_table} (pack_id, pack_title, cost_usd, credit_cost)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (pack_id) DO UPDATE SET
                            pack_title = EXCLUDED.pack_title,
                            cost_usd = EXCLUDED.cost_usd,
                            credit_cost = EXCLUDED.credit_cost
                    """, (int(item["pack_id"]), str(item.get("pack_title", "")), float(item.get("cost_usd", 0)), cc_val))
                conn.commit()

    def get_pack_stats(self, date_from: str | None = None, date_to: str | None = None) -> list[dict]:
        """Статистика по пакам: pack_id, pack_title, generations (количество завершённых)."""
        if not self._use_pg:
            return []
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if date_from and date_to:
                    d_from = str(date_from) + " 00:00:00+03"
                    d_to = str(date_to) + " 23:59:59+03"
                    cur.execute(f"""
                        SELECT event_data->>'pack_id' as pack_id,
                               MAX(event_data->>'pack_title') as pack_title,
                               COUNT(*) as generations,
                               COALESCE(SUM((event_data->>'images_sent')::int), 0) as total_images
                        FROM {self._user_events_table}
                        WHERE event_type IN ('pack_generation', 'pack_callback', 'pack_fallback')
                          AND created_at >= %s AND created_at <= %s
                        GROUP BY event_data->>'pack_id'
                        ORDER BY generations DESC
                    """, (d_from, d_to))
                else:
                    cur.execute(f"""
                        SELECT event_data->>'pack_id' as pack_id,
                               MAX(event_data->>'pack_title') as pack_title,
                               COUNT(*) as generations,
                               COALESCE(SUM((event_data->>'images_sent')::int), 0) as total_images
                        FROM {self._user_events_table}
                        WHERE event_type IN ('pack_generation', 'pack_callback', 'pack_fallback')
                        GROUP BY event_data->>'pack_id'
                        ORDER BY generations DESC
                    """)
                return [dict(row) for row in cur.fetchall()]

    def upsert_training(
        self,
        *,
        user_id: int,
        training_id: str | None,
        training_status: str | None,
        trigger_word: str | None = None,
    ) -> None:
        self._run(
            f"""
            INSERT INTO {self._users_table} (user_id, training_id, training_status, personal_trigger_word, updated_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                training_id = EXCLUDED.training_id,
                training_status = EXCLUDED.training_status,
                personal_trigger_word = COALESCE(EXCLUDED.personal_trigger_word, {self._users_table}.personal_trigger_word),
                updated_at = CURRENT_TIMESTAMP
            """,
            (int(user_id), training_id, training_status, trigger_word),
        )

    def set_personal_model(
        self,
        *,
        user_id: int,
        model_version: str,
        trigger_word: str,
    ) -> None:
        self._run(
            f"""
            INSERT INTO {self._users_table} (user_id, personal_model_version, personal_trigger_word, training_status, updated_at)
            VALUES (%s, %s, %s, 'succeeded', CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                personal_model_version = EXCLUDED.personal_model_version,
                personal_trigger_word = EXCLUDED.personal_trigger_word,
                training_status = 'succeeded',
                updated_at = CURRENT_TIMESTAMP
            """,
            (int(user_id), str(model_version), str(trigger_word)),
        )

    def set_astria_tune(self, *, user_id: int, tune_id: str | None) -> None:
        if tune_id is None:
            self._run(
                f"UPDATE {self._users_table} SET astria_tune_id = NULL, updated_at = CURRENT_TIMESTAMP WHERE user_id = %s",
                (int(user_id),),
            )
        else:
            self._run(
                f"""
                INSERT INTO {self._users_table} (user_id, astria_tune_id, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    astria_tune_id = EXCLUDED.astria_tune_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (int(user_id), str(tune_id)),
            )

    def set_astria_lora_tune(
        self, *, user_id: int, tune_id: str | None, class_name: str | None = None
    ) -> None:
        if tune_id is None:
            self._run(
                f"UPDATE {self._users_table} SET astria_lora_tune_id = NULL, astria_lora_tune_id_pending = NULL, persona_lora_class_name = NULL, astria_lora_pack_tune_id = NULL, astria_lora_pack_tune_id_pending = NULL, updated_at = CURRENT_TIMESTAMP WHERE user_id = %s",
                (int(user_id),),
            )
        else:
            if class_name:
                self._run(
                    f"""
                    INSERT INTO {self._users_table} (user_id, astria_lora_tune_id, astria_lora_tune_id_pending, persona_lora_class_name, astria_lora_pack_tune_id, astria_lora_pack_tune_id_pending, updated_at)
                    VALUES (%s, %s, NULL, %s, NULL, NULL, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                        astria_lora_tune_id = EXCLUDED.astria_lora_tune_id,
                        astria_lora_tune_id_pending = NULL,
                        persona_lora_class_name = EXCLUDED.persona_lora_class_name,
                        astria_lora_pack_tune_id = NULL,
                        astria_lora_pack_tune_id_pending = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (int(user_id), str(tune_id), str(class_name)),
                )
            else:
                self._run(
                    f"""
                    INSERT INTO {self._users_table} (user_id, astria_lora_tune_id, astria_lora_tune_id_pending, astria_lora_pack_tune_id, astria_lora_pack_tune_id_pending, updated_at)
                    VALUES (%s, %s, NULL, NULL, NULL, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                        astria_lora_tune_id = EXCLUDED.astria_lora_tune_id,
                        astria_lora_tune_id_pending = NULL,
                        astria_lora_pack_tune_id = NULL,
                        astria_lora_pack_tune_id_pending = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (int(user_id), str(tune_id)),
                )

    def set_astria_lora_pack_tune(self, *, user_id: int, tune_id: str | None) -> None:
        if tune_id is None:
            self._run(
                f"UPDATE {self._users_table} SET astria_lora_pack_tune_id = NULL, astria_lora_pack_tune_id_pending = NULL, updated_at = CURRENT_TIMESTAMP WHERE user_id = %s",
                (int(user_id),),
            )
        else:
            self._run(
                f"""
                INSERT INTO {self._users_table} (user_id, astria_lora_pack_tune_id, astria_lora_pack_tune_id_pending, updated_at)
                VALUES (%s, %s, NULL, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    astria_lora_pack_tune_id = EXCLUDED.astria_lora_pack_tune_id,
                    astria_lora_pack_tune_id_pending = NULL,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (int(user_id), str(tune_id)),
            )

    def set_astria_lora_pack_tune_pending(self, *, user_id: int, tune_id: str) -> None:
        self._run(
            f"""
            INSERT INTO {self._users_table} (user_id, astria_lora_pack_tune_id_pending, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                astria_lora_pack_tune_id_pending = EXCLUDED.astria_lora_pack_tune_id_pending,
                updated_at = CURRENT_TIMESTAMP
            """,
            (int(user_id), str(tune_id)),
        )

    def clear_astria_lora_pack_tune_pending(self, user_id: int) -> None:
        self._run(
            f"UPDATE {self._users_table} SET astria_lora_pack_tune_id_pending = NULL, updated_at = CURRENT_TIMESTAMP WHERE user_id = %s",
            (int(user_id),),
        )

    def set_astria_lora_tune_pending(self, *, user_id: int, tune_id: str) -> None:
        """Сохраняет tune_id сразу после создания (до завершения обучения). Защита от потери при рестарте бота."""
        self._run(
            f"""
            INSERT INTO {self._users_table} (user_id, astria_lora_tune_id_pending, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                astria_lora_tune_id_pending = EXCLUDED.astria_lora_tune_id_pending,
                updated_at = CURRENT_TIMESTAMP
            """,
            (int(user_id), str(tune_id)),
        )

    def set_persona_lora_class_name(self, *, user_id: int, class_name: str | None) -> None:
        """Сохраняет trigger token (person/woman/man) для LoRA inference."""
        self._run(
            f"""
            INSERT INTO {self._users_table} (user_id, persona_lora_class_name, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                persona_lora_class_name = EXCLUDED.persona_lora_class_name,
                updated_at = CURRENT_TIMESTAMP
            """,
            (int(user_id), str(class_name) if class_name else None),
        )

    def clear_astria_lora_tune_pending(self, user_id: int) -> None:
        """Очищает pending tune (при ошибке обучения или после успешного завершения)."""
        self._run(
            f"UPDATE {self._users_table} SET astria_lora_tune_id_pending = NULL, updated_at = CURRENT_TIMESTAMP WHERE user_id = %s",
            (int(user_id),),
        )

    def spend_free_generation(self, user_id: int) -> None:
        self._run(
            f"""
            INSERT INTO {self._users_table} (user_id, free_generation_used, updated_at)
            VALUES (%s, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                free_generation_used = 1,
                updated_at = CURRENT_TIMESTAMP
            """,
            (int(user_id),),
        )

    def set_paid_generations_remaining(self, user_id: int, count: int) -> None:
        self._run(
            f"""
            INSERT INTO {self._users_table} (user_id, paid_generations_remaining, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                paid_generations_remaining = EXCLUDED.paid_generations_remaining,
                updated_at = CURRENT_TIMESTAMP
            """,
            (int(user_id), max(0, int(count))),
        )

    def reset_fast_generations(self, user_id: int, *, paid: int = 10) -> None:
        """Сброс генераций Быстрое фото: free не потрачена, paid = заданное значение."""
        self._run(
            f"""
            INSERT INTO {self._users_table} (user_id, free_generation_used, paid_generations_remaining, updated_at)
            VALUES (%s, 0, %s, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                free_generation_used = 0,
                paid_generations_remaining = EXCLUDED.paid_generations_remaining,
                updated_at = CURRENT_TIMESTAMP
            """,
            (int(user_id), max(0, int(paid))),
        )

    def set_persona_credits(self, user_id: int, count: int) -> None:
        self._run(
            f"""
            INSERT INTO {self._users_table} (user_id, persona_credits_remaining, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                persona_credits_remaining = EXCLUDED.persona_credits_remaining,
                updated_at = CURRENT_TIMESTAMP
            """,
            (int(user_id), max(0, int(count))),
        )

    def decrement_persona_credits(self, user_id: int, count: int = 1) -> int:
        """Атомарно вычитает count кредитов. Возвращает новый баланс или 0, если недостаточно."""
        count = max(1, count)
        with self._connect() as conn:
            uid = int(user_id)
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        UPDATE {self._users_table}
                        SET persona_credits_remaining = persona_credits_remaining - %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = %s AND persona_credits_remaining >= %s
                        RETURNING persona_credits_remaining
                        """,
                        (count, uid, count),
                    )
                    row = cur.fetchone()
                conn.commit()
                return int(row["persona_credits_remaining"]) if row else 0
            else:
                cur = conn.execute(
                    f"UPDATE {self._users_table} SET persona_credits_remaining = persona_credits_remaining - ?, updated_at = CURRENT_TIMESTAMP "
                    f"WHERE user_id = ? AND persona_credits_remaining >= ? RETURNING persona_credits_remaining",
                    (count, uid, count),
                )
                row = cur.fetchone()
                conn.commit()
                return int(row[0]) if row else 0

    def reserve_persona_credits(self, user_id: int, amount: int) -> bool:
        """Атомарно списывает amount кредитов. Возвращает True если успешно, False если недостаточно."""
        if amount <= 0:
            return True
        uid = int(user_id)
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        UPDATE {self._users_table}
                        SET persona_credits_remaining = persona_credits_remaining - %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = %s AND persona_credits_remaining >= %s
                        RETURNING persona_credits_remaining
                        """,
                        (amount, uid, amount),
                    )
                    row = cur.fetchone()
                    conn.commit()
                    return row is not None
            else:
                cur = conn.execute(
                    f"UPDATE {self._users_table} SET persona_credits_remaining = persona_credits_remaining - ?, "
                    f"updated_at = CURRENT_TIMESTAMP "
                    f"WHERE user_id = ? AND persona_credits_remaining >= ?",
                    (amount, uid, amount),
                )
                conn.commit()
                return cur.rowcount > 0

    def refund_persona_credits(self, user_id: int, amount: int) -> None:
        """Возвращает amount кредитов пользователю."""
        if amount <= 0:
            return
        uid = int(user_id)
        with self._connect() as conn:
            if self._use_pg:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        UPDATE {self._users_table}
                        SET persona_credits_remaining = persona_credits_remaining + %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = %s
                        """,
                        (amount, uid),
                    )
                    conn.commit()
            else:
                conn.execute(
                    f"UPDATE {self._users_table} SET persona_credits_remaining = persona_credits_remaining + ?, "
                    f"updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                    (amount, uid),
                )
                conn.commit()

    def set_pending_persona_batch(self, user_id: int, styles_json: str) -> None:
        """Сохраняет pending batch генерации персоны (JSON-строка со списком стилей)."""
        self._run(
            f"""
            UPDATE {self._users_table}
            SET pending_persona_batch = %s, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = %s
            """,
            (styles_json, int(user_id)),
        )

    def get_pending_persona_batch(self, user_id: int) -> str | None:
        """Возвращает pending batch JSON или None."""
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"SELECT pending_persona_batch FROM {self._users_table} WHERE user_id = %s",
                        (int(user_id),),
                    )
                    row = cur.fetchone()
            else:
                row = conn.execute(
                    f"SELECT pending_persona_batch FROM {self._users_table} WHERE user_id = ?",
                    (int(user_id),),
                ).fetchone()
            if not row:
                return None
            return _row_get(row, "pending_persona_batch")

    def clear_pending_persona_batch(self, user_id: int) -> None:
        """Очищает pending batch."""
        self._run(
            f"""
            UPDATE {self._users_table}
            SET pending_persona_batch = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = %s
            """,
            (int(user_id),),
        )

    def set_subject_gender(self, user_id: int, gender: str) -> None:
        if gender not in ("male", "female"):
            return
        self._run(
            f"""
            INSERT INTO {self._users_table} (user_id, subject_gender, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                subject_gender = EXCLUDED.subject_gender,
                updated_at = CURRENT_TIMESTAMP
            """,
            (int(user_id), gender),
        )

    def clear_subject_gender(self, user_id: int) -> None:
        """Снять пол у пользователя (subject_gender = NULL)."""
        self._run(
            f"""
            UPDATE {self._users_table}
            SET subject_gender = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = %s
            """,
            (int(user_id),),
        )

    def clear_user_data(self, *, user_id: int) -> None:
        self._run(
            f"""
            UPDATE {self._users_table}
            SET personal_model_version = NULL,
                personal_trigger_word = NULL,
                astria_tune_id = NULL,
                astria_lora_tune_id = NULL,
                persona_lora_class_name = NULL,
                astria_lora_pack_tune_id = NULL,
                astria_lora_pack_tune_id_pending = NULL,
                training_status = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = %s
            """,
            (int(user_id),),
        )

    # --- Pending pack upload (Mini App: оплата в другом потоке, user_data не доходит) ---

    def set_pending_pack_upload(self, *, user_id: int, pack_id: int) -> None:
        """Юзер оплатил пак — ждём 10 фото. Fallback когда application.user_data не доходит."""
        self._run(
            f"""
            INSERT INTO {self._users_table} (user_id, pending_pack_id, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET pending_pack_id = EXCLUDED.pending_pack_id, updated_at = CURRENT_TIMESTAMP
            """,
            (int(user_id), int(pack_id)),
        )

    def get_pending_pack_upload(self, user_id: int) -> int | None:
        """pack_id если юзер ждёт загрузку фото для пака, иначе None."""
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"SELECT pending_pack_id FROM {self._users_table} WHERE user_id = %s AND pending_pack_id IS NOT NULL",
                        (int(user_id),),
                    )
                    row = cur.fetchone()
            else:
                sql = f"SELECT pending_pack_id FROM {self._users_table} WHERE user_id = ? AND pending_pack_id IS NOT NULL"
                cur = conn.execute(sql, (int(user_id),))
                row = cur.fetchone()
            if row:
                val = row["pending_pack_id"] if hasattr(row, "keys") else row[0]
                return int(val) if val is not None else None
            return None

    def clear_pending_pack_upload(self, user_id: int) -> None:
        """Очистить после приёма 10 фото или сброса."""
        self._run(
            f"UPDATE {self._users_table} SET pending_pack_id = NULL, updated_at = CURRENT_TIMESTAMP WHERE user_id = %s",
            (int(user_id),),
        )

    # --- Pending pack run (восстановление после рестарта во время обучения pack tune) ---

    def set_pending_pack_run(
        self,
        *,
        user_id: int,
        pack_id: int,
        chat_id: int,
        run_id: str,
        expected: int,
        class_name: str,
        offer_title: str = "",
    ) -> None:
        """Сохранить состояние pack run перед блокирующим ожиданием обучения pack tune."""
        if not self._use_pg:
            return
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._pending_pack_runs_table} (user_id, pack_id, chat_id, run_id, expected, class_name, offer_title)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        pack_id = EXCLUDED.pack_id,
                        chat_id = EXCLUDED.chat_id,
                        run_id = EXCLUDED.run_id,
                        expected = EXCLUDED.expected,
                        class_name = EXCLUDED.class_name,
                        offer_title = EXCLUDED.offer_title,
                        created_at = NOW()
                    """,
                    (int(user_id), int(pack_id), int(chat_id), str(run_id), int(expected), str(class_name), str(offer_title or "")),
                )
                conn.commit()

    def clear_pending_pack_run(self, user_id: int) -> None:
        """Очистить после завершения pack run (успех или ошибка)."""
        if not self._use_pg:
            return
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"DELETE FROM {self._pending_pack_runs_table} WHERE user_id = %s", (int(user_id),))
                conn.commit()

    def get_pending_pack_runs_to_recover(self) -> list[dict]:
        """Юзеры с astria_lora_pack_tune_id_pending и записью в pending_pack_runs — кандидаты на восстановление."""
        if not self._use_pg:
            return []
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT p.user_id, p.pack_id, p.chat_id, p.run_id, p.expected, p.class_name, p.offer_title,
                           u.astria_lora_pack_tune_id_pending as tune_id
                    FROM {self._pending_pack_runs_table} p
                    JOIN {self._users_table} u ON u.user_id = p.user_id
                    WHERE u.astria_lora_pack_tune_id_pending IS NOT NULL
                    """,
                )
                return [dict(row) for row in cur.fetchall()]

    def delete_user(self, user_id: int) -> bool:
        """Удалить пользователя полностью из БД. Возвращает True если строка была удалена."""
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"DELETE FROM {self._users_table} WHERE user_id = %s", (int(user_id),))
                    deleted = cur.rowcount > 0
            else:
                cur = conn.execute(
                    f"DELETE FROM {self._users_table} WHERE user_id = ?", (int(user_id),)
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    # ========================================
    # АДМИНКА: таблицы и методы
    # ========================================

    def _init_admin_tables(self) -> None:
        """Создаёт таблицы для админки: payments, user_events, admins."""
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Таблица платежей
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._payments_table} (
                            id SERIAL PRIMARY KEY,
                            user_id BIGINT NOT NULL,
                            payment_id TEXT,
                            payment_method TEXT NOT NULL,
                            product_type TEXT NOT NULL,
                            credits INTEGER NOT NULL,
                            amount_rub DECIMAL(10,2) NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    # Таблица событий пользователей
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._user_events_table} (
                            id SERIAL PRIMARY KEY,
                            user_id BIGINT NOT NULL,
                            event_type TEXT NOT NULL,
                            event_data JSONB,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    # Таблица админов
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._admins_table} (
                            id SERIAL PRIMARY KEY,
                            username TEXT UNIQUE NOT NULL,
                            password_hash TEXT NOT NULL,
                            display_name TEXT,
                            is_active BOOLEAN DEFAULT TRUE,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    # Таблица стилей персоны
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._persona_styles_table} (
                            id SERIAL PRIMARY KEY,
                            slug TEXT UNIQUE NOT NULL,
                            title TEXT NOT NULL,
                            description TEXT,
                            gender TEXT NOT NULL DEFAULT 'female',
                            image_url TEXT,
                            prompt TEXT,
                            sort_order INTEGER DEFAULT 0,
                            is_active BOOLEAN DEFAULT TRUE,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    conn.commit()
                    # Таблица превью стилей персоны (нормализованная)
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._persona_style_previews_table} (
                            id SERIAL PRIMARY KEY,
                            style_id INTEGER NOT NULL REFERENCES {self._persona_styles_table}(id) ON DELETE CASCADE,
                            image_url TEXT NOT NULL,
                            sort_order INTEGER DEFAULT 0
                        )
                    """)
                    conn.commit()
                    # Таблица экспресс-стилей
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._express_styles_table} (
                            id SERIAL PRIMARY KEY,
                            slug TEXT UNIQUE NOT NULL,
                            title TEXT NOT NULL,
                            emoji TEXT DEFAULT '',
                            theme TEXT NOT NULL DEFAULT 'general',
                            gender TEXT NOT NULL DEFAULT 'female',
                            prompt TEXT,
                            negative_prompt TEXT,
                            provider TEXT NOT NULL DEFAULT 'seedream',
                            model TEXT DEFAULT 'seedream',
                            image_url TEXT,
                            model_params TEXT,
                            sort_order INTEGER DEFAULT 0,
                            is_active BOOLEAN DEFAULT TRUE,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    conn.commit()
                    # Миграции express_styles: новые колонки для существующих таблиц
                    for col, col_def in [
                        ("theme", "TEXT NOT NULL DEFAULT 'general'"),
                        ("provider", "TEXT NOT NULL DEFAULT 'seedream'"),
                        ("negative_prompt", "TEXT"),
                        ("image_url", "TEXT"),
                        ("model_params", "TEXT"),
                    ]:
                        try:
                            cur.execute(f"ALTER TABLE {self._express_styles_table} ADD COLUMN {col} {col_def}")
                            conn.commit()
                        except Exception:
                            conn.rollback()
                    # Миграция: model → provider (one-time, для строк где model отличается от provider)
                    try:
                        cur.execute(f"""
                            UPDATE {self._express_styles_table}
                            SET provider = model
                            WHERE (provider IS NULL OR provider = '' OR provider = 'seedream')
                              AND model IS NOT NULL AND model != '' AND model != provider
                        """)
                        conn.commit()
                    except Exception:
                        conn.rollback()
                    # --- V3: Категории, теги и junction-таблицы для экспресс-стилей ---
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._express_categories_table} (
                            id SERIAL PRIMARY KEY,
                            slug TEXT UNIQUE NOT NULL,
                            title TEXT NOT NULL,
                            sort_order INTEGER DEFAULT 0,
                            is_active BOOLEAN DEFAULT TRUE,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._express_tags_table} (
                            id SERIAL PRIMARY KEY,
                            slug TEXT UNIQUE NOT NULL,
                            title TEXT NOT NULL,
                            sort_order INTEGER DEFAULT 0,
                            is_active BOOLEAN DEFAULT TRUE,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._express_style_categories_table} (
                            style_id INTEGER REFERENCES {self._express_styles_table}(id) ON DELETE CASCADE,
                            category_id INTEGER REFERENCES {self._express_categories_table}(id) ON DELETE CASCADE,
                            PRIMARY KEY(style_id, category_id),
                            UNIQUE(style_id, category_id)
                        )
                    """)
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._express_style_tags_table} (
                            style_id INTEGER REFERENCES {self._express_styles_table}(id) ON DELETE CASCADE,
                            tag_id INTEGER REFERENCES {self._express_tags_table}(id) ON DELETE CASCADE,
                            PRIMARY KEY(style_id, tag_id),
                            UNIQUE(style_id, tag_id)
                        )
                    """)
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._express_category_tags_table} (
                            category_id INTEGER REFERENCES {self._express_categories_table}(id) ON DELETE CASCADE,
                            tag_id INTEGER REFERENCES {self._express_tags_table}(id) ON DELETE CASCADE,
                            PRIMARY KEY(category_id, tag_id),
                            UNIQUE(category_id, tag_id)
                        )
                    """)
                    conn.commit()
                    # Индексы на junction-таблицы
                    try:
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._prefix}esc_style ON {self._express_style_categories_table}(style_id)")
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._prefix}esc_cat ON {self._express_style_categories_table}(category_id)")
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._prefix}est_style ON {self._express_style_tags_table}(style_id)")
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._prefix}est_tag ON {self._express_style_tags_table}(tag_id)")
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._prefix}ect_cat ON {self._express_category_tags_table}(category_id)")
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._prefix}ect_tag ON {self._express_category_tags_table}(tag_id)")
                        conn.commit()
                    except Exception:
                        conn.rollback()
                    # Миграция users: last_express_provider
                    try:
                        cur.execute(f"ALTER TABLE {self._users_table} ADD COLUMN last_express_provider TEXT DEFAULT 'seedream'")
                        conn.commit()
                    except Exception:
                        conn.rollback()
                    # Добавить prompt в persona_styles если нет
                    try:
                        cur.execute(f"ALTER TABLE {self._persona_styles_table} ADD COLUMN prompt TEXT")
                        conn.commit()
                    except Exception:
                        conn.rollback()
                    # Добавить credit_cost в persona_styles если нет
                    try:
                        cur.execute(f"ALTER TABLE {self._persona_styles_table} ADD COLUMN credit_cost INTEGER DEFAULT 4")
                        conn.commit()
                    except Exception:
                        conn.rollback()
                    # Добавить cost_usd в persona_styles если нет
                    try:
                        cur.execute(f"ALTER TABLE {self._persona_styles_table} ADD COLUMN cost_usd DECIMAL(10,4) DEFAULT 0")
                        conn.commit()
                    except Exception:
                        conn.rollback()
                    # Миграция: перенести persona_styles.image_url → persona_style_previews
                    try:
                        cur.execute(f"""
                            INSERT INTO {self._persona_style_previews_table} (style_id, image_url, sort_order)
                            SELECT id, image_url, 0 FROM {self._persona_styles_table}
                            WHERE image_url IS NOT NULL AND image_url != ''
                            AND id NOT IN (SELECT style_id FROM {self._persona_style_previews_table})
                        """)
                        conn.commit()
                    except Exception:
                        conn.rollback()
                    # Добавить created_at в users если нет
                    try:
                        cur.execute(f"ALTER TABLE {self._users_table} ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW()")
                        conn.commit()
                    except Exception:
                        conn.rollback()
                    # Дедуп платежей и уникальность payment_id для идемпотентности
                    try:
                        cur.execute(
                            f"""
                            DELETE FROM {self._payments_table} p1
                            USING {self._payments_table} p2
                            WHERE p1.id < p2.id
                              AND p1.payment_id IS NOT NULL
                              AND p1.payment_id = p2.payment_id
                            """
                        )
                        cur.execute(
                            f"""
                            CREATE UNIQUE INDEX IF NOT EXISTS idx_{self._prefix}payments_payment_id_uniq
                            ON {self._payments_table}(payment_id)
                            WHERE payment_id IS NOT NULL
                            """
                        )
                        conn.commit()
                    except Exception as e:
                        conn.rollback()
                        logger.warning("Не удалось обеспечить уникальность payment_id: %s", e)
                    # Индексы
                    try:
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._prefix}payments_user_id ON {self._payments_table}(user_id)")
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_payments_created_at ON {self._payments_table}(created_at DESC)")
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_user_events_user_id ON {self._user_events_table}(user_id)")
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_user_events_created_at ON {self._user_events_table}(created_at DESC)")
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_user_events_type ON {self._user_events_table}(event_type)")
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._prefix}user_events_type_created ON {self._user_events_table}(event_type, created_at DESC)")
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._prefix}payments_type_created ON {self._payments_table}(product_type, created_at DESC)")
                        conn.commit()
                    except Exception:
                        conn.rollback()
                    # --- Phase 4: generation_history ---
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._generation_history_table} (
                            id SERIAL PRIMARY KEY,
                            user_id BIGINT NOT NULL,
                            mode TEXT NOT NULL DEFAULT 'express',
                            style_slug TEXT,
                            style_title TEXT,
                            provider TEXT,
                            image_url TEXT,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._prefix}gh_user_created ON {self._generation_history_table}(user_id, created_at DESC)")
                    # Миграция generation_history: prompt_preview, refs_count, request_id
                    for col, col_def in [
                        ("prompt_preview", "TEXT"),
                        ("refs_count", "INTEGER DEFAULT 0"),
                        ("request_id", "TEXT"),
                    ]:
                        try:
                            cur.execute(f"ALTER TABLE {self._generation_history_table} ADD COLUMN {col} {col_def}")
                            conn.commit()
                        except Exception:
                            conn.rollback()
                    # --- Custom generation: generation_requests (idempotency) ---
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self._generation_requests_table} (
                            id SERIAL PRIMARY KEY,
                            request_id TEXT NOT NULL,
                            user_id BIGINT NOT NULL,
                            task_id TEXT NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            UNIQUE (user_id, request_id)
                        )
                    """)
                    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._prefix}gr_user ON {self._generation_requests_table}(user_id)")
                    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self._prefix}gr_created ON {self._generation_requests_table}(created_at)")
                    conn.commit()
                logger.info("Таблицы админки созданы/проверены")
            else:
                # SQLite версия
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._payments_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        payment_id TEXT,
                        payment_method TEXT NOT NULL,
                        product_type TEXT NOT NULL,
                        credits INTEGER NOT NULL,
                        amount_rub REAL NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Дедуп и уникальность payment_id для идемпотентности
                try:
                    conn.execute(f"""
                        DELETE FROM {self._payments_table}
                        WHERE rowid NOT IN (
                            SELECT MAX(rowid)
                            FROM {self._payments_table}
                            WHERE payment_id IS NOT NULL
                            GROUP BY payment_id
                        )
                        AND payment_id IS NOT NULL
                    """)
                    conn.execute(
                        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{self._prefix}payments_payment_id_uniq ON {self._payments_table}(payment_id)"
                    )
                except Exception as e:
                    logger.warning("SQLite: не удалось обеспечить уникальность payment_id: %s", e)
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._user_events_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        event_data TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._admins_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        display_name TEXT,
                        is_active INTEGER DEFAULT 1,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                try:
                    conn.execute(f"ALTER TABLE {self._users_table} ADD COLUMN created_at TEXT DEFAULT CURRENT_TIMESTAMP")
                except Exception:
                    pass
                # Таблица стилей персоны (SQLite)
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._persona_styles_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        slug TEXT UNIQUE NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        gender TEXT NOT NULL DEFAULT 'female',
                        image_url TEXT,
                        prompt TEXT,
                        credit_cost INTEGER DEFAULT 4,
                        sort_order INTEGER DEFAULT 0,
                        is_active INTEGER DEFAULT 1,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Добавить credit_cost если нет (миграция)
                try:
                    conn.execute(f"ALTER TABLE {self._persona_styles_table} ADD COLUMN credit_cost INTEGER DEFAULT 4")
                except Exception:
                    pass
                # Добавить cost_usd если нет (миграция)
                try:
                    conn.execute(f"ALTER TABLE {self._persona_styles_table} ADD COLUMN cost_usd REAL DEFAULT 0")
                except Exception:
                    pass
                # Таблица превью стилей персоны (SQLite)
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._persona_style_previews_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        style_id INTEGER NOT NULL REFERENCES {self._persona_styles_table}(id) ON DELETE CASCADE,
                        image_url TEXT NOT NULL,
                        sort_order INTEGER DEFAULT 0
                    )
                """)
                # Миграция: перенести persona_styles.image_url → persona_style_previews
                try:
                    conn.execute(f"""
                        INSERT INTO {self._persona_style_previews_table} (style_id, image_url, sort_order)
                        SELECT id, image_url, 0 FROM {self._persona_styles_table}
                        WHERE image_url IS NOT NULL AND image_url != ''
                        AND id NOT IN (SELECT style_id FROM {self._persona_style_previews_table})
                    """)
                except Exception:
                    pass
                # Таблица экспресс-стилей (SQLite)
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._express_styles_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        slug TEXT UNIQUE NOT NULL,
                        title TEXT NOT NULL,
                        emoji TEXT DEFAULT '',
                        theme TEXT NOT NULL DEFAULT 'general',
                        gender TEXT NOT NULL DEFAULT 'female',
                        prompt TEXT,
                        negative_prompt TEXT,
                        provider TEXT NOT NULL DEFAULT 'seedream',
                        model TEXT DEFAULT 'seedream',
                        image_url TEXT,
                        model_params TEXT,
                        sort_order INTEGER DEFAULT 0,
                        is_active INTEGER DEFAULT 1,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Миграции express_styles (SQLite)
                for col, col_def in [
                    ("theme", "TEXT NOT NULL DEFAULT 'general'"),
                    ("provider", "TEXT NOT NULL DEFAULT 'seedream'"),
                    ("negative_prompt", "TEXT"),
                    ("image_url", "TEXT"),
                    ("model_params", "TEXT"),
                ]:
                    try:
                        conn.execute(f"ALTER TABLE {self._express_styles_table} ADD COLUMN {col} {col_def}")
                    except Exception:
                        pass
                # Миграция: model → provider (one-time, для строк где model отличается от provider)
                try:
                    conn.execute(f"""
                        UPDATE {self._express_styles_table}
                        SET provider = model
                        WHERE (provider IS NULL OR provider = '' OR provider = 'seedream')
                          AND model IS NOT NULL AND model != '' AND model != provider
                    """)
                except Exception:
                    pass
                # --- V3: Категории, теги и junction-таблицы для экспресс-стилей (SQLite) ---
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._express_categories_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        slug TEXT UNIQUE NOT NULL,
                        title TEXT NOT NULL,
                        sort_order INTEGER DEFAULT 0,
                        is_active INTEGER DEFAULT 1,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._express_tags_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        slug TEXT UNIQUE NOT NULL,
                        title TEXT NOT NULL,
                        sort_order INTEGER DEFAULT 0,
                        is_active INTEGER DEFAULT 1,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._express_style_categories_table} (
                        style_id INTEGER REFERENCES {self._express_styles_table}(id) ON DELETE CASCADE,
                        category_id INTEGER REFERENCES {self._express_categories_table}(id) ON DELETE CASCADE,
                        PRIMARY KEY(style_id, category_id),
                        UNIQUE(style_id, category_id)
                    )
                """)
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._express_style_tags_table} (
                        style_id INTEGER REFERENCES {self._express_styles_table}(id) ON DELETE CASCADE,
                        tag_id INTEGER REFERENCES {self._express_tags_table}(id) ON DELETE CASCADE,
                        PRIMARY KEY(style_id, tag_id),
                        UNIQUE(style_id, tag_id)
                    )
                """)
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._express_category_tags_table} (
                        category_id INTEGER REFERENCES {self._express_categories_table}(id) ON DELETE CASCADE,
                        tag_id INTEGER REFERENCES {self._express_tags_table}(id) ON DELETE CASCADE,
                        PRIMARY KEY(category_id, tag_id),
                        UNIQUE(category_id, tag_id)
                    )
                """)
                # Индексы на junction-таблицы (SQLite)
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_esc_style ON {self._express_style_categories_table}(style_id)")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_esc_cat ON {self._express_style_categories_table}(category_id)")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_est_style ON {self._express_style_tags_table}(style_id)")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_est_tag ON {self._express_style_tags_table}(tag_id)")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_ect_cat ON {self._express_category_tags_table}(category_id)")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_ect_tag ON {self._express_category_tags_table}(tag_id)")
                # Миграция users: last_express_provider (SQLite)
                try:
                    conn.execute(f"ALTER TABLE {self._users_table} ADD COLUMN last_express_provider TEXT DEFAULT 'seedream'")
                except Exception:
                    pass
                # --- Phase 4: generation_history (SQLite) ---
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._generation_history_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        mode TEXT NOT NULL DEFAULT 'express',
                        style_slug TEXT,
                        style_title TEXT,
                        provider TEXT,
                        image_url TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_gh_user_created ON {self._generation_history_table}(user_id, created_at DESC)")
                # Миграция generation_history: prompt_preview, refs_count, request_id (SQLite)
                for col, col_def in [
                    ("prompt_preview", "TEXT"),
                    ("refs_count", "INTEGER DEFAULT 0"),
                    ("request_id", "TEXT"),
                ]:
                    try:
                        conn.execute(f"ALTER TABLE {self._generation_history_table} ADD COLUMN {col} {col_def}")
                    except Exception:
                        pass
                # --- Custom generation: generation_requests (SQLite) ---
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._generation_requests_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT NOT NULL,
                        user_id INTEGER NOT NULL,
                        task_id TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (user_id, request_id)
                    )
                """)
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_gr_user ON {self._generation_requests_table}(user_id)")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_gr_created ON {self._generation_requests_table}(created_at)")
                # --- admin_settings (SQLite) ---
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._admin_settings_table} (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()

    def init_admin_tables(self) -> None:
        """Публичный метод для инициализации таблиц админки."""
        self._init_admin_tables()

    # --- Логирование платежей ---

    def is_payment_processed(self, payment_id: str) -> bool:
        """Проверяет, был ли платёж уже обработан (защита от дублирования)."""
        if not payment_id:
            return False
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"SELECT 1 FROM {self._payments_table} WHERE payment_id = %s LIMIT 1",
                        (payment_id,),
                    )
                    return cur.fetchone() is not None
            else:
                cur = conn.execute(
                    f"SELECT 1 FROM {self._payments_table} WHERE payment_id = ? LIMIT 1",
                    (payment_id,),
                )
                return cur.fetchone() is not None

    def log_payment(
        self,
        user_id: int,
        payment_id: str | None,
        payment_method: str,
        product_type: str,
        credits: int,
        amount_rub: float,
    ) -> int | None:
        """Записывает платёж в БД. Возвращает ID записи или None при дубле payment_id."""
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    try:
                        cur.execute(
                            f"""
                            INSERT INTO {self._payments_table} (user_id, payment_id, payment_method, product_type, credits, amount_rub)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            RETURNING id
                            """,
                            (int(user_id), payment_id, payment_method, product_type, int(credits), float(amount_rub)),
                        )
                        row = cur.fetchone()
                        conn.commit()
                        return row["id"] if row else None
                    except Exception as e:
                        conn.rollback()
                        if getattr(e, "pgcode", None) == "23505":
                            return None
                        raise
            else:
                try:
                    cur = conn.execute(
                        f"""
                        INSERT INTO {self._payments_table} (user_id, payment_id, payment_method, product_type, credits, amount_rub)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (int(user_id), payment_id, payment_method, product_type, int(credits), float(amount_rub)),
                    )
                    conn.commit()
                    return cur.lastrowid
                except sqlite3.IntegrityError:
                    conn.rollback()
                    return None

    # --- Логирование событий ---

    def log_event(self, user_id: int, event_type: str, event_data: dict | None = None) -> int | None:
        """Записывает событие пользователя. Возвращает ID записи."""
        import json
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import Json, RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        INSERT INTO {self._user_events_table} (user_id, event_type, event_data)
                        VALUES (%s, %s, %s)
                        RETURNING id
                        """,
                        (int(user_id), event_type, Json(event_data) if event_data else None),
                    )
                    row = cur.fetchone()
                    conn.commit()
                    return row["id"] if row else None
            else:
                cur = conn.execute(
                    f"""
                    INSERT INTO {self._user_events_table} (user_id, event_type, event_data)
                    VALUES (?, ?, ?)
                    """,
                    (int(user_id), event_type, json.dumps(event_data) if event_data else None),
                )
                conn.commit()
                return cur.lastrowid

    # --- Аналитика воронок ---

    def _funnel_count(self, cur, event_type: str, d_from: str | None, d_to: str | None,
                      data_filter: dict | None = None) -> int:
        """Считает DISTINCT user_id по event_type за период с опциональным фильтром event_data."""
        conditions = ["event_type = %s"]
        params: list = [event_type]
        if d_from and d_to:
            conditions.append("created_at >= %s AND created_at <= %s")
            params.extend([d_from, d_to])
        if data_filter:
            for key, val in data_filter.items():
                conditions.append(f"event_data->>'{key}' = %s")
                params.append(val)
        sql = f"SELECT COUNT(DISTINCT user_id) as cnt FROM {self._user_events_table} WHERE " + " AND ".join(conditions)
        cur.execute(sql, params)
        row = cur.fetchone()
        return int(row["cnt"]) if row else 0

    def get_funnel_data(self, date_from: str | None = None, date_to: str | None = None) -> dict:
        """Данные воронок: Персона, Экспресс, Фотосеты, Mini App."""
        result = {"persona": [], "fast": [], "packs": [], "miniapp": []}
        if not self._use_pg:
            return result
        d_from = (str(date_from) + " 00:00:00+03") if date_from else None
        d_to = (str(date_to) + " 23:59:59+03") if date_to else None
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Воронка Персоны
                persona_steps = [
                    ("Зашёл в раздел", "nav_persona", None),
                    ("Нажал Создать", "persona_create_start", None),
                    ("Выбрал тариф", "persona_buy_init", None),
                    ("Оплатил", "payment_success", {"product_type": "persona_create"}),
                    ("Начал загрузку фото", "persona_upload_start", None),
                    ("Получил фото", "generation", {"mode": "persona"}),
                ]
                for label, evt, filt in persona_steps:
                    cnt = self._funnel_count(cur, evt, d_from, d_to, filt)
                    result["persona"].append({"label": label, "count": cnt})

                # Воронка Экспресса
                fast_steps = [
                    ("Зашёл в раздел", "nav_fast", None),
                    ("Выбрал пол", "fast_gender_select", None),
                    ("Выбрал стиль", "fast_style_select", None),
                    ("Нажал купить", "fast_buy_init", None),
                    ("Оплатил", "payment_success", {"product_type": "fast"}),
                    ("Получил фото", "generation", {"mode": "fast"}),
                ]
                for label, evt, filt in fast_steps:
                    cnt = self._funnel_count(cur, evt, d_from, d_to, filt)
                    result["fast"].append({"label": label, "count": cnt})

                # Воронка Фотосетов
                pack_steps = [
                    ("Зашёл в раздел", "nav_packs", None),
                    ("Нажал купить", "pack_buy_init", None),
                    ("Оплатил", "payment_success", {"product_type": "persona_pack"}),
                    ("Получил фотосет", "pack_generation", None),
                ]
                for label, evt, filt in pack_steps:
                    cnt = self._funnel_count(cur, evt, d_from, d_to, filt)
                    result["packs"].append({"label": label, "count": cnt})

                # Воронка Mini App
                miniapp_steps = [
                    ("Открыл аппку", "miniapp_open", None),
                    ("Перешёл в Персону", "nav_persona", {"source": "miniapp"}),
                    ("Просмотрел образ", "persona_style_view", {"source": "miniapp"}),
                    ("Выбрал образ", "persona_style_select", {"source": "miniapp"}),
                    ("Выбрал тариф", "persona_buy_init", {"source": "miniapp"}),
                    ("Нажал Оплатить", "persona_buy_confirm", {"source": "miniapp"}),
                ]
                for label, evt, filt in miniapp_steps:
                    cnt = self._funnel_count(cur, evt, d_from, d_to, filt)
                    result["miniapp"].append({"label": label, "count": cnt})
        return result

    def get_popularity_data(self, date_from: str | None = None, date_to: str | None = None) -> dict:
        """Топ стилей и действий за период."""
        result = {"top_styles": [], "top_actions": []}
        if not self._use_pg:
            return result
        d_from = (str(date_from) + " 00:00:00+03") if date_from else None
        d_to = (str(date_to) + " 23:59:59+03") if date_to else None
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Топ стилей
                date_cond = ""
                params: list = []
                if d_from and d_to:
                    date_cond = " AND created_at >= %s AND created_at <= %s"
                    params = [d_from, d_to]
                cur.execute(f"""
                    SELECT event_data->>'style_id' as style_id, COUNT(*) as cnt
                    FROM {self._user_events_table}
                    WHERE event_type IN ('persona_style_select', 'fast_style_select')
                      AND event_data->>'style_id' IS NOT NULL
                      {date_cond}
                    GROUP BY event_data->>'style_id'
                    ORDER BY cnt DESC
                    LIMIT 15
                """, params)
                result["top_styles"] = [dict(r) for r in cur.fetchall()]

                # Топ действий
                cur.execute(f"""
                    SELECT event_type, COUNT(*) as cnt
                    FROM {self._user_events_table}
                    WHERE 1=1 {date_cond}
                    GROUP BY event_type
                    ORDER BY cnt DESC
                    LIMIT 20
                """, params)
                result["top_actions"] = [dict(r) for r in cur.fetchall()]
        return result

    # --- Админы ---

    def get_admin_by_username(self, username: str) -> dict | None:
        """Получить админа по username."""
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"SELECT * FROM {self._admins_table} WHERE username = %s AND is_active = TRUE", (username,))
                    row = cur.fetchone()
                    return dict(row) if row else None
            else:
                row = conn.execute(f"SELECT * FROM {self._admins_table} WHERE username = ? AND is_active = 1", (username,)).fetchone()
                return dict(row) if row else None

    def create_admin(self, username: str, password_hash: str, display_name: str | None = None) -> int | None:
        """Создать нового админа."""
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        INSERT INTO {self._admins_table} (username, password_hash, display_name)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (username) DO UPDATE SET
                            password_hash = EXCLUDED.password_hash,
                            display_name = COALESCE(EXCLUDED.display_name, {self._admins_table}.display_name)
                        RETURNING id
                        """,
                        (username, password_hash, display_name),
                    )
                    row = cur.fetchone()
                    conn.commit()
                    return row["id"] if row else None
            else:
                try:
                    cur = conn.execute(
                        f"INSERT INTO {self._admins_table} (username, password_hash, display_name) VALUES (?, ?, ?)",
                        (username, password_hash, display_name),
                    )
                    conn.commit()
                    return cur.lastrowid
                except Exception:
                    return None

    # --- Аналитика для админки ---

    def get_users_paginated(self, limit: int = 50, offset: int = 0, search: str | None = None) -> list[dict]:
        """Получить список пользователей с пагинацией."""
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if search:
                        cur.execute(
                            f"SELECT * FROM {self._users_table} WHERE CAST(user_id AS TEXT) LIKE %s ORDER BY updated_at DESC LIMIT %s OFFSET %s",
                            (f"%{search}%", limit, offset),
                        )
                    else:
                        cur.execute(f"SELECT * FROM {self._users_table} ORDER BY updated_at DESC LIMIT %s OFFSET %s", (limit, offset))
                    return [dict(row) for row in cur.fetchall()]
            else:
                if search:
                    rows = conn.execute(
                        f"SELECT * FROM {self._users_table} WHERE CAST(user_id AS TEXT) LIKE ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                        (f"%{search}%", limit, offset),
                    ).fetchall()
                else:
                    rows = conn.execute(f"SELECT * FROM {self._users_table} ORDER BY updated_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
                return [dict(row) for row in rows]

    def get_users_count(self, search: str | None = None) -> int:
        """Получить количество пользователей."""
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if search:
                        cur.execute(f"SELECT COUNT(*) as cnt FROM {self._users_table} WHERE CAST(user_id AS TEXT) LIKE %s", (f"%{search}%",))
                    else:
                        cur.execute(f"SELECT COUNT(*) as cnt FROM {self._users_table}")
                    row = cur.fetchone()
                    return int(row["cnt"]) if row else 0
            else:
                if search:
                    row = conn.execute(f"SELECT COUNT(*) as cnt FROM {self._users_table} WHERE CAST(user_id AS TEXT) LIKE ?", (f"%{search}%",)).fetchone()
                else:
                    row = conn.execute(f"SELECT COUNT(*) as cnt FROM {self._users_table}").fetchone()
                return int(row["cnt"]) if row else 0

    # ========== Рассылка ==========

    def get_user_ids_for_broadcast(self, filter_type: str = "all", user_ids: list[int] | None = None) -> list[int]:
        """
        Получить список user_id по фильтру для рассылки.
        filter_type: "all" | "has_persona" | "no_persona" | "specific"
        """
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if filter_type == "has_persona":
                        cur.execute(f"SELECT user_id FROM {self._users_table} WHERE astria_lora_tune_id IS NOT NULL")
                    elif filter_type == "no_persona":
                        cur.execute(f"SELECT user_id FROM {self._users_table} WHERE astria_lora_tune_id IS NULL")
                    elif filter_type == "specific" and user_ids:
                        cur.execute(f"SELECT user_id FROM {self._users_table} WHERE user_id = ANY(%s)", (user_ids,))
                    else:  # all
                        cur.execute(f"SELECT user_id FROM {self._users_table}")
                    return [int(r["user_id"]) for r in cur.fetchall()]
            else:
                if filter_type == "has_persona":
                    rows = conn.execute(f"SELECT user_id FROM {self._users_table} WHERE astria_lora_tune_id IS NOT NULL").fetchall()
                elif filter_type == "no_persona":
                    rows = conn.execute(f"SELECT user_id FROM {self._users_table} WHERE astria_lora_tune_id IS NULL").fetchall()
                elif filter_type == "specific" and user_ids:
                    placeholders = ",".join("?" * len(user_ids))
                    rows = conn.execute(f"SELECT user_id FROM {self._users_table} WHERE user_id IN ({placeholders})", user_ids).fetchall()
                else:
                    rows = conn.execute(f"SELECT user_id FROM {self._users_table}").fetchall()
                return [int(r["user_id"]) for r in rows]

    def get_broadcast_counts(self) -> dict:
        """Количество юзеров по фильтрам для рассылки."""
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"""
                        SELECT
                            COUNT(*) as total,
                            COUNT(astria_lora_tune_id) as has_persona,
                            COUNT(*) - COUNT(astria_lora_tune_id) as no_persona
                        FROM {self._users_table}
                    """)
                    row = cur.fetchone()
                    return dict(row) if row else {"total": 0, "has_persona": 0, "no_persona": 0}
            else:
                row = conn.execute(f"""
                    SELECT
                        COUNT(*) as total,
                        COUNT(astria_lora_tune_id) as has_persona,
                        COUNT(*) - COUNT(astria_lora_tune_id) as no_persona
                    FROM {self._users_table}
                """).fetchone()
                return dict(row) if row else {"total": 0, "has_persona": 0, "no_persona": 0}

    def get_payments_paginated(
        self,
        limit: int = 50,
        offset: int = 0,
        user_id: int | None = None,
        product_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict]:
        """Получить список платежей с фильтрами."""
        with self._connect() as conn:
            conditions = []
            params: list = []
            if user_id:
                conditions.append("user_id = %s" if self._use_pg else "user_id = ?")
                params.append(int(user_id))
            if product_type:
                conditions.append("product_type = %s" if self._use_pg else "product_type = ?")
                params.append(product_type)
            if date_from:
                conditions.append("created_at >= %s" if self._use_pg else "created_at >= ?")
                params.append(str(date_from) + " 00:00:00+03")
            if date_to:
                conditions.append("created_at <= %s" if self._use_pg else "created_at <= ?")
                params.append(str(date_to) + " 23:59:59+03")

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            table = self._payments_table

            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"SELECT * FROM {table} WHERE {where_clause} ORDER BY created_at DESC LIMIT %s OFFSET %s",
                        (*params, limit, offset),
                    )
                    return [dict(row) for row in cur.fetchall()]
            else:
                sql = f"SELECT * FROM {table} WHERE {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?"
                rows = conn.execute(sql, (*params, limit, offset)).fetchall()
                return [dict(row) for row in rows]

    def get_payments_count(self, user_id: int | None = None, product_type: str | None = None, date_from: str | None = None, date_to: str | None = None) -> int:
        """Получить количество платежей."""
        with self._connect() as conn:
            conditions = []
            params: list = []
            if user_id:
                conditions.append("user_id = %s" if self._use_pg else "user_id = ?")
                params.append(int(user_id))
            if product_type:
                conditions.append("product_type = %s" if self._use_pg else "product_type = ?")
                params.append(product_type)
            if date_from:
                conditions.append("created_at >= %s" if self._use_pg else "created_at >= ?")
                params.append(str(date_from) + " 00:00:00+03")
            if date_to:
                conditions.append("created_at <= %s" if self._use_pg else "created_at <= ?")
                params.append(str(date_to) + " 23:59:59+03")

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            table = self._payments_table

            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"SELECT COUNT(*) as cnt FROM {table} WHERE {where_clause}", params)
                    row = cur.fetchone()
                    return int(row["cnt"]) if row else 0
            else:
                row = conn.execute(f"SELECT COUNT(*) as cnt FROM {table} WHERE {where_clause}", params).fetchone()
                return int(row["cnt"]) if row else 0

    def get_all_time_stats(self, _cost_settings: dict | None = None, _pack_costs_map: dict | None = None) -> dict:
        """Получить статистику за всё время (для постоянного блока)."""
        # Себестоимость из настроек (принимаем кэш или грузим)
        cost_settings = _cost_settings or self.get_cost_settings()
        COST_PERSONA_CREATE = cost_settings["cost_persona_create"]
        COST_FAST_PHOTO = cost_settings["cost_fast_photo"]
        COST_NANO_BANANA = cost_settings.get("cost_nano_banana", COST_FAST_PHOTO)
        COST_PERSONA_PHOTO = cost_settings["cost_persona_photo"]
        USD_RUB = cost_settings["usd_rub"]

        stats = {
            "users_total": 0,
            "total_revenue": 0.0,
            "total_cost": 0.0,
            "avg_check": 0.0,
            "paid_users": 0,
            "conversion": 0.0,
            "margin": {"amount": 0.0, "percent": 0.0},
            "gens_per_paying_user": 0.0,
            "express_purchases": 0,
            "persona_purchases": 0,  # оставляем для обратной совместимости (итого)
            "persona_create_purchases": 0,
            "persona_topup_purchases": 0,
            "pack_purchases": 0,
            "gender": {"male": 0.0, "female": 0.0, "unknown": 0.0},
            "days_to_first_purchase": 0.0,
            "fast_generations_by_provider": {"seedream": 0, "nano_banana": 0},
            "fast_costs_by_provider": {"seedream": 0.0, "nano_banana": 0.0},
        }

        with self._connect() as conn:
            if not self._use_pg:
                row = conn.execute(f"SELECT COUNT(*) as cnt FROM {self._users_table}").fetchone()
                stats["users_total"] = int(row["cnt"]) if row else 0
                row = conn.execute(f"SELECT COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM {self._payments_table}").fetchone()
                if row:
                    stats["total_revenue"] = float(row["total"])
                    if int(row["cnt"]) > 0:
                        stats["avg_check"] = round(float(row["total"]) / int(row["cnt"]), 2)
                return stats

            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Всего юзеров (кто нажал /start)
                cur.execute(f"SELECT COUNT(DISTINCT user_id) as cnt FROM {self._user_events_table} WHERE event_type = 'start'")
                row = cur.fetchone()
                stats["users_total"] = int(row["cnt"]) if row else 0

                # Общая выручка и кол-во платежей
                cur.execute(f"SELECT COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM {self._payments_table}")
                row = cur.fetchone()
                if row:
                    total_payments = int(row["cnt"])
                    stats["total_revenue"] = float(row["total"])
                    if total_payments > 0:
                        stats["avg_check"] = round(stats["total_revenue"] / total_payments, 2)

                # Платящие юзеры
                cur.execute(f"SELECT COUNT(DISTINCT user_id) as cnt FROM {self._payments_table}")
                row = cur.fetchone()
                stats["paid_users"] = int(row["cnt"]) if row else 0

                # Конверсия
                if stats["users_total"] > 0:
                    stats["conversion"] = round((stats["paid_users"] / stats["users_total"]) * 100, 2)

                # Покупок по типам (один запрос)
                cur.execute(f"""SELECT product_type, COUNT(*) as cnt FROM {self._payments_table}
                               GROUP BY product_type""")
                for row in cur.fetchall():
                    pt = row["product_type"]
                    cnt = int(row["cnt"])
                    if pt == "fast":
                        stats["express_purchases"] = cnt
                    elif pt == "persona_create":
                        stats["persona_create_purchases"] = cnt
                    elif pt == "persona_topup":
                        stats["persona_topup_purchases"] = cnt
                    elif pt == "persona_pack":
                        stats["pack_purchases"] = cnt
                stats["persona_purchases"] = stats["persona_create_purchases"] + stats["persona_topup_purchases"] + stats["pack_purchases"]

                # Для маржи: генерации по типам и создания персон
                cur.execute(f"""SELECT
                    COUNT(*) FILTER (
                        WHERE event_type = 'generation'
                          AND event_data->>'mode' = 'fast'
                          AND event_data->>'provider' = 'nano-banana-pro'
                    ) as fast_nano,
                    COUNT(*) FILTER (
                        WHERE event_type = 'generation'
                          AND event_data->>'mode' = 'fast'
                          AND (event_data->>'provider' IS NULL OR event_data->>'provider' = 'seedream')
                    ) as fast_seedream,
                    COUNT(*) FILTER (WHERE event_data->>'mode' = 'persona') as persona
                FROM {self._user_events_table} WHERE event_type = 'generation'""")
                row = cur.fetchone()
                fast_nano = int(row["fast_nano"] or 0) if row else 0
                fast_seedream = int(row["fast_seedream"] or 0) if row else 0
                persona_gens = int(row["persona"] or 0) if row else 0
                stats["fast_generations_by_provider"]["seedream"] = fast_seedream
                stats["fast_generations_by_provider"]["nano_banana"] = fast_nano

                cur.execute(f"SELECT COUNT(*) as cnt FROM {self._payments_table} WHERE product_type = 'persona_create'")
                row = cur.fetchone()
                persona_creates = int(row["cnt"]) if row else 0

                # Себестоимость паков из admin_pack_costs + user_events
                pack_cost_total_usd = 0.0
                try:
                    pack_costs_map = _pack_costs_map if _pack_costs_map is not None else self.get_pack_costs_map()
                    if pack_costs_map:
                        cur.execute(f"""
                            SELECT event_data->>'pack_id' as pack_id, COUNT(*) as cnt
                            FROM {self._user_events_table}
                            WHERE event_type IN ('pack_generation', 'pack_callback', 'pack_fallback')
                            GROUP BY event_data->>'pack_id'
                        """)
                        for row in cur.fetchall():
                            pid = int(row["pack_id"]) if row["pack_id"] and row["pack_id"].isdigit() else 0
                            if pid in pack_costs_map:
                                pack_cost_total_usd += pack_costs_map[pid] * int(row["cnt"])
                except Exception:
                    pass

                # Расчёт себестоимости и маржи
                fast_costs = self._calculate_fast_costs(
                    fast_seedream=fast_seedream,
                    fast_nano=fast_nano,
                    cost_fast_photo=COST_FAST_PHOTO,
                    cost_nano_banana=COST_NANO_BANANA,
                    usd_rub=USD_RUB,
                )
                stats["fast_costs_by_provider"]["seedream"] = fast_costs["fast_seedream"]
                stats["fast_costs_by_provider"]["nano_banana"] = fast_costs["fast_nano"]
                total_cost = (
                    persona_creates * COST_PERSONA_CREATE * USD_RUB
                    + fast_costs["fast_total"]
                    + persona_gens * COST_PERSONA_PHOTO * USD_RUB
                    + pack_cost_total_usd * USD_RUB
                )
                stats["total_cost"] = round(total_cost, 2)
                margin = stats["total_revenue"] - total_cost
                stats["margin"]["amount"] = round(margin, 2)
                stats["margin"]["percent"] = round((margin / stats["total_revenue"]) * 100, 2) if stats["total_revenue"] > 0 else 0.0

                # Генераций на платящего юзера — INNER JOIN вместо EXISTS
                if stats["paid_users"] > 0:
                    cur.execute(f"""SELECT COUNT(*) as cnt
                                   FROM {self._user_events_table} ue
                                   INNER JOIN (SELECT DISTINCT user_id FROM {self._payments_table}) p ON p.user_id = ue.user_id
                                   WHERE ue.event_type = 'generation'""")
                    row = cur.fetchone()
                    paying_gens = int(row["cnt"]) if row else 0
                    stats["gens_per_paying_user"] = round(paying_gens / stats["paid_users"], 1)

                # Статистика по полу
                cur.execute(f"""SELECT
                    COUNT(*) FILTER (WHERE subject_gender = 'male') as male,
                    COUNT(*) FILTER (WHERE subject_gender = 'female') as female,
                    COUNT(*) FILTER (WHERE subject_gender IS NULL) as unknown,
                    COUNT(*) as total
                FROM {self._users_table}""")
                row = cur.fetchone()
                if row and int(row["total"] or 0) > 0:
                    total = int(row["total"])
                    stats["gender"]["male"] = round((int(row["male"] or 0) / total) * 100, 1)
                    stats["gender"]["female"] = round((int(row["female"] or 0) / total) * 100, 1)
                    stats["gender"]["unknown"] = round((int(row["unknown"] or 0) / total) * 100, 1)

                # Среднее количество дней до первой покупки
                cur.execute(f"""
                    SELECT AVG(days) as avg_days FROM (
                        SELECT
                            p.user_id,
                            EXTRACT(EPOCH FROM (MIN(p.created_at) - MIN(ue.created_at))) / 86400 as days
                        FROM {self._payments_table} p
                        JOIN {self._user_events_table} ue ON ue.user_id = p.user_id AND ue.event_type = 'start'
                        GROUP BY p.user_id
                    ) t
                    WHERE days >= 0
                """)
                row = cur.fetchone()
                if row and row["avg_days"] is not None:
                    stats["days_to_first_purchase"] = round(float(row["avg_days"]), 1)

        return stats

    def get_hourly_activity(self, date_from: str | None = None, date_to: str | None = None) -> list[int]:
        """Получить статистику уникальных пользователей по часам (МСК)."""
        # Возвращаем список из 24 чисел (по количеству уникальных юзеров в каждый час)
        result = [0] * 24

        with self._connect() as conn:
            if not self._use_pg:
                return result

            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Конвертируем UTC в МСК (UTC+3)
                if date_from and date_to:
                    cur.execute(f"""
                        SELECT
                            EXTRACT(HOUR FROM created_at AT TIME ZONE 'Europe/Moscow')::int as hour,
                            COUNT(DISTINCT user_id) as cnt
                        FROM {self._user_events_table}
                        WHERE created_at >= %s AND created_at <= %s
                        GROUP BY EXTRACT(HOUR FROM created_at AT TIME ZONE 'Europe/Moscow')
                        ORDER BY hour
                    """, (date_from, date_to + " 23:59:59"))
                else:
                    cur.execute(f"""
                        SELECT
                            EXTRACT(HOUR FROM created_at AT TIME ZONE 'Europe/Moscow')::int as hour,
                            COUNT(DISTINCT user_id) as cnt
                        FROM {self._user_events_table}
                        GROUP BY EXTRACT(HOUR FROM created_at AT TIME ZONE 'Europe/Moscow')
                        ORDER BY hour
                    """)

                for row in cur.fetchall():
                    hour = int(row["hour"])
                    if 0 <= hour < 24:
                        result[hour] = int(row["cnt"])

        return result

    def get_dashboard_stats(self, date_from: str | None = None, date_to: str | None = None, _cost_settings: dict | None = None, _pack_costs_map: dict | None = None) -> dict:
        """Получить статистику для дашборда с опциональной фильтрацией по датам."""
        # Себестоимость из настроек (принимаем кэш или грузим)
        cost_settings = _cost_settings or self.get_cost_settings()
        COST_PERSONA_CREATE = cost_settings["cost_persona_create"]
        COST_FAST_PHOTO = cost_settings["cost_fast_photo"]
        COST_NANO_BANANA = cost_settings.get("cost_nano_banana", COST_FAST_PHOTO)
        COST_PERSONA_PHOTO = cost_settings["cost_persona_photo"]
        USD_RUB = cost_settings["usd_rub"]

        stats = {
            "users": {"new": 0},
            "payments": {
                "total_count": 0, "total_revenue": 0.0, "avg_check": 0.0,
                "express": {"count": 0, "revenue": 0.0},
                "persona": {"count": 0, "revenue": 0.0},  # обратная совместимость (итого persona*)
                "persona_create": {"count": 0, "revenue": 0.0},
                "persona_topup": {"count": 0, "revenue": 0.0},
                "pack": {"count": 0, "revenue": 0.0},
            },
            "generations": {"total": 0, "free": 0, "fast": 0, "fast_seedream": 0, "fast_nano": 0, "persona": 0},
            "costs": {
                "total": 0.0,
                "persona_create": 0.0,
                "fast_photos": 0.0,
                "fast_seedream": 0.0,
                "fast_nano": 0.0,
                "persona_photos": 0.0,
                "packs": 0.0,
            },
            "margin": {"amount": 0.0, "percent": 0.0},
            "conversion": {"paid_users": 0, "percent": 0.0},
        }

        # Подготовка дат (МСК → UTC)
        has_dates = bool(date_from and date_to)
        if has_dates:
            d_from = str(date_from) + " 00:00:00+03"
            d_to = str(date_to) + " 23:59:59+03"

        with self._connect() as conn:
            if not self._use_pg:
                # SQLite: упрощённая версия
                row = conn.execute(f"SELECT COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM {self._payments_table}").fetchone()
                if row:
                    stats["payments"]["total_count"] = int(row["cnt"])
                    stats["payments"]["total_revenue"] = float(row["total"])
                return stats

            # PostgreSQL
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Новые юзеры за период: считаем по дате ПЕРВОГО события start для каждого юзера
                if has_dates:
                    cur.execute(
                        f"""
                        SELECT COUNT(*) as cnt FROM (
                            SELECT user_id, MIN(created_at) as first_start
                            FROM {self._user_events_table}
                            WHERE event_type = 'start'
                            GROUP BY user_id
                            HAVING MIN(created_at) >= %s AND MIN(created_at) <= %s
                        ) t
                        """,
                        (d_from, d_to),
                    )
                    row = cur.fetchone()
                    stats["users"]["new"] = int(row["cnt"]) if row else 0
                else:
                    # Без дат — все уникальные юзеры с хотя бы одним start
                    cur.execute(f"SELECT COUNT(DISTINCT user_id) as cnt FROM {self._user_events_table} WHERE event_type = 'start'")
                    row = cur.fetchone()
                    stats["users"]["new"] = int(row["cnt"]) if row else 0

                # Платежи
                if has_dates:
                    cur.execute(
                        f"SELECT COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM {self._payments_table} WHERE created_at >= %s AND created_at <= %s",
                        (d_from, d_to),
                    )
                else:
                    cur.execute(f"SELECT COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM {self._payments_table}")
                row = cur.fetchone()
                if row:
                    stats["payments"]["total_count"] = int(row["cnt"])
                    stats["payments"]["total_revenue"] = float(row["total"])
                    if stats["payments"]["total_count"] > 0:
                        stats["payments"]["avg_check"] = round(stats["payments"]["total_revenue"] / stats["payments"]["total_count"], 2)

                # Платежи по типам (один запрос вместо нескольких)
                if has_dates:
                    cur.execute(
                        f"SELECT product_type, COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM {self._payments_table} WHERE created_at >= %s AND created_at <= %s GROUP BY product_type",
                        (d_from, d_to),
                    )
                else:
                    cur.execute(f"SELECT product_type, COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM {self._payments_table} GROUP BY product_type")
                persona_total_cnt, persona_total_rev = 0, 0.0
                for row in cur.fetchall():
                    pt = row["product_type"]
                    cnt = int(row["cnt"])
                    rev = float(row["total"])
                    if pt == "fast":
                        stats["payments"]["express"]["count"] = cnt
                        stats["payments"]["express"]["revenue"] = rev
                    elif pt == "persona_create":
                        stats["payments"]["persona_create"]["count"] = cnt
                        stats["payments"]["persona_create"]["revenue"] = rev
                        persona_total_cnt += cnt
                        persona_total_rev += rev
                    elif pt == "persona_topup":
                        stats["payments"]["persona_topup"]["count"] = cnt
                        stats["payments"]["persona_topup"]["revenue"] = rev
                        persona_total_cnt += cnt
                        persona_total_rev += rev
                    elif pt == "persona_pack":
                        stats["payments"]["pack"]["count"] = cnt
                        stats["payments"]["pack"]["revenue"] = rev
                        persona_total_cnt += cnt
                        persona_total_rev += rev
                stats["payments"]["persona"]["count"] = persona_total_cnt
                stats["payments"]["persona"]["revenue"] = persona_total_rev

                # Генерации по типам (fast / persona)
                if has_dates:
                    cur.execute(
                        f"""SELECT
                            COUNT(*) as total,
                            COUNT(*) FILTER (WHERE event_data->>'mode' = 'fast') as fast,
                            COUNT(*) FILTER (
                                WHERE event_data->>'mode' = 'fast'
                                  AND event_data->>'provider' = 'nano-banana-pro'
                            ) as fast_nano,
                            COUNT(*) FILTER (
                                WHERE event_data->>'mode' = 'fast'
                                  AND (event_data->>'provider' IS NULL OR event_data->>'provider' = 'seedream')
                            ) as fast_seedream,
                            COUNT(*) FILTER (WHERE event_data->>'mode' = 'persona') as persona
                        FROM {self._user_events_table}
                        WHERE event_type = 'generation' AND created_at >= %s AND created_at <= %s""",
                        (d_from, d_to),
                    )
                else:
                    cur.execute(
                        f"""SELECT
                            COUNT(*) as total,
                            COUNT(*) FILTER (WHERE event_data->>'mode' = 'fast') as fast,
                            COUNT(*) FILTER (
                                WHERE event_data->>'mode' = 'fast'
                                  AND event_data->>'provider' = 'nano-banana-pro'
                            ) as fast_nano,
                            COUNT(*) FILTER (
                                WHERE event_data->>'mode' = 'fast'
                                  AND (event_data->>'provider' IS NULL OR event_data->>'provider' = 'seedream')
                            ) as fast_seedream,
                            COUNT(*) FILTER (WHERE event_data->>'mode' = 'persona') as persona
                        FROM {self._user_events_table}
                        WHERE event_type = 'generation'"""
                    )
                row = cur.fetchone()
                if row:
                    stats["generations"]["total"] = int(row["total"] or 0)
                    stats["generations"]["fast"] = int(row["fast"] or 0)
                    stats["generations"]["fast_nano"] = int(row["fast_nano"] or 0)
                    stats["generations"]["fast_seedream"] = int(row["fast_seedream"] or 0)
                    stats["generations"]["persona"] = int(row["persona"] or 0)

                # Бесплатные генерации (юзеры без платежей) — LEFT JOIN вместо NOT EXISTS
                if has_dates:
                    cur.execute(
                        f"""SELECT COUNT(*) as cnt
                            FROM {self._user_events_table} ue
                            LEFT JOIN (SELECT DISTINCT user_id FROM {self._payments_table}) p ON p.user_id = ue.user_id
                            WHERE ue.event_type = 'generation'
                              AND ue.created_at >= %s AND ue.created_at <= %s
                              AND p.user_id IS NULL""",
                        (d_from, d_to),
                    )
                else:
                    cur.execute(f"""SELECT COUNT(*) as cnt
                                   FROM {self._user_events_table} ue
                                   LEFT JOIN (SELECT DISTINCT user_id FROM {self._payments_table}) p ON p.user_id = ue.user_id
                                   WHERE ue.event_type = 'generation'
                                     AND p.user_id IS NULL""")
                row = cur.fetchone()
                stats["generations"]["free"] = int(row["cnt"]) if row else 0

                # Создания персон (покупки persona_create)
                if has_dates:
                    cur.execute(
                        f"SELECT COUNT(*) as cnt FROM {self._payments_table} WHERE product_type = 'persona_create' AND created_at >= %s AND created_at <= %s",
                        (d_from, d_to),
                    )
                else:
                    cur.execute(f"SELECT COUNT(*) as cnt FROM {self._payments_table} WHERE product_type = 'persona_create'")
                row = cur.fetchone()
                persona_creates = int(row["cnt"]) if row else 0

                # Себестоимость паков за период
                pack_cost_usd = 0.0
                try:
                    pack_costs_map = _pack_costs_map if _pack_costs_map is not None else self.get_pack_costs_map()
                    if pack_costs_map:
                        if has_dates:
                            cur.execute(f"""
                                SELECT event_data->>'pack_id' as pack_id, COUNT(*) as cnt
                                FROM {self._user_events_table}
                                WHERE event_type IN ('pack_generation', 'pack_callback', 'pack_fallback')
                                  AND created_at >= %s AND created_at <= %s
                                GROUP BY event_data->>'pack_id'
                            """, (d_from, d_to))
                        else:
                            cur.execute(f"""
                                SELECT event_data->>'pack_id' as pack_id, COUNT(*) as cnt
                                FROM {self._user_events_table}
                                WHERE event_type IN ('pack_generation', 'pack_callback', 'pack_fallback')
                                GROUP BY event_data->>'pack_id'
                            """)
                        for row in cur.fetchall():
                            pid = int(row["pack_id"]) if row["pack_id"] and row["pack_id"].isdigit() else 0
                            if pid in pack_costs_map:
                                pack_cost_usd += pack_costs_map[pid] * int(row["cnt"])
                except Exception:
                    pass

                # Расчёт себестоимости
                cost_persona_create = persona_creates * COST_PERSONA_CREATE * USD_RUB
                fast_costs = self._calculate_fast_costs(
                    fast_seedream=stats["generations"]["fast_seedream"],
                    fast_nano=stats["generations"]["fast_nano"],
                    cost_fast_photo=COST_FAST_PHOTO,
                    cost_nano_banana=COST_NANO_BANANA,
                    usd_rub=USD_RUB,
                )
                cost_fast_photos = fast_costs["fast_total"]
                cost_persona_photos = stats["generations"]["persona"] * COST_PERSONA_PHOTO * USD_RUB
                cost_packs = pack_cost_usd * USD_RUB
                total_cost = cost_persona_create + cost_fast_photos + cost_persona_photos + cost_packs

                stats["costs"]["persona_create"] = round(cost_persona_create, 2)
                stats["costs"]["fast_photos"] = round(cost_fast_photos, 2)
                stats["costs"]["fast_seedream"] = fast_costs["fast_seedream"]
                stats["costs"]["fast_nano"] = fast_costs["fast_nano"]
                stats["costs"]["persona_photos"] = round(cost_persona_photos, 2)
                stats["costs"]["packs"] = round(cost_packs, 2)
                stats["costs"]["total"] = round(total_cost, 2)

                # Маржа
                revenue = stats["payments"]["total_revenue"]
                margin = revenue - total_cost
                stats["margin"]["amount"] = round(margin, 2)
                stats["margin"]["percent"] = round((margin / revenue) * 100, 2) if revenue > 0 else 0.0

                # Конверсия за период: платящие юзеры / новые юзеры
                if has_dates:
                    cur.execute(
                        f"SELECT COUNT(DISTINCT user_id) as cnt FROM {self._payments_table} WHERE created_at >= %s AND created_at <= %s",
                        (d_from, d_to),
                    )
                else:
                    cur.execute(f"SELECT COUNT(DISTINCT user_id) as cnt FROM {self._payments_table}")
                row = cur.fetchone()
                paid_users = int(row["cnt"]) if row else 0
                stats["conversion"]["paid_users"] = paid_users
                new_users = stats["users"]["new"]
                if new_users > 0:
                    stats["conversion"]["percent"] = round((paid_users / new_users) * 100, 2)

        return stats

    def get_chart_data(self, days: int = 30) -> dict:
        """Получить данные для графиков за последние N дней."""
        with self._connect() as conn:
            result = {"dates": [], "revenue": [], "users": [], "generations": []}

            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # PostgreSQL: INTERVAL '1 day' * N вместо INTERVAL '%s days'
                    # Выручка по дням
                    cur.execute(
                        f"""
                        SELECT DATE(created_at) as day, COALESCE(SUM(amount_rub), 0) as total
                        FROM {self._payments_table}
                        WHERE created_at >= NOW() - INTERVAL '1 day' * %s
                        GROUP BY DATE(created_at)
                        ORDER BY day
                        """,
                        (days,),
                    )
                    revenue_data = {str(row["day"]): float(row["total"]) for row in cur.fetchall()}

                    # Новые юзеры по дням (по событию start)
                    cur.execute(
                        f"""
                        SELECT DATE(created_at) as day, COUNT(DISTINCT user_id) as cnt
                        FROM {self._user_events_table}
                        WHERE event_type = 'start' AND created_at >= NOW() - INTERVAL '1 day' * %s
                        GROUP BY DATE(created_at)
                        ORDER BY day
                        """,
                        (days,),
                    )
                    users_data = {str(row["day"]): int(row["cnt"]) for row in cur.fetchall()}

                    # Генерации по дням
                    cur.execute(
                        f"""
                        SELECT DATE(created_at) as day, COUNT(*) as cnt
                        FROM {self._user_events_table}
                        WHERE event_type = 'generation' AND created_at >= NOW() - INTERVAL '1 day' * %s
                        GROUP BY DATE(created_at)
                        ORDER BY day
                        """,
                        (days,),
                    )
                    gen_data = {str(row["day"]): int(row["cnt"]) for row in cur.fetchall()}

                    # Заполняем все дни
                    from datetime import datetime, timedelta
                    for i in range(days):
                        day = (datetime.now() - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
                        result["dates"].append(day)
                        result["revenue"].append(revenue_data.get(day, 0))
                        result["users"].append(users_data.get(day, 0))
                        result["generations"].append(gen_data.get(day, 0))
            else:
                # SQLite версия
                from datetime import datetime, timedelta
                for i in range(days):
                    day = (datetime.now() - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
                    result["dates"].append(day)

                    row = conn.execute(f"SELECT COALESCE(SUM(amount_rub), 0) as total FROM {self._payments_table} WHERE DATE(created_at) = ?", (day,)).fetchone()
                    result["revenue"].append(float(row["total"]) if row else 0)

                    row = conn.execute(f"SELECT COUNT(*) as cnt FROM {self._users_table} WHERE DATE(created_at) = ?", (day,)).fetchone()
                    result["users"].append(int(row["cnt"]) if row else 0)

                    row = conn.execute(f"SELECT COUNT(*) as cnt FROM {self._user_events_table} WHERE event_type = 'generation' AND DATE(created_at) = ?", (day,)).fetchone()
                    result["generations"].append(int(row["cnt"]) if row else 0)

            return result

    def get_user_history(self, user_id: int) -> dict:
        """Получить историю платежей и событий пользователя."""
        with self._connect() as conn:
            result = {"payments": [], "events": []}

            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"SELECT * FROM {self._payments_table} WHERE user_id = %s ORDER BY created_at DESC LIMIT 100",
                        (int(user_id),),
                    )
                    result["payments"] = [dict(row) for row in cur.fetchall()]

                    cur.execute(
                        f"SELECT * FROM {self._user_events_table} WHERE user_id = %s ORDER BY created_at DESC LIMIT 100",
                        (int(user_id),),
                    )
                    result["events"] = [dict(row) for row in cur.fetchall()]
            else:
                rows = conn.execute(f"SELECT * FROM {self._payments_table} WHERE user_id = ? ORDER BY created_at DESC LIMIT 100", (int(user_id),)).fetchall()
                result["payments"] = [dict(row) for row in rows]

                rows = conn.execute(f"SELECT * FROM {self._user_events_table} WHERE user_id = ? ORDER BY created_at DESC LIMIT 100", (int(user_id),)).fetchall()
                result["events"] = [dict(row) for row in rows]

            return result

    def adjust_user_credits(self, user_id: int, credit_type: str, delta: int) -> bool:
        """Изменить кредиты пользователя. credit_type: 'fast' или 'persona'."""
        profile = self.get_user(user_id)
        if credit_type == "fast":
            new_value = max(0, profile.paid_generations_remaining + delta)
            self.set_paid_generations_remaining(user_id, new_value)
            return True
        elif credit_type == "persona":
            new_value = max(0, profile.persona_credits_remaining + delta)
            self.set_persona_credits(user_id, new_value)
            return True
        return False

    # ========================================
    # СТИЛИ ПЕРСОНЫ
    # ========================================

    def get_persona_styles(self, *, active_only: bool = False, gender: str | None = None) -> list[dict]:
        """Получить список стилей персоны."""
        conditions = []
        params: list = []
        if active_only:
            conditions.append("is_active = " + ("TRUE" if self._use_pg else "1"))
        if gender:
            conditions.append("gender = %s")
            params.append(gender)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT * FROM {self._persona_styles_table}{where} ORDER BY sort_order, id"
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, params)
                    return [dict(row) for row in cur.fetchall()]
        else:
            with self._connect() as conn:
                rows = conn.execute(sql.replace("%s", "?"), params).fetchall()
                return [dict(row) for row in rows]

    def get_persona_style(self, style_id: int) -> dict | None:
        """Получить один стиль по id."""
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"SELECT * FROM {self._persona_styles_table} WHERE id = %s", (int(style_id),))
                    row = cur.fetchone()
                    return dict(row) if row else None
        else:
            with self._connect() as conn:
                row = conn.execute(
                    f"SELECT * FROM {self._persona_styles_table} WHERE id = ?", (int(style_id),)
                ).fetchone()
                return dict(row) if row else None

    def get_persona_style_by_slug(self, slug: str) -> dict | None:
        """Получить один стиль по slug."""
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"SELECT * FROM {self._persona_styles_table} WHERE slug = %s", (slug,))
                    row = cur.fetchone()
                    return dict(row) if row else None
        else:
            with self._connect() as conn:
                row = conn.execute(
                    f"SELECT * FROM {self._persona_styles_table} WHERE slug = ?", (slug,)
                ).fetchone()
                return dict(row) if row else None

    def swap_persona_style_order(self, style_id_a: int, style_id_b: int) -> bool:
        """Поменять sort_order двух стилей местами и перенумеровать."""
        if not self._use_pg:
            return False
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"SELECT id, sort_order FROM {self._persona_styles_table} WHERE id IN (%s, %s)",
                    (int(style_id_a), int(style_id_b)),
                )
                rows = cur.fetchall()
                if len(rows) != 2:
                    return False
                a, b = rows[0], rows[1]
                cur.execute(
                    f"UPDATE {self._persona_styles_table} SET sort_order = %s, updated_at = NOW() WHERE id = %s",
                    (b["sort_order"], a["id"]),
                )
                cur.execute(
                    f"UPDATE {self._persona_styles_table} SET sort_order = %s, updated_at = NOW() WHERE id = %s",
                    (a["sort_order"], b["id"]),
                )
                conn.commit()
        self._renumber_persona_styles()
        return True

    def _renumber_persona_styles(self) -> None:
        """Перенумеровать sort_order всех стилей подряд 1, 2, 3, ..."""
        if not self._use_pg:
            return
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"SELECT id FROM {self._persona_styles_table} ORDER BY sort_order, id")
                rows = cur.fetchall()
                for i, row in enumerate(rows, start=1):
                    cur.execute(
                        f"UPDATE {self._persona_styles_table} SET sort_order = %s WHERE id = %s",
                        (i, row["id"]),
                    )
                conn.commit()

    def _shift_persona_style_sort_order(self, sort_order: int, exclude_id: int | None = None) -> None:
        """Сдвинуть sort_order >= заданного на +1, чтобы освободить место."""
        if not self._use_pg:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                if exclude_id is not None:
                    cur.execute(
                        f"UPDATE {self._persona_styles_table} SET sort_order = sort_order + 1 WHERE sort_order >= %s AND id != %s",
                        (int(sort_order), int(exclude_id)),
                    )
                else:
                    cur.execute(
                        f"UPDATE {self._persona_styles_table} SET sort_order = sort_order + 1 WHERE sort_order >= %s",
                        (int(sort_order),),
                    )
                conn.commit()
        self._renumber_persona_styles()

    def create_persona_style(self, *, slug: str, title: str, description: str = "",
                             gender: str = "female", image_url: str = "",
                             prompt: str = "", sort_order: int = 0,
                             credit_cost: int = 4) -> int | None:
        """Создать стиль. Возвращает id."""
        sql = f"""
            INSERT INTO {self._persona_styles_table} (slug, title, description, gender, image_url, prompt, sort_order, credit_cost)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (slug, title, description, gender, image_url, prompt, int(sort_order), int(credit_cost))
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql + " RETURNING id", params)
                    row = cur.fetchone()
                    conn.commit()
                    return int(row["id"]) if row else None
        else:
            with self._connect() as conn:
                cur = conn.execute(sql.replace("%s", "?"), params)
                conn.commit()
                return cur.lastrowid

    def update_persona_style(self, style_id: int, **kwargs) -> bool:
        """Обновить стиль. Допустимые поля: slug, title, description, gender, image_url, sort_order, is_active, credit_cost."""
        allowed = {"slug", "title", "description", "gender", "image_url", "prompt", "sort_order", "is_active", "credit_cost", "cost_usd"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        if self._use_pg:
            fields["updated_at"] = "NOW()"
        set_parts = []
        params: list = []
        for k, v in fields.items():
            if v == "NOW()":
                set_parts.append(f"{k} = NOW()")
            else:
                set_parts.append(f"{k} = %s")
                params.append(v)
        params.append(int(style_id))
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"UPDATE {self._persona_styles_table} SET {', '.join(set_parts)} WHERE id = %s",
                        params,
                    )
                    updated = cur.rowcount > 0
                    conn.commit()
                    return updated
        else:
            sql = f"UPDATE {self._persona_styles_table} SET {', '.join(set_parts)} WHERE id = ?"
            with self._connect() as conn:
                cur = conn.execute(sql.replace("%s", "?"), params)
                conn.commit()
                return cur.rowcount > 0

    def delete_persona_style(self, style_id: int) -> bool:
        """Удалить стиль."""
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"DELETE FROM {self._persona_styles_table} WHERE id = %s", (int(style_id),))
                    deleted = cur.rowcount > 0
                    conn.commit()
                    return deleted
        else:
            with self._connect() as conn:
                cur = conn.execute(
                    f"DELETE FROM {self._persona_styles_table} WHERE id = ?", (int(style_id),)
                )
                conn.commit()
                return cur.rowcount > 0

    # ========================================
    # Persona Style Previews (превью-фото для стилей)
    # ========================================

    def get_style_previews(self, style_id: int) -> list[str]:
        """Получить список URL превью для стиля, отсортированных по sort_order."""
        sql = f"SELECT image_url FROM {self._persona_style_previews_table} WHERE style_id = %s ORDER BY sort_order, id"
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, (int(style_id),))
                    return [row["image_url"] for row in cur.fetchall()]
        else:
            with self._connect() as conn:
                rows = conn.execute(sql.replace("%s", "?"), (int(style_id),)).fetchall()
                return [row["image_url"] if isinstance(row, dict) else row[0] for row in rows]

    def set_style_previews(self, style_id: int, urls: list[str]) -> None:
        """Заменить превью стиля. Удаляет старые, вставляет новые (max 4)."""
        urls = urls[:4]
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"DELETE FROM {self._persona_style_previews_table} WHERE style_id = %s", (int(style_id),))
                    for i, url in enumerate(urls):
                        cur.execute(
                            f"INSERT INTO {self._persona_style_previews_table} (style_id, image_url, sort_order) VALUES (%s, %s, %s)",
                            (int(style_id), url, i),
                        )
                    conn.commit()
        else:
            with self._connect() as conn:
                conn.execute(f"DELETE FROM {self._persona_style_previews_table} WHERE style_id = ?", (int(style_id),))
                for i, url in enumerate(urls):
                    conn.execute(
                        f"INSERT INTO {self._persona_style_previews_table} (style_id, image_url, sort_order) VALUES (?, ?, ?)",
                        (int(style_id), url, i),
                    )
                conn.commit()

    def get_all_style_previews_map(self) -> dict[int, list[str]]:
        """Получить превью для всех стилей (batch, без N+1). Возвращает {style_id: [url, ...]}."""
        sql = f"SELECT style_id, image_url FROM {self._persona_style_previews_table} ORDER BY style_id, sort_order, id"
        result: dict[int, list[str]] = {}
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql)
                    for row in cur.fetchall():
                        result.setdefault(int(row["style_id"]), []).append(row["image_url"])
        else:
            with self._connect() as conn:
                rows = conn.execute(sql).fetchall()
                for row in rows:
                    sid = row["style_id"] if isinstance(row, dict) else row[0]
                    url = row["image_url"] if isinstance(row, dict) else row[1]
                    result.setdefault(int(sid), []).append(url)
        return result

    def get_persona_style_stats(self, date_from: str | None = None, date_to: str | None = None) -> dict[int, dict]:
        """Генерации по стилям persona: {style_id: {generations: N}}.

        log_event в bot.py пишет числовой style_id в event_data->>'style'.
        """
        if not self._use_pg:
            return {}
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = f"""
                    SELECT event_data->>'style' AS style_id, COUNT(*) AS generations
                    FROM {self._user_events_table}
                    WHERE event_type = 'generation' AND event_data->>'mode' = 'persona'
                """
                params: list = []
                if date_from:
                    sql += " AND created_at >= %s"
                    params.append(date_from)
                if date_to:
                    sql += " AND created_at <= %s"
                    params.append(date_to)
                sql += " GROUP BY style_id"
                cur.execute(sql, params)
                result: dict[int, dict] = {}
                for row in cur.fetchall():
                    sid = row.get("style_id")
                    if sid and str(sid).isdigit():
                        result[int(sid)] = {"generations": int(row["generations"])}
                return result

    def set_persona_style_costs_bulk(self, items: list[dict]) -> None:
        """Batch update cost_usd for persona styles. items: [{style_id, cost_usd}]."""
        for item in items:
            self.update_persona_style(int(item["style_id"]), cost_usd=float(item.get("cost_usd", 0)))

    # ========================================
    # Express Styles (экспресс-фото стили из БД)
    # ========================================

    def get_express_styles(self, *, active_only: bool = False, gender: str | None = None) -> list[dict]:
        """Получить список экспресс-стилей."""
        conditions = []
        params: list = []
        if active_only:
            conditions.append("is_active = " + ("TRUE" if self._use_pg else "1"))
        if gender:
            conditions.append("gender = %s")
            params.append(gender)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT * FROM {self._express_styles_table}{where} ORDER BY sort_order, id"
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, params)
                    return [dict(row) for row in cur.fetchall()]
        else:
            with self._connect() as conn:
                rows = conn.execute(sql.replace("%s", "?"), params).fetchall()
                return [dict(row) for row in rows]

    def get_express_style(self, style_id: int) -> dict | None:
        """Получить один экспресс-стиль по id."""
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"SELECT * FROM {self._express_styles_table} WHERE id = %s", (int(style_id),))
                    row = cur.fetchone()
                    return dict(row) if row else None
        else:
            with self._connect() as conn:
                row = conn.execute(
                    f"SELECT * FROM {self._express_styles_table} WHERE id = ?", (int(style_id),)
                ).fetchone()
                return dict(row) if row else None

    def get_express_style_by_slug(self, slug: str) -> dict | None:
        """Получить один экспресс-стиль по slug."""
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"SELECT * FROM {self._express_styles_table} WHERE slug = %s", (slug,))
                    row = cur.fetchone()
                    return dict(row) if row else None
        else:
            with self._connect() as conn:
                row = conn.execute(
                    f"SELECT * FROM {self._express_styles_table} WHERE slug = ?", (slug,)
                ).fetchone()
                return dict(row) if row else None

    def get_express_themes(self, *, gender: str | None = None, active_only: bool = True) -> list[str]:
        """Получить список уникальных тем экспресс-стилей (для drill-down)."""
        conditions = []
        params: list = []
        if active_only:
            conditions.append("is_active = " + ("TRUE" if self._use_pg else "1"))
        if gender:
            conditions.append("gender = %s")
            params.append(gender)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT DISTINCT theme FROM {self._express_styles_table}{where} ORDER BY theme"
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, params)
                    return [row["theme"] for row in cur.fetchall()]
        else:
            with self._connect() as conn:
                rows = conn.execute(sql.replace("%s", "?"), params).fetchall()
                return [row["theme"] if isinstance(row, dict) else row[0] for row in rows]

    def create_express_style(self, *, slug: str, title: str, emoji: str = "",
                             theme: str = "general", gender: str = "female",
                             prompt: str = "", negative_prompt: str = "",
                             provider: str = "", image_url: str = "",
                             model_params: str = "", sort_order: int = 0,
                             # backward compat: model= falls back to provider
                             model: str = "") -> int | None:
        """Создать экспресс-стиль. Возвращает id."""
        actual_provider = provider or model or "seedream"
        sql = f"""
            INSERT INTO {self._express_styles_table}
            (slug, title, emoji, theme, gender, prompt, negative_prompt, provider, model, image_url, model_params, sort_order)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (slug, title, emoji, theme, gender, prompt, negative_prompt,
                  actual_provider, actual_provider, image_url, model_params, int(sort_order))
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql + " RETURNING id", params)
                    row = cur.fetchone()
                    conn.commit()
                    return int(row["id"]) if row else None
        else:
            with self._connect() as conn:
                cur = conn.execute(sql.replace("%s", "?"), params)
                conn.commit()
                return cur.lastrowid

    def update_express_style(self, style_id: int, **kwargs) -> bool:
        """Обновить экспресс-стиль."""
        allowed = {"slug", "title", "emoji", "theme", "gender", "prompt", "negative_prompt",
                   "provider", "model", "image_url", "model_params", "sort_order", "is_active"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        # Если передан provider — синхронизируем model для совместимости
        if "provider" in fields and "model" not in fields:
            fields["model"] = fields["provider"]
        if self._use_pg:
            fields["updated_at"] = "NOW()"
        set_parts = []
        params: list = []
        for k, v in fields.items():
            if v == "NOW()":
                set_parts.append(f"{k} = NOW()")
            else:
                set_parts.append(f"{k} = %s")
                params.append(v)
        params.append(int(style_id))
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"UPDATE {self._express_styles_table} SET {', '.join(set_parts)} WHERE id = %s",
                        params,
                    )
                    updated = cur.rowcount > 0
                    conn.commit()
                    return updated
        else:
            sql = f"UPDATE {self._express_styles_table} SET {', '.join(set_parts)} WHERE id = ?"
            with self._connect() as conn:
                cur = conn.execute(sql.replace("%s", "?"), params)
                conn.commit()
                return cur.rowcount > 0

    def upsert_express_style(self, *, slug: str, title: str, emoji: str = "",
                              theme: str = "general", gender: str = "female",
                              prompt: str = "", negative_prompt: str = "",
                              provider: str = "seedream", image_url: str = "",
                              model_params: str = "", sort_order: int = 0,
                              model: str = "") -> int | None:
        """Upsert экспресс-стиля по slug (для сидера). Возвращает id."""
        existing = self.get_express_style_by_slug(slug)
        if existing:
            self.update_express_style(
                existing["id"], title=title, emoji=emoji, theme=theme,
                gender=gender, prompt=prompt, negative_prompt=negative_prompt,
                provider=provider or model or "seedream", image_url=image_url,
                model_params=model_params, sort_order=sort_order,
            )
            return existing["id"]
        return self.create_express_style(
            slug=slug, title=title, emoji=emoji, theme=theme,
            gender=gender, prompt=prompt, negative_prompt=negative_prompt,
            provider=provider or model or "seedream", image_url=image_url,
            model_params=model_params, sort_order=sort_order,
        )

    def delete_express_style(self, style_id: int) -> bool:
        """Удалить экспресс-стиль."""
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"DELETE FROM {self._express_styles_table} WHERE id = %s", (int(style_id),))
                    deleted = cur.rowcount > 0
                    conn.commit()
                    return deleted
        else:
            with self._connect() as conn:
                cur = conn.execute(
                    f"DELETE FROM {self._express_styles_table} WHERE id = ?", (int(style_id),)
                )
                conn.commit()
                return cur.rowcount > 0

    def swap_express_style_order(self, style_id_a: int, style_id_b: int) -> bool:
        """Поменять sort_order двух экспресс-стилей местами."""
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"SELECT id, sort_order FROM {self._express_styles_table} WHERE id IN (%s, %s)",
                        (int(style_id_a), int(style_id_b)),
                    )
                    rows = cur.fetchall()
                    if len(rows) != 2:
                        return False
                    a, b = rows[0], rows[1]
                    cur.execute(
                        f"UPDATE {self._express_styles_table} SET sort_order = %s, updated_at = NOW() WHERE id = %s",
                        (b["sort_order"], a["id"]),
                    )
                    cur.execute(
                        f"UPDATE {self._express_styles_table} SET sort_order = %s, updated_at = NOW() WHERE id = %s",
                        (a["sort_order"], b["id"]),
                    )
                    conn.commit()
        else:
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT id, sort_order FROM {self._express_styles_table} WHERE id IN (?, ?)",
                    (int(style_id_a), int(style_id_b)),
                ).fetchall()
                if len(rows) != 2:
                    return False
                a, b = dict(rows[0]), dict(rows[1])
                conn.execute(
                    f"UPDATE {self._express_styles_table} SET sort_order = ? WHERE id = ?",
                    (b["sort_order"], a["id"]),
                )
                conn.execute(
                    f"UPDATE {self._express_styles_table} SET sort_order = ? WHERE id = ?",
                    (a["sort_order"], b["id"]),
                )
                conn.commit()
        self._renumber_express_styles()
        return True

    def _renumber_express_styles(self) -> None:
        """Перенумеровать sort_order всех экспресс-стилей подряд 1, 2, 3, ..."""
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"SELECT id FROM {self._express_styles_table} ORDER BY sort_order, id")
                    rows = cur.fetchall()
                    for i, row in enumerate(rows, start=1):
                        cur.execute(
                            f"UPDATE {self._express_styles_table} SET sort_order = %s WHERE id = %s",
                            (i, row["id"]),
                        )
                    conn.commit()
        else:
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT id FROM {self._express_styles_table} ORDER BY sort_order, id"
                ).fetchall()
                for i, row in enumerate(rows, start=1):
                    conn.execute(
                        f"UPDATE {self._express_styles_table} SET sort_order = ? WHERE id = ?",
                        (i, row["id"]),
                    )
                conn.commit()

    def _shift_express_style_sort_order(self, sort_order: int, exclude_id: int | None = None) -> None:
        """Сдвинуть sort_order >= заданного на +1, чтобы освободить место."""
        if self._use_pg:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    if exclude_id is not None:
                        cur.execute(
                            f"UPDATE {self._express_styles_table} SET sort_order = sort_order + 1 WHERE sort_order >= %s AND id != %s",
                            (int(sort_order), int(exclude_id)),
                        )
                    else:
                        cur.execute(
                            f"UPDATE {self._express_styles_table} SET sort_order = sort_order + 1 WHERE sort_order >= %s",
                            (int(sort_order),),
                        )
                    conn.commit()
        else:
            with self._connect() as conn:
                if exclude_id is not None:
                    conn.execute(
                        f"UPDATE {self._express_styles_table} SET sort_order = sort_order + 1 WHERE sort_order >= ? AND id != ?",
                        (int(sort_order), int(exclude_id)),
                    )
                else:
                    conn.execute(
                        f"UPDATE {self._express_styles_table} SET sort_order = sort_order + 1 WHERE sort_order >= ?",
                        (int(sort_order),),
                    )
                conn.commit()

    def move_express_style_to_order(self, style_id: int, new_order: int) -> None:
        """Переместить стиль на позицию new_order (directional shift).

        Move up (old=4 → new=2): сдвинуть [new..old-1] на +1, поставить style на new.
        Move down (old=2 → new=4): сдвинуть [old+1..new] на -1, поставить style на new.
        """
        sid = int(style_id)
        existing = self.get_express_style(sid)
        if not existing:
            return
        old_order = existing.get("sort_order", 0) or 0
        if old_order == new_order:
            return

        if self._use_pg:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    if new_order < old_order:
                        # Move up: shift [new..old-1] → +1
                        cur.execute(
                            f"UPDATE {self._express_styles_table} SET sort_order = sort_order + 1 "
                            f"WHERE sort_order >= %s AND sort_order < %s AND id != %s",
                            (new_order, old_order, sid),
                        )
                    else:
                        # Move down: shift [old+1..new] → -1
                        cur.execute(
                            f"UPDATE {self._express_styles_table} SET sort_order = sort_order - 1 "
                            f"WHERE sort_order > %s AND sort_order <= %s AND id != %s",
                            (old_order, new_order, sid),
                        )
                    cur.execute(
                        f"UPDATE {self._express_styles_table} SET sort_order = %s, updated_at = NOW() WHERE id = %s",
                        (new_order, sid),
                    )
                    conn.commit()
        else:
            with self._connect() as conn:
                if new_order < old_order:
                    conn.execute(
                        f"UPDATE {self._express_styles_table} SET sort_order = sort_order + 1 "
                        f"WHERE sort_order >= ? AND sort_order < ? AND id != ?",
                        (new_order, old_order, sid),
                    )
                else:
                    conn.execute(
                        f"UPDATE {self._express_styles_table} SET sort_order = sort_order - 1 "
                        f"WHERE sort_order > ? AND sort_order <= ? AND id != ?",
                        (old_order, new_order, sid),
                    )
                conn.execute(
                    f"UPDATE {self._express_styles_table} SET sort_order = ? WHERE id = ?",
                    (new_order, sid),
                )
                conn.commit()
        self._renumber_express_styles()

    # ========== Helper query methods ==========

    def _fetch_all(self, sql: str, params: tuple | list = ()) -> list[dict]:
        """Выполнить SELECT, вернуть list[dict]. PG/SQLite-совместимый."""
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, tuple(params))
                    return [dict(row) for row in cur.fetchall()]
        else:
            with self._connect() as conn:
                rows = conn.execute(sql.replace("%s", "?"), tuple(params)).fetchall()
                return [dict(row) for row in rows]

    def _fetch_one(self, sql: str, params: tuple | list = ()) -> dict | None:
        """Выполнить SELECT, вернуть первую строку или None."""
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, tuple(params))
                    row = cur.fetchone()
                    return dict(row) if row else None
        else:
            with self._connect() as conn:
                row = conn.execute(sql.replace("%s", "?"), tuple(params)).fetchone()
                return dict(row) if row else None

    def _insert_returning_id(self, sql: str, params: tuple | list = ()) -> int | None:
        """INSERT ... RETURNING id (PG) или lastrowid (SQLite)."""
        if self._use_pg:
            with self._connect() as conn:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql + " RETURNING id", tuple(params))
                    row = cur.fetchone()
                    conn.commit()
                    return int(row["id"]) if row else None
        else:
            with self._connect() as conn:
                cur = conn.execute(sql.replace("%s", "?"), tuple(params))
                conn.commit()
                return cur.lastrowid

    # ========== Express Categories CRUD ==========

    def get_express_categories(self, active_only: bool = True) -> list[dict]:
        """Список категорий, отсортированных по sort_order."""
        conditions = []
        if active_only:
            conditions.append("is_active = " + ("TRUE" if self._use_pg else "1"))
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        return self._fetch_all(
            f"SELECT * FROM {self._express_categories_table}{where} ORDER BY sort_order, id"
        )

    def get_express_category(self, category_id: int) -> dict | None:
        return self._fetch_one(
            f"SELECT * FROM {self._express_categories_table} WHERE id = %s",
            (int(category_id),),
        )

    def get_express_category_by_slug(self, slug: str) -> dict | None:
        return self._fetch_one(
            f"SELECT * FROM {self._express_categories_table} WHERE slug = %s",
            (slug,),
        )

    def create_express_category(self, slug: str, title: str,
                                 sort_order: int = 0, is_active: bool = True) -> int | None:
        return self._insert_returning_id(
            f"INSERT INTO {self._express_categories_table} (slug, title, sort_order, is_active) "
            f"VALUES (%s, %s, %s, %s)",
            (slug, title, int(sort_order), is_active),
        )

    def update_express_category(self, category_id: int, **kwargs) -> bool:
        allowed = {"slug", "title", "sort_order", "is_active"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        set_parts = [f"{k} = %s" for k in updates]
        if self._use_pg:
            set_parts.append("updated_at = NOW()")
        params = list(updates.values()) + [int(category_id)]
        sql = f"UPDATE {self._express_categories_table} SET {', '.join(set_parts)} WHERE id = %s"
        self._run(sql, tuple(params))
        return True

    def delete_express_category(self, category_id: int) -> bool:
        self._run(
            f"DELETE FROM {self._express_categories_table} WHERE id = %s",
            (int(category_id),),
        )
        return True

    # ========== Express Tags CRUD ==========

    def get_express_tags(self, active_only: bool = True) -> list[dict]:
        """Список тегов, отсортированных по sort_order."""
        conditions = []
        if active_only:
            conditions.append("is_active = " + ("TRUE" if self._use_pg else "1"))
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        return self._fetch_all(
            f"SELECT * FROM {self._express_tags_table}{where} ORDER BY sort_order, id"
        )

    def get_express_tag(self, tag_id: int) -> dict | None:
        return self._fetch_one(
            f"SELECT * FROM {self._express_tags_table} WHERE id = %s",
            (int(tag_id),),
        )

    def get_express_tag_by_slug(self, slug: str) -> dict | None:
        return self._fetch_one(
            f"SELECT * FROM {self._express_tags_table} WHERE slug = %s",
            (slug,),
        )

    def create_express_tag(self, slug: str, title: str,
                            sort_order: int = 0, is_active: bool = True) -> int | None:
        return self._insert_returning_id(
            f"INSERT INTO {self._express_tags_table} (slug, title, sort_order, is_active) "
            f"VALUES (%s, %s, %s, %s)",
            (slug, title, int(sort_order), is_active),
        )

    def update_express_tag(self, tag_id: int, **kwargs) -> bool:
        allowed = {"slug", "title", "sort_order", "is_active"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        set_parts = [f"{k} = %s" for k in updates]
        if self._use_pg:
            set_parts.append("updated_at = NOW()")
        params = list(updates.values()) + [int(tag_id)]
        sql = f"UPDATE {self._express_tags_table} SET {', '.join(set_parts)} WHERE id = %s"
        self._run(sql, tuple(params))
        return True

    def delete_express_tag(self, tag_id: int) -> bool:
        self._run(
            f"DELETE FROM {self._express_tags_table} WHERE id = %s",
            (int(tag_id),),
        )
        return True

    # ========== Junction: style ↔ categories ==========

    def set_style_categories(self, style_id: int, category_ids: list[int]) -> None:
        """Заменить все категории стиля. DELETE + INSERT."""
        sid = int(style_id)
        if self._use_pg:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {self._express_style_categories_table} WHERE style_id = %s",
                        (sid,),
                    )
                    for cid in category_ids:
                        cur.execute(
                            f"INSERT INTO {self._express_style_categories_table} (style_id, category_id) VALUES (%s, %s)",
                            (sid, int(cid)),
                        )
                    conn.commit()
        else:
            with self._connect() as conn:
                conn.execute(
                    f"DELETE FROM {self._express_style_categories_table} WHERE style_id = ?",
                    (sid,),
                )
                for cid in category_ids:
                    conn.execute(
                        f"INSERT INTO {self._express_style_categories_table} (style_id, category_id) VALUES (?, ?)",
                        (sid, int(cid)),
                    )
                conn.commit()

    def get_style_categories(self, style_id: int) -> list[dict]:
        """Категории конкретного стиля."""
        return self._fetch_all(
            f"SELECT c.* FROM {self._express_categories_table} c "
            f"JOIN {self._express_style_categories_table} sc ON sc.category_id = c.id "
            f"WHERE sc.style_id = %s ORDER BY c.sort_order, c.id",
            (int(style_id),),
        )

    # ========== Junction: style ↔ tags ==========

    def set_style_tags(self, style_id: int, tag_ids: list[int]) -> None:
        """Заменить все теги стиля. DELETE + INSERT."""
        sid = int(style_id)
        if self._use_pg:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {self._express_style_tags_table} WHERE style_id = %s",
                        (sid,),
                    )
                    for tid in tag_ids:
                        cur.execute(
                            f"INSERT INTO {self._express_style_tags_table} (style_id, tag_id) VALUES (%s, %s)",
                            (sid, int(tid)),
                        )
                    conn.commit()
        else:
            with self._connect() as conn:
                conn.execute(
                    f"DELETE FROM {self._express_style_tags_table} WHERE style_id = ?",
                    (sid,),
                )
                for tid in tag_ids:
                    conn.execute(
                        f"INSERT INTO {self._express_style_tags_table} (style_id, tag_id) VALUES (?, ?)",
                        (sid, int(tid)),
                    )
                conn.commit()

    def get_style_tags(self, style_id: int) -> list[dict]:
        """Теги конкретного стиля."""
        return self._fetch_all(
            f"SELECT t.* FROM {self._express_tags_table} t "
            f"JOIN {self._express_style_tags_table} st ON st.tag_id = t.id "
            f"WHERE st.style_id = %s ORDER BY t.sort_order, t.id",
            (int(style_id),),
        )

    # ========== Junction: category ↔ tags ==========

    def set_category_tags(self, category_id: int, tag_ids: list[int]) -> None:
        """Заменить разрешённые теги категории. DELETE + INSERT."""
        cid = int(category_id)
        if self._use_pg:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {self._express_category_tags_table} WHERE category_id = %s",
                        (cid,),
                    )
                    for tid in tag_ids:
                        cur.execute(
                            f"INSERT INTO {self._express_category_tags_table} (category_id, tag_id) VALUES (%s, %s)",
                            (cid, int(tid)),
                        )
                    conn.commit()
        else:
            with self._connect() as conn:
                conn.execute(
                    f"DELETE FROM {self._express_category_tags_table} WHERE category_id = ?",
                    (cid,),
                )
                for tid in tag_ids:
                    conn.execute(
                        f"INSERT INTO {self._express_category_tags_table} (category_id, tag_id) VALUES (?, ?)",
                        (cid, int(tid)),
                    )
                conn.commit()

    def get_category_tags(self, category_id: int) -> list[dict]:
        """Разрешённые теги для категории."""
        return self._fetch_all(
            f"SELECT t.* FROM {self._express_tags_table} t "
            f"JOIN {self._express_category_tags_table} ct ON ct.tag_id = t.id "
            f"WHERE ct.category_id = %s ORDER BY t.sort_order, t.id",
            (int(category_id),),
        )

    def get_allowed_tag_ids_for_categories(self, category_ids: list[int]) -> set[int]:
        """Разрешённые tag_id для набора категорий (union)."""
        if not category_ids:
            return set()
        placeholders = ", ".join(["%s"] * len(category_ids))
        rows = self._fetch_all(
            f"SELECT DISTINCT tag_id FROM {self._express_category_tags_table} "
            f"WHERE category_id IN ({placeholders})",
            category_ids,
        )
        return {r["tag_id"] for r in rows}

    def get_tag_style_counts(self) -> dict[int, int]:
        """Количество стилей для каждого тега. {tag_id: count}."""
        rows = self._fetch_all(
            f"SELECT tag_id, COUNT(*) as cnt FROM {self._express_style_tags_table} GROUP BY tag_id"
        )
        return {r["tag_id"]: r["cnt"] for r in rows}

    def get_category_style_counts(self) -> dict[int, int]:
        """Количество стилей для каждой категории. {category_id: count}."""
        rows = self._fetch_all(
            f"SELECT category_id, COUNT(*) as cnt FROM {self._express_style_categories_table} GROUP BY category_id"
        )
        return {r["category_id"]: r["cnt"] for r in rows}

    def get_all_style_categories_map(self) -> dict[int, list[dict]]:
        """Batch: style_id → list[{id, slug, title}]. Один запрос вместо N+1."""
        rows = self._fetch_all(
            f"SELECT sc.style_id, c.id, c.slug, c.title "
            f"FROM {self._express_style_categories_table} sc "
            f"JOIN {self._express_categories_table} c ON c.id = sc.category_id "
            f"ORDER BY c.sort_order, c.id"
        )
        result: dict[int, list[dict]] = {}
        for r in rows:
            sid = r["style_id"]
            if sid not in result:
                result[sid] = []
            result[sid].append({"id": r["id"], "slug": r["slug"], "title": r["title"]})
        return result

    def get_all_style_tags_map(self) -> dict[int, list[dict]]:
        """Batch: style_id → list[{id, slug, title}]. Один запрос вместо N+1."""
        rows = self._fetch_all(
            f"SELECT st.style_id, t.id, t.slug, t.title "
            f"FROM {self._express_style_tags_table} st "
            f"JOIN {self._express_tags_table} t ON t.id = st.tag_id "
            f"ORDER BY t.sort_order, t.id"
        )
        result: dict[int, list[dict]] = {}
        for r in rows:
            sid = r["style_id"]
            if sid not in result:
                result[sid] = []
            result[sid].append({"id": r["id"], "slug": r["slug"], "title": r["title"]})
        return result

    # ========== Filtered styles query ==========

    def get_styles_filtered(
        self,
        category_slugs: list[str] | None = None,
        tag_slugs: list[str] | None = None,
        active_only: bool = True,
    ) -> list[dict]:
        """Получить стили с фильтрацией по категориям и тегам.

        - Без category_slugs → все стили (виртуальная категория "Все")
        - category_slugs=["all"] → то же что без фильтра
        - Категории: стиль принадлежит хотя бы одной из указанных категорий
        - Теги: OR-логика (хотя бы один из указанных тегов)
        - Теги валидируются: учитываются только те, которые разрешены для указанных
          категорий через express_category_tags. Невалидные тихо игнорируются.
        - is_active проверяется на стилях, категориях и тегах.
        - Несуществующие slug'и → пустой результат (не ошибка, не сброс фильтра).
        """
        sql = f"SELECT DISTINCT s.* FROM {self._express_styles_table} s"
        joins = []
        conditions = []
        params: list = []

        # Фильтр по категориям
        cat_slugs = [slug for slug in (category_slugs or []) if slug and slug != "all"]
        if cat_slugs:
            joins.append(
                f" JOIN {self._express_style_categories_table} sc ON sc.style_id = s.id"
                f" JOIN {self._express_categories_table} c ON c.id = sc.category_id"
            )
            placeholders = ", ".join(["%s"] * len(cat_slugs))
            conditions.append(f"c.slug IN ({placeholders})")
            params.extend(cat_slugs)
            if active_only:
                conditions.append("c.is_active = " + ("TRUE" if self._use_pg else "1"))

        # Фильтр по тегам (валидация через category_tags)
        t_slugs = [slug for slug in (tag_slugs or []) if slug]
        if t_slugs:
            joins.append(
                f" JOIN {self._express_style_tags_table} st ON st.style_id = s.id"
                f" JOIN {self._express_tags_table} t ON t.id = st.tag_id"
            )
            tag_placeholders = ", ".join(["%s"] * len(t_slugs))
            tag_condition = f"t.slug IN ({tag_placeholders})"
            params.extend(t_slugs)
            if active_only:
                tag_condition += " AND t.is_active = " + ("TRUE" if self._use_pg else "1")
            # Валидация: теги должны быть разрешены для выбранных категорий
            if cat_slugs:
                tag_condition += (
                    f" AND t.id IN ("
                    f"SELECT ct.tag_id FROM {self._express_category_tags_table} ct "
                    f"JOIN {self._express_categories_table} cv ON cv.id = ct.category_id "
                    f"WHERE cv.slug IN ({placeholders})"
                    f")"
                )
                params.extend(cat_slugs)
            conditions.append(tag_condition)

        # Активность стилей
        if active_only:
            conditions.append("s.is_active = " + ("TRUE" if self._use_pg else "1"))

        sql += "".join(joins)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY s.sort_order, s.id"

        return self._fetch_all(sql, params)

    # ========== Provider memory ==========

    _VALID_PROVIDERS = {"seedream", "nano-banana-pro"}

    def set_user_last_express_provider(self, user_id: int, provider: str) -> None:
        """Запомнить последний выбранный провайдер. Невалидный → seedream."""
        safe_provider = provider if provider in self._VALID_PROVIDERS else "seedream"
        self._run(
            f"""
            INSERT INTO {self._users_table} (user_id, last_express_provider)
            VALUES (%s, %s)
            ON CONFLICT(user_id) DO UPDATE SET
                last_express_provider = EXCLUDED.last_express_provider
            """,
            (int(user_id), safe_provider),
        )

    def get_user_last_express_provider(self, user_id: int) -> str:
        """Последний провайдер юзера. Default/fallback: seedream."""
        row = self._fetch_one(
            f"SELECT last_express_provider FROM {self._users_table} WHERE user_id = %s",
            (int(user_id),),
        )
        if not row:
            return "seedream"
        val = row.get("last_express_provider") or ""
        return val if val in self._VALID_PROVIDERS else "seedream"

    # --- Generation History (Phase 4) ---

    _VALID_MODES = {"express", "photoset", "custom"}

    def save_generation_history(
        self,
        user_id: int,
        mode: str,
        style_slug: str,
        style_title: str,
        provider: str,
        image_url: str | None = None,
        prompt_preview: str | None = None,
        refs_count: int = 0,
        request_id: str | None = None,
    ) -> int:
        """Сохраняет запись в generation_history. Возвращает id."""
        if mode not in self._VALID_MODES:
            mode = "express"
        # Sanitize prompt_preview: trim, max 100 chars, no HTML
        if prompt_preview:
            prompt_preview = prompt_preview.strip().replace("<", "&lt;").replace(">", "&gt;")[:100]
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""INSERT INTO {self._generation_history_table}
                            (user_id, mode, style_slug, style_title, provider, image_url, prompt_preview, refs_count, request_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING id""",
                        (int(user_id), mode, style_slug, style_title, provider, image_url, prompt_preview, refs_count, request_id),
                    )
                    row = cur.fetchone()
                    conn.commit()
                    return row["id"]
            else:
                cur = conn.execute(
                    f"""INSERT INTO {self._generation_history_table}
                        (user_id, mode, style_slug, style_title, provider, image_url, prompt_preview, refs_count, request_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (int(user_id), mode, style_slug, style_title, provider, image_url, prompt_preview, refs_count, request_id),
                )
                conn.commit()
                return cur.lastrowid

    def update_generation_history_url(self, history_id: int, image_url: str) -> None:
        """Обновляет image_url после upload в storage."""
        with self._connect() as conn:
            self._execute(conn, f"UPDATE {self._generation_history_table} SET image_url = %s WHERE id = %s", (image_url, int(history_id)))
            conn.commit()

    def get_generation_history(
        self,
        user_id: int,
        mode: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """Возвращает историю генераций. mode=None → все."""
        limit = max(1, min(limit, 100))
        offset = max(0, offset)
        if mode and mode not in self._VALID_MODES:
            mode = None
        if mode:
            rows = self._fetch_all(
                f"SELECT * FROM {self._generation_history_table} "
                f"WHERE user_id = %s AND mode = %s "
                f"ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s",
                (int(user_id), mode, limit, offset),
            )
        else:
            rows = self._fetch_all(
                f"SELECT * FROM {self._generation_history_table} "
                f"WHERE user_id = %s "
                f"ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s",
                (int(user_id), limit, offset),
            )
        return [dict(r) for r in rows]

    def get_generation_history_total(self, user_id: int, mode: str | None = None) -> int:
        """Общее количество записей истории пользователя (для пагинации)."""
        if mode and mode not in self._VALID_MODES:
            mode = None
        if mode:
            row = self._fetch_one(
                f"SELECT COUNT(*) AS cnt FROM {self._generation_history_table} WHERE user_id = %s AND mode = %s",
                (int(user_id), mode),
            )
        else:
            row = self._fetch_one(
                f"SELECT COUNT(*) AS cnt FROM {self._generation_history_table} WHERE user_id = %s",
                (int(user_id),),
            )
        return int((row or {}).get("cnt", 0) or 0)

    # --- Generation Requests (idempotency) ---

    def check_request_id(self, user_id: int, request_id: str) -> str | None:
        """Проверяет (user_id, request_id). Возвращает task_id если есть."""
        row = self._fetch_one(
            f"SELECT task_id FROM {self._generation_requests_table} WHERE user_id = %s AND request_id = %s",
            (int(user_id), request_id),
        )
        return row["task_id"] if row else None

    def save_request_id(self, request_id: str, user_id: int, task_id: str) -> str:
        """Атомарно сохраняет request_id. Возвращает task_id (свой или существующий при гонке)."""
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""INSERT INTO {self._generation_requests_table} (request_id, user_id, task_id)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (user_id, request_id) DO NOTHING
                            RETURNING task_id""",
                        (request_id, int(user_id), task_id),
                    )
                    row = cur.fetchone()
                    conn.commit()
                    if row:
                        return row["task_id"]
                    # Conflict — read existing
                    existing = self.check_request_id(user_id, request_id)
                    return existing or task_id
            else:
                try:
                    conn.execute(
                        f"INSERT INTO {self._generation_requests_table} (request_id, user_id, task_id) VALUES (?, ?, ?)",
                        (request_id, int(user_id), task_id),
                    )
                    conn.commit()
                    return task_id
                except Exception:
                    conn.rollback()
                    existing = self.check_request_id(user_id, request_id)
                    return existing or task_id

    def cleanup_old_requests(self, max_age_hours: int = 1) -> None:
        """Удаляет записи старше max_age_hours."""
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"DELETE FROM {self._generation_requests_table} WHERE created_at < NOW() - INTERVAL '1 hour' * %s",
                        (max_age_hours,),
                    )
                conn.commit()
            else:
                conn.execute(
                    f"DELETE FROM {self._generation_requests_table} WHERE created_at < datetime('now', ?)",
                    (f"-{max_age_hours} hours",),
                )
                conn.commit()
