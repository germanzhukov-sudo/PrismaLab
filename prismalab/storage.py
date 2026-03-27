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
        return conn

    def _connect_pg(self):
        """Возвращает контекст-менеджер: при выходе соединение возвращается в пул."""
        conn = self._pg_pool.getconn()
        return _PooledConnection(self._pg_pool, conn)

    def _connect(self):
        if self._use_pg:
            return self._connect_pg()
        return self._connect_sqlite()

    def _execute(self, conn, sql: str, params: tuple = ()) -> None:
        if self._use_pg:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
        else:
            conn.execute(sql.replace("%s", "?"), params)

    def _init(self) -> None:
        if self._use_pg:
            self._init_pg()
        else:
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        with self._connect() as conn:
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS users (
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
            ]:
                try:
                    self._execute(conn, f"ALTER TABLE users ADD COLUMN {col} {col_type}")
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
                        cost_usd DECIMAL(10,4) DEFAULT 0
                    )
                    """,
                )
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
                    "SELECT * FROM users WHERE user_id = ?", (int(user_id),)
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
                    if key in ("cost_persona_create", "cost_fast_photo", "cost_persona_photo", "usd_rub"):
                        cur.execute(f"""
                            INSERT INTO {self._admin_settings_table} (key, value, updated_at)
                            VALUES (%s, %s, NOW())
                            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                        """, (key, str(value)))
                conn.commit()

    # --- Себестоимость паков ---

    def get_pack_costs(self) -> list[dict]:
        """Получить себестоимость всех паков из admin_pack_costs."""
        if not self._use_pg:
            return []
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"SELECT pack_id, pack_title, cost_usd FROM {self._admin_pack_costs_table} ORDER BY pack_id")
                return [dict(row) for row in cur.fetchall()]

    def get_pack_costs_map(self) -> dict[int, float]:
        """Получить словарь {pack_id: cost_usd} для расчёта себестоимости."""
        rows = self.get_pack_costs()
        return {int(r["pack_id"]): float(r["cost_usd"] or 0) for r in rows}

    def set_pack_costs_bulk(self, items: list[dict]) -> None:
        """Массовое обновление себестоимости паков. items = [{pack_id, pack_title, cost_usd}, ...]"""
        if not self._use_pg or not items:
            return
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for item in items:
                    cur.execute(f"""
                        INSERT INTO {self._admin_pack_costs_table} (pack_id, pack_title, cost_usd)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (pack_id) DO UPDATE SET
                            pack_title = EXCLUDED.pack_title,
                            cost_usd = EXCLUDED.cost_usd
                    """, (int(item["pack_id"]), str(item.get("pack_title", "")), float(item.get("cost_usd", 0))))
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

    def decrement_persona_credits(self, user_id: int) -> int:
        """Атомарно вычитает 1 кредит. Возвращает новый баланс или 0, если не было кредитов."""
        with self._connect() as conn:
            uid = int(user_id)
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        UPDATE {self._users_table}
                        SET persona_credits_remaining = persona_credits_remaining - 1,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = %s AND persona_credits_remaining > 0
                        RETURNING persona_credits_remaining
                        """,
                        (uid,),
                    )
                    row = cur.fetchone()
                conn.commit()
                return int(row["persona_credits_remaining"]) if row else 0
            else:
                cur = conn.execute(
                    "UPDATE users SET persona_credits_remaining = persona_credits_remaining - 1, updated_at = CURRENT_TIMESTAMP "
                    "WHERE user_id = ? AND persona_credits_remaining > 0 RETURNING persona_credits_remaining",
                    (uid,),
                )
                row = cur.fetchone()
                conn.commit()
                return int(row[0]) if row else 0

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
                    "SELECT pending_persona_batch FROM users WHERE user_id = ?",
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
                    "DELETE FROM users WHERE user_id = ?", (int(user_id),)
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
                    # Добавить prompt в persona_styles если нет
                    try:
                        cur.execute(f"ALTER TABLE {self._persona_styles_table} ADD COLUMN prompt TEXT")
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
                logger.info("Таблицы админки созданы/проверены")
            else:
                # SQLite версия
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS payments (
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
                    conn.execute(
                        """
                        DELETE FROM payments
                        WHERE rowid NOT IN (
                            SELECT MAX(rowid)
                            FROM payments
                            WHERE payment_id IS NOT NULL
                            GROUP BY payment_id
                        )
                        AND payment_id IS NOT NULL
                        """
                    )
                    conn.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS idx_payments_payment_id_uniq ON payments(payment_id)"
                    )
                except Exception as e:
                    logger.warning("SQLite: не удалось обеспечить уникальность payment_id: %s", e)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        event_data TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS admins (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        display_name TEXT,
                        is_active INTEGER DEFAULT 1,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                try:
                    conn.execute("ALTER TABLE users ADD COLUMN created_at TEXT DEFAULT CURRENT_TIMESTAMP")
                except Exception:
                    pass
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
                    "SELECT 1 FROM payments WHERE payment_id = ? LIMIT 1",
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
                        """
                        INSERT INTO payments (user_id, payment_id, payment_method, product_type, credits, amount_rub)
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
                    """
                    INSERT INTO user_events (user_id, event_type, event_data)
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
                row = conn.execute("SELECT * FROM admins WHERE username = ? AND is_active = 1", (username,)).fetchone()
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
                        "INSERT INTO admins (username, password_hash, display_name) VALUES (?, ?, ?)",
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
                        "SELECT * FROM users WHERE CAST(user_id AS TEXT) LIKE ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                        (f"%{search}%", limit, offset),
                    ).fetchall()
                else:
                    rows = conn.execute("SELECT * FROM users ORDER BY updated_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
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
                    row = conn.execute("SELECT COUNT(*) as cnt FROM users WHERE CAST(user_id AS TEXT) LIKE ?", (f"%{search}%",)).fetchone()
                else:
                    row = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()
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
                    rows = conn.execute("SELECT user_id FROM users WHERE astria_lora_tune_id IS NOT NULL").fetchall()
                elif filter_type == "no_persona":
                    rows = conn.execute("SELECT user_id FROM users WHERE astria_lora_tune_id IS NULL").fetchall()
                elif filter_type == "specific" and user_ids:
                    placeholders = ",".join("?" * len(user_ids))
                    rows = conn.execute(f"SELECT user_id FROM users WHERE user_id IN ({placeholders})", user_ids).fetchall()
                else:
                    rows = conn.execute("SELECT user_id FROM users").fetchall()
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
                row = conn.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(astria_lora_tune_id) as has_persona,
                        COUNT(*) - COUNT(astria_lora_tune_id) as no_persona
                    FROM users
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
        }

        with self._connect() as conn:
            if not self._use_pg:
                row = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()
                stats["users_total"] = int(row["cnt"]) if row else 0
                row = conn.execute("SELECT COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM payments").fetchone()
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
                    COUNT(*) FILTER (WHERE event_data->>'mode' = 'fast') as fast,
                    COUNT(*) FILTER (WHERE event_data->>'mode' = 'persona') as persona
                FROM {self._user_events_table} WHERE event_type = 'generation'""")
                row = cur.fetchone()
                fast_gens = int(row["fast"] or 0) if row else 0
                persona_gens = int(row["persona"] or 0) if row else 0

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
                total_cost = (persona_creates * COST_PERSONA_CREATE + fast_gens * COST_FAST_PHOTO + persona_gens * COST_PERSONA_PHOTO + pack_cost_total_usd) * USD_RUB
                stats["total_cost"] = round(total_cost, 2)
                margin = stats["total_revenue"] - total_cost
                stats["margin"]["amount"] = round(margin, 2)
                stats["margin"]["percent"] = round((margin / stats["total_revenue"]) * 100, 2) if stats["total_revenue"] > 0 else 0.0

                # Генераций на платящего юзера — INNER JOIN вместо EXISTS
                total_gens = fast_gens + persona_gens
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
            "generations": {"total": 0, "free": 0, "fast": 0, "persona": 0},
            "costs": {"total": 0.0, "persona_create": 0.0, "fast_photos": 0.0, "persona_photos": 0.0, "packs": 0.0},
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
                row = conn.execute("SELECT COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM payments").fetchone()
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
                            COUNT(*) FILTER (WHERE event_data->>'mode' = 'persona') as persona
                        FROM {self._user_events_table}
                        WHERE event_type = 'generation'"""
                    )
                row = cur.fetchone()
                if row:
                    stats["generations"]["total"] = int(row["total"] or 0)
                    stats["generations"]["fast"] = int(row["fast"] or 0)
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
                cost_fast_photos = stats["generations"]["fast"] * COST_FAST_PHOTO * USD_RUB
                cost_persona_photos = stats["generations"]["persona"] * COST_PERSONA_PHOTO * USD_RUB
                cost_packs = pack_cost_usd * USD_RUB
                total_cost = cost_persona_create + cost_fast_photos + cost_persona_photos + cost_packs

                stats["costs"]["persona_create"] = round(cost_persona_create, 2)
                stats["costs"]["fast_photos"] = round(cost_fast_photos, 2)
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

                    row = conn.execute("SELECT COALESCE(SUM(amount_rub), 0) as total FROM payments WHERE DATE(created_at) = ?", (day,)).fetchone()
                    result["revenue"].append(float(row["total"]) if row else 0)

                    row = conn.execute("SELECT COUNT(*) as cnt FROM users WHERE DATE(created_at) = ?", (day,)).fetchone()
                    result["users"].append(int(row["cnt"]) if row else 0)

                    row = conn.execute("SELECT COUNT(*) as cnt FROM user_events WHERE event_type = 'generation' AND DATE(created_at) = ?", (day,)).fetchone()
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
                rows = conn.execute("SELECT * FROM payments WHERE user_id = ? ORDER BY created_at DESC LIMIT 100", (int(user_id),)).fetchall()
                result["payments"] = [dict(row) for row in rows]

                rows = conn.execute("SELECT * FROM user_events WHERE user_id = ? ORDER BY created_at DESC LIMIT 100", (int(user_id),)).fetchall()
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
        if not self._use_pg:
            return []
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                conditions = []
                params: list = []
                if active_only:
                    conditions.append("is_active = TRUE")
                if gender:
                    conditions.append("gender = %s")
                    params.append(gender)
                where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
                cur.execute(f"SELECT * FROM {self._persona_styles_table}{where} ORDER BY sort_order, id", params)
                return [dict(row) for row in cur.fetchall()]

    def get_persona_style(self, style_id: int) -> dict | None:
        """Получить один стиль по id."""
        if not self._use_pg:
            return None
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"SELECT * FROM {self._persona_styles_table} WHERE id = %s", (int(style_id),))
                row = cur.fetchone()
                return dict(row) if row else None

    def get_persona_style_by_slug(self, slug: str) -> dict | None:
        """Получить один стиль по slug."""
        if not self._use_pg:
            return None
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"SELECT * FROM {self._persona_styles_table} WHERE slug = %s", (slug,))
                row = cur.fetchone()
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
                             prompt: str = "", sort_order: int = 0) -> int | None:
        """Создать стиль. Возвращает id."""
        if not self._use_pg:
            return None
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    INSERT INTO {self._persona_styles_table} (slug, title, description, gender, image_url, prompt, sort_order)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (slug, title, description, gender, image_url, prompt, int(sort_order)))
                row = cur.fetchone()
                conn.commit()
                return int(row["id"]) if row else None

    def update_persona_style(self, style_id: int, **kwargs) -> bool:
        """Обновить стиль. Допустимые поля: slug, title, description, gender, image_url, sort_order, is_active."""
        if not self._use_pg:
            return False
        allowed = {"slug", "title", "description", "gender", "image_url", "prompt", "sort_order", "is_active"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
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

    def delete_persona_style(self, style_id: int) -> bool:
        """Удалить стиль."""
        if not self._use_pg:
            return False
        with self._connect() as conn:
            from psycopg2.extras import RealDictCursor
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"DELETE FROM {self._persona_styles_table} WHERE id = %s", (int(style_id),))
                deleted = cur.rowcount > 0
                conn.commit()
                return deleted
