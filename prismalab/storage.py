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
        self._users_table = "public.users" if self._use_pg else "users"
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
                    free_generation_used INTEGER DEFAULT 0,
                    paid_generations_remaining INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """,
            )
            for col, col_type in [
                ("astria_tune_id", "TEXT"),
                ("astria_lora_tune_id", "TEXT"),
                ("free_generation_used", "INTEGER DEFAULT 0"),
                ("paid_generations_remaining", "INTEGER DEFAULT 0"),
                ("persona_credits_remaining", "INTEGER DEFAULT 0"),
                ("subject_gender", "TEXT"),
            ]:
                try:
                    self._execute(conn, f"ALTER TABLE users ADD COLUMN {col} {col_type}")
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
                    free_generation_used INTEGER DEFAULT 0,
                    paid_generations_remaining INTEGER DEFAULT 0,
                    subject_gender TEXT,
                    persona_credits_remaining INTEGER DEFAULT 0,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
                """,
            )
            try:
                self._execute(conn, f"ALTER TABLE {self._users_table} ADD COLUMN persona_credits_remaining INTEGER DEFAULT 0")
                conn.commit()
            except Exception:
                conn.rollback()
            # Таблица настроек (себестоимость, курс)
            try:
                self._execute(
                    conn,
                    """
                    CREATE TABLE IF NOT EXISTS public.admin_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
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
                cur.execute("SELECT key, value FROM public.admin_settings WHERE key LIKE 'cost_%' OR key = 'usd_rub'")
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
                        cur.execute("""
                            INSERT INTO public.admin_settings (key, value, updated_at)
                            VALUES (%s, %s, NOW())
                            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                        """, (key, str(value)))
                conn.commit()

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

    def set_astria_lora_tune(self, *, user_id: int, tune_id: str | None) -> None:
        if tune_id is None:
            self._run(
                f"UPDATE {self._users_table} SET astria_lora_tune_id = NULL, updated_at = CURRENT_TIMESTAMP WHERE user_id = %s",
                (int(user_id),),
            )
        else:
            self._run(
                f"""
                INSERT INTO {self._users_table} (user_id, astria_lora_tune_id, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    astria_lora_tune_id = EXCLUDED.astria_lora_tune_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (int(user_id), str(tune_id)),
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

    def clear_user_data(self, *, user_id: int) -> None:
        self._run(
            f"""
            UPDATE {self._users_table}
            SET personal_model_version = NULL,
                personal_trigger_word = NULL,
                astria_tune_id = NULL,
                astria_lora_tune_id = NULL,
                training_status = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = %s
            """,
            (int(user_id),),
        )

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
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS public.payments (
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
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS public.user_events (
                            id SERIAL PRIMARY KEY,
                            user_id BIGINT NOT NULL,
                            event_type TEXT NOT NULL,
                            event_data JSONB,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    # Таблица админов
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS public.admins (
                            id SERIAL PRIMARY KEY,
                            username TEXT UNIQUE NOT NULL,
                            password_hash TEXT NOT NULL,
                            display_name TEXT,
                            is_active BOOLEAN DEFAULT TRUE,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                    # Добавить created_at в users если нет
                    try:
                        cur.execute(f"ALTER TABLE {self._users_table} ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW()")
                    except Exception:
                        pass
                    # Индексы
                    try:
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_payments_user_id ON public.payments(user_id)")
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_payments_created_at ON public.payments(created_at DESC)")
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_user_events_user_id ON public.user_events(user_id)")
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_user_events_created_at ON public.user_events(created_at DESC)")
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_user_events_type ON public.user_events(event_type)")
                    except Exception:
                        pass
                conn.commit()
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
                        "SELECT 1 FROM public.payments WHERE payment_id = %s LIMIT 1",
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
        """Записывает платёж в БД. Возвращает ID записи."""
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        INSERT INTO public.payments (user_id, payment_id, payment_method, product_type, credits, amount_rub)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (int(user_id), payment_id, payment_method, product_type, int(credits), float(amount_rub)),
                    )
                    row = cur.fetchone()
                    conn.commit()
                    return row["id"] if row else None
            else:
                cur = conn.execute(
                    """
                    INSERT INTO payments (user_id, payment_id, payment_method, product_type, credits, amount_rub)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (int(user_id), payment_id, payment_method, product_type, int(credits), float(amount_rub)),
                )
                conn.commit()
                return cur.lastrowid

    # --- Логирование событий ---

    def log_event(self, user_id: int, event_type: str, event_data: dict | None = None) -> int | None:
        """Записывает событие пользователя. Возвращает ID записи."""
        import json
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor, Json
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        INSERT INTO public.user_events (user_id, event_type, event_data)
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

    # --- Админы ---

    def get_admin_by_username(self, username: str) -> dict | None:
        """Получить админа по username."""
        with self._connect() as conn:
            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM public.admins WHERE username = %s AND is_active = TRUE", (username,))
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
                        """
                        INSERT INTO public.admins (username, password_hash, display_name)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (username) DO UPDATE SET
                            password_hash = EXCLUDED.password_hash,
                            display_name = COALESCE(EXCLUDED.display_name, public.admins.display_name)
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
                params.append(date_from)
            if date_to:
                conditions.append("created_at <= %s" if self._use_pg else "created_at <= ?")
                params.append(date_to + " 23:59:59")

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            table = "public.payments" if self._use_pg else "payments"

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
                params.append(date_from)
            if date_to:
                conditions.append("created_at <= %s" if self._use_pg else "created_at <= ?")
                params.append(date_to + " 23:59:59")

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            table = "public.payments" if self._use_pg else "payments"

            if self._use_pg:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"SELECT COUNT(*) as cnt FROM {table} WHERE {where_clause}", params)
                    row = cur.fetchone()
                    return int(row["cnt"]) if row else 0
            else:
                row = conn.execute(f"SELECT COUNT(*) as cnt FROM {table} WHERE {where_clause}", params).fetchone()
                return int(row["cnt"]) if row else 0

    def get_all_time_stats(self) -> dict:
        """Получить статистику за всё время (для постоянного блока)."""
        # Себестоимость из настроек
        cost_settings = self.get_cost_settings()
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
            "persona_purchases": 0,
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
                cur.execute("SELECT COUNT(DISTINCT user_id) as cnt FROM public.user_events WHERE event_type = 'start'")
                row = cur.fetchone()
                stats["users_total"] = int(row["cnt"]) if row else 0

                # Общая выручка и кол-во платежей
                cur.execute("SELECT COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM public.payments")
                row = cur.fetchone()
                if row:
                    total_payments = int(row["cnt"])
                    stats["total_revenue"] = float(row["total"])
                    if total_payments > 0:
                        stats["avg_check"] = round(stats["total_revenue"] / total_payments, 2)

                # Платящие юзеры
                cur.execute("SELECT COUNT(DISTINCT user_id) as cnt FROM public.payments")
                row = cur.fetchone()
                stats["paid_users"] = int(row["cnt"]) if row else 0

                # Конверсия
                if stats["users_total"] > 0:
                    stats["conversion"] = round((stats["paid_users"] / stats["users_total"]) * 100, 2)

                # Покупок Экспресс
                cur.execute("SELECT COUNT(*) as cnt FROM public.payments WHERE product_type = 'fast'")
                row = cur.fetchone()
                stats["express_purchases"] = int(row["cnt"]) if row else 0

                # Покупок Персона
                cur.execute("SELECT COUNT(*) as cnt FROM public.payments WHERE product_type LIKE 'persona%%'")
                row = cur.fetchone()
                stats["persona_purchases"] = int(row["cnt"]) if row else 0

                # Для маржи: генерации по типам и создания персон
                cur.execute("""SELECT
                    COUNT(*) FILTER (WHERE event_data->>'mode' = 'fast') as fast,
                    COUNT(*) FILTER (WHERE event_data->>'mode' = 'persona') as persona
                FROM public.user_events WHERE event_type = 'generation'""")
                row = cur.fetchone()
                fast_gens = int(row["fast"] or 0) if row else 0
                persona_gens = int(row["persona"] or 0) if row else 0

                cur.execute("SELECT COUNT(*) as cnt FROM public.payments WHERE product_type = 'persona_create'")
                row = cur.fetchone()
                persona_creates = int(row["cnt"]) if row else 0

                # Расчёт себестоимости и маржи
                total_cost = (persona_creates * COST_PERSONA_CREATE + fast_gens * COST_FAST_PHOTO + persona_gens * COST_PERSONA_PHOTO) * USD_RUB
                stats["total_cost"] = round(total_cost, 2)
                margin = stats["total_revenue"] - total_cost
                stats["margin"]["amount"] = round(margin, 2)
                stats["margin"]["percent"] = round((margin / stats["total_revenue"]) * 100, 2) if stats["total_revenue"] > 0 else 0.0

                # Генераций на платящего юзера
                total_gens = fast_gens + persona_gens
                if stats["paid_users"] > 0:
                    # Считаем генерации только от платящих юзеров
                    cur.execute("""SELECT COUNT(*) as cnt FROM public.user_events ue
                                   WHERE ue.event_type = 'generation'
                                   AND EXISTS (SELECT 1 FROM public.payments p WHERE p.user_id = ue.user_id)""")
                    row = cur.fetchone()
                    paying_gens = int(row["cnt"]) if row else 0
                    stats["gens_per_paying_user"] = round(paying_gens / stats["paid_users"], 1)

                # Статистика по полу
                cur.execute("""SELECT
                    COUNT(*) FILTER (WHERE subject_gender = 'male') as male,
                    COUNT(*) FILTER (WHERE subject_gender = 'female') as female,
                    COUNT(*) FILTER (WHERE subject_gender IS NULL) as unknown,
                    COUNT(*) as total
                FROM public.users""")
                row = cur.fetchone()
                if row and int(row["total"] or 0) > 0:
                    total = int(row["total"])
                    stats["gender"]["male"] = round((int(row["male"] or 0) / total) * 100, 1)
                    stats["gender"]["female"] = round((int(row["female"] or 0) / total) * 100, 1)
                    stats["gender"]["unknown"] = round((int(row["unknown"] or 0) / total) * 100, 1)

                # Среднее количество дней до первой покупки
                cur.execute("""
                    SELECT AVG(days) as avg_days FROM (
                        SELECT
                            p.user_id,
                            EXTRACT(EPOCH FROM (MIN(p.created_at) - MIN(ue.created_at))) / 86400 as days
                        FROM public.payments p
                        JOIN public.user_events ue ON ue.user_id = p.user_id AND ue.event_type = 'start'
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
                    cur.execute("""
                        SELECT
                            EXTRACT(HOUR FROM created_at AT TIME ZONE 'Europe/Moscow')::int as hour,
                            COUNT(DISTINCT user_id) as cnt
                        FROM public.user_events
                        WHERE created_at >= %s AND created_at <= %s
                        GROUP BY EXTRACT(HOUR FROM created_at AT TIME ZONE 'Europe/Moscow')
                        ORDER BY hour
                    """, (date_from, date_to + " 23:59:59"))
                else:
                    cur.execute("""
                        SELECT
                            EXTRACT(HOUR FROM created_at AT TIME ZONE 'Europe/Moscow')::int as hour,
                            COUNT(DISTINCT user_id) as cnt
                        FROM public.user_events
                        GROUP BY EXTRACT(HOUR FROM created_at AT TIME ZONE 'Europe/Moscow')
                        ORDER BY hour
                    """)

                for row in cur.fetchall():
                    hour = int(row["hour"])
                    if 0 <= hour < 24:
                        result[hour] = int(row["cnt"])

        return result

    def get_dashboard_stats(self, date_from: str | None = None, date_to: str | None = None) -> dict:
        """Получить статистику для дашборда с опциональной фильтрацией по датам."""
        # Себестоимость из настроек
        cost_settings = self.get_cost_settings()
        COST_PERSONA_CREATE = cost_settings["cost_persona_create"]
        COST_FAST_PHOTO = cost_settings["cost_fast_photo"]
        COST_PERSONA_PHOTO = cost_settings["cost_persona_photo"]
        USD_RUB = cost_settings["usd_rub"]

        stats = {
            "users": {"new": 0},
            "payments": {"total_count": 0, "total_revenue": 0.0, "avg_check": 0.0, "express": {"count": 0, "revenue": 0.0}, "persona": {"count": 0, "revenue": 0.0}},
            "generations": {"total": 0, "free": 0, "fast": 0, "persona": 0},
            "costs": {"total": 0.0, "persona_create": 0.0, "fast_photos": 0.0, "persona_photos": 0.0},
            "margin": {"amount": 0.0, "percent": 0.0},
        }

        # Подготовка дат
        has_dates = bool(date_from and date_to)
        if has_dates:
            d_from = str(date_from)
            d_to = str(date_to) + " 23:59:59"

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
                        """
                        SELECT COUNT(*) as cnt FROM (
                            SELECT user_id, MIN(created_at) as first_start
                            FROM public.user_events
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
                    cur.execute("SELECT COUNT(DISTINCT user_id) as cnt FROM public.user_events WHERE event_type = 'start'")
                    row = cur.fetchone()
                    stats["users"]["new"] = int(row["cnt"]) if row else 0

                # Платежи
                if has_dates:
                    cur.execute(
                        "SELECT COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM public.payments WHERE created_at >= %s AND created_at <= %s",
                        (d_from, d_to),
                    )
                else:
                    cur.execute("SELECT COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM public.payments")
                row = cur.fetchone()
                if row:
                    stats["payments"]["total_count"] = int(row["cnt"])
                    stats["payments"]["total_revenue"] = float(row["total"])
                    if stats["payments"]["total_count"] > 0:
                        stats["payments"]["avg_check"] = round(stats["payments"]["total_revenue"] / stats["payments"]["total_count"], 2)

                # Экспресс платежи
                if has_dates:
                    cur.execute(
                        "SELECT COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM public.payments WHERE product_type = 'fast' AND created_at >= %s AND created_at <= %s",
                        (d_from, d_to),
                    )
                else:
                    cur.execute("SELECT COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM public.payments WHERE product_type = 'fast'")
                row = cur.fetchone()
                if row:
                    stats["payments"]["express"]["count"] = int(row["cnt"])
                    stats["payments"]["express"]["revenue"] = float(row["total"])

                # Персона платежи (%% для экранирования % в Python)
                if has_dates:
                    cur.execute(
                        "SELECT COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM public.payments WHERE product_type LIKE 'persona%%' AND created_at >= %s AND created_at <= %s",
                        (d_from, d_to),
                    )
                else:
                    cur.execute("SELECT COUNT(*) as cnt, COALESCE(SUM(amount_rub), 0) as total FROM public.payments WHERE product_type LIKE 'persona%%'")
                row = cur.fetchone()
                if row:
                    stats["payments"]["persona"]["count"] = int(row["cnt"])
                    stats["payments"]["persona"]["revenue"] = float(row["total"])

                # Генерации по типам (fast / persona)
                if has_dates:
                    cur.execute(
                        """SELECT
                            COUNT(*) as total,
                            COUNT(*) FILTER (WHERE event_data->>'mode' = 'fast') as fast,
                            COUNT(*) FILTER (WHERE event_data->>'mode' = 'persona') as persona
                        FROM public.user_events
                        WHERE event_type = 'generation' AND created_at >= %s AND created_at <= %s""",
                        (d_from, d_to),
                    )
                else:
                    cur.execute(
                        """SELECT
                            COUNT(*) as total,
                            COUNT(*) FILTER (WHERE event_data->>'mode' = 'fast') as fast,
                            COUNT(*) FILTER (WHERE event_data->>'mode' = 'persona') as persona
                        FROM public.user_events
                        WHERE event_type = 'generation'"""
                    )
                row = cur.fetchone()
                if row:
                    stats["generations"]["total"] = int(row["total"] or 0)
                    stats["generations"]["fast"] = int(row["fast"] or 0)
                    stats["generations"]["persona"] = int(row["persona"] or 0)

                # Бесплатные генерации (юзеры без платежей за период)
                if has_dates:
                    cur.execute(
                        """SELECT COUNT(*) as cnt FROM public.user_events ue
                           WHERE ue.event_type = 'generation'
                           AND ue.created_at >= %s AND ue.created_at <= %s
                           AND NOT EXISTS (SELECT 1 FROM public.payments p WHERE p.user_id = ue.user_id)""",
                        (d_from, d_to),
                    )
                else:
                    cur.execute("""SELECT COUNT(*) as cnt FROM public.user_events ue
                                   WHERE ue.event_type = 'generation'
                                   AND NOT EXISTS (SELECT 1 FROM public.payments p WHERE p.user_id = ue.user_id)""")
                row = cur.fetchone()
                stats["generations"]["free"] = int(row["cnt"]) if row else 0

                # Создания персон (покупки persona_create)
                if has_dates:
                    cur.execute(
                        "SELECT COUNT(*) as cnt FROM public.payments WHERE product_type = 'persona_create' AND created_at >= %s AND created_at <= %s",
                        (d_from, d_to),
                    )
                else:
                    cur.execute("SELECT COUNT(*) as cnt FROM public.payments WHERE product_type = 'persona_create'")
                row = cur.fetchone()
                persona_creates = int(row["cnt"]) if row else 0

                # Расчёт себестоимости
                cost_persona_create = persona_creates * COST_PERSONA_CREATE * USD_RUB
                cost_fast_photos = stats["generations"]["fast"] * COST_FAST_PHOTO * USD_RUB
                cost_persona_photos = stats["generations"]["persona"] * COST_PERSONA_PHOTO * USD_RUB
                total_cost = cost_persona_create + cost_fast_photos + cost_persona_photos

                stats["costs"]["persona_create"] = round(cost_persona_create, 2)
                stats["costs"]["fast_photos"] = round(cost_fast_photos, 2)
                stats["costs"]["persona_photos"] = round(cost_persona_photos, 2)
                stats["costs"]["total"] = round(total_cost, 2)

                # Маржа
                revenue = stats["payments"]["total_revenue"]
                margin = revenue - total_cost
                stats["margin"]["amount"] = round(margin, 2)
                stats["margin"]["percent"] = round((margin / revenue) * 100, 2) if revenue > 0 else 0.0

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
                        """
                        SELECT DATE(created_at) as day, COALESCE(SUM(amount_rub), 0) as total
                        FROM public.payments
                        WHERE created_at >= NOW() - INTERVAL '1 day' * %s
                        GROUP BY DATE(created_at)
                        ORDER BY day
                        """,
                        (days,),
                    )
                    revenue_data = {str(row["day"]): float(row["total"]) for row in cur.fetchall()}

                    # Новые юзеры по дням (по событию start)
                    cur.execute(
                        """
                        SELECT DATE(created_at) as day, COUNT(DISTINCT user_id) as cnt
                        FROM public.user_events
                        WHERE event_type = 'start' AND created_at >= NOW() - INTERVAL '1 day' * %s
                        GROUP BY DATE(created_at)
                        ORDER BY day
                        """,
                        (days,),
                    )
                    users_data = {str(row["day"]): int(row["cnt"]) for row in cur.fetchall()}

                    # Генерации по дням
                    cur.execute(
                        """
                        SELECT DATE(created_at) as day, COUNT(*) as cnt
                        FROM public.user_events
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
                        "SELECT * FROM public.payments WHERE user_id = %s ORDER BY created_at DESC LIMIT 100",
                        (int(user_id),),
                    )
                    result["payments"] = [dict(row) for row in cur.fetchall()]

                    cur.execute(
                        "SELECT * FROM public.user_events WHERE user_id = %s ORDER BY created_at DESC LIMIT 100",
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
