"""
Бот поддержки PrismaLab.
Пользователи пишут сюда → сообщения пересылаются админу.
Админ отвечает (reply) на сообщение → ответ уходит пользователю от имени бота.
Маппинг reply→пользователь хранится в памяти и в PostgreSQL (если DATABASE_URL),
чтобы переживать рестарты контейнера.
"""
import asyncio
import logging
import os

import dotenv
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

dotenv.load_dotenv()

SUPPORT_BOT_TOKEN = os.getenv("PRISMALAB_SUPPORT_BOT_TOKEN", "").strip()
ADMIN_ID = int(os.getenv("PRISMALAB_SUPPORT_ADMIN_ID", "0") or "0")
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()

# (admin_chat_id, our_message_id) -> (user_id, user_chat_id) — кэш в памяти
_reply_map: dict[tuple[int, int], tuple[int, int]] = {}

logger = logging.getLogger("prismalab.support_bot")


def _pg_url_with_ssl(url: str) -> str:
    if not url or "sslmode=" in url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}sslmode=require"


def _reply_map_pg_init() -> None:
    """Создать таблицу support_reply_map при наличии DATABASE_URL."""
    if not DATABASE_URL:
        return
    import psycopg2
    conn = psycopg2.connect(_pg_url_with_ssl(DATABASE_URL))
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS support_reply_map (
                    admin_chat_id BIGINT NOT NULL,
                    admin_message_id INT NOT NULL,
                    user_id BIGINT NOT NULL,
                    user_chat_id BIGINT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (admin_chat_id, admin_message_id)
                )
            """)
        conn.commit()
        logger.info("Таблица support_reply_map готова (PostgreSQL)")
    finally:
        conn.close()


def _reply_map_pg_save(admin_chat_id: int, admin_message_id: int, user_id: int, user_chat_id: int) -> None:
    """Сохранить маппинг в PostgreSQL (синхронно)."""
    import psycopg2
    conn = psycopg2.connect(_pg_url_with_ssl(DATABASE_URL))
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO support_reply_map (admin_chat_id, admin_message_id, user_id, user_chat_id)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (admin_chat_id, admin_message_id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    user_chat_id = EXCLUDED.user_chat_id,
                    created_at = NOW()
                """,
                (admin_chat_id, admin_message_id, user_id, user_chat_id),
            )
        conn.commit()
    finally:
        conn.close()


def _reply_map_pg_get_and_remove(admin_chat_id: int, admin_message_id: int) -> tuple[int, int] | None:
    """Прочитать маппинг из PostgreSQL и удалить. Возвращает (user_id, user_chat_id) или None."""
    import psycopg2
    conn = psycopg2.connect(_pg_url_with_ssl(DATABASE_URL))
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM support_reply_map
                WHERE admin_chat_id = %s AND admin_message_id = %s
                RETURNING user_id, user_chat_id
                """,
                (admin_chat_id, admin_message_id),
            )
            row = cur.fetchone()
        conn.commit()
        return (int(row[0]), int(row[1])) if row else None
    finally:
        conn.close()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def _user_label(user) -> str:
    if not user:
        return "Unknown"
    parts = []
    if user.username:
        parts.append(f"@{user.username}")
    parts.append(f"ID: {user.id}")
    if user.first_name or user.last_name:
        name = " ".join(filter(None, [user.first_name, user.last_name])).strip()
        if name:
            parts.append(f"({name})")
    return " ".join(parts)


async def _forward_to_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Пересылаем сообщение пользователя админу."""
    user = update.effective_user
    chat_id = update.effective_chat.id if update.effective_chat else 0
    logger.info("Сообщение в поддержку: user_id=%s chat_id=%s", getattr(user, "id", None), chat_id)

    if not user:
        logger.warning("Пропуск: нет effective_user в update")
        return
    if not SUPPORT_BOT_TOKEN:
        logger.warning("Пропуск: PRISMALAB_SUPPORT_BOT_TOKEN не задан")
        return
    if not ADMIN_ID:
        logger.warning("Пропуск: PRISMALAB_SUPPORT_ADMIN_ID не задан или 0")
        return

    label = _user_label(user)
    caption = update.message.caption or ""
    text = update.message.text or ""

    try:
        sent = None
        # Фото
        if update.message.photo:
            photo = update.message.photo[-1]  # лучшее качество
            admin_caption = f"📩 От {label}:\n\n{caption}" if caption else f"📩 От {label}"
            sent = await context.bot.send_photo(chat_id=ADMIN_ID, photo=photo.file_id, caption=admin_caption)
        # Документ (в т.ч. фото как файл)
        elif update.message.document:
            admin_caption = f"📩 От {label}:\n\n{caption}" if caption else f"📩 От {label}"
            sent = await context.bot.send_document(chat_id=ADMIN_ID, document=update.message.document.file_id, caption=admin_caption)
        # Видео
        elif update.message.video:
            admin_caption = f"📩 От {label}:\n\n{caption}" if caption else f"📩 От {label}"
            sent = await context.bot.send_video(chat_id=ADMIN_ID, video=update.message.video.file_id, caption=admin_caption)
        # Голосовое
        elif update.message.voice:
            sent = await context.bot.send_voice(chat_id=ADMIN_ID, voice=update.message.voice.file_id, caption=f"📩 От {label}")
        # Стикер
        elif update.message.sticker:
            await context.bot.send_message(chat_id=ADMIN_ID, text=f"📩 От {label}: (стикер)")
            sent = await context.bot.send_sticker(chat_id=ADMIN_ID, sticker=update.message.sticker.file_id)
        # Текст
        elif text:
            admin_text = f"📩 От {label}:\n\n{text}"
            sent = await context.bot.send_message(chat_id=ADMIN_ID, text=admin_text)
        else:
            admin_text = f"📩 От {label}:\n\n(неподдерживаемый тип сообщения)"
            sent = await context.bot.send_message(chat_id=ADMIN_ID, text=admin_text)

        if sent:
            _reply_map[(ADMIN_ID, sent.message_id)] = (user.id, chat_id)
            if DATABASE_URL:
                await asyncio.to_thread(
                    _reply_map_pg_save, ADMIN_ID, sent.message_id, user.id, chat_id
                )
            logger.info("Переслано админу (message_id=%s), от user_id=%s", sent.message_id, user.id)
    except Exception as e:
        logger.error("Не удалось переслать админу: %s", e, exc_info=True)


async def _handle_admin_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Админ ответил на сообщение — отправляем ответ пользователю."""
    user = update.effective_user
    if not user or user.id != ADMIN_ID:
        return

    reply_to = update.message.reply_to_message
    if not reply_to or not reply_to.from_user:
        return

    # Ответ на наше сообщение (бот — автор)
    if reply_to.from_user.is_bot:
        key = (update.effective_chat.id, reply_to.message_id)
        pair = _reply_map.get(key)
        if pair is None and DATABASE_URL:
            pair = await asyncio.to_thread(
                _reply_map_pg_get_and_remove, key[0], key[1]
            )
        if pair is not None:
            target_user_id, target_chat_id = pair
            text = update.message.text or update.message.caption
            if text:
                try:
                    await context.bot.send_message(chat_id=target_chat_id, text=text)
                    logger.info("Ответ админа отправлен пользователю chat_id=%s", target_chat_id)
                except Exception as e:
                    logger.error("Не удалось отправить пользователю: %s", e, exc_info=True)
            if key in _reply_map:
                del _reply_map[key]
        else:
            logger.warning("Ответ админа: ключ (chat_id=%s, msg_id=%s) не найден (рестарт?)", *key)


def main() -> None:
    if not SUPPORT_BOT_TOKEN or not ADMIN_ID:
        raise SystemExit(
            "Задайте PRISMALAB_SUPPORT_BOT_TOKEN и PRISMALAB_SUPPORT_ADMIN_ID в .env"
        )

    app = Application.builder().token(SUPPORT_BOT_TOKEN).build()

    # Сообщения от НЕ админа — пересылаем админу
    app.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE & ~filters.User(ADMIN_ID),
            _forward_to_admin,
        )
    )

    # Сообщения от админа с reply — отправляем пользователю
    app.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE & filters.User(ADMIN_ID) & filters.REPLY,
            _handle_admin_reply,
        )
    )

    _reply_map_pg_init()
    logger.info("Бот поддержки запущен. Админ ID: %s, PG: %s", ADMIN_ID, bool(DATABASE_URL))
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
