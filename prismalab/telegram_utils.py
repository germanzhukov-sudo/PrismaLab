"""Утилиты Telegram: retry, safe_send, locks."""

from __future__ import annotations

import asyncio
import io
import logging
import threading
from typing import Any

from telegram.error import BadRequest, RetryAfter, TimedOut

logger = logging.getLogger("prismalab")


# ---------------------------------------------------------------------------
# Блокировка генерации на уровне user_id
# ---------------------------------------------------------------------------

_user_locks: dict[int, asyncio.Lock] = {}
_lock_dict_mutex = threading.Lock()


def _get_user_lock(user_id: int) -> asyncio.Lock:
    with _lock_dict_mutex:
        if user_id not in _user_locks:
            _user_locks[user_id] = asyncio.Lock()
        return _user_locks[user_id]


async def _acquire_user_generation_lock(user_id: int) -> asyncio.Lock | None:
    """Пытается захватить lock. Возвращает lock при успехе, None если уже занят."""
    lock = _get_user_lock(user_id)
    # timeout=0.05: даём шанс acquire выполниться; при занятом lock — TimeoutError
    try:
        await asyncio.wait_for(lock.acquire(), timeout=0.05)
        return lock
    except asyncio.TimeoutError:
        return None


# ---------------------------------------------------------------------------
# Безопасные операции с Telegram API
# ---------------------------------------------------------------------------

async def _safe_get_file_bytes(
    bot: Any,
    file_id: str,
    *,
    max_retries: int = 2,
    timeout: int = 20,
) -> bytes:
    """
    Безопасное скачивание файла из Telegram с обработкой таймаутов и retry.
    ПРОСТАЯ версия без лишних обёрток.
    """
    logger.info(f"[СКАЧИВАНИЕ] Начинаю скачивание файла {file_id[:15]}...")

    for attempt in range(max_retries):
        try:
            logger.info(f"[СКАЧИВАНИЕ] Попытка {attempt + 1}/{max_retries}: вызываю get_file...")
            tg_file = await bot.get_file(file_id, read_timeout=timeout, write_timeout=timeout, connect_timeout=timeout)

            logger.info("[СКАЧИВАНИЕ] get_file OK, вызываю download_as_bytearray...")
            image_bytes = bytes(await tg_file.download_as_bytearray(read_timeout=timeout, write_timeout=timeout, connect_timeout=timeout))

            logger.info(f"[СКАЧИВАНИЕ] ✅ Файл скачан! Размер: {len(image_bytes)} байт")
            return image_bytes

        except (TimedOut, asyncio.TimeoutError) as e:
            error_type = "TimedOut" if isinstance(e, TimedOut) else "asyncio.TimeoutError"
            logger.warning(f"[СКАЧИВАНИЕ] ❌ {error_type} на попытке {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                wait_time = 3
                logger.info(f"[СКАЧИВАНИЕ] Жду {wait_time}с перед повтором...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"[СКАЧИВАНИЕ] ❌ Таймаут после {max_retries} попыток")
                raise
        except Exception as e:
            logger.warning(f"[СКАЧИВАНИЕ] ❌ Ошибка {type(e).__name__}: {e} на попытке {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                wait_time = 3
                logger.info(f"[СКАЧИВАНИЕ] Жду {wait_time}с перед повтором...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"[СКАЧИВАНИЕ] ❌ Ошибка после {max_retries} попыток: {type(e).__name__}: {e}")
                raise


async def _safe_edit_status(bot: Any, chat_id: int, message_id: int, text: str, **kwargs: Any) -> None:
    """edit_message_text с подавлением 'Message to edit not found' и 'message is not modified'."""
    try:
        await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, **kwargs)
    except BadRequest as e:
        err = str(e).lower()
        if "message to edit not found" in err or "message is not modified" in err:
            logger.debug("edit_status пропущен: %s", e)
        else:
            raise


async def _safe_send_document(
    bot: Any,
    chat_id: int,
    document: io.BytesIO,
    caption: str,
    *,
    max_retries: int = 3,
    timeout: int = 90,
) -> None:
    """
    Безопасная отправка документа с обработкой таймаутов и retry.
    Если отправка документа не удалась, пробует отправить как фото.
    """
    document.seek(0)

    for attempt in range(max_retries):
        try:
            await bot.send_document(
                chat_id=chat_id,
                document=document,
                caption=caption,
                read_timeout=timeout,
                write_timeout=timeout,
                connect_timeout=timeout,
            )
            return  # Успешно отправлено
        except RetryAfter as e:
            retry_after = float(getattr(e, "retry_after", 1.0) or 1.0)
            wait_time = max(1.0, retry_after) + 0.5
            if attempt < max_retries - 1:
                logger.warning(
                    "Flood limit при отправке документа (попытка %s/%s), жду %.1fс...",
                    attempt + 1,
                    max_retries,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
                document.seek(0)
            else:
                logger.warning("Flood limit после %s попыток, пробую отправить как фото...", max_retries)
                document.seek(0)
                try:
                    await bot.send_photo(
                        chat_id=chat_id,
                        photo=document,
                        caption=caption,
                        read_timeout=timeout,
                        write_timeout=timeout,
                        connect_timeout=timeout,
                    )
                    return
                except RetryAfter as photo_retry:
                    photo_wait = max(1.0, float(getattr(photo_retry, "retry_after", 1.0) or 1.0)) + 0.5
                    logger.warning("Flood limit и на фото fallback, жду %.1fс и повторяю...", photo_wait)
                    await asyncio.sleep(photo_wait)
                    document.seek(0)
                    await bot.send_photo(
                        chat_id=chat_id,
                        photo=document,
                        caption=caption,
                        read_timeout=timeout,
                        write_timeout=timeout,
                        connect_timeout=timeout,
                    )
                    return
                except Exception as photo_err:
                    logger.error(f"Ошибка при отправке фото (fallback): {photo_err}")
                    raise
        except TimedOut:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.warning(f"Таймаут при отправке документа (попытка {attempt + 1}/{max_retries}), жду {wait_time}с...")
                await asyncio.sleep(wait_time)
                document.seek(0)
            else:
                logger.warning(f"Таймаут при отправке документа после {max_retries} попыток, пробую отправить как фото...")
                # Fallback: пробуем отправить как фото
                document.seek(0)
                try:
                    await bot.send_photo(
                        chat_id=chat_id,
                        photo=document,
                        caption=caption,
                        read_timeout=timeout,
                        write_timeout=timeout,
                        connect_timeout=timeout,
                    )
                    return
                except Exception as photo_err:
                    logger.error(f"Ошибка при отправке фото (fallback): {photo_err}")
                    raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.warning(f"Ошибка при отправке документа (попытка {attempt + 1}/{max_retries}): {e}, жду {wait_time}с...")
                await asyncio.sleep(wait_time)
                document.seek(0)
            else:
                logger.error(f"Ошибка при отправке документа после {max_retries} попыток: {e}")
                # Последняя попытка: пробуем фото
                document.seek(0)
                try:
                    await bot.send_photo(
                        chat_id=chat_id,
                        photo=document,
                        caption=caption,
                        read_timeout=timeout,
                        write_timeout=timeout,
                        connect_timeout=timeout,
                    )
                    return
                except Exception as photo_err:
                    logger.error(f"Ошибка при отправке фото (fallback): {photo_err}")
                    raise
