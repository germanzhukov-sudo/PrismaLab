"""User generation locks for Mini App.

Отдельный от telegram_utils — не смешиваем Telegram-layer и Web-layer.
"""
from __future__ import annotations

import asyncio
import threading

_user_locks: dict[int, tuple[asyncio.Lock, asyncio.AbstractEventLoop]] = {}
_dict_mutex = threading.Lock()


def _get_lock(user_id: int) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    with _dict_mutex:
        existing = _user_locks.get(user_id)
        if existing and existing[1] is loop:
            return existing[0]
        # New lock for current event loop
        lock = asyncio.Lock()
        _user_locks[user_id] = (lock, loop)
        return lock


async def acquire_generation_lock(user_id: int) -> asyncio.Lock | None:
    """Try to acquire lock. Returns lock on success, None if already locked."""
    lock = _get_lock(user_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=0.05)
        return lock
    except asyncio.TimeoutError:
        return None


def release_generation_lock(user_id: int, lock: asyncio.Lock) -> None:
    """Release previously acquired lock."""
    if lock.locked():
        lock.release()
