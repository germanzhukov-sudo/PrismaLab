"""Supabase Storage — загрузка файлов через REST API.

Использует переменные окружения:
  SUPABASE_URL          — URL проекта (https://xxx.supabase.co)
  SUPABASE_SERVICE_KEY  — service_role key (для серверных операций)

Bucket: persona-styles (создаётся вручную в Supabase Dashboard, public).
"""
from __future__ import annotations

import logging
import os
import uuid

import requests

logger = logging.getLogger("prismalab.supabase_storage")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

BUCKET = "persona-styles"


def _headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }


def upload_image(file_bytes: bytes, filename: str, content_type: str = "image/jpeg") -> str | None:
    """Загружает файл в Supabase Storage. Возвращает публичный URL или None при ошибке.

    filename — имя файла в бакете (например, 'wedding.jpg'). Для уникальности
    добавляется uuid-префикс.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        logger.error("SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
        return None

    # Уникальное имя для предотвращения конфликтов
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "jpg"
    storage_path = f"{uuid.uuid4().hex[:12]}.{ext}"

    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{storage_path}"
    headers = _headers()
    headers["Content-Type"] = content_type

    try:
        resp = requests.post(url, headers=headers, data=file_bytes, timeout=30)
        if resp.status_code in (200, 201):
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{storage_path}"
            logger.info("Uploaded %s → %s", filename, public_url)
            return public_url
        logger.error("Upload failed: %s %s", resp.status_code, resp.text[:200])
        return None
    except Exception as e:
        logger.error("Upload error: %s", e)
        return None


def delete_image(public_url: str) -> bool:
    """Удаляет файл из Supabase Storage по публичному URL."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return False

    # Извлекаем путь из URL: .../object/public/persona-styles/abc123.jpg → abc123.jpg
    prefix = f"/storage/v1/object/public/{BUCKET}/"
    idx = public_url.find(prefix)
    if idx < 0:
        logger.warning("Cannot parse storage path from URL: %s", public_url)
        return False
    storage_path = public_url[idx + len(prefix):]

    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}"
    headers = _headers()
    headers["Content-Type"] = "application/json"

    try:
        resp = requests.delete(url, headers=headers, json={"prefixes": [storage_path]}, timeout=15)
        ok = resp.status_code in (200, 201)
        if ok:
            logger.info("Deleted %s", storage_path)
        else:
            logger.warning("Delete failed: %s %s", resp.status_code, resp.text[:200])
        return ok
    except Exception as e:
        logger.error("Delete error: %s", e)
        return False
