"""
Одноразовый скрипт: доставка пака, промпты которого УЖЕ созданы.
Запускать на сервере:
  docker exec prismalab-prismalab-1 python run_pack_manual.py
"""
import asyncio
import io
import os
import time
import requests

ASTRIA_API_KEY = os.getenv("PRISMALAB_ASTRIA_API_KEY", "")
BOT_TOKEN = os.getenv("PRISMALAB_BOT_TOKEN", "")

# === ПАРАМЕТРЫ ===
USER_ID = 511428366
CHAT_ID = 511428366
PACK_ID = 3161
TUNE_ID = 4222787
CLASS_NAME = "woman"
PACK_TITLE = "Портрет художника"
EXPECTED_IMAGES = 52
# =================

HEADERS = {"Authorization": f"Bearer {ASTRIA_API_KEY}"}


def poll_images(max_minutes: int = 15):
    """
    Поллим промпты tune. Берём только те, у которых ещё нет картинок
    (= недавно созданные, от нашего POST) + те, у которых картинки
    только что появились. Ждём пока наберётся EXPECTED_IMAGES новых.
    """
    url = f"https://api.astria.ai/tunes/{TUNE_ID}/prompts"
    deadline = time.time() + max_minutes * 60

    # Первый запрос — смотрим что есть
    r = requests.get(url, headers=HEADERS, timeout=30)
    prompts = r.json() if r.status_code == 200 else []

    # Находим промпты без картинок (= наш пак, ещё генерируются)
    pending_ids = set()
    already_done_ids = set()
    for p in prompts:
        pid = str(p.get("id", ""))
        images = p.get("images") or []
        if not images:
            pending_ids.add(pid)
        else:
            already_done_ids.add(pid)

    print(f"На tune {TUNE_ID}: {len(already_done_ids)} готовых промптов, {len(pending_ids)} ещё генерируются")
    if not pending_ids:
        print("Нет pending промптов! Возможно, все уже готовы.")
        print("Ищу промпты с картинками, созданные недавно...")

    # Отслеживаем: нам нужны картинки от промптов, которые были pending
    # + любые новые промпты, которых не было в already_done_ids
    target_ids = pending_ids.copy()

    print(f"Ждём {len(target_ids) if target_ids else EXPECTED_IMAGES} фото (до {max_minutes} мин)...")

    while time.time() < deadline:
        time.sleep(8)
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code >= 400:
            print(f"  poll error: HTTP {r.status_code}")
            continue

        prompts = r.json()
        if not isinstance(prompts, list):
            continue

        ready_urls = []
        still_pending = 0

        for p in prompts:
            pid = str(p.get("id", ""))
            images = p.get("images") or []

            # Интересуют только наши промпты (были pending или появились после старта)
            is_ours = pid in target_ids or pid not in already_done_ids
            if not is_ours:
                continue

            if images:
                for img in images:
                    img_url = img if isinstance(img, str) else img.get("url", "")
                    if img_url:
                        ready_urls.append(img_url)
            else:
                still_pending += 1

        print(f"  наших фото готово: {len(ready_urls)}, ещё генерируется: {still_pending}")

        if still_pending == 0 and len(ready_urls) > 0:
            print(f"  Все наши промпты завершены! Фото: {len(ready_urls)}")
            return ready_urls
        if len(ready_urls) >= EXPECTED_IMAGES:
            return ready_urls

    print(f"TIMEOUT: получили {len(ready_urls)}")
    return ready_urls


async def send_photos(urls: list[str]):
    """Отправляем фотки юзеру через Telegram Bot API."""
    from telegram import Bot
    bot = Bot(token=BOT_TOKEN)

    total = len(urls)
    sent = 0
    for i, url in enumerate(urls, 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code != 200:
                print(f"  [{i}/{total}] download fail: HTTP {r.status_code}")
                continue
            bio = io.BytesIO(r.content)
            bio.name = f"pack_{TUNE_ID}_{i}.png"
            caption = f"Фотосет «{PACK_TITLE}» ({i}/{total})" if i == 1 else ""
            await bot.send_document(
                chat_id=CHAT_ID,
                document=bio,
                caption=caption,
                read_timeout=60,
                write_timeout=60,
            )
            sent += 1
            print(f"  [{i}/{total}] sent OK")
        except Exception as e:
            print(f"  [{i}/{total}] error: {e}")
        if i < total:
            await asyncio.sleep(0.5)

    await bot.send_message(
        chat_id=CHAT_ID,
        text=f"Фотосет «{PACK_TITLE}» готов! Отправлено {sent}/{total} фото.",
    )
    print(f"Done: {sent}/{total} фото отправлено.")


async def main():
    if not ASTRIA_API_KEY:
        print("ERROR: PRISMALAB_ASTRIA_API_KEY не задан")
        return
    if not BOT_TOKEN:
        print("ERROR: PRISMALAB_BOT_TOKEN не задан")
        return

    print(f"=== Ручной запуск пака ===")
    print(f"User: {USER_ID}, Tune: {TUNE_ID}, Pack: {PACK_TITLE}")
    print()

    # 0. Запоминаем текущие промпты (старые)
    check_url = f"https://api.astria.ai/tunes/{TUNE_ID}/prompts"
    r = requests.get(check_url, headers=HEADERS, timeout=30)
    old_prompts = r.json() if r.status_code == 200 and isinstance(r.json(), list) else []
    old_ids = {str(p.get("id")) for p in old_prompts if p.get("id")}
    pending_count = sum(1 for p in old_prompts if not (p.get("images") or []))
    print(f"Текущие промпты на tune: {len(old_prompts)} всего, {pending_count} pending")

    if pending_count == 0:
        # Нет pending — первый запуск не сработал, создаём заново
        print("Нет pending промптов — создаём промпты пака заново...")
        pack_url = f"https://api.astria.ai/p/{PACK_ID}/tunes"
        payload = {
            "tune": {
                "tune_ids": [TUNE_ID],
                "title": f"{PACK_TITLE} user:{USER_ID} manual:{int(time.time())}",
                "name": CLASS_NAME,
            }
        }
        pr = requests.post(
            pack_url,
            headers={**HEADERS, "Content-Type": "application/json"},
            json=payload,
            timeout=90,
        )
        if pr.status_code >= 400:
            print(f"ERROR: HTTP {pr.status_code}: {pr.text[:500]}")
            return
        print(f"OK! Ответ: {pr.json()}")
    else:
        print(f"Есть {pending_count} pending промптов, ждём их...")

    # Ждём фотки
    urls = poll_images()
    if not urls:
        print("Нет фото для отправки!")
        return

    # Отправляем
    await send_photos(urls)


if __name__ == "__main__":
    asyncio.run(main())
