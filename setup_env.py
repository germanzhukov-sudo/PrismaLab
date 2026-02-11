#!/usr/bin/env python3
"""
Интерактивная настройка PrismaLab: создаёт .env в текущей папке.

Запуск:
  cd "/Users/germanzukov/PrismaLab"
  python3 setup_env.py
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PROMPT_STRENGTH = "0.7"


def _ask(name: str, example: str | None = None, default: str | None = None) -> str:
    hint = []
    if example:
        hint.append(f"пример: {example}")
    if default is not None:
        hint.append(f"по умолчанию: {default}")
    hint_s = f" ({', '.join(hint)})" if hint else ""
    raw = input(f"{name}{hint_s}\n> ").strip()
    if not raw and default is not None:
        return default
    return raw


def main() -> None:
    env_path = Path(".env")
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    existing_map: dict[str, str] = {}
    if existing:
        for line in existing.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            existing_map[k.strip()] = v.strip()

    if env_path.exists():
        ans = input(".env уже существует. Обновить, сохранив токены? (Y/n)\n> ").strip().lower()
        if ans in {"n", "no"}:
            ans2 = input("Перезаписать полностью (попросит токены заново)? (y/N)\n> ").strip().lower()
            if ans2 not in {"y", "yes"}:
                print("Ок, ничего не делаю.")
                return

    bot_token = existing_map.get("PRISMALAB_BOT_TOKEN") or _ask(
        "Вставь PRISMALAB_BOT_TOKEN", example="123456:ABCDEF..."
    )
    astria_key = existing_map.get("PRISMALAB_ASTRIA_API_KEY") or _ask(
        "Вставь PRISMALAB_ASTRIA_API_KEY (Astria)", example="..."
    )
    kie_key = _ask(
        "PRISMALAB_KIE_API_KEY (KIE, опционально)",
        default=existing_map.get("PRISMALAB_KIE_API_KEY") or "",
    )
    prompt_strength = _ask(
        "PRISMALAB_PROMPT_STRENGTH (0..1)",
        default=existing_map.get("PRISMALAB_PROMPT_STRENGTH") or DEFAULT_PROMPT_STRENGTH,
    )

    if not bot_token:
        print("Нужен PRISMALAB_BOT_TOKEN.")
        return

    content = (
        "PRISMALAB_APP_NAME=PrismaLab\n"
        f"PRISMALAB_BOT_TOKEN={bot_token}\n"
        f"PRISMALAB_ASTRIA_API_KEY={astria_key}\n"
        f"PRISMALAB_KIE_API_KEY={kie_key}\n"
        f"PRISMALAB_PROMPT_STRENGTH={prompt_strength}\n"
    )

    try:
        fd = os.open(str(env_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        env_path.write_text(content, encoding="utf-8")

    print("Готово: .env создан.")
    print("Запуск бота:")
    print('  python3 -m prismalab.bot')


if __name__ == "__main__":
    main()
