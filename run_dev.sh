#!/bin/bash
# Скрипт для запуска dev-бота локально
# Использует .env.dev вместо .env

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Проверяем что .env.dev существует
if [ ! -f ".env.dev" ]; then
    echo "Ошибка: .env.dev не найден!"
    echo "Создай .env.dev с токеном dev-бота и TABLE_PREFIX=dev_"
    exit 1
fi

# Загружаем переменные из .env.dev
set -a
source .env.dev
set +a

echo "=== DEV MODE ==="
echo "TABLE_PREFIX: $TABLE_PREFIX"
echo "ALLOWED_USERS: $ALLOWED_USERS"
echo "Bot token: ${PRISMALAB_BOT_TOKEN:0:10}..."
echo "================"

# Запускаем бота
python3 -m prismalab.bot
