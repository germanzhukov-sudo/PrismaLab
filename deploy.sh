#!/bin/bash
# Деплой PrismaLab на сервер
# Использование: ./deploy.sh        — задеплоить и запустить
#              ./deploy.sh stop     — остановить бота на сервере (чтобы не было Conflict)
# Нужно: .env в папке проекта, SSH-доступ к серверу

set -e

SERVER="root@194.87.133.7"
REMOTE_DIR="/root/PrismaLab"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "stop" ]; then
  echo "Останавливаю бота на сервере..."
  ssh "$SERVER" "cd $REMOTE_DIR && docker compose down"
  echo "Готово. Бот на сервере остановлен."
  exit 0
fi

cd "$LOCAL_DIR"

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)
if [ -n "$branch" ] && [ "$branch" != "main" ]; then
  echo "Ошибка: деплой только с ветки main. Сейчас: $branch"
  exit 1
fi

if [ ! -f .env ]; then
  echo "Ошибка: создай .env в папке проекта с токенами и ключами"
  exit 1
fi

echo "Копирую проект на сервер..."
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='*.db' --exclude='.git' --exclude='*.log' --exclude='*.bak' --exclude='.env' --exclude='.env.*' --exclude='.env.dev' . "$SERVER:$REMOTE_DIR/"

echo "Запускаю миграцию БД (создание public.users при наличии DATABASE_URL)..."
ssh "$SERVER" "cd $REMOTE_DIR && docker compose run --rm prismalab python -m prismalab.migrate_db" || true

echo "Запускаю бота на сервере..."
ssh "$SERVER" "cd $REMOTE_DIR && docker compose up -d --build"

echo "Готово. Бот запущен."
echo "Логи: ssh $SERVER 'cd $REMOTE_DIR && docker compose logs -f'"
