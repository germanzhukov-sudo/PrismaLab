#!/bin/bash
# Останавливает бота на сервере (чтобы не было Conflict при запуске на Mac)
SERVER="root@194.87.133.7"
REMOTE_DIR="/root/PrismaLab"
ssh "$SERVER" "cd $REMOTE_DIR && docker compose down"
echo "Бот на сервере остановлен. Можешь запускать бота на Mac."
