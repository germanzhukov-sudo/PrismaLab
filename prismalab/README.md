## PrismaLab

Telegram-бот: пользователь отправляет фото → выбирает стиль → получает стилизованную картинку через Astria и KIE.

### Установка

Используй зависимости из корня репозитория (`requirements.txt`) или установи минимум:

```bash
pip install python-telegram-bot[job-queue]==20.7 python-dotenv==1.0.0 requests==2.31.0
```

### Переменные окружения

Скопируй пример:

```bash
cp prismalab/env.prismalab.example .env
```

Минимум нужно:

```env
PRISMALAB_BOT_TOKEN=...
PRISMALAB_ASTRIA_API_KEY=...

# Опционально: KIE API для тестирования моделей
PRISMALAB_KIE_API_KEY=...

# Сила промпта (по умолчанию 0.7)
PRISMALAB_PROMPT_STRENGTH=0.7
```

### Запуск

Из корня репозитория:

```bash
python3 -m prismalab.bot
```

