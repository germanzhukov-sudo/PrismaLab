# PrismaLab

Telegram-бот для AI-фотосессий. python-telegram-bot 20.7, Supabase PostgreSQL, Starlette-админка.

## Запуск

```bash
# Dev-среда (TABLE_PREFIX=dev_, только owner)
./run_dev.sh

# Тесты
python3 -m pytest tests/ -v

# Проверка импорта (быстрая валидация без запуска)
python3 -c "from prismalab.bot import main"

# Деплой (ТОЛЬКО с ветки main!)
git checkout main && ./deploy.sh
```

## Архитектура

```
prismalab/
├── bot.py                  — точка входа, main(), регистрация хэндлеров, _run_style_job
├── config.py               — константы, env, USERDATA_* ключи, feature flags
├── pack_offers.py          — офферы паков (цены, ID)
├── messages.py             — тексты сообщений, форматирование баланса, склонение
├── keyboards.py            — все InlineKeyboardMarkup, стили (FAST_STYLES_*, PERSONA_STYLES_*)
├── telegram_utils.py       — retry, safe_send, user generation locks
├── image_utils.py          — resize, crop, padding, postprocess
├── storage.py              — БД (Supabase PostgreSQL), все CRUD-операции
├── payment.py              — ЮKassa, Telegram Payments, вебхуки, поллинг
├── astria_client.py        — API Astria (LoRA обучение, генерация)
├── kie_client.py           — API KIE (Seedream генерация)
├── persona_prompts.py      — промпты стилей Персоны
├── styles.py               — каталог стилей
├── alerts.py               — алерты владельцу (ошибки, медленная генерация)
├── settings.py             — загрузка настроек из admin_settings
├── supabase_storage.py     — Supabase Storage (загрузка файлов)
├── handlers/
│   ├── navigation.py       — /start, /menu, профиль, примеры, тарифы, FAQ, «Назад»
│   ├── fast_photo.py       — Экспресс-фото: выбор стиля, оплата, генерация
│   ├── persona.py          — Персона: оплата, правила, загрузка фото, обучение, стили
│   ├── packs.py            — Фотосеты: покупка, генерация, recovery
│   └── photos.py           — Приём фото, документов, текстовых сообщений
├── admin/                  — Starlette-админка (Jinja2)
└── miniapp/                — Telegram Mini App (API + фронт)
```

## Куда класть новый код

| Что | Куда |
|-----|------|
| Новый callback-хэндлер навигации | `handlers/navigation.py` |
| Хэндлер Экспресс-фото | `handlers/fast_photo.py` |
| Хэндлер Персоны (оплата, обучение, стили) | `handlers/persona.py` |
| Хэндлер фотосетов (паков) | `handlers/packs.py` |
| Обработка входящих фото/текста | `handlers/photos.py` |
| Новая клавиатура | `keyboards.py` |
| Новое текстовое сообщение/константа | `messages.py` |
| Новая USERDATA_* константа | `config.py` |
| Работа с БД | `storage.py` |
| Работа с платежами | `payment.py` |

## Правила написания кода

### Импорты
- Хэндлеры обращаются к `store` через `import prismalab.bot as _bot; _bot.store`
- Клавиатуры и сообщения импортируются напрямую из `keyboards.py` и `messages.py`
- Константы USERDATA_* — из `config.py`

### Куда НЕ класть новый код
- **НЕ в bot.py** — bot.py только точка входа (main, регистрация хэндлеров, _run_style_job). Новая логика — в handlers/ или shared-модули
- **НЕ хардкодить тексты в хэндлерах** — все пользовательские сообщения в `messages.py`
- **НЕ хардкодить клавиатуры в хэндлерах** — все клавиатуры в `keyboards.py`
- **НЕ дублировать константы** — USERDATA_*, URLs, лимиты → `config.py`

### Размер файлов
- До 500 строк — норма
- 500–1000 строк — допустимо, следить
- Больше 1000 строк — обсудить разбивку на подмодули
- `storage.py` (2200 строк) — кандидат на разбивку при росте

### При добавлении новой фичи
1. Определить модуль по таблице «Куда класть новый код»
2. Текстовые константы → `messages.py`
3. Клавиатуры → `keyboards.py`
4. USERDATA_* ключи → `config.py`
5. Новый хэндлер зарегистрировать в `bot.py:main()`
6. Запустить тесты: `python3 -m pytest tests/ -v`
7. Проверить импорт: `python3 -c "from prismalab.bot import main"`

### Документация библиотек
При написании кода с python-telegram-bot, Supabase, Starlette — используй Context7 для проверки актуальности API.

### Деплой
- `deploy.sh` = rsync, ему плевать на git-ветку. Всегда проверять ветку перед деплоем

## Тесты

29 тестов: storage, payment, keyboards, bot_utils. Запуск:

```bash
python3 -m pytest tests/ -v
```

Перед любым рефакторингом или изменением логики — сначала тесты.
