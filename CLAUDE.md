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
├── settings.py             — загрузка настроек из переменных окружения (env)
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

### Размер файлов и функций
- До 500 строк — норма
- 500–1000 строк — допустимо, следить
- Больше 1000 строк — обсудить разбивку на подмодули
- `storage.py` — приоритет на разбивку. При следующем крупном рефакторинге разделить по доменам: `storage/users.py`, `storage/photos.py`, `storage/payments.py`, `storage/packs.py`, `storage/personas.py`. Общее (подключение, хелперы) — в `storage/base.py`. Все публичные функции реэкспортировать из `storage/__init__.py`, чтобы импорты в остальном коде не менялись
- Ориентир — 40 строк на функцию. Если длиннее и можно разбить без потери читаемости — разбей на приватные функции с понятными именами. Не `_helper()`, а `_calculate_remaining_credits()`. Но не нарезай искусственно ради цифры

### Обработка ошибок
- Никаких голых `except: pass`. Каждый except без действия — баг
- Внутри бизнес-логики — ловить конкретные исключения (`KeyError`, `ValueError`, `psycopg2.Error` и т.д.)
- На границах системы (webhook-хэндлеры, API-эндпоинты, воркеры генерации) — допустим `except Exception` с обязательным `logger.exception(...)`, чтобы один кривой запрос не уронил весь процесс
- Каждая ошибка логируется с контекстом: `logger.error(f"Payment failed user={user_id}: {e}")`
- Все обращения к внешним API (Astria, KIE, Supabase, ЮKassa) обёрнуты в try/except с retry или внятным fallback

### SQL и безопасность данных
- Пользовательские значения в SQL — ТОЛЬКО через параметризацию (`%s`). Никаких f-строк для значений
- Интерполяция допустима только для доверенных идентификаторов: имена таблиц с `TABLE_PREFIX`, имена колонок. Никогда — для данных от пользователя
- Каждая операция с деньгами/кредитами — в транзакции (`BEGIN ... COMMIT`, `ROLLBACK` при ошибке)
- Списание кредитов: проверяй баланс в том же UPDATE: `UPDATE {TABLE_PREFIX}users SET credits = credits - %s WHERE id = %s AND credits >= %s RETURNING credits`. Не делай отдельный SELECT перед UPDATE

### Константы
- Никаких магических чисел в коде. `if credits < 4` → `if credits < MIN_EXPRESS_CREDITS`. Все константы — в `config.py`

### Фронтенд (Vanilla JS / Mini App)
- Никогда не используй innerHTML с пользовательскими данными — это XSS. Используй textContent для текста, createElement для структуры. innerHTML допустим только для статических шаблонов без пользовательских данных
- Все fetch-запросы — через единый модуль/функцию с обработкой loading/success/error/empty
- При росте app_v2.js — разбивать на модули: `ui/cards.js`, `ui/navigation.js`, `api.js` и т.д.

### Документация
- Каждая публичная функция в storage — с docstring (что делает, параметры, возвращаемое значение, исключения)
- Не дублируй паттерны. Если подключение→курсор→запрос→закрытие повторяется — вынеси в контекстный менеджер

### Логирование
- Логируй все важные действия: создание генерации, платёж, ошибка, webhook — с user_id, типом действия, результатом
- Без логов ты слепой при дебаге на проде

### Документация библиотек
При написании кода с python-telegram-bot, Supabase, Starlette — используй Context7 для проверки актуальности API.

## Требования к изменениям

### Перед написанием кода
- Если фича затрагивает 5+ файлов — сначала опиши план изменений, потом пиши код. Не начинай молча
- Не добавляй абстракции «на будущее». Не делай фабрику, если один класс. Не делай event bus, если достаточно прямого вызова. Усложняй только когда текущий подход объективно мешает

### После написания кода
- Запусти `python3 -m pytest tests/ -v`. Сломанные тесты — чини до того, как двигаться дальше
- Новая фича = новые тесты. Минимум: happy path + один error case

### При добавлении новой фичи
1. Определить модуль по таблице «Куда класть новый код»
2. Текстовые константы → `messages.py`
3. Клавиатуры → `keyboards.py`
4. USERDATA_* ключи → `config.py`
5. Новый хэндлер зарегистрировать в `bot.py:main()`
6. Запустить тесты: `python3 -m pytest tests/ -v`
7. Проверить импорт: `python3 -c "from prismalab.bot import main"`

## Миграции БД

- Любая новая таблица или колонка = SQL-файл в `migrations/` с датой и описанием: `2025_01_15_add_pack_status.sql`
- Миграция должна быть обратно совместимой: сначала добавь колонку (с дефолтом или nullable), задеплой код который её использует, потом убирай старое
- Не меняй существующие колонки напрямую (переименование, смена типа) — это сломает прод между деплоями
- Каждая миграция проверяется на dev-таблицах (`TABLE_PREFIX=dev_`) перед применением на проде

## Тесты

281 тест: storage, payment, keyboards, miniapp routes, custom generation, tariffs, bot_utils. Запуск:

```bash
python3 -m pytest tests/ -v
```

Перед любым рефакторингом или изменением логики — сначала тесты.

## Деплой

- `deploy.sh` = rsync, ему плевать на git-ветку. Всегда проверять ветку перед деплоем
