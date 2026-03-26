# План рефакторинга PrismaLab (v2)

## Контекст

PrismaLab — Telegram-бот для AI-фотосессий. Владелец не разработчик, работает один. Впереди новые фичи (CRM, рефералка, сайт). `bot.py` — 7272 строки, 184 функции. Круговые импорты. 423 ошибки линтера. 0 тестов.

**Цель:** код понятный новому разработчику за день, новые фичи без страха сломать.

**Принцип:** НЕ переписывать, а нарезать. Логика та же, меняется расположение.

---

## Этап A: Стабилизация и тестовый каркас

### A.1 Подготовка ✅ DONE
- [x] Установить pytest, pytest-asyncio
- [x] Создать `requirements-dev.txt`
- [x] Создать `tests/__init__.py`, `tests/conftest.py`
- [x] Фикстура SQLite store (tmp_path, не :memory: — т.к. каждый connect создаёт новую БД)
- [x] Проверить: `python -m pytest` запускается

### A.2 Зафиксировать состояние до рефакторинга ✅ DONE
- [x] Сохранить `HANDLERS_CHECKLIST.md` — 63 хэндлера задокументированы
- [x] Записать число хэндлеров: 63
- [x] Записать число ошибок линтера: 423

### A.3 Убрать сайд-эффекты на импорте ✅ DONE
- [x] `store = PrismaLabStore()` → `store: PrismaLabStore | None = None` + `_get_store()` lazy init
- [x] `_get_store()` вызывается в `main()` при старте бота
- [x] `import prismalab.bot` больше не создаёт подключение к БД

**Коммит:** `"remove import side-effects for safe testing"`

### A.4 Тесты на критичное ✅ DONE (29/29 passed)

**storage (test_storage.py) — 13 тестов:**
- [x] get_user, credits, generations, payment dedup, pending_pack, styles, events, gender, clear

**payment (test_payment.py) — 6 тестов:**
- [x] HMAC tokens, callback URL, _amount_rub, config checks

**bot utils (test_bot_utils.py) — только чистые функции:**
- _find_pack_offer
- _pack_offers (env + defaults)
- _format_balance_persona

**snapshot/contract (test_keyboards.py):**
- Проверить callback_data всех критичных клавиатур
- Проверить что _start_keyboard возвращает ожидаемые кнопки
- Проверить что тексты ключевых сообщений не пустые

**Коммит:** `"add critical tests before refactoring"`

---

## Этап B: Вынос общих модулей (разрыв импортов)

### B.1 `prismalab/config.py` ✅ DONE
- [x] MINIAPP_URL, OWNER_ID, ALLOWED_USERS, MAX_IMAGE_SIZE_BYTES, USER_FRIENDLY_ERROR
- [x] Все 37 USERDATA_* констант
- [x] Feature flags: _use_unified_pack_persona_flow(), _is_dev_runtime(), etc.
- [x] _guard_dev_only_flags()

### B.2 `prismalab/pack_offers.py` ✅ DONE
- [x] _DEFAULT_PACK_OFFERS, _pack_offers(), _find_pack_offer()
- [x] **Разрывает круговой импорт** payment.py → bot.py

### B.3 `prismalab/messages.py` (~15 мин)
Вынести из bot.py:
- PERSONA_RULES_MESSAGE, PERSONA_UPLOAD_WAIT_MESSAGE, TARIFFS_MESSAGE
- _start_message_text(), _format_balance_persona(), _photoset_done_message()
- Все длинные строковые константы

### B.4 `prismalab/keyboards.py` (~20 мин)
Вынести из bot.py (30+ функций):
- Все _*_keyboard() функции
- FAST_STYLES_FEMALE, FAST_STYLES_MALE, PERSONA_STYLES_*

### B.5 `prismalab/telegram_utils.py` (~10 мин)
- _safe_get_file_bytes(), _safe_edit_status(), _safe_send_document()
- _acquire_user_generation_lock() / release

### B.6 `prismalab/image_utils.py` (~10 мин)
- _prepare_image_for_photomaker/instantid/instantid_zoom()
- _round_to_64(), _ceil_to_64(), _postprocess_output()

**После каждого выноса:**
- [ ] `python3 -c "from prismalab.bot import main"` — проверка импорта
- [ ] `python -m pytest tests/ -v` — тесты проходят

**Коммит:** `"extract shared modules: config, pack_offers, messages, keyboards, telegram_utils, image_utils"`

---

## Этап C: Чистка payment.py (разрыв круговых импортов)

- [ ] Заменить все `from prismalab.bot import ...` на импорты из новых модулей
- [ ] `from prismalab.pack_offers import _find_pack_offer`
- [ ] `from prismalab.keyboards import _persona_rules_keyboard, ...`
- [ ] `from prismalab.messages import _format_balance_persona, ...`
- [ ] Проверить: `python3 -c "from prismalab.payment import run_webhook_server"` — без ошибок
- [ ] Тесты проходят

**Коммит:** `"break circular imports in payment.py"`

---

## Этап D: Разбивка bot.py на хэндлеры

Самый важный этап. Делаем по одному файлу, коммит после каждого.

### D.1 `prismalab/handlers/navigation.py` (~500 строк)
- start_command, menu_command
- handle_start_fast/persona/examples/tariffs/faq_callback
- handle_profile_callback

**Коммит:** `"extract navigation handlers"`

### D.2 `prismalab/handlers/fast_photo.py` (~1500 строк)
- _run_style_job — основная генерация
- handle_fast_buy/gender/style/show_ready_callback
- Все handle_fast_* хэндлеры

**Коммит:** `"extract fast photo handlers"`

### D.3 `prismalab/handlers/persona.py` (~2000 строк)
- Весь флоу: оплата → правила → фото → обучение
- handle_persona_buy/got_it/confirm_pay/recreate_callback
- handle_persona_topup_* хэндлеры
- _start_astria_lora, _run_persona_batch

**Коммит:** `"extract persona handlers"`

### D.4 `prismalab/handlers/packs.py` (~1000 строк)
- _run_persona_pack_generation
- _ensure_pack_lora_tune_id
- _fallback_to_pack_photo_upload
- handle_persona_pack_buy/retry_callback
- Pack recovery логика

**Коммит:** `"extract pack handlers"`

### D.5 `prismalab/handlers/photos.py` (~500 строк)
- handle_photo — роутинг фото по режимам
- handle_document — загрузка документов
- handle_text — текстовые сообщения

**Коммит:** `"extract photo/document/text handlers"`

### D.6 `prismalab/bot.py` остаётся (~300 строк)
- Импорты всех хэндлеров
- main() — регистрация хэндлеров + запуск
- post_init() — инициализация
- Глобальный error handler

**После каждого шага D.1-D.5:**
- [ ] `python3 -c "from prismalab.bot import main"` — импорт ок
- [ ] `python -m pytest tests/ -v` — тесты проходят
- [ ] Сверка: количество хэндлеров = HANDLERS_CHECKLIST.md

---

## Этап E: Линтер (ограниченный профиль)

- [ ] `ruff check prismalab/ --fix` — автоисправление безопасных
- [ ] Добавить в ruff.toml ignore для S608 (SQL f-строки — наш паттерн)
- [ ] Исправить F821 (undefined names — реальные баги)
- [ ] Исправить F841 (unused variables)
- [ ] Исправить F401 (unused imports)
- [ ] **Цель:** no new lint errors, критичные (F-серия) = 0
- [ ] НЕ цель: ноль ошибок по всем правилам

**Коммит:** `"fix critical lint errors"`

---

## Этап F: CLAUDE.md

- Описание проекта
- Как запустить (run_dev.sh, .env.dev)
- Архитектура: какой файл за что
- Правила: функция ≤50 строк, файл ≤500 строк (handlers), переиспользовать утилиты
- Куда класть новый код
- Как запускать тесты
- Пометка: storage.py (2200 строк) — кандидат на разбивку при росте

**Коммит:** `"add CLAUDE.md"`

---

## Этап G: Финальная проверка и мерж

- [ ] `python -m pytest tests/ -v` — все тесты зелёные
- [ ] `ruff check prismalab/` — без критичных ошибок
- [ ] `python3 -c "from prismalab.bot import main"` — бот стартует
- [ ] Сверка хэндлеров с HANDLERS_CHECKLIST.md
- [ ] `./run_dev.sh` — пройти /start, экспресс, персона
- [ ] `git checkout main && git merge refactoring`
- [ ] `./deploy.sh`

---

## Структура после рефакторинга

```
prismalab/
├── bot.py                  (~300 строк — точка входа)
├── config.py               (~100 строк — константы, env, flags)
├── pack_offers.py          (~80 строк — офферы паков)
├── keyboards.py            (~400 строк — все клавиатуры)
├── messages.py             (~300 строк — тексты сообщений)
├── telegram_utils.py       (~200 строк — retry, safe_send, locks)
├── image_utils.py          (~150 строк — обработка фото)
├── storage.py              (~2200 строк — БД, без изменений*)
├── payment.py              (~1300 строк — без круговых импортов)
├── astria_client.py        (без изменений)
├── kie_client.py           (без изменений)
├── settings.py             (без изменений)
├── alerts.py               (без изменений)
├── persona_prompts.py      (без изменений)
├── styles.py               (без изменений)
├── supabase_storage.py     (без изменений)
├── handlers/
│   ├── __init__.py
│   ├── navigation.py       (~500 строк)
│   ├── fast_photo.py       (~1500 строк)
│   ├── persona.py          (~2000 строк)
│   ├── packs.py            (~1000 строк)
│   └── photos.py           (~500 строк)
├── admin/                  (без изменений)
├── miniapp/                (без изменений)
tests/
├── conftest.py
├── test_storage.py
├── test_payment.py
├── test_bot_utils.py
└── test_keyboards.py
```

*storage.py помечен в CLAUDE.md как кандидат на разбивку при росте

## Что НЕ входит в этот рефакторинг
- Alembic миграции (отдельный этап после)
- Переписывание логики
- CRM, рефералка, сайт (после рефакторинга)
- Полная ликвидация всех ошибок линтера

## Риски

| Риск | Митигация |
|------|-----------|
| Сломанный импорт | `python3 -c "from prismalab.bot import main"` после каждого шага |
| Потерянный хэндлер | HANDLERS_CHECKLIST.md — сверка до/после |
| Сломанный callback_data | test_keyboards.py — snapshot тесты |
| Сайд-эффекты при импорте | Этап A.3 — убираем до начала рефакторинга |
| Круговой импорт | pack_offers.py + config.py разрывают цепочку |
| Сломанный прод | Ветка refactoring, main не трогаем до финала |
