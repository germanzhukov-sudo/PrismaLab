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

### B.3 `prismalab/messages.py` ✅ DONE
- [x] _fast_credits_word(), _generations_count_fast(), _generations_line(), _fast_generations_line()
- [x] _format_balance_express(), _format_balance_persona()
- [x] _start_message_text(), _photoset_done_message()
- [x] STYLE_EXAMPLES_FOOTER, PERSONA_INTRO_MESSAGE, PERSONA_CREDITS_OUT_MESSAGE
- [x] PERSONA_RULES_MESSAGE, PERSONA_UPLOAD_WAIT_MESSAGE, PERSONA_PACK_UPLOAD_WAIT_MESSAGE
- [x] PERSONA_TRAINING_MESSAGE, PHOTOSET_PROGRESS_ALERT
- [x] FAST_TARIFFS_TARIFFS_MESSAGE, TARIFFS_MESSAGE
- [x] Импорты в payment.py обновлены (PERSONA_RULES_MESSAGE, _format_balance_persona, etc.)
- [x] Тесты обновлены на прямые импорты из messages.py

### B.4 `prismalab/keyboards.py` ✅ DONE
- [x] 33 keyboard-функции вынесены из bot.py
- [x] FAST_STYLES_FEMALE/MALE, PERSONA_STYLES_FEMALE/MALE, *_PER_PAGE
- [x] _express_button_label, _fast_style_label (хелперы клавиатур)
- [x] Импорты в payment.py обновлены (keyboards + pack_offers + config)
- [x] Тесты обновлены на прямые импорты из keyboards.py

### B.5 `prismalab/telegram_utils.py` ✅ DONE
- [x] _safe_get_file_bytes(), _safe_edit_status(), _safe_send_document()
- [x] _get_user_lock(), _acquire_user_generation_lock() + глобальные _user_locks, _lock_dict_mutex
- [x] Импорт в payment.py обновлён (_safe_send_document)

### B.6 `prismalab/image_utils.py` ✅ DONE
- [x] _prepare_image_for_photomaker/instantid/instantid_zoom()
- [x] _round_to_64(), _ceil_to_64(), _postprocess_output()
- [x] _guess_aspect_ratio(), _format_strength(), _subject_prompt_prefix()

**После каждого выноса:**
- [ ] `python3 -c "from prismalab.bot import main"` — проверка импорта
- [ ] `python -m pytest tests/ -v` — тесты проходят

**Коммит:** `"extract shared modules: config, pack_offers, messages, keyboards, telegram_utils, image_utils"`

---

## Этап C: Чистка payment.py (разрыв круговых импортов) ✅ DONE

- [x] USERDATA_* → config.py (2 вхождения)
- [x] _find_pack_offer → pack_offers.py (3 вхождения)
- [x] MINIAPP_URL → config.py
- [x] _format_balance_persona, _format_balance_express, _generations_count_fast, STYLE_EXAMPLES_FOOTER → messages.py
- [x] PERSONA_RULES_MESSAGE → messages.py (3 вхождения)
- [x] _photoset_done_message → messages.py
- [x] _persona_rules_keyboard, _persona_training_keyboard → keyboards.py
- [x] _fast_style_choice_keyboard → keyboards.py
- [x] _photoset_done_keyboard, _photoset_retry_keyboard → keyboards.py
- [x] _safe_send_document → telegram_utils.py
- [x] Остались 3 обоснованных ленивых импорта: _run_persona_pack_generation (x2), _pack_polling_active
- [x] `python3 -c "from prismalab.payment import run_webhook_server"` — OK
- [x] 29/29 тестов

---

## Этап D: Разбивка bot.py на хэндлеры

Самый важный этап. Делаем по одному файлу, коммит после каждого.

### D.1 `prismalab/handlers/navigation.py` ✅ DONE (752 строк)
- [x] start_command, menu_command, tips_command, profile_command, help_command
- [x] getfileid_command, handle_getfileid_album_callback
- [x] handle_start_fast/persona/examples/tariffs/faq_callback
- [x] handle_profile/profile_toggle_gender/profile_fast_tariffs_callback
- [x] handle_fast_back_callback, handle_help_callback
- [x] _show_examples_page, handle_examples_show_albums/page_callback

### D.2 `prismalab/handlers/fast_photo.py` ✅ DONE (787 строк)
- [x] 16 функций: handle_fast_*, handle_pre_checkout, handle_successful_payment
- [x] _run_fast_generation_impl, _send_fast_tariffs_two_messages, _fast_after_gender_content
- [x] _fast_ready_to_upload_text, _fast_style_screen_text, _update_fast_style_message

### D.3 `prismalab/handlers/persona.py` ✅ DONE (1282 строки)
- [x] 23 функции: handle_persona_*, _start_astria_lora, _run_persona_batch, newpersona_command, _clear_persona_flow_state

### D.4 `prismalab/handlers/packs.py` ✅ DONE (1399 строк)
- [x] 16 функций: _run_persona_pack_generation, _ensure_pack_lora_tune_id, handle_persona_pack_*, recovery логика

### D.5 `prismalab/handlers/photos.py` ✅ DONE (612 строк)
- [x] handle_photo, handle_document, handle_text

### D.6 `prismalab/bot.py` — точка входа (1785 строк)
- Импорты всех хэндлеров
- main() — регистрация хэндлеров + запуск
- Legacy хэндлеры (handle_kie_test, handle_style, etc.)
- _run_style_job — используется из нескольких модулей
- Утилиты и message-константы, не вынесенные в отдельные модули

**После каждого шага D.1-D.5:**
- [ ] `python3 -c "from prismalab.bot import main"` — импорт ок
- [ ] `python -m pytest tests/ -v` — тесты проходят
- [ ] Сверка: количество хэндлеров = HANDLERS_CHECKLIST.md

---

## Этап E: Линтер (ограниченный профиль) ✅ DONE

- [x] `ruff check prismalab/ --fix` — 199 автоисправлений (F401 unused imports, F541 f-strings)
- [x] F821 (undefined names) = 0 (было 105)
- [x] F401 (unused imports) = 0 (было 186)
- [x] Исправлен баг: сломанный except блок в _run_fast_generation_impl
- [x] Остаток: F841=11 (unused vars), F811=1 — не баги, warning-уровень
- [x] S608 (SQL f-strings) — наш паттерн, не исправляем

---

## Этап F: CLAUDE.md ✅ DONE

- [x] Описание проекта, запуск, тесты
- [x] Архитектура: файловая структура с описанием
- [x] Таблица «куда класть новый код»
- [x] Правила: паттерн доступа к store, деплой

---

## Этап G: Финальная проверка и мерж

- [x] `python -m pytest tests/ -v` — 29/29 passed
- [x] `ruff check prismalab/ --select F821` — 0 ошибок
- [x] `python3 -c "from prismalab.bot import main"` — OK
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
