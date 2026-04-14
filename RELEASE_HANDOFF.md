# Release Handoff — miniapp-phase0 → main → prod

**Дата:** 2026-04-14 ночь → 2026-04-15
**Статус:** ✅ **DEPLOYED, прод работает.** Smoke test in progress (юзер делает вручную).

---

## Что задеплоено

Огромный релиз: tab switcher на main, Express V3 (категории/теги/провайдеры), Custom Prompt, Unified Photosets, atomic credit contract, generation_history, persona_style_previews, tariff refactor, admin cleanup. **17k+ строк, 58 файлов, 36 коммитов.**

**MERGE_HASH:** `2a9efabaa6a161be2805dd37e75e606ea97b974c` (`--no-ff` merge на `main`, для revert если что)

**Backup TS:** `20260414-230157` — code/env/volume tarballs в `~/prismalab-backups/` (локально) + `/root/prismalab-backups/` (на проде).

**Git tags для отката (на origin):**
- `pre-prod-20260414-230157` → origin/main `f44434c` (публичная история до релиза)
- `pre-release-main-20260414-230157` → main `b9a1dd1` (что стояло на проде до)
- `pre-merge-20260414-230157` → miniapp-phase0 `062cbcd` (готовый к merge HEAD)

---

## 4 hotfix'а после merge (все baseline issue, не от кода релиза)

| # | Commit | Что сломалось | Что починили |
|---|---|---|---|
| 1 | `dff5c5c` | `TypeError: unhashable type: 'dict'` — Starlette 1.0 поменял `TemplateResponse(name, context)` → `(request, name, context)` | `starlette<1.0` в requirements.txt |
| 2 | `6575302` | `telegram.error.TimedOut` — DNS lottery выдал заблокированный `149.154.166.110` | `extra_hosts: api.telegram.org→149.154.167.220` в docker-compose.yml |
| 3 | `402b21d` | `RuntimeError: Using chunked encoding is forbidden for HTTP/1.0` — попытка №1 (не помогла) | `starlette<0.40` |
| 4 | **NOT IN GIT** — nginx config на проде | Тот же chunked encoding (Starlette 0.39 тоже проверял) | `proxy_http_version 1.1;` в `/etc/nginx/sites-available/default` обоих location/. **Backup:** `~/prismalab-backups/nginx-default-20260414-200854.bak`. Если nginx переустанавливается — пере-применить вручную! |

**Visual fixes (после hotfix'ов):**
- `c283650` — `.style-card-name { font-size: 13px; line-height: 1.3 }` (вернуть старый компактный шрифт)
- `0d1ab40` — убраны эмодзи у Express стилей в admin form, list, miniapp catalog (БД колонка `emoji` оставлена для отката)

---

## Prod `.env` флаги (upserted, каждый ровно 1 раз)

```
MINIAPP_V2=1
EXPRESS_FILTERS_V3=1
CUSTOM_REQUEST_V1=1
PRISMALAB_PERSONA_LORA_NAME_MODE=gender
PACKS_USE_CREDITS=1
```

---

## Что сидено в prod БД

### Express styles (новая таблица V3)
- **39 стилей** (10 male + 29 female) залиты через `scripts/seed_express_styles.py` (через `docker cp` + `exec`, т.к. `scripts/` не в Dockerfile COPY)
- **Промпты у всех 39** (взяты из `prismalab/persona_prompts.py::PERSONA_STYLE_PROMPTS`)
- **Картинка только у `evening_glamour`** (скопирована из `dev_express_styles` через SQL UPDATE)
- **Категории**: 1 (`zhenskie/Женские`, создана юзером через `/admin/express-categories/new`)
- **Тегов**: 0
- **Привязки стилей к категориям**: 1 (`Студийный дым` id=14 → Женские)

### Persona styles (фотосет-стили, БД V1+V3)
- **Было до релиза:** 78 female + 23 male = 101 active (заведены ранее через `/admin/persona-styles`)
- **Восстановили 24 мужских** из старого `keyboards.py PERSONA_STYLES_MALE` (Байкер, Пилот, Ночной бар, Утренний кофе, ...). id 104-127, sort_order 102-127, gender=male, с промптами, **без image_url**.
- **Сейчас:** 78 female + 47 male = **125 active**

---

## TODO юзеру в админке (НЕ блокеры релиза)

1. `/admin/express-styles` — залить картинки для 38 Express styles (все кроме `evening_glamour`)
2. `/admin/express-categories` — создать остальные категории + теги
3. Привязать Express стили → categories через edit form каждого стиля
4. `/admin/persona-styles` — залить картинки для 24 восстановленных мужских (id 104-127)

---

## Smoke test status

✅ **Подтверждено работает:**
- Mini App открывается (`GET /app HTTP/1.1 200`)
- Статика грузится (CSS/JS 200)
- `/api/auth` 200, `/api/v3/express/catalog` 200
- Бот отвечает на команды (после hotfix #2 `extra_hosts`)
- Категория "Женские" создана и привязана к стилю — фильтрация работает
- Шрифт style-card вернулся к компактному 13px/1.3
- Эмодзи у Express стилей убраны

⏳ **НЕ подтверждено** (нужно от юзера):
- Реальная **Express-генерация** end-to-end (выбор стиля → загрузка фото → результат в боте)
- **Custom Prompt** генерация (textarea → фото)
- **Photoset pack** генерация за credits

После 1 успешной Express + 1 успешной Photoset генерации — релиз закрыт.

---

## Active monitors / processes

- **`b1q1gow1a`** persistent monitor live-логов прод-бота (любые `error|exception|traceback|5xx` прилетают в чат). НЕ убивать.
- Все deploy.sh завершены. На проде работают `prismalab-prismalab-1` + `prismalab-support-1`, healthy.

---

## Follow-up tickets (после релиза, отдельные PRs)

1. **Migrate все 22 TemplateResponse calls** на новую сигнатуру + unpin starlette (можно идти на 1.x)
2. **Persona batch abandonment TTL job** (Task 5e — был отложен из основного релиза)
3. **Pack background failure refund** (Task 5d не покрывает фоновые ошибки генерации после возврата 200)
4. **featured_custom URLs hardcoded** в `routes.py` → в админку/БД
5. **Декомпозиция `app_v2.js`** (~2700 строк, классическое спагетти, `ui/express.js`, `ui/custom.js`, `ui/photosets.js` etc.)
6. **nginx config в git** — текущий fix `proxy_http_version 1.1` живёт только на проде, не закреплён
7. **Залить фото к persona styles male** (24 шт)

---

## Как продолжить в новом чате

В новом чате **первое сообщение**:

> Привет. Продолжаем релиз PrismaLab miniapp-phase0. Прочитай эти два файла перед ответом:
> 1. `~/.claude/projects/-Users-germanzukov-PrismaLab/memory/MEMORY.md` — общий контекст проекта + краткая сводка релиза
> 2. `/Users/germanzukov/PrismaLab/RELEASE_HANDOFF.md` — детальный handoff текущего состояния
>
> После прочтения подтверди что подхватил: статус релиза, какие 4 hotfix'а сделаны, что осталось в smoke. Дальше я скажу что делать.

После этого Claude автоматически прочитает оба файла и будет знать:
- Что задеплоено и какие фиксы применены
- Какие backup'ы и tag'и для отката
- Что прод флаги установлены
- Что seeded в prod БД (Express + Persona styles)
- Что у юзера в TODO (картинки в админке)
- Какие follow-up тикеты

И сразу сможет продолжать работу, не задавая вопросов "а что было".
