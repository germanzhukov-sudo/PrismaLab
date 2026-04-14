"""Навигационные хэндлеры: /start, /menu, профиль, примеры, тарифы, FAQ."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, Update, WebAppInfo
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from prismalab.config import (
    ALLOWED_USERS,
    MINIAPP_URL,
    OWNER_ID,
    USERDATA_EXAMPLES_INTRO_MSG_ID,
    USERDATA_EXAMPLES_MEDIA_IDS,
    USERDATA_EXAMPLES_NAV_MSG_ID,
    USERDATA_EXAMPLES_PAGE,
    USERDATA_FAST_CUSTOM_PROMPT,
    USERDATA_FAST_LAST_MSG_ID,
    USERDATA_FAST_PERSONA_MSG_ID,
    USERDATA_FAST_SELECTED_STYLE,
    USERDATA_FAST_STYLE_MSG_ID,
    USERDATA_FAST_STYLE_PAGE,
    USERDATA_GETFILEID_EXPECTING_PHOTO,
    USERDATA_MODE,
    USERDATA_PERSONA_PACK_PHOTOS,
    USERDATA_PERSONA_PHOTOS,
    USERDATA_PERSONA_STYLE_MSG_ID,
    USERDATA_PERSONA_TRAINING_STATUS,
    USERDATA_PERSONA_WAITING_UPLOAD,
    USERDATA_PHOTO_FILE_IDS,
    USERDATA_SUBJECT_GENDER,
    _use_unified_pack_persona_flow,
)
from prismalab.keyboards import (
    _examples_intro_keyboard,
    _examples_nav_keyboard,
    _express_button,
    _express_button_label,
    _fast_gender_keyboard,
    _fast_style_choice_keyboard,
    _persona_app_keyboard,
    _persona_gender_keyboard,
    _persona_rules_keyboard,
    _persona_training_keyboard,
    _persona_upload_keyboard,
    _profile_keyboard,
    _start_keyboard,
)
from prismalab.messages import (
    PERSONA_INTRO_MESSAGE,
    PERSONA_RULES_MESSAGE,
    PERSONA_TRAINING_MESSAGE,
    PERSONA_UPLOAD_WAIT_MESSAGE,
    TARIFFS_MESSAGE,
    _fast_credits_word,
    _format_balance_persona,
    _generations_count_fast,
    _start_message_text,
)

logger = logging.getLogger("prismalab")


# ---------------------------------------------------------------------------
# Доступ к bot.py (store и функции, которые ещё не вынесены)
# ---------------------------------------------------------------------------
# Используем import модуля (не from ... import), чтобы избежать circular import.
# Всё обращение к _bot.xxx происходит в рантайме (внутри функций), не при импорте.
import prismalab.bot as _bot  # noqa: E402
from prismalab.handlers.persona import _clear_persona_flow_state, _run_persona_batch


# ---------------------------------------------------------------------------
# /start, /menu
# ---------------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = int(update.effective_user.id) if update.effective_user else 0
    if ALLOWED_USERS and user_id not in ALLOWED_USERS:
        return
    store = _bot.store
    profile = store.get_user(user_id)
    try:
        store.log_event(user_id, "start")
    except Exception:
        pass

    # Deep link: /start persona_batch
    args = context.args
    if args and args[0] == "persona_batch":
        _clear_persona_flow_state(context)
        # Task 5a: атомарный claim (get+clear за один DB-запрос) — устраняет race
        # между этим handler'ом и api_persona_generate cleanup.
        pending_json = store.claim_and_clear_pending_persona_batch(user_id)
        if pending_json:
            try:
                styles_list = json.loads(pending_json)
            except (json.JSONDecodeError, TypeError):
                styles_list = []
            if styles_list:
                await _run_persona_batch(update, context, user_id, styles_list)
                return
        await update.message.reply_text(
            "Нет запланированных генераций. Откройте Mini App и выберите стили.",
        )
        return

    # Если оплатил персону но модель не обучена — вернуть в флоу загрузки фото
    if not getattr(profile, "astria_lora_tune_id", None):
        has_credits = (getattr(profile, "persona_credits_remaining", 0) or 0) > 0
        has_pending = store.get_pending_pack_upload(user_id) is not None
        pending_training = getattr(profile, "astria_lora_tune_id_pending", None)
        if has_credits or has_pending:
            if pending_training:
                context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "training"
                await update.message.reply_text(
                    PERSONA_TRAINING_MESSAGE,
                    reply_markup=_persona_training_keyboard(),
                    parse_mode="HTML",
                )
            elif context.user_data.get(USERDATA_PERSONA_WAITING_UPLOAD):
                photos_count = len(list(context.user_data.get(USERDATA_PERSONA_PHOTOS, [])))
                pack_photos_count = len(list(context.user_data.get(USERDATA_PERSONA_PACK_PHOTOS, [])))
                count = photos_count or pack_photos_count
                if count > 0:
                    await update.message.reply_text(
                        f"Загружено {count}/10 фото. Продолжайте отправлять!",
                        reply_markup=_persona_upload_keyboard(),
                    )
                else:
                    await update.message.reply_text(PERSONA_UPLOAD_WAIT_MESSAGE, parse_mode="HTML")
            else:
                context.user_data[USERDATA_MODE] = "persona"
                await update.message.reply_text(
                    PERSONA_RULES_MESSAGE,
                    reply_markup=_persona_rules_keyboard(),
                    parse_mode="HTML",
                )
            return

    _clear_persona_flow_state(context)
    await update.message.reply_text(
        _start_message_text(profile),
        reply_markup=_start_keyboard(profile),
        parse_mode="HTML",
    )


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /menu — то же, что /start (главное меню)."""
    await start_command(update, context)


async def tips_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /tips: шпаргалка с командами деплоя и логов (только для owner)."""
    user_id = update.effective_user.id if update.effective_user else None
    if OWNER_ID and user_id != OWNER_ID:
        await update.message.reply_text("Команда недоступна.")
        return
    await update.message.reply_text(_bot.TIPS_MESSAGE, parse_mode="HTML")


# ---------------------------------------------------------------------------
# /profile, /help
# ---------------------------------------------------------------------------

async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /profile — экран Профиль."""
    user_id = int(update.effective_user.id) if update.effective_user else 0
    profile = _bot.store.get_user(user_id)
    msg = await update.message.reply_text(
        _bot._profile_text(profile),
        reply_markup=_profile_keyboard(profile),
        parse_mode="HTML",
    )
    _bot._schedule_profile_delete(context, msg.chat_id, msg.message_id, user_id)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /help — Помощь и ссылка на бот поддержки."""
    support_url = f"https://t.me/{_bot.SUPPORT_BOT_USERNAME}"
    await update.message.reply_text(
        _bot.HELP_MESSAGE,
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("Написать в поддержку", url=support_url)],
            [InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")],
        ]),
    )


# ---------------------------------------------------------------------------
# /getfileid (owner only)
# ---------------------------------------------------------------------------

async def getfileid_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /getfileid: ответь на фото и получи file_id."""
    user_id = update.effective_user.id if update.effective_user else None
    if OWNER_ID and user_id != OWNER_ID:
        await update.message.reply_text("Команда недоступна.")
        return
    reply = update.message.reply_to_message
    caption = (context.args or [])
    caption_str = " ".join(caption).strip() if caption else ""
    if not reply:
        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Анастасия", callback_data="pl_getfileid_album:Анастасия"),
                InlineKeyboardButton("Мария", callback_data="pl_getfileid_album:Мария"),
            ],
            [
                InlineKeyboardButton("Наталья", callback_data="pl_getfileid_album:Наталья"),
                InlineKeyboardButton("Иван", callback_data="pl_getfileid_album:Иван"),
            ],
        ])
        await update.message.reply_text(
            "1. Выбери альбом\n2. Отправь фото",
            reply_markup=kb,
        )
        return
    fid: str | None = None
    if reply.photo:
        fid = reply.photo[-1].file_id
    elif reply.document and (reply.document.mime_type or "").startswith("image/"):
        fid = reply.document.file_id
    if not fid:
        await update.message.reply_text("Ответь на фото (или изображение как файл)")
        return
    if caption_str:
        albums = _bot._load_examples_albums()
        found = next((a for a in albums if (a.get("caption") or "").strip() == caption_str), None)
        if found:
            ids_list = found.setdefault("file_ids", [])
            if len(ids_list) < 10:
                ids_list.append(fid)
                _bot._save_examples_albums(albums)
                await update.message.reply_text(f"Добавлено в альбом «{caption_str}» ({len(ids_list)}/10)")
            else:
                await update.message.reply_text(f"В альбоме «{caption_str}» уже 10 фото — максимум")
        else:
            albums.append({"caption": caption_str, "file_ids": [fid]})
            _bot._save_examples_albums(albums)
            await update.message.reply_text(f"Создан альбом «{caption_str}» и добавлено фото")
    else:
        await update.message.reply_text(f"<code>{fid}</code>", parse_mode="HTML")


async def handle_getfileid_album_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка выбора альбома для /getfileid."""
    query = update.callback_query
    if not query or not query.data:
        return
    user_id = query.from_user.id if query.from_user else 0
    if OWNER_ID and user_id != OWNER_ID:
        await query.answer("Недоступно")
        return
    await query.answer()
    _, album_name = query.data.split(":", 1)
    context.user_data[USERDATA_GETFILEID_EXPECTING_PHOTO] = album_name
    await query.edit_message_text(f"Теперь отправь фото — добавлю в «{album_name}»")


# ---------------------------------------------------------------------------
# Callback: Быстрое фото, Персона (навигация)
# ---------------------------------------------------------------------------

async def handle_start_fast_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Быстрое фото»."""
    from prismalab.handlers.fast_photo import _fast_after_gender_content, _send_fast_tariffs_two_messages

    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "nav_fast")
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    known_gender = getattr(profile, "subject_gender", None) or context.user_data.get(USERDATA_SUBJECT_GENDER)
    if known_gender in ("male", "female"):
        context.user_data[USERDATA_SUBJECT_GENDER] = known_gender
        context.user_data[USERDATA_MODE] = "fast"
        has_photo = bool(context.user_data.get(USERDATA_PHOTO_FILE_IDS))
        text, reply_markup = _fast_after_gender_content(profile, gender=known_gender, has_photo=has_photo)
        if text is not None:
            extra = {"parse_mode": "HTML", "disable_web_page_preview": True}
            await query.edit_message_text(text, reply_markup=reply_markup, **extra)
            context.user_data[USERDATA_FAST_STYLE_MSG_ID] = query.message.message_id
        else:
            await _send_fast_tariffs_two_messages(context.bot, query.message.chat_id, context, edit_message=query.message)
        return
    await query.edit_message_text(
        "Выбери пол – так нейросеть точнее настроит результат",
        reply_markup=_fast_gender_keyboard(),
    )


async def handle_start_persona_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Персона»."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    _bot._cancel_profile_delete_job(context)
    tariffs_msg_id = context.user_data.pop(USERDATA_FAST_STYLE_MSG_ID, None)
    if tariffs_msg_id is not None and tariffs_msg_id != query.message.message_id:
        try:
            await context.bot.delete_message(chat_id=query.message.chat_id, message_id=tariffs_msg_id)
        except Exception:
            pass
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "nav_persona")
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    context.user_data[USERDATA_MODE] = "persona"
    logger.info("handle_start_persona: user_id=%s astria_lora_tune_id=%s", user_id, getattr(profile, "astria_lora_tune_id", None))

    if _use_unified_pack_persona_flow() and _bot.store.get_pending_pack_upload(user_id) is not None and not getattr(profile, "astria_lora_tune_id", None):
        if getattr(profile, "astria_lora_tune_id_pending", None):
            context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "training"
            await query.edit_message_text(
                PERSONA_TRAINING_MESSAGE,
                reply_markup=_persona_training_keyboard(),
                parse_mode="HTML",
            )
        else:
            await query.edit_message_text(
                PERSONA_RULES_MESSAGE,
                reply_markup=_persona_rules_keyboard(),
                parse_mode="HTML",
            )
        return

    if getattr(profile, "astria_lora_tune_id", None):
        credits = profile.persona_credits_remaining
        if credits <= 0:
            text, kb = _bot._persona_credits_out_content(profile)
            await query.edit_message_text(text, reply_markup=kb, parse_mode="HTML")
            return
        await query.edit_message_text(
            f"Выберите образ в приложении <b>Персона</b> 👇\n\n{_format_balance_persona(credits)}",
            reply_markup=_persona_app_keyboard(),
            parse_mode="HTML",
        )
        context.user_data[USERDATA_PERSONA_STYLE_MSG_ID] = 0
        return

    credits = getattr(profile, "persona_credits_remaining", 0) or 0
    pending = getattr(profile, "astria_lora_tune_id_pending", None)
    if credits > 0:
        if pending:
            context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "training"
            await query.edit_message_text(
                PERSONA_TRAINING_MESSAGE,
                reply_markup=_persona_training_keyboard(),
                parse_mode="HTML",
            )
            return
        await query.edit_message_text(
            PERSONA_RULES_MESSAGE,
            reply_markup=_persona_rules_keyboard(),
            parse_mode="HTML",
        )
        return

    known_gender = getattr(profile, "subject_gender", None) or context.user_data.get(USERDATA_SUBJECT_GENDER)
    if known_gender not in ("male", "female"):
        await query.edit_message_text(
            "Выбери пол – так нейросеть точнее настроит результат",
            reply_markup=_persona_gender_keyboard(),
        )
        return
    context.user_data[USERDATA_SUBJECT_GENDER] = known_gender
    await query.edit_message_text(
        PERSONA_INTRO_MESSAGE,
        reply_markup=_persona_app_keyboard(),
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# Callback: Примеры работ
# ---------------------------------------------------------------------------

async def _show_examples_page(
    bot: Any,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    page: int,
    *,
    delete_previous: bool = False,
    nav_msg_id_to_delete: int | None = None,
) -> None:
    """Показать альбом примеров на странице page (0-based)."""
    albums = [a for a in _bot._load_examples_albums() if (a.get("file_ids") or [])]
    total = len(albums)
    if total == 0:
        return
    page = max(0, min(page, total - 1))
    album = albums[page]
    caption = (album.get("caption") or "")[:1024]
    file_ids = (album.get("file_ids") or [])[:10]
    if not file_ids:
        return
    if delete_previous:
        to_delete = list(context.user_data.pop(USERDATA_EXAMPLES_MEDIA_IDS, []) or [])
        if nav_msg_id_to_delete is not None:
            to_delete.append(nav_msg_id_to_delete)
        await asyncio.gather(*[
            bot.delete_message(chat_id=chat_id, message_id=mid) for mid in to_delete
        ], return_exceptions=True)
    media = [InputMediaPhoto(media=fid, caption=caption if i == 0 else None) for i, fid in enumerate(file_ids)]
    sent = await bot.send_media_group(chat_id=chat_id, media=media)
    context.user_data[USERDATA_EXAMPLES_MEDIA_IDS] = [m.message_id for m in sent]
    nav_text = f"{page + 1} альбом из {total}"
    kb = _examples_nav_keyboard(page, total)
    nav_msg = await bot.send_message(chat_id=chat_id, text=nav_text, reply_markup=kb)
    context.user_data[USERDATA_EXAMPLES_NAV_MSG_ID] = nav_msg.message_id
    context.user_data[USERDATA_EXAMPLES_PAGE] = page


async def handle_start_examples_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Примеры работ»."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "nav_examples")
    except Exception:
        pass
    chat_id = query.message.chat_id if query.message else 0
    bot = context.bot

    intro_msg_id = context.user_data.pop(USERDATA_EXAMPLES_INTRO_MSG_ID, None)
    media_ids = context.user_data.pop(USERDATA_EXAMPLES_MEDIA_IDS, None) or []
    to_delete = []
    if intro_msg_id is not None and intro_msg_id != query.message.message_id:
        to_delete.append(intro_msg_id)
    to_delete.extend(media_ids)
    if to_delete:
        await asyncio.gather(*[bot.delete_message(chat_id=chat_id, message_id=mid) for mid in to_delete], return_exceptions=True)

    intro = (
        "<b>Примеры персональных фотосессий</b>\n\n"
        "Ниже – результаты работы <b>Персоны</b> ✨\n"
        "Каждый пример – <b>реальный человек</b>, его исходные фото и результат после обучения персональной модели\n"
        "Все карусели размещены <b>с разрешения людей</b>, которые сами поделились результатами\n\n"
        "Мы не делаем абстрактную «красивую картинку»\n"
        "Мы усиливаем внешний вид: на снимках – вы, просто <b>спокойнее, смелее и свободнее</b>\n\n"
        "Если вы тоже решите поделиться результатами, мы <b>с радостью подарим приятный бонус</b> 🤍\n\n"
        "<b>Листайте, присматривайтесь и представляйте себя. Всё остальное сделаем мы</b>"
    )
    albums = [a for a in _bot._load_examples_albums() if (a.get("file_ids") or [])]
    if not albums:
        profile = _bot.store.get_user(user_id)
        empty_rows: list[list[InlineKeyboardButton]] = []
        if MINIAPP_URL:
            empty_rows.append([InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})])
        empty_rows.append([_express_button(profile)])
        empty_rows.append([InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")])
        empty_kb = InlineKeyboardMarkup(empty_rows)
        await query.edit_message_text(
            intro + "\n\n<b>Хотите попробовать?</b>",
            reply_markup=empty_kb,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    else:
        await query.edit_message_text(
            intro,
            reply_markup=_examples_intro_keyboard(),
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    context.user_data[USERDATA_EXAMPLES_INTRO_MSG_ID] = query.message.message_id


async def handle_examples_show_albums_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Смотреть примеры»."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    chat_id = query.message.chat_id if query.message else 0
    bot = context.bot
    saved_page = context.user_data.get(USERDATA_EXAMPLES_PAGE, 0)
    nav_msg_id = context.user_data.get(USERDATA_EXAMPLES_NAV_MSG_ID)
    await _show_examples_page(
        bot, chat_id, context, saved_page,
        delete_previous=True,
        nav_msg_id_to_delete=nav_msg_id,
    )


async def handle_examples_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Навигация по альбомам примеров."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    try:
        page = int(query.data.split(":")[1])
    except (IndexError, ValueError):
        return
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "examples_page", {"page": page})
    except Exception:
        pass
    chat_id = query.message.chat_id if query.message else 0
    bot = context.bot
    nav_msg_id = context.user_data.get(USERDATA_EXAMPLES_NAV_MSG_ID)
    await _show_examples_page(bot, chat_id, context, page, delete_previous=True, nav_msg_id_to_delete=nav_msg_id)


# ---------------------------------------------------------------------------
# Callback: Тарифы, FAQ, Помощь
# ---------------------------------------------------------------------------

async def handle_start_tariffs_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Тарифы»."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "nav_tariffs")
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    rows = []
    if MINIAPP_URL:
        rows.append([InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})])
    rows.append([_express_button(profile)])
    rows.append([InlineKeyboardButton("Назад", callback_data="pl_fast_back")])
    kb = InlineKeyboardMarkup(rows)
    await query.edit_message_text(TARIFFS_MESSAGE, reply_markup=kb, parse_mode="HTML")


async def handle_start_faq_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «А точно ли получится круто?»."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "nav_faq")
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    text = (
        "<b>Точно ли получится круто?</b>\n\n"
        "Круто получится – вопрос насколько и каким способом\n\n"
        "✨ <b>Персона (с вас 10 фото)</b>\n\n"
        "<b>Главное отличие PrismaLab</b>. Мы обучаем модель на ваших фото, и лицо сохраняется максимально точно: мимика, черты, ощущение «это я». <b>Результат – как полноценная фотосессия.</b>\n"
        "Бывает, наша нейросеть шалит с пальцами и мелкими деталями – за такие фото <b>мы начисляем дополнительные кредиты</b>\n\n"
        "⚡️ <b>Быстрое фото (с вас 1 фото)</b>\n\n"
        "Загружаете один снимок, выбираете стиль, получаете результат.\n"
        "Есть элемент случайности, но при хорошем исходнике <b>наша нейросеть выжмет максимум</b>: красиво, аккуратно и часто очень эффектно\n\n"
        "Иногда <b>Экспресс-фото с первого раза выдаёт шедевр</b> – особенно если исходное фото удачное.\n"
        "А если хочется <b>стабильно максимального сходства</b> и результата «как фотосессия» – выбирайте <b>Персону</b>\n\n"
        "Самый простой путь – попробовать <b>Экспресс-фото</b>, а если хотите включить «вау-режим» надолго – перейти на <b>Персону</b>"
    )
    rows = []
    if MINIAPP_URL:
        rows.append([InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})])
    rows.append([_express_button(profile)])
    rows.append([InlineKeyboardButton("Примеры работ", callback_data="pl_start_examples")])
    rows.append([InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")])
    kb = InlineKeyboardMarkup(rows)
    await query.edit_message_text(text, reply_markup=kb, parse_mode="HTML")


async def handle_help_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Помощь»."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "nav_help")
    except Exception:
        pass
    support_url = f"https://t.me/{_bot.SUPPORT_BOT_USERNAME}"
    await query.edit_message_text(
        _bot.HELP_MESSAGE,
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("Написать в поддержку", url=support_url)],
            [InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")],
        ]),
    )


# ---------------------------------------------------------------------------
# Callback: Профиль
# ---------------------------------------------------------------------------

async def handle_profile_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Профиль»."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    chat_id = query.message.chat_id if query.message else 0
    bot = context.bot
    persona_msg_id = context.user_data.pop(USERDATA_FAST_PERSONA_MSG_ID, None)
    if persona_msg_id is not None and persona_msg_id != query.message.message_id:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=persona_msg_id)
        except Exception:
            pass
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "profile_view")
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    await query.edit_message_text(
        _bot._profile_text(profile),
        reply_markup=_profile_keyboard(profile),
        parse_mode="HTML",
    )
    _bot._schedule_profile_delete(context, query.message.chat_id, query.message.message_id, int(query.from_user.id) if query.from_user else 0)


async def handle_profile_toggle_gender_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Изменить пол»."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "profile_toggle_gender")
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    current = profile.subject_gender or "female"
    new_gender = "male" if current == "female" else "female"
    _bot.store.set_subject_gender(user_id, new_gender)
    context.user_data[USERDATA_SUBJECT_GENDER] = new_gender
    profile = _bot.store.get_user(user_id)
    style_msg_id = context.user_data.pop(USERDATA_FAST_STYLE_MSG_ID, None)
    if style_msg_id is not None:
        try:
            await context.bot.delete_message(chat_id=query.message.chat_id, message_id=style_msg_id)
        except Exception:
            pass
    persona_style_msg_id = context.user_data.pop(USERDATA_PERSONA_STYLE_MSG_ID, None)
    if persona_style_msg_id is not None:
        try:
            await context.bot.delete_message(chat_id=query.message.chat_id, message_id=persona_style_msg_id)
        except Exception:
            pass
    await query.edit_message_text(
        _bot._profile_text(profile),
        reply_markup=_profile_keyboard(profile),
        parse_mode="HTML",
    )
    _bot._schedule_profile_delete(context, query.message.chat_id, query.message.message_id, user_id)


async def handle_profile_fast_tariffs_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Экспресс-фото» из Профиля."""
    from prismalab.handlers.fast_photo import _fast_style_screen_text, _send_fast_tariffs_two_messages

    query = update.callback_query
    if not query:
        return
    await query.answer()
    _bot._cancel_profile_delete_job(context)
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "profile_fast_tariffs")
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    if _generations_count_fast(profile) > 0:
        context.user_data[USERDATA_MODE] = "fast"
        gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or profile.subject_gender or "female"
        credits = _generations_count_fast(profile)
        credits_word = _fast_credits_word(credits)
        has_photo = bool(context.user_data.get(USERDATA_PHOTO_FILE_IDS))
        text = _fast_style_screen_text(credits, credits_word, has_photo=has_photo)
        page = context.user_data.get(USERDATA_FAST_STYLE_PAGE, 0)
        kb = _fast_style_choice_keyboard(gender, include_tariffs=True, from_profile=True, page=page)
        await query.edit_message_text(text, reply_markup=kb, parse_mode="HTML", disable_web_page_preview=True)
        context.user_data[USERDATA_FAST_STYLE_MSG_ID] = query.message.message_id
    else:
        await _send_fast_tariffs_two_messages(
            context.bot, query.message.chat_id, context, edit_message=query.message, back_callback="pl_profile"
        )


# ---------------------------------------------------------------------------
# Callback: Назад / Главное меню
# ---------------------------------------------------------------------------

async def handle_fast_back_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Назад» / «Главное меню»."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    chat_id = query.message.chat_id if query.message else 0
    bot = context.bot

    intro_msg_id = context.user_data.pop(USERDATA_EXAMPLES_INTRO_MSG_ID, None)
    media_ids = context.user_data.pop(USERDATA_EXAMPLES_MEDIA_IDS, None) or []
    context.user_data.pop(USERDATA_EXAMPLES_NAV_MSG_ID, None)
    if intro_msg_id is not None or media_ids:
        to_delete = []
        if intro_msg_id is not None and intro_msg_id != query.message.message_id:
            to_delete.append(intro_msg_id)
        to_delete.extend(media_ids)
        if to_delete:
            await asyncio.gather(*[bot.delete_message(chat_id=chat_id, message_id=mid) for mid in to_delete], return_exceptions=True)

    context.user_data.pop(USERDATA_FAST_SELECTED_STYLE, None)
    context.user_data.pop(USERDATA_FAST_CUSTOM_PROMPT, None)
    context.user_data.pop(USERDATA_FAST_LAST_MSG_ID, None)
    context.user_data.pop(USERDATA_FAST_STYLE_MSG_ID, None)
    persona_msg_id = context.user_data.pop(USERDATA_FAST_PERSONA_MSG_ID, None)
    if persona_msg_id is not None and persona_msg_id != query.message.message_id:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=persona_msg_id)
        except Exception:
            pass
    _clear_persona_flow_state(context)
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "nav_back_main")
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    text = _start_message_text(profile)
    kb = _start_keyboard(profile)
    try:
        await query.edit_message_text(text, reply_markup=kb, parse_mode="HTML")
    except BadRequest as e:
        logger.warning("handle_fast_back: edit_message_text failed (%s), sending new message", e)
        try:
            await query.message.reply_text(text, reply_markup=kb, parse_mode="HTML")
        except Exception as send_err:
            logger.error("handle_fast_back: send failed: %s", send_err)
            raise
