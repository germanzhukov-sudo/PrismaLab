"""Хэндлеры фото, документов и текстовых сообщений."""

from __future__ import annotations

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, WebAppInfo
from telegram.ext import ContextTypes

from prismalab.config import (
    ALLOWED_USERS,
    MAX_IMAGE_SIZE_BYTES,
    MINIAPP_URL,
    OWNER_ID,
    USERDATA_FAST_CUSTOM_PROMPT,
    USERDATA_FAST_SELECTED_STYLE,
    USERDATA_FAST_STYLE_PAGE,
    USERDATA_GETFILEID_EXPECTING_PHOTO,
    USERDATA_MODE,
    USERDATA_PERSONA_PACK_PHOTOS,
    USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS,
    USERDATA_PERSONA_PACK_WAITING_UPLOAD,
    USERDATA_PERSONA_PHOTOS,
    USERDATA_PERSONA_SELECTED_PACK_ID,
    USERDATA_PERSONA_TRAINING_MSG_ID,
    USERDATA_PERSONA_TRAINING_STATUS,
    USERDATA_PERSONA_UPLOAD_MSG_IDS,
    USERDATA_PERSONA_WAITING_UPLOAD,
    USERDATA_PHOTO_FILE_IDS,
    USERDATA_ASTRIA_LORA_FILE_IDS,
    USERDATA_FAST_LAST_MSG_ID,
    USERDATA_SUBJECT_GENDER,
    _use_unified_pack_persona_flow,
)
from prismalab.keyboards import (
    _express_button_label,
    _fast_style_choice_keyboard,
    _fast_style_label,
    _persona_pack_upload_keyboard,
    _persona_training_keyboard,
    _persona_upload_keyboard,
    _start_keyboard,
)
from prismalab.pack_offers import _find_pack_offer
from prismalab.messages import (
    PERSONA_TRAINING_MESSAGE,
    STYLE_EXAMPLES_FOOTER,
    _fast_credits_word,
    _format_balance_express,
    _generations_count_fast,
)
from prismalab.telegram_utils import _acquire_user_generation_lock

logger = logging.getLogger("prismalab")

import prismalab.bot as _bot  # noqa: E402


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Приём своего запроса в Экспресс-фото (стиль «Свой запрос»)."""
    if not update.message or not update.message.text:
        return
    user_id = int(update.effective_user.id) if update.effective_user else 0
    if ALLOWED_USERS and user_id not in ALLOWED_USERS:
        return
    mode = context.user_data.get(USERDATA_MODE) or "normal"
    if mode != "fast":
        return
    try:
        _bot.store.log_event(user_id, "text_input", {"mode": mode})
    except Exception:
        pass
    selected_style = context.user_data.get(USERDATA_FAST_SELECTED_STYLE)
    if selected_style == "custom":
        text = (update.message.text or "").strip()
        if not text:
            await update.message.reply_text("Напишите непустое описание картинки.")
            return
        context.user_data[USERDATA_FAST_CUSTOM_PROMPT] = text[:2000]
        photo_file_ids = list(context.user_data.get(USERDATA_PHOTO_FILE_IDS, []))
        if photo_file_ids:
            user_id = int(update.effective_user.id) if update.effective_user else 0
            profile = _bot.store.get_user(user_id)
            if _generations_count_fast(profile) <= 0:
                chat_id = update.effective_chat.id if update.effective_chat else 0
                await _bot._send_fast_tariffs_two_messages(context.bot, chat_id, context)
                return
            gen_lock = await _acquire_user_generation_lock(user_id)
            if gen_lock is None:
                await update.message.reply_text("Пожалуйста, немного подождите. Ещё обрабатываю прошлый запрос")
                return
            chat_id = update.effective_chat.id if update.effective_chat else 0
            status_msg = await update.message.reply_text("🎨 <i>Создаю изображение...</i>", parse_mode="HTML")

            async def _run_and_release() -> None:
                try:
                    await _bot._run_fast_generation_impl(
                        context=context,
                        chat_id=chat_id,
                        user_id=user_id,
                        style_id="custom",
                        style_label="Свой запрос",
                        prompt=text[:2000],
                        photo_file_ids=photo_file_ids,
                        profile=profile,
                        status_msg=status_msg,
                    )
                finally:
                    gen_lock.release()

            context.application.create_task(_run_and_release())
        else:
            await update.message.reply_text(
                "Запрос принят ✅ Теперь загрузите фото по правилам ниже.\n\n" + _bot.FAST_PHOTO_RULES_MESSAGE,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("Назад", callback_data="pl_fast_change_style"),
                ]]),
                parse_mode="HTML",
            )
    else:
        await update.message.reply_text(
            "Чтобы отправить свой запрос, нажмите «✏️ Свой запрос» в меню стилей.",
        )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return

    user_id = int(update.effective_user.id) if update.effective_user else 0
    if ALLOWED_USERS and user_id not in ALLOWED_USERS:
        return
    album_name = context.user_data.pop(USERDATA_GETFILEID_EXPECTING_PHOTO, None)
    if OWNER_ID and user_id == OWNER_ID and album_name:
        photo = update.message.photo[-1]
        albums = _bot._load_examples_albums()
        found = next((a for a in albums if (a.get("caption") or "").strip() == album_name), None)
        if found:
            ids_list = found.setdefault("file_ids", [])
            if len(ids_list) < 10:
                ids_list.append(photo.file_id)
                _bot._save_examples_albums(albums)
                await update.message.reply_text(f"Добавлено в «{album_name}» ({len(ids_list)}/10)")
            else:
                await update.message.reply_text(f"В «{album_name}» уже 10 фото — максимум")
        else:
            albums.append({"caption": album_name, "file_ids": [photo.file_id]})
            _bot._save_examples_albums(albums)
            await update.message.reply_text(f"Создан альбом «{album_name}» и добавлено фото")
        return

    mode = context.user_data.get(USERDATA_MODE) or "normal"
    # Fallback: Mini App оплата — user_data не доходит из webhook/poll, проверяем БД
    if mode != "persona_pack_upload" and not _use_unified_pack_persona_flow():
        pending_pack_id = _bot.store.get_pending_pack_upload(user_id)
        if pending_pack_id is not None:
            context.user_data[USERDATA_MODE] = "persona_pack_upload"
            context.user_data[USERDATA_PERSONA_SELECTED_PACK_ID] = pending_pack_id
            context.user_data[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = True
            context.user_data[USERDATA_PERSONA_PACK_PHOTOS] = []
            context.user_data[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = []
            mode = "persona_pack_upload"
    try:
        _bot.store.log_event(user_id, "photo_upload", {"mode": mode})
    except Exception:
        pass
    logger.info(f"[Photo Handler] Режим: {mode}, user {update.effective_user.id}")
    photo = update.message.photo[-1]  # самое большое

    if photo.file_size is not None and photo.file_size > MAX_IMAGE_SIZE_BYTES:
        await update.message.reply_text(
            f"Слишком большой файл ({photo.file_size // (1024 * 1024)} МБ). Максимум 15 МБ. Сожмите фото или отправьте в меньшем разрешении."
        )
        return

    if mode == "persona_pack_upload" and context.user_data.get(USERDATA_PERSONA_PACK_WAITING_UPLOAD):
        ids = list(context.user_data.get(USERDATA_PERSONA_PACK_PHOTOS, []))
        if len(ids) >= 10:
            await update.message.reply_text("Уже загружено 10 фото для фотосета. Нажмите «Сбросить фото фотосета» или дождитесь результата.")
            return
        ids.append(photo.file_id)
        context.user_data[USERDATA_PERSONA_PACK_PHOTOS] = ids
        count = len(ids)
        if count < 10:
            text = f"Фото для фотосета {count}/10 получено. Осталось {10 - count}."
            msg = await update.message.reply_text(text, reply_markup=_persona_pack_upload_keyboard())
            upload_msg_ids = list(context.user_data.get(USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS, []))
            upload_msg_ids.append(msg.message_id)
            context.user_data[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = upload_msg_ids
        else:
            context.user_data[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = False
            context.user_data[USERDATA_MODE] = "persona"
            _bot.store.clear_pending_pack_upload(user_id)
            await update.message.reply_text("Все 10 фото получил ✅\n\nЗапускаю генерацию фотосета…")
            pack_id = int(context.user_data.get(USERDATA_PERSONA_SELECTED_PACK_ID) or 0)
            offer = _find_pack_offer(pack_id)
            if not offer:
                await update.message.reply_text("❌ Не удалось найти выбранный фотосет. Откройте «Персона» и выберите заново.")
                return
            context.application.create_task(
                _bot._run_persona_pack_generation(
                    context=context,
                    chat_id=update.effective_chat.id,
                    user_id=user_id,
                    pack_id=pack_id,
                    offer=offer,
                    train_file_ids=list(ids),
                )
            )
            return

    if mode == "persona" and context.user_data.get(USERDATA_PERSONA_WAITING_UPLOAD):
        ids = list(context.user_data.get(USERDATA_PERSONA_PHOTOS, []))
        if len(ids) >= 10:
            await update.message.reply_text("Уже загружено 10 фото. Жду или нажми «Сбросить и начать заново».")
            return
        ids.append(photo.file_id)
        context.user_data[USERDATA_PERSONA_PHOTOS] = ids
        count = len(ids)
        if count < 10:
            text = f"Фото {count}/10 получено. Осталось {10 - count}."
            kb = _persona_upload_keyboard() if count >= 1 else None
            msg = await update.message.reply_text(text, reply_markup=kb)
            upload_msg_ids = list(context.user_data.get(USERDATA_PERSONA_UPLOAD_MSG_IDS, []))
            upload_msg_ids.append(msg.message_id)
            context.user_data[USERDATA_PERSONA_UPLOAD_MSG_IDS] = upload_msg_ids
        else:
            chat_id = update.effective_chat.id
            for msg_id in context.user_data.get(USERDATA_PERSONA_UPLOAD_MSG_IDS, []):
                try:
                    await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
                except Exception as e:
                    logger.warning("Не удалось удалить сообщение загрузки: chat_id=%s msg_id=%s err=%s", chat_id, msg_id, e)
            context.user_data[USERDATA_PERSONA_UPLOAD_MSG_IDS] = []
            context.user_data[USERDATA_PERSONA_WAITING_UPLOAD] = False
            context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "training"
            lora_file_ids = list(ids)
            context.user_data[USERDATA_ASTRIA_LORA_FILE_IDS] = lora_file_ids
            user_id = int(update.effective_user.id) if update.effective_user else 0
            msg = await update.message.reply_text(
                PERSONA_TRAINING_MESSAGE,
                reply_markup=_persona_training_keyboard(),
            )
            if _use_unified_pack_persona_flow() and _bot.store.get_pending_pack_upload(user_id) is not None:
                context.user_data[USERDATA_PERSONA_TRAINING_MSG_ID] = msg.message_id
            context.application.create_task(
                _bot._start_astria_lora(context, update.effective_chat.id, user_id, from_persona=True, file_ids=lora_file_ids)
            )
        return

    # Режим Persona: оплатил, но не нажал «Всё понятно» — напомнить про правила
    if mode == "persona" and not context.user_data.get(USERDATA_PERSONA_WAITING_UPLOAD):
        user_id = int(update.effective_user.id) if update.effective_user else 0
        profile = _bot.store.get_user(user_id)
        credits = getattr(profile, "persona_credits_remaining", 0) or 0
        pending = getattr(profile, "astria_lora_tune_id_pending", None)
        has_pending_paid_photoset = _use_unified_pack_persona_flow() and _bot.store.get_pending_pack_upload(user_id) is not None
        if (credits > 0 or has_pending_paid_photoset) and not profile.astria_lora_tune_id:
            # Есть pending (обучение шло, бот рестартовал) — показать «Проверить статус»
            if pending:
                context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "training"
                await update.message.reply_text(
                    PERSONA_TRAINING_MESSAGE,
                    reply_markup=_persona_training_keyboard(),
                )
                return
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("Да, всё понятно!", callback_data="pl_persona_got_it")],
            ])
            await update.message.reply_text("Правила прочитали? 🫶", reply_markup=kb)
            return

    # Режим Persona (превью): показать редирект в Персону или Экспресс
    if mode == "persona":
        user_id = int(update.effective_user.id) if update.effective_user else 0
        profile = _bot.store.get_user(user_id)
        if profile.astria_lora_tune_id:
            text = (
                "Если хотите создать новую Персону, нажмите <b>«Создать персону»</b> и загрузите <b>10 фото</b>\n\n"
                "Или перейдите в <b>«Экспресс-фото»</b> для быстрой стилизации одного снимка"
            )
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("Создать новую Персону", callback_data="pl_persona_recreate")],
                [InlineKeyboardButton(_express_button_label(profile), callback_data="pl_start_fast")],
                [InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")],
            ])
        else:
            text = (
                "Сначала создайте <b>Персону</b> и следуйте инструкциям\n\n"
                "Или перейдите в раздел <b>Экспресс-фото</b>"
            )
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})],
                [InlineKeyboardButton(_express_button_label(profile), callback_data="pl_start_fast")],
                [InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")],
            ])
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")
        return

    # Режим fast или fallback: есть генерации — обрабатываем как Быстрое фото
    user_id = int(update.effective_user.id) if update.effective_user else 0
    profile = _bot.store.get_user(user_id)
    use_fast = mode == "fast" or _generations_count_fast(profile) > 0
    if use_fast:
        context.user_data[USERDATA_MODE] = "fast"
        gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or profile.subject_gender or "female"
        context.user_data[USERDATA_SUBJECT_GENDER] = gender
        selected_style = context.user_data.pop(USERDATA_FAST_SELECTED_STYLE, None)
        if selected_style:
            if _generations_count_fast(profile) <= 0:
                context.user_data[USERDATA_FAST_SELECTED_STYLE] = selected_style
                chat_id = update.effective_chat.id if update.effective_chat else 0
                await _bot._send_fast_tariffs_two_messages(context.bot, chat_id, context)
                return
            # Стиль уже выбран — сохраняем фото и запускаем генерацию
            gen_lock = await _acquire_user_generation_lock(user_id)
            if gen_lock is None:
                await update.message.reply_text("Пожалуйста, немного подождите. Ещё обрабатываю прошлый запрос")
                return
            style_id = selected_style
            style_label = _fast_style_label(style_id)
            if style_id == "custom":
                prompt = (context.user_data.get(USERDATA_FAST_CUSTOM_PROMPT) or "").strip()
                if not prompt:
                    gen_lock.release()
                    await update.message.reply_text(
                        "Сначала напишите текстом описание картинки (выберите «✏️ Свой запрос» и отправьте сообщение).",
                    )
                    context.user_data[USERDATA_FAST_SELECTED_STYLE] = style_id
                    return
            else:
                prompt = _bot._persona_style_prompt(style_id, style_label)
            context.user_data[USERDATA_PHOTO_FILE_IDS] = [photo.file_id]
            last_msg_id = context.user_data.pop(USERDATA_FAST_LAST_MSG_ID, None)
            chat_id = update.effective_chat.id if update.effective_chat else 0
            if last_msg_id:
                try:
                    await context.bot.delete_message(chat_id=chat_id, message_id=last_msg_id)
                except Exception:
                    pass
            status_msg = await update.message.reply_text("🎨 <i>Создаю изображение...</i>", parse_mode="HTML")

            async def _run_and_release() -> None:
                try:
                    await _bot._run_fast_generation_impl(
                        context=context,
                        chat_id=chat_id,
                        user_id=user_id,
                        style_id=style_id,
                        style_label=style_label,
                        prompt=prompt,
                        photo_file_ids=[photo.file_id],
                        profile=profile,
                        status_msg=status_msg,
                    )
                finally:
                    gen_lock.release()

            context.application.create_task(_run_and_release())
        else:
            # Новое фото — показываем выбор стиля
            context.user_data[USERDATA_PHOTO_FILE_IDS] = [photo.file_id]
            credits = _generations_count_fast(profile)
            credits_word = _fast_credits_word(credits)
            page = context.user_data.get(USERDATA_FAST_STYLE_PAGE, 0)
            text = f"Отлично! <b>Выберите стиль</b> или введите <b>свой запрос</b> 👇\n\n{_format_balance_express(credits)}\n\n<b>1 кредит = 1 фото</b>\n\n{STYLE_EXAMPLES_FOOTER}"
            msg = await update.message.reply_text(
                text,
                reply_markup=_fast_style_choice_keyboard(gender, include_tariffs=True, page=page),
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            chat_id = update.effective_chat.id if update.effective_chat else 0
            await _bot._update_fast_style_message(context, chat_id, msg)
        return

    user_id = int(update.effective_user.id) if update.effective_user else 0
    profile = _bot.store.get_user(user_id)
    await update.message.reply_text(
        "Перед загрузкой фото нужно выбрать раздел: <b>Экспресс-фото</b> или <b>Персона</b>",
        reply_markup=_start_keyboard(profile),
        parse_mode="HTML",
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.document:
        return
    user_id = int(update.effective_user.id) if update.effective_user else 0
    if ALLOWED_USERS and user_id not in ALLOWED_USERS:
        return
    doc = update.message.document
    if not (doc.mime_type or "").startswith("image/"):
        await update.message.reply_text("Пришли картинку (файл должен быть изображением).")
        return
    if doc.file_size is not None and doc.file_size > MAX_IMAGE_SIZE_BYTES:
        await update.message.reply_text(
            f"Слишком большой файл ({doc.file_size // (1024 * 1024)} МБ). Максимум 15 МБ. Сожмите изображение."
        )
        return

    mode = context.user_data.get(USERDATA_MODE) or "normal"
    if mode != "persona_pack_upload" and not _use_unified_pack_persona_flow():
        pending_pack_id = _bot.store.get_pending_pack_upload(user_id)
        if pending_pack_id is not None:
            context.user_data[USERDATA_MODE] = "persona_pack_upload"
            context.user_data[USERDATA_PERSONA_SELECTED_PACK_ID] = pending_pack_id
            context.user_data[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = True
            context.user_data[USERDATA_PERSONA_PACK_PHOTOS] = []
            context.user_data[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = []
            mode = "persona_pack_upload"
    try:
        _bot.store.log_event(user_id, "document_upload", {"mode": mode})
    except Exception:
        pass
    if mode == "persona_pack_upload" and context.user_data.get(USERDATA_PERSONA_PACK_WAITING_UPLOAD):
        ids = list(context.user_data.get(USERDATA_PERSONA_PACK_PHOTOS, []))
        if len(ids) >= 10:
            await update.message.reply_text("Уже загружено 10 фото для фотосета. Нажмите «Сбросить фото фотосета» или дождитесь результата.")
            return
        ids.append(doc.file_id)
        context.user_data[USERDATA_PERSONA_PACK_PHOTOS] = ids
        count = len(ids)
        if count < 10:
            text = f"Фото для фотосета {count}/10 получено. Осталось {10 - count}."
            msg = await update.message.reply_text(text, reply_markup=_persona_pack_upload_keyboard())
            upload_msg_ids = list(context.user_data.get(USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS, []))
            upload_msg_ids.append(msg.message_id)
            context.user_data[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = upload_msg_ids
        else:
            context.user_data[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = False
            context.user_data[USERDATA_MODE] = "persona"
            _bot.store.clear_pending_pack_upload(user_id)
            await update.message.reply_text("Все 10 фото получил ✅\n\nЗапускаю генерацию фотосета…")
            pack_id = int(context.user_data.get(USERDATA_PERSONA_SELECTED_PACK_ID) or 0)
            offer = _find_pack_offer(pack_id)
            if not offer:
                await update.message.reply_text("❌ Не удалось найти выбранный фотосет. Откройте «Персона» и выберите заново.")
                return
            context.application.create_task(
                _bot._run_persona_pack_generation(
                    context=context,
                    chat_id=update.effective_chat.id,
                    user_id=user_id,
                    pack_id=pack_id,
                    offer=offer,
                    train_file_ids=list(ids),
                )
            )
            return

    if mode == "persona" and context.user_data.get(USERDATA_PERSONA_WAITING_UPLOAD):
        ids = list(context.user_data.get(USERDATA_PERSONA_PHOTOS, []))
        if len(ids) >= 10:
            await update.message.reply_text("Уже загружено 10 фото. Жду или нажми «Сбросить и начать заново».")
            return
        ids.append(doc.file_id)
        context.user_data[USERDATA_PERSONA_PHOTOS] = ids
        count = len(ids)
        if count < 10:
            text = f"Фото {count}/10 получено. Осталось {10 - count}."
            kb = _persona_upload_keyboard() if count >= 1 else None
            msg = await update.message.reply_text(text, reply_markup=kb)
            upload_msg_ids = list(context.user_data.get(USERDATA_PERSONA_UPLOAD_MSG_IDS, []))
            upload_msg_ids.append(msg.message_id)
            context.user_data[USERDATA_PERSONA_UPLOAD_MSG_IDS] = upload_msg_ids
        else:
            chat_id = update.effective_chat.id
            for msg_id in context.user_data.get(USERDATA_PERSONA_UPLOAD_MSG_IDS, []):
                try:
                    await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
                except Exception as e:
                    logger.warning("Не удалось удалить сообщение загрузки: chat_id=%s msg_id=%s err=%s", chat_id, msg_id, e)
            context.user_data[USERDATA_PERSONA_UPLOAD_MSG_IDS] = []
            context.user_data[USERDATA_PERSONA_WAITING_UPLOAD] = False
            context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "training"
            lora_file_ids = list(ids)
            context.user_data[USERDATA_ASTRIA_LORA_FILE_IDS] = lora_file_ids
            user_id = int(update.effective_user.id) if update.effective_user else 0
            msg = await update.message.reply_text(
                PERSONA_TRAINING_MESSAGE,
                reply_markup=_persona_training_keyboard(),
            )
            if _use_unified_pack_persona_flow() and _bot.store.get_pending_pack_upload(user_id) is not None:
                context.user_data[USERDATA_PERSONA_TRAINING_MSG_ID] = msg.message_id
            context.application.create_task(
                _bot._start_astria_lora(context, update.effective_chat.id, user_id, from_persona=True, file_ids=lora_file_ids)
            )
        return

    # Режим Persona: оплата фотосета прошла, но пользователь ещё не нажал «Всё понятно»
    if mode == "persona" and not context.user_data.get(USERDATA_PERSONA_WAITING_UPLOAD):
        _user_id = int(update.effective_user.id) if update.effective_user else 0
        _profile = _bot.store.get_user(_user_id)
        has_pending_paid_photoset = _use_unified_pack_persona_flow() and _bot.store.get_pending_pack_upload(_user_id) is not None
        if has_pending_paid_photoset and not getattr(_profile, "astria_lora_tune_id", None):
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("Да, всё понятно!", callback_data="pl_persona_got_it")],
            ])
            await update.message.reply_text("Правила прочитали? 🫶", reply_markup=kb)
            return

    # Режим Persona (превью): показать редирект в Персону или Экспресс
    if mode == "persona":
        _user_id = int(update.effective_user.id) if update.effective_user else 0
        _profile = _bot.store.get_user(_user_id)
        if _profile.astria_lora_tune_id:
            text = (
                "Если хотите создать новую Персону, нажмите <b>«Создать персону»</b> и загрузите <b>10 фото</b>\n\n"
                "Или перейдите в <b>«Экспресс-фото»</b> для быстрой стилизации одного снимка"
            )
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("Создать новую Персону", callback_data="pl_persona_recreate")],
                [InlineKeyboardButton(_express_button_label(_profile), callback_data="pl_start_fast")],
                [InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")],
            ])
        else:
            text = (
                "Сначала создайте <b>Персону</b> и следуйте инструкциям\n\n"
                "Или перейдите в раздел <b>Экспресс-фото</b>"
            )
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})],
                [InlineKeyboardButton(_express_button_label(_profile), callback_data="pl_start_fast")],
                [InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")],
            ])
        await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")
        return

    # Режим fast или fallback: есть генерации — обрабатываем как Быстрое фото
    _user_id = int(update.effective_user.id) if update.effective_user else 0
    _profile = _bot.store.get_user(_user_id)
    _use_fast = mode == "fast" or _generations_count_fast(_profile) > 0
    if _use_fast:
        context.user_data[USERDATA_MODE] = "fast"
        gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or _profile.subject_gender or "female"
        context.user_data[USERDATA_SUBJECT_GENDER] = gender
        selected_style = context.user_data.pop(USERDATA_FAST_SELECTED_STYLE, None)
        if selected_style:
            if _generations_count_fast(_profile) <= 0:
                context.user_data[USERDATA_FAST_SELECTED_STYLE] = selected_style
                _chat_id = update.effective_chat.id if update.effective_chat else 0
                await _bot._send_fast_tariffs_two_messages(context.bot, _chat_id, context)
                return
            gen_lock = await _acquire_user_generation_lock(_user_id)
            if gen_lock is None:
                await update.message.reply_text("Пожалуйста, немного подождите. Ещё обрабатываю прошлый запрос")
                return
            style_id = selected_style
            style_label = _fast_style_label(style_id)
            if style_id == "custom":
                prompt = (context.user_data.get(USERDATA_FAST_CUSTOM_PROMPT) or "").strip()
                if not prompt:
                    gen_lock.release()
                    await update.message.reply_text(
                        "Сначала напишите текстом описание картинки (выберите «✏️ Свой запрос» и отправьте сообщение).",
                    )
                    context.user_data[USERDATA_FAST_SELECTED_STYLE] = style_id
                    return
            else:
                prompt = _bot._persona_style_prompt(style_id, style_label)
            context.user_data[USERDATA_PHOTO_FILE_IDS] = [doc.file_id]
            last_msg_id = context.user_data.pop(USERDATA_FAST_LAST_MSG_ID, None)
            _chat_id = update.effective_chat.id if update.effective_chat else 0
            if last_msg_id:
                try:
                    await context.bot.delete_message(chat_id=_chat_id, message_id=last_msg_id)
                except Exception:
                    pass
            status_msg = await update.message.reply_text("🎨 <i>Создаю изображение...</i>", parse_mode="HTML")

            async def _run_and_release() -> None:
                try:
                    await _bot._run_fast_generation_impl(
                        context=context,
                        chat_id=_chat_id,
                        user_id=_user_id,
                        style_id=style_id,
                        style_label=style_label,
                        prompt=prompt,
                        photo_file_ids=[doc.file_id],
                        profile=_profile,
                        status_msg=status_msg,
                    )
                finally:
                    gen_lock.release()

            context.application.create_task(_run_and_release())
        else:
            context.user_data[USERDATA_PHOTO_FILE_IDS] = [doc.file_id]
            credits = _generations_count_fast(_profile)
            credits_word = _fast_credits_word(credits)
            page = context.user_data.get(USERDATA_FAST_STYLE_PAGE, 0)
            text = f"Отлично! <b>Выберите стиль</b> или введите <b>свой запрос</b> 👇\n\n{_format_balance_express(credits)}\n\n<b>1 кредит = 1 фото</b>\n\n{STYLE_EXAMPLES_FOOTER}"
            msg = await update.message.reply_text(
                text,
                reply_markup=_fast_style_choice_keyboard(gender, include_tariffs=True, page=page),
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            chat_id = update.effective_chat.id if update.effective_chat else 0
            await _bot._update_fast_style_message(context, chat_id, msg)
        return

    user_id = int(update.effective_user.id) if update.effective_user else 0
    profile = _bot.store.get_user(user_id)
    await update.message.reply_text(
        "Перед загрузкой фото нужно выбрать раздел: <b>Экспресс-фото</b> или <b>Персона</b>",
        reply_markup=_start_keyboard(profile),
        parse_mode="HTML",
    )
