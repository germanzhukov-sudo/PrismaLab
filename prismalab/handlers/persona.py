"""Хэндлеры Персоны: оплата, обучение, стили, генерация."""

from __future__ import annotations

import asyncio
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice, Update, WebAppInfo
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from prismalab.config import (
    MINIAPP_URL,
    USERDATA_ASTRIA_LORA_FILE_IDS,
    USERDATA_MODE,
    USERDATA_PERSONA_CREDITS,
    USERDATA_PERSONA_PACK_GIFT_APPLIED,
    USERDATA_PERSONA_PACK_IN_PROGRESS,
    USERDATA_PERSONA_PACK_PHOTOS,
    USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS,
    USERDATA_PERSONA_PACK_WAITING_UPLOAD,
    USERDATA_PERSONA_PHOTOS,
    USERDATA_PERSONA_RECREATING,
    USERDATA_PERSONA_SELECTED_PACK_ID,
    USERDATA_PERSONA_SELECTED_STYLE,
    USERDATA_PERSONA_STYLE_MSG_ID,
    USERDATA_PERSONA_STYLE_PAGE,
    USERDATA_PERSONA_TRAINING_STATUS,
    USERDATA_PERSONA_UPLOAD_MSG_IDS,
    USERDATA_PERSONA_WAITING_UPLOAD,
    USERDATA_SUBJECT_GENDER,
    USERDATA_EXAMPLES_INTRO_MSG_ID,
    USERDATA_EXAMPLES_MEDIA_IDS,
    USERDATA_EXAMPLES_NAV_MSG_ID,
    USER_FRIENDLY_ERROR,
)
from prismalab.keyboards import (
    PERSONA_STYLES_FEMALE,
    PERSONA_STYLES_MALE,
    _persona_app_keyboard,
    _persona_intro_keyboard,
    _persona_pack_upload_keyboard,
    _persona_recreate_confirm_keyboard,
    _persona_rules_keyboard,
    _persona_topup_keyboard,
    _persona_training_keyboard,
    _persona_gender_keyboard,
    _payment_yookassa_keyboard,
    _start_keyboard,
)
from prismalab.messages import (
    PERSONA_CREDITS_OUT_MESSAGE,
    PERSONA_INTRO_MESSAGE,
    PERSONA_PACK_UPLOAD_WAIT_MESSAGE,
    PERSONA_RULES_MESSAGE,
    PERSONA_TRAINING_MESSAGE,
    PERSONA_UPLOAD_WAIT_MESSAGE,
    _fast_credits_word,
    _format_balance_persona,
    PHOTOSET_PROGRESS_ALERT,
    _start_message_text,
)
from prismalab.image_utils import _prepare_image_for_photomaker
from prismalab.telegram_utils import _acquire_user_generation_lock, _safe_get_file_bytes
from prismalab.alerts import alert_payment_error
from prismalab.payment import (
    TELEGRAM_PROVIDER_TOKEN,
    _amount_rub,
    create_payment,
    poll_payment_status,
)
from prismalab.handlers.packs import _start_pending_paid_photoset_after_persona

logger = logging.getLogger("prismalab")

import prismalab.bot as _bot  # noqa: E402


async def _run_persona_batch(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int, styles_list: list) -> None:
    """Запускает батч-генерацию стилей персоны."""
    profile = _bot.store.get_user(user_id)
    has_persona = bool(
        getattr(profile, "astria_lora_tune_id", None)
        or getattr(profile, "astria_lora_pack_tune_id", None)
    )
    if not has_persona:
        await update.message.reply_text("Для генерации нужна Персона. Нажмите /newpersona.")
        return

    credits = profile.persona_credits_remaining
    if credits <= 0:
        await update.message.reply_text("У вас 0 кредитов Персоны. Докупите кредиты в Mini App.")
        return

    batch = styles_list[:credits]
    total = len(batch)

    _bot.store.log_event(user_id, "persona_generate_batch", {
        "styles_count": total,
        "styles": [{"slug": s.get("slug"), "title": s.get("title")} for s in batch],
    })

    gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
    settings = _bot.load_settings()

    async def batch_runner():
        for i, style_data in enumerate(batch, 1):
            slug = style_data.get("slug", "")
            title = style_data.get("title", slug)
            credit_cost = int(style_data.get("credit_cost", 4))
            # Промпт из БД (обогащён API endpoint), фоллбэк на словарь
            prompt = style_data.get("prompt") or _bot._persona_style_prompt(slug, title)

            # Проверяем кредиты перед каждой генерацией (по credit_cost, не по 1)
            current_profile = _bot.store.get_user(user_id)
            if current_profile.persona_credits_remaining < credit_cost:
                await context.bot.send_message(
                    chat_id=user_id,
                    text=f"Кредиты закончились после {i-1}/{total} фото."
                )
                break

            gen_lock = await _acquire_user_generation_lock(user_id)
            if gen_lock is None:
                await asyncio.sleep(5)
                gen_lock = await _acquire_user_generation_lock(user_id)
                if gen_lock is None:
                    await context.bot.send_message(
                        chat_id=user_id,
                        text=f"⏳ Не удалось запустить «{title}» — предыдущая ещё выполняется. Пропускаю."
                    )
                    continue

            status_msg = await context.bot.send_message(
                chat_id=user_id,
                text=f"🎨 <i>Создаю изображение {i}/{total}: {title}...</i>",
                parse_mode="HTML",
            )

            try:
                await _bot._run_style_job(
                    bot=context.bot,
                    chat_id=user_id,
                    photo_file_ids=[],
                    style_id=slug,
                    settings=settings,
                    status_message_id=status_msg.message_id,
                    prompt_strength=0.7,
                    user_id=user_id,
                    subject_gender=gender,
                    use_personal_requested=False,
                    test_prompt=None,
                    lora_prompt_override=prompt,
                    style_title_override=title,
                    is_persona_style=True,
                    context=context,
                    skip_post_message=True,
                    num_images=4,
                    credits_to_spend=credit_cost,
                )
            except Exception as e:
                logger.error("Batch gen error user %s style %s: %s", user_id, slug, e, exc_info=True)
                try:
                    await context.bot.send_message(chat_id=user_id, text=f"Ошибка при «{title}». Продолжаю...")
                except Exception:
                    pass
            finally:
                gen_lock.release()

        final_profile = _bot.store.get_user(user_id)
        remaining = final_profile.persona_credits_remaining if final_profile else 0
        if remaining <= 0:
            # Кредиты закончились — тарифы + Персона + Главное меню
            text = PERSONA_CREDITS_OUT_MESSAGE
            kb_rows = [
                [InlineKeyboardButton("✨ 10 кредитов – 229 руб", callback_data="pl_persona_topup_buy:10")],
                [InlineKeyboardButton("✨ 20 кредитов – 439 руб", callback_data="pl_persona_topup_buy:20")],
                [InlineKeyboardButton("✨ 30 кредитов – 629 руб", callback_data="pl_persona_topup_buy:30")],
            ]
            if MINIAPP_URL:
                kb_rows.append([InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})])
            kb_rows.append([InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")])
            kb = InlineKeyboardMarkup(kb_rows)
        else:
            text = f"<b>Готово!</b>\n\nМожете вернуться в приложение ✨<b>Персона</b> и попробовать новые стили\n\n{_format_balance_persona(remaining)}"
            kb_rows = []
            if MINIAPP_URL:
                kb_rows.append([InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})])
            kb_rows.append([InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")])
            kb = InlineKeyboardMarkup(kb_rows)
        await context.bot.send_message(
            chat_id=user_id,
            text=text,
            reply_markup=kb,
            parse_mode="HTML",
        )

    context.application.create_task(batch_runner())

async def newpersona_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /newpersona: если есть персона — подтверждение пересоздания, иначе — сразу флоу Персоны."""
    user_id = int(update.effective_user.id) if update.effective_user else 0
    _clear_persona_flow_state(context)
    chat_id = update.message.chat_id if update.message else 0
    bot = context.bot
    intro_msg_id = context.user_data.pop(USERDATA_EXAMPLES_INTRO_MSG_ID, None)
    media_ids = context.user_data.pop(USERDATA_EXAMPLES_MEDIA_IDS, None) or []
    nav_msg_id = context.user_data.pop(USERDATA_EXAMPLES_NAV_MSG_ID, None)
    if intro_msg_id is not None or media_ids or nav_msg_id is not None:
        to_delete = []
        if intro_msg_id is not None and intro_msg_id != (update.message.message_id if update.message else None):
            to_delete.append(intro_msg_id)
        to_delete.extend(media_ids)
        if nav_msg_id is not None:
            to_delete.append(nav_msg_id)
        if to_delete:
            await asyncio.gather(*[bot.delete_message(chat_id=chat_id, message_id=mid) for mid in to_delete], return_exceptions=True)
    profile = _bot.store.get_user(user_id)
    if not getattr(profile, "astria_lora_tune_id", None):
        context.user_data[USERDATA_MODE] = "persona"
        known_gender = getattr(profile, "subject_gender", None) or context.user_data.get(USERDATA_SUBJECT_GENDER)
        if known_gender not in ("male", "female"):
            await update.message.reply_text(
                "Выбери пол – так нейросеть точнее настроит результат",
                reply_markup=_persona_gender_keyboard(),
            )
        else:
            context.user_data[USERDATA_SUBJECT_GENDER] = known_gender
            await update.message.reply_text(
                PERSONA_INTRO_MESSAGE,
                reply_markup=_persona_app_keyboard(),
                parse_mode="HTML",
            )
        return
    await update.message.reply_text(
        _bot.PERSONA_RECREATE_CONFIRM_MESSAGE,
        reply_markup=_persona_recreate_confirm_keyboard(),
        parse_mode="HTML",
    )

async def handle_persona_create_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Создать персону» (из профиля): если пол неизвестен — выбор пола, иначе — intro с тарифами."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    _bot._cancel_profile_delete_job(context)
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "persona_create_start")
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    known_gender = getattr(profile, "subject_gender", None) or context.user_data.get(USERDATA_SUBJECT_GENDER)
    if known_gender in ("male", "female"):
        context.user_data[USERDATA_SUBJECT_GENDER] = known_gender
        context.user_data[USERDATA_MODE] = "persona"
        await query.edit_message_text(
            PERSONA_INTRO_MESSAGE,
            reply_markup=_persona_app_keyboard(),
            parse_mode="HTML",
        )
    else:
        await query.edit_message_text(
            "Выбери пол – так нейросеть точнее настроит результат",
            reply_markup=_persona_gender_keyboard(),
        )

async def handle_persona_gender_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """После выбора пола в «Персона»: сохраняем в профиль, показываем intro с тарифами."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    _, gender = query.data.split(":", 1)
    context.user_data[USERDATA_SUBJECT_GENDER] = gender
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "persona_gender_select", {"gender": gender})
    except Exception:
        pass
    _bot.store.set_subject_gender(user_id, gender)
    context.user_data[USERDATA_MODE] = "persona"
    await query.edit_message_text(
        PERSONA_INTRO_MESSAGE,
        reply_markup=_persona_app_keyboard(),
        parse_mode="HTML",
    )

async def handle_persona_buy_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Выбор тарифа Персоны: сразу инвойс/ссылка/симуляция (без экрана «Нажмите Оплатить»)."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    _, count_str = query.data.split(":", 1)
    credits = int(count_str) if count_str in ("5", "20", "40") else 20
    context.user_data[USERDATA_PERSONA_CREDITS] = credits
    user_id = int(query.from_user.id) if query.from_user else 0
    chat_id = query.message.chat_id if query.message else 0
    try:
        _bot.store.log_event(user_id, "persona_buy_init", {"credits": credits})
    except Exception:
        pass

    if _bot.use_yookassa():
        amount = _amount_rub(_bot.store, "persona_create", credits)
        me = await context.bot.get_me()
        return_url = f"https://t.me/{me.username}" if me and me.username else None
        url, payment_id = create_payment(
            amount_rub=amount,
            description=f"Персона: {credits} кредитов",
            metadata={
                "user_id": str(user_id),
                "chat_id": str(chat_id),
                "credits": str(credits),
                "product_type": "persona_create",
            },
            return_url=return_url,
        )
        if url and payment_id:
            await query.edit_message_text(
                f"<b>Создание Персоны + {credits} кредитов за {amount:.0f} ₽</b>\n\nОплатите удобным способом, после оплаты загрузите 10 фото",
                parse_mode="HTML",
                reply_markup=_payment_yookassa_keyboard(url, "pl_persona_back"),
            )
            asyncio.create_task(poll_payment_status(
                payment_id=payment_id,
                bot=context.bot,
                store=_bot.store,
                user_id=user_id,
                chat_id=chat_id,
                credits=credits,
                product_type="persona_create",
                amount_rub=amount,
            ))
            return
        else:
            logger.warning("Ошибка создания платежа (persona_create): %s", payment_id)
            asyncio.create_task(alert_payment_error(user_id, "persona_create", str(payment_id)))
            await query.edit_message_text("❌ Ошибка оплаты. Попробуйте еще раз.")
            return

    if _bot.use_telegram_payments():
        amount = _amount_rub(_bot.store, "persona_create", credits)
        amount_kop = max(8800, int(amount * 100))  # минимум 88 ₽ для Telegram
        payload = f"pl:persona_create:{credits}:{user_id}"
        try:
            await context.bot.send_invoice(
                chat_id=chat_id,
                title="Персона: создание",
                description=f"Персона: {credits} кредитов",
                payload=payload,
                provider_token=TELEGRAM_PROVIDER_TOKEN,
                currency="RUB",
                prices=[LabeledPrice(label="Оплата", amount=amount_kop)],
            )
        except BadRequest as e:
            logger.exception("send_invoice (persona_create) BadRequest: %s", e)
            await query.edit_message_text("❌ Не удалось отправить счёт. Проверьте в BotFather → Payments, что подключена платёжная система.")
        return

    _bot.store.set_persona_credits(user_id, credits)
    if context.user_data.pop(USERDATA_PERSONA_RECREATING, None):
        _bot.store.set_astria_lora_tune(user_id=user_id, tune_id=None)
    context.user_data[USERDATA_MODE] = "persona"
    await query.edit_message_text(
        PERSONA_RULES_MESSAGE,
        reply_markup=_persona_rules_keyboard(),
        parse_mode="HTML",
    )

async def handle_persona_back_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Назад» из экранов Персоны (тарифы, выбор пола): возврат на вводный экран Персоны."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "nav_back_persona")
    except Exception:
        pass
    context.user_data[USERDATA_MODE] = "persona"
    context.user_data.pop(USERDATA_PERSONA_WAITING_UPLOAD, None)
    context.user_data.pop(USERDATA_PERSONA_PHOTOS, None)
    context.user_data.pop(USERDATA_PERSONA_CREDITS, None)
    await query.edit_message_text(
        PERSONA_INTRO_MESSAGE,
        reply_markup=_persona_app_keyboard(),
        parse_mode="HTML",
    )

async def handle_persona_show_credits_out_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать экран «кредиты закончились» (Назад из докупки или подтверждения)."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "persona_credits_out")
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    text, kb = _bot._persona_credits_out_content(profile)
    await query.edit_message_text(text, reply_markup=kb, parse_mode="HTML")

async def handle_persona_topup_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Докупить кредиты»: показываем тарифы 10/229, 20/439, 30/629."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "persona_topup_view")
    except Exception:
        pass
    await query.edit_message_text(
        _bot.PERSONA_TOPUP_MESSAGE,
        reply_markup=_persona_topup_keyboard(),
    )

async def handle_persona_topup_buy_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Выбор пакета докупки: сразу инвойс/ссылка/симуляция (без экрана «Нажмите Оплатить»)."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    _, count_str = query.data.split(":", 1)
    credits = int(count_str) if count_str in ("10", "20", "30") else 10
    context.user_data[USERDATA_PERSONA_CREDITS] = credits
    user_id = int(query.from_user.id) if query.from_user else 0
    chat_id = query.message.chat_id if query.message else 0
    try:
        _bot.store.log_event(user_id, "persona_topup_init", {"credits": credits})
    except Exception:
        pass

    if _bot.use_yookassa():
        amount = _amount_rub(_bot.store, "persona_topup", credits)
        me = await context.bot.get_me()
        return_url = f"https://t.me/{me.username}" if me and me.username else None
        url, payment_id = create_payment(
            amount_rub=amount,
            description=f"Докупка Персона: {credits} кредитов",
            metadata={
                "user_id": str(user_id),
                "chat_id": str(chat_id),
                "credits": str(credits),
                "product_type": "persona_topup",
            },
            return_url=return_url,
        )
        if url and payment_id:
            await query.edit_message_text(
                f"<b>{credits} кредитов Персона за {amount:.0f} ₽</b>\n\nОплатите удобным способом, кредиты зачислим моментально",
                parse_mode="HTML",
                reply_markup=_payment_yookassa_keyboard(url, "pl_persona_show_credits_out"),
            )
            asyncio.create_task(poll_payment_status(
                payment_id=payment_id,
                bot=context.bot,
                store=_bot.store,
                user_id=user_id,
                chat_id=chat_id,
                credits=credits,
                product_type="persona_topup",
                amount_rub=amount,
            ))
            return
        else:
            logger.warning("Ошибка создания платежа (persona_topup): %s", payment_id)
            asyncio.create_task(alert_payment_error(user_id, "persona_topup", str(payment_id)))
            await query.edit_message_text("❌ Ошибка оплаты. Попробуйте еще раз.")
            return

    if _bot.use_telegram_payments():
        amount = _amount_rub(_bot.store, "persona_topup", credits)
        amount_kop = max(8800, int(amount * 100))  # минимум 88 ₽ для Telegram
        payload = f"pl:persona_topup:{credits}:{user_id}"
        try:
            await context.bot.send_invoice(
                chat_id=chat_id,
                title="Докупка кредитов Персоны",
                description=f"{credits} кредитов Персоны",
                payload=payload,
                provider_token=TELEGRAM_PROVIDER_TOKEN,
                currency="RUB",
                prices=[LabeledPrice(label="Оплата", amount=amount_kop)],
            )
        except BadRequest as e:
            logger.exception("send_invoice (persona_topup) BadRequest: %s", e)
            await query.edit_message_text("❌ Не удалось отправить счёт. Проверьте в BotFather → Payments, что подключена платёжная система.")
        return

    profile = _bot.store.get_user(user_id)
    new_total = profile.persona_credits_remaining + credits
    _bot.store.set_persona_credits(user_id, new_total)
    profile = _bot.store.get_user(user_id)
    gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"

    selected = context.user_data.pop(USERDATA_PERSONA_SELECTED_STYLE, None)
    gen_lock = await _acquire_user_generation_lock(user_id) if selected else None
    if selected and gen_lock is None:
        selected = None

    if selected and gen_lock is not None:
        style_id, label = selected
        prompt = _bot._persona_style_prompt(style_id, label)
        status_text = f"<b>Оплата получена</b> ✅\n\n{_format_balance_persona(new_total)}\n\nВыбран стиль: «{label}»\n\n🎨 <i>Создаю изображение...</i>"
        await query.edit_message_text(status_text, parse_mode="HTML")
        settings = _bot.load_settings()

        async def runner() -> None:
            try:
                await _bot._run_style_job(
                    bot=context.bot,
                    chat_id=query.message.chat_id,
                    photo_file_ids=[],
                    style_id=style_id,
                    settings=settings,
                    status_message_id=query.message.message_id,
                    prompt_strength=0.7,
                    user_id=user_id,
                    subject_gender=gender,
                    use_personal_requested=False,
                    test_prompt=None,
                    lora_prompt_override=prompt,
                    style_title_override=label,
                    is_persona_style=True,
                    context=context,
                )
            finally:
                gen_lock.release()

        context.application.create_task(runner())
    else:
        text = f"<b>Оплата получена</b> ✅\n\nВыберите образ в приложении <b>Персона</b> 👇\n\n{_format_balance_persona(new_total)}"
        kb_rows = []
        if MINIAPP_URL:
            kb_rows.append([InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})])
        kb_rows.append([InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")])
        await query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup(kb_rows),
            parse_mode="HTML",
        )
    context.user_data[USERDATA_PERSONA_STYLE_PAGE] = 0

async def handle_persona_topup_confirm_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Оплата докупки: Telegram Payments (инвойс) или ЮKassa (ссылка) или симуляция."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    _, count_str = query.data.split(":", 1)
    credits = int(count_str) if count_str in ("10", "20", "30") else 10
    user_id = int(query.from_user.id) if query.from_user else 0
    chat_id = query.message.chat_id if query.message else 0
    try:
        _bot.store.log_event(user_id, "persona_topup_confirm", {"credits": credits})
    except Exception:
        pass

    if _bot.use_yookassa():
        amount = _amount_rub(_bot.store, "persona_topup", credits)
        me = await context.bot.get_me()
        return_url = f"https://t.me/{me.username}" if me and me.username else None
        url, payment_id = create_payment(
            amount_rub=amount,
            description=f"Докупка Персона: {credits} кредитов",
            metadata={
                "user_id": str(user_id),
                "chat_id": str(chat_id),
                "credits": str(credits),
                "product_type": "persona_topup",
            },
            return_url=return_url,
        )
        if url and payment_id:
            await query.edit_message_text(
                f"<b>{credits} кредитов Персона за {amount:.0f} ₽</b>\n\nОплатите удобным способом, кредиты зачислим моментально",
                parse_mode="HTML",
                reply_markup=_payment_yookassa_keyboard(url, "pl_persona_show_credits_out"),
            )
            asyncio.create_task(poll_payment_status(
                payment_id=payment_id,
                bot=context.bot,
                store=_bot.store,
                user_id=user_id,
                chat_id=chat_id,
                credits=credits,
                product_type="persona_topup",
                amount_rub=amount,
            ))
            return
        else:
            logger.warning("Ошибка создания платежа (persona_topup): %s", payment_id)
            asyncio.create_task(alert_payment_error(user_id, "persona_topup", str(payment_id)))
            await query.edit_message_text("❌ Ошибка оплаты. Попробуйте еще раз.")
            return

    if _bot.use_telegram_payments():
        amount = _amount_rub(_bot.store, "persona_topup", credits)
        amount_kop = max(8800, int(amount * 100))  # минимум 88 ₽ для Telegram
        payload = f"pl:persona_topup:{credits}:{user_id}"
        try:
            await context.bot.send_invoice(
                chat_id=chat_id,
                title="Докупка кредитов Персоны",
                description=f"{credits} кредитов Персоны",
                payload=payload,
                provider_token=TELEGRAM_PROVIDER_TOKEN,
                currency="RUB",
                prices=[LabeledPrice(label="Оплата", amount=amount_kop)],
            )
        except BadRequest as e:
            logger.exception("send_invoice (persona_topup) BadRequest: %s", e)
            await query.edit_message_text("❌ Не удалось отправить счёт. Проверьте в BotFather → Payments, что подключена платёжная система.")
        return

    profile = _bot.store.get_user(user_id)
    new_total = profile.persona_credits_remaining + credits
    _bot.store.set_persona_credits(user_id, new_total)
    profile = _bot.store.get_user(user_id)
    gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"

    selected = context.user_data.pop(USERDATA_PERSONA_SELECTED_STYLE, None)
    gen_lock = await _acquire_user_generation_lock(user_id) if selected else None
    if selected and gen_lock is None:
        selected = None  # не запускаем, если уже идёт генерация

    if selected and gen_lock is not None:
        style_id, label = selected
        prompt = _bot._persona_style_prompt(style_id, label)
        status_text = f"<b>Оплата получена</b> ✅\n\n{_format_balance_persona(new_total)}\n\nВыбран стиль: «{label}»\n\n🎨 <i>Создаю изображение...</i>"
        await query.edit_message_text(status_text, parse_mode="HTML")
        settings = _bot.load_settings()

        async def runner() -> None:
            try:
                await _bot._run_style_job(
                    bot=context.bot,
                    chat_id=query.message.chat_id,
                    photo_file_ids=[],
                    style_id=style_id,
                    settings=settings,
                    status_message_id=query.message.message_id,
                    prompt_strength=0.7,
                    user_id=user_id,
                    subject_gender=gender,
                    use_personal_requested=False,
                    test_prompt=None,
                    lora_prompt_override=prompt,
                    style_title_override=label,
                    is_persona_style=True,
                    context=context,
                )
            finally:
                gen_lock.release()

        context.application.create_task(runner())
    else:
        text = f"<b>Оплата получена</b> ✅\n\nВыберите образ в приложении <b>Персона</b> 👇\n\n{_format_balance_persona(new_total)}"
        kb_rows = []
        if MINIAPP_URL:
            kb_rows.append([InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})])
        kb_rows.append([InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")])
        await query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup(kb_rows),
            parse_mode="HTML",
        )
    context.user_data[USERDATA_PERSONA_STYLE_PAGE] = 0

async def handle_persona_recreate_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Создать новую Персону»: показываем подтверждение."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "persona_recreate_start")
    except Exception:
        pass
    chat_id = query.message.chat_id if query.message else 0
    bot = context.bot
    intro_msg_id = context.user_data.pop(USERDATA_EXAMPLES_INTRO_MSG_ID, None)
    media_ids = context.user_data.pop(USERDATA_EXAMPLES_MEDIA_IDS, None) or []
    nav_msg_id = context.user_data.pop(USERDATA_EXAMPLES_NAV_MSG_ID, None)
    if intro_msg_id is not None or media_ids or nav_msg_id is not None:
        to_delete = []
        if intro_msg_id is not None and intro_msg_id != query.message.message_id:
            to_delete.append(intro_msg_id)
        to_delete.extend(media_ids)
        if nav_msg_id is not None:
            to_delete.append(nav_msg_id)
        if to_delete:
            await asyncio.gather(*[bot.delete_message(chat_id=chat_id, message_id=mid) for mid in to_delete], return_exceptions=True)
    await query.edit_message_text(
        _bot.PERSONA_RECREATE_CONFIRM_MESSAGE,
        reply_markup=_persona_recreate_confirm_keyboard(),
        parse_mode="HTML",
    )

async def handle_persona_recreate_cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """«Главное меню» из подтверждения пересоздания: сбрасываем флоу, возвращаем в главное меню."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    _clear_persona_flow_state(context)
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "persona_recreate_cancel")
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    await query.edit_message_text(
        _start_message_text(profile),
        reply_markup=_start_keyboard(profile),
        parse_mode="HTML",
    )

def _clear_persona_flow_state(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сброс состояния флоу Персоны при выходе/отмене."""
    for key in (
        USERDATA_MODE,
        USERDATA_PERSONA_WAITING_UPLOAD,
        USERDATA_PERSONA_PHOTOS,
        USERDATA_PERSONA_CREDITS,
        USERDATA_PERSONA_TRAINING_STATUS,
        USERDATA_PERSONA_UPLOAD_MSG_IDS,
        USERDATA_PERSONA_STYLE_MSG_ID,
        USERDATA_PERSONA_STYLE_PAGE,
        USERDATA_PERSONA_SELECTED_STYLE,
        USERDATA_PERSONA_SELECTED_PACK_ID,
        USERDATA_PERSONA_PACK_WAITING_UPLOAD,
        USERDATA_PERSONA_PACK_PHOTOS,
        USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS,
        USERDATA_PERSONA_PACK_IN_PROGRESS,
        USERDATA_PERSONA_PACK_GIFT_APPLIED,
        USERDATA_PERSONA_RECREATING,
    ):
        context.user_data.pop(key, None)

async def handle_persona_recreate_confirm_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Подтверждение: НЕ удаляем модель, переходим к тарифам. Удаление только при оплате."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "persona_recreate_confirm")
    except Exception:
        pass
    context.user_data[USERDATA_PERSONA_RECREATING] = True
    profile = _bot.store.get_user(user_id)
    known_gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
    context.user_data[USERDATA_SUBJECT_GENDER] = known_gender
    context.user_data[USERDATA_MODE] = "persona"
    await query.edit_message_text(
        _bot.PERSONA_RECREATE_TARIFF_MESSAGE,
        reply_markup=_persona_intro_keyboard(),
        parse_mode="HTML",
    )

async def handle_persona_confirm_pay_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Оплатить»: Telegram Payments (инвойс) или ЮKassa (ссылка) или симуляция."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    _, count_str = query.data.split(":", 1)
    credits = int(count_str) if count_str in ("5", "20", "40") else 20
    context.user_data[USERDATA_PERSONA_CREDITS] = credits
    user_id = int(query.from_user.id) if query.from_user else 0
    chat_id = query.message.chat_id if query.message else 0

    if _bot.use_yookassa():
        amount = _amount_rub(_bot.store, "persona_create", credits)
        me = await context.bot.get_me()
        return_url = f"https://t.me/{me.username}" if me and me.username else None
        url, payment_id = create_payment(
            amount_rub=amount,
            description=f"Персона: {credits} кредитов",
            metadata={
                "user_id": str(user_id),
                "chat_id": str(chat_id),
                "credits": str(credits),
                "product_type": "persona_create",
            },
            return_url=return_url,
        )
        if url and payment_id:
            await query.edit_message_text(
                f"<b>Создание Персоны + {credits} кредитов за {amount:.0f} ₽</b>\n\nОплатите удобным способом, после оплаты загрузите 10 фото",
                parse_mode="HTML",
                reply_markup=_payment_yookassa_keyboard(url, "pl_persona_back"),
            )
            asyncio.create_task(poll_payment_status(
                payment_id=payment_id,
                bot=context.bot,
                store=_bot.store,
                user_id=user_id,
                chat_id=chat_id,
                credits=credits,
                product_type="persona_create",
                amount_rub=amount,
            ))
            return
        else:
            logger.warning("Ошибка создания платежа (persona_create): %s", payment_id)
            asyncio.create_task(alert_payment_error(user_id, "persona_create", str(payment_id)))
            await query.edit_message_text("❌ Ошибка оплаты. Попробуйте еще раз.")
            return

    if _bot.use_telegram_payments():
        amount = _amount_rub(_bot.store, "persona_create", credits)
        amount_kop = max(8800, int(amount * 100))  # минимум 88 ₽ для Telegram
        payload = f"pl:persona_create:{credits}:{user_id}"
        try:
            await context.bot.send_invoice(
                chat_id=chat_id,
                title="Персона: создание",
                description=f"Персона: {credits} кредитов",
                payload=payload,
                provider_token=TELEGRAM_PROVIDER_TOKEN,
                currency="RUB",
                prices=[LabeledPrice(label="Оплата", amount=amount_kop)],
            )
        except BadRequest as e:
            logger.exception("send_invoice (persona_create) BadRequest: %s", e)
            await query.edit_message_text("❌ Не удалось отправить счёт. Проверьте в BotFather → Payments, что подключена платёжная система.")
        return

    _bot.store.set_persona_credits(user_id, credits)
    if context.user_data.pop(USERDATA_PERSONA_RECREATING, None):
        _bot.store.set_astria_lora_tune(user_id=user_id, tune_id=None)
    context.user_data[USERDATA_MODE] = "persona"
    await query.edit_message_text(
        PERSONA_RULES_MESSAGE,
        reply_markup=_persona_rules_keyboard(),
        parse_mode="HTML",
    )

async def handle_persona_got_it_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Всё понятно, погнали!»: переходим к загрузке 10 фото."""
    query = update.callback_query
    if not query:
        return

    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "persona_upload_start")
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    training_status = context.user_data.get(USERDATA_PERSONA_TRAINING_STATUS)
    photos_count = len(list(context.user_data.get(USERDATA_PERSONA_PHOTOS, [])))
    pending_tune_id = getattr(profile, "astria_lora_tune_id_pending", None)
    lora_tune_id = getattr(profile, "astria_lora_tune_id", None)
    pack_tune_pending = getattr(profile, "astria_lora_pack_tune_id_pending", None)
    pack_in_progress = bool(context.user_data.get(USERDATA_PERSONA_PACK_IN_PROGRESS))
    message_id = query.message.message_id if query.message else None

    logger.info(
        "persona_got_it: user_id=%s message_id=%s training_status=%s photos_count=%s pending_tune_id=%s lora_tune_id=%s pack_tune_pending=%s pack_in_progress=%s",
        user_id,
        message_id,
        training_status,
        photos_count,
        bool(pending_tune_id),
        bool(lora_tune_id),
        bool(pack_tune_pending),
        pack_in_progress,
    )

    # Guard от повторного callback: если обучение/пак уже запущены (или уже набрали 10 фото),
    # не возвращаем пользователя на этап повторной загрузки.
    if (
        pending_tune_id
        or lora_tune_id
        or training_status == "training"
        or photos_count >= 10
        or pack_tune_pending
        or pack_in_progress
    ):
        await query.answer("Обработка уже запущена, проверьте статус.", show_alert=False)
        context.user_data[USERDATA_MODE] = "persona"
        context.user_data[USERDATA_PERSONA_WAITING_UPLOAD] = False
        try:
            await query.edit_message_text(
                PERSONA_TRAINING_MESSAGE,
                reply_markup=_persona_training_keyboard(),
                parse_mode="HTML",
            )
        except Exception:
            await context.bot.send_message(
                chat_id=query.message.chat_id if query.message else user_id,
                text=PERSONA_TRAINING_MESSAGE,
                reply_markup=_persona_training_keyboard(),
                parse_mode="HTML",
            )
        return

    await query.answer()
    context.user_data[USERDATA_PERSONA_WAITING_UPLOAD] = True
    context.user_data[USERDATA_PERSONA_PHOTOS] = []
    context.user_data[USERDATA_MODE] = "persona"
    await query.edit_message_text(PERSONA_UPLOAD_WAIT_MESSAGE, parse_mode="HTML")

async def handle_persona_reset_photos_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сброс загруженных фото Персоны."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "persona_reset_photos")
    except Exception:
        pass
    context.user_data[USERDATA_PERSONA_PHOTOS] = []
    context.user_data[USERDATA_PERSONA_WAITING_UPLOAD] = True
    context.user_data[USERDATA_PERSONA_UPLOAD_MSG_IDS] = []
    await query.edit_message_text(PERSONA_UPLOAD_WAIT_MESSAGE, parse_mode="HTML")

async def handle_miniapp_pack_upload_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка из Mini App после оплаты пака: начинаем загрузку фото."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    # Формат: pl_pack_upload:{pack_id}:{credits}
    parts = query.data.split(":")
    pack_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    credits = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 20
    try:
        user_id_miniapp = int(query.from_user.id) if query.from_user else 0
        _bot.store.log_event(user_id_miniapp, "miniapp_pack_upload", {"pack_id": pack_id})
    except Exception:
        pass

    context.user_data[USERDATA_PERSONA_SELECTED_PACK_ID] = pack_id
    context.user_data[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = True
    context.user_data[USERDATA_PERSONA_PACK_PHOTOS] = []
    context.user_data[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = []
    context.user_data[USERDATA_MODE] = "persona_pack_upload"

    await query.edit_message_text(
        PERSONA_PACK_UPLOAD_WAIT_MESSAGE,
        reply_markup=_persona_pack_upload_keyboard(),
        parse_mode="HTML",
    )

async def handle_persona_pack_reset_photos_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сброс загруженных фото для pack-run в dev-режиме."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "pack_reset_photos")
    except Exception:
        pass
    context.user_data[USERDATA_PERSONA_PACK_PHOTOS] = []
    context.user_data[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = True
    context.user_data[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = []
    context.user_data[USERDATA_MODE] = "persona_pack_upload"
    await query.edit_message_text(
        PERSONA_PACK_UPLOAD_WAIT_MESSAGE,
        reply_markup=_persona_pack_upload_keyboard(),
        parse_mode="HTML",
    )

async def handle_persona_check_status_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Проверить статус»: проверяет обучение Персоны и фоновые этапы фотосета."""
    query = update.callback_query
    if not query:
        return
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "persona_check_status")
    except Exception:
        pass

    async def _safe_answer(text: str = "", show_alert: bool = False):
        try:
            await query.answer(text, show_alert=show_alert)
        except Exception:
            pass

    async def _safe_edit(text: str, reply_markup=None):
        try:
            await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
        except Exception:
            pass

    profile = _bot.store.get_user(user_id)
    pending_tune_id = getattr(profile, "astria_lora_tune_id_pending", None)

    # Восстановление: есть pending tune (бот рестартовал во время обучения)
    if pending_tune_id:
        try:
            from prismalab.astria_client import _get_tune, _timeout_s
            settings = _bot.load_settings()
            last = await asyncio.to_thread(
                _get_tune,
                api_key=settings.astria_api_key,
                tune_id=pending_tune_id,
                timeout_s=_timeout_s(30.0),
            )
            status = str(last.get("status") or last.get("state") or "").lower()
            trained_at = last.get("trained_at")
            if status in {"completed", "succeeded", "ready", "trained", "finished"} or trained_at:
                _bot.store.set_astria_lora_tune(user_id=user_id, tune_id=pending_tune_id, class_name=getattr(profile, "persona_lora_class_name", None) or "person")
                context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "done"
                if await _start_pending_paid_photoset_after_persona(
                    context=context,
                    chat_id=query.message.chat_id if query.message else user_id,
                    user_id=user_id,
                ):
                    await _safe_answer("Персональная модель готова. Запускаю фотосет.")
                    return
                credits = profile.persona_credits_remaining
                text = f"Готово! 🎉 Персональная модель обучена\n\nВыберите образ в приложении <b>Персона</b> – у вас {credits} {_fast_credits_word(credits)}"
                await _safe_answer("Готово! 🎉")
                await _safe_edit(text, reply_markup=_persona_app_keyboard())
                return
            if status in {"failed", "error", "cancelled"}:
                _bot.store.clear_astria_lora_tune_pending(user_id)
                context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "error"
                await _safe_answer("При обучении возникла ошибка.", show_alert=True)
                await _safe_edit("При обучении возникла ошибка. Загрузите 10 фото заново или напишите в поддержку.")
                return
        except Exception as e:
            logger.warning("Ошибка проверки pending tune %s: %s", pending_tune_id, e)

    if context.user_data.get(USERDATA_PERSONA_PACK_IN_PROGRESS) or getattr(profile, "astria_lora_pack_tune_id_pending", None):
        await _safe_answer(PHOTOSET_PROGRESS_ALERT, show_alert=True)
        return

    status = context.user_data.get(USERDATA_PERSONA_TRAINING_STATUS) or "training"
    if status == "training":
        await _safe_answer(
            "Модель ещё обучается ⏳ Обычно это занимает около 10 минут. Напишу, когда будет готово.",
            show_alert=True,
        )
    elif status == "error":
        await _safe_answer(
            "При обучении возникла ошибка. Напиши нам в поддержку — разберёмся.",
            show_alert=True,
        )
    else:
        await _safe_answer("Модель готова! Скоро появится возможность генерировать.", show_alert=True)

async def handle_persona_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Переключение страницы стилей Персоны (← Пред / След →)."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    _, page_str = query.data.split(":", 1)
    if page_str == "noop":
        return
    try:
        page = int(page_str)
    except ValueError:
        return
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "persona_page", {"page": page})
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    credits = profile.persona_credits_remaining
    gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
    text = f"Выберите образ в приложении <b>Персона</b> 👇\n\n{_format_balance_persona(credits)}"
    await query.edit_message_text(
        text,
        reply_markup=_persona_app_keyboard(),
        parse_mode="HTML",
    )
    context.user_data[USERDATA_PERSONA_STYLE_MSG_ID] = query.message.message_id
    context.user_data[USERDATA_PERSONA_STYLE_PAGE] = page

async def handle_persona_style_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Выбор стиля в Персоне: генерация через Astria LoRA, списание кредита."""
    query = update.callback_query
    if not query or not query.data or "pl_persona_style:" not in query.data:
        return
    _, style_id = query.data.split(":", 1)
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "persona_style_select", {"style_id": style_id})
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    credits = profile.persona_credits_remaining
    if credits <= 0:
        await query.answer()
        gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
        label = next((l for l, s in (PERSONA_STYLES_FEMALE if gender == "female" else PERSONA_STYLES_MALE) if s == style_id), style_id)
        context.user_data[USERDATA_PERSONA_SELECTED_STYLE] = (style_id, label)
        text, kb = _bot._persona_credits_out_content(profile)
        await query.edit_message_text(text, reply_markup=kb, parse_mode="HTML")
        return
    gen_lock = await _acquire_user_generation_lock(user_id)
    if gen_lock is None:
        await query.answer("Пожалуйста, немного подождите. Ещё обрабатываю прошлый запрос", show_alert=True)
        return
    await query.answer()
    context.user_data.pop(USERDATA_PERSONA_STYLE_MSG_ID, None)
    try:
        await context.bot.delete_message(chat_id=query.message.chat_id, message_id=query.message.message_id)
    except Exception:
        pass
    gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
    label = next((l for l, s in (PERSONA_STYLES_FEMALE if gender == "female" else PERSONA_STYLES_MALE) if s == style_id), style_id)
    prompt = _bot._persona_style_prompt(style_id, label)
    status_msg = await context.bot.send_message(chat_id=query.message.chat_id, text="🎨 <i>Создаю изображение...</i>", parse_mode="HTML")
    settings = _bot.load_settings()

    async def runner() -> None:
        try:
            await _bot._run_style_job(
                bot=context.bot,
                chat_id=query.message.chat_id,
                photo_file_ids=[],
                style_id=style_id,
                settings=settings,
                status_message_id=status_msg.message_id,
                prompt_strength=0.7,
                user_id=user_id,
                subject_gender=gender,
                use_personal_requested=False,
                test_prompt=None,
                lora_prompt_override=prompt,
                style_title_override=label,
                is_persona_style=True,
                context=context,
            )
        finally:
            gen_lock.release()

    context.application.create_task(runner())

async def _start_astria_lora(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int, from_persona: bool = False,
    file_ids: list | None = None,
) -> None:
    """Создаёт LoRA tune из 10 фото через Astria API. from_persona=True — поток Персоны."""
    logger.info(f"[LoRA] ========== НАЧАЛО создания LoRA для user {user_id} ==========")
    gen_lock = await _acquire_user_generation_lock(user_id)
    if gen_lock is None:
        logger.warning(f"[LoRA] ❌ Lock уже занят для user {user_id}, пропускаю")
        return

    try:
        settings = _bot.load_settings()
        logger.info("[LoRA] Настройки загружены, проверяю API ключ...")
        if not settings.astria_api_key:
            logger.error("[LoRA] ❌ Нет API ключа Astria")
            await context.bot.send_message(chat_id=chat_id, text=USER_FRIENDLY_ERROR)
            return
        logger.info(f"[LoRA] API ключ найден (длина: {len(settings.astria_api_key)})")

        if file_ids is None:
            file_ids = list(context.user_data.get(USERDATA_ASTRIA_LORA_FILE_IDS, []))
            logger.info(f"[LoRA] file_ids из user_data: {len(file_ids)}")
        else:
            logger.info(f"[LoRA] file_ids переданы явно: {len(file_ids)}")
        if len(file_ids) < 10:
            logger.warning(f"[LoRA] ❌ Недостаточно фото: {len(file_ids)}/10, user_id={user_id}")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Нужно 10 фото. Сейчас {len(file_ids)}/10. Отправь ещё {10 - len(file_ids)} фото."
            )
            return

        # Имя класса: woman/man (при PERSONA_LORA_NAME_MODE=gender) или person (default).
        from prismalab.config import persona_lora_name
        profile = _bot.store.get_user(user_id)
        gender = getattr(profile, "subject_gender", None) or context.user_data.get(USERDATA_SUBJECT_GENDER)
        name = persona_lora_name(gender, user_id=user_id)

        # Скачиваем все 10 фото (с обработкой таймаутов)
        image_bytes_list = []
        if not from_persona:
            await context.bot.send_message(chat_id=chat_id, text="Скачиваю 10 фото…")
        for idx, fid in enumerate(file_ids, 1):
            try:
                image_bytes = await _safe_get_file_bytes(context.bot, fid)
                # Подготавливаем фото
                image_bytes = _prepare_image_for_photomaker(image_bytes)
                image_bytes_list.append(image_bytes)
                if not from_persona and idx % 3 == 0:  # Обновляем статус каждые 3 фото
                    await context.bot.send_message(chat_id=chat_id, text=f"Скачано {idx}/10 фото…")
            except Exception as e:
                logger.error(f"Ошибка при скачивании фото {idx}/10: {e}")
                await context.bot.send_message(chat_id=chat_id, text=USER_FRIENDLY_ERROR)
                raise

        if not from_persona:
            await context.bot.send_message(chat_id=chat_id, text="Создаю LoRA tune…\n\nЭто может занять 10–60 минут (иногда до 2 часов при загрузке). Я напишу, когда будет готово.")

        logger.info("[LoRA] Начинаю создание LoRA tune через Astria API...")
        from prismalab.astria_client import create_lora_tune_and_wait

        def _on_lora_created(tid: str) -> None:
            _bot.store.set_astria_lora_tune_pending(user_id=user_id, tune_id=tid)
            _bot.store.set_persona_lora_class_name(user_id=user_id, class_name=name)

        logger.info(f"[LoRA] Вызываю create_lora_tune_and_wait с параметрами: name={name}, base_tune_id=1504944, preset=flux-lora-portrait")
        result = await create_lora_tune_and_wait(
            api_key=settings.astria_api_key,
            name=name,
            title=f"LoRA user {user_id}",
            image_bytes_list=image_bytes_list,
            base_tune_id="1504944",  # Flux1.dev из галереи (проверенная конфигурация для LoRA)
            preset="flux-lora-portrait",  # Рекомендуется для людей
            on_created=_on_lora_created,
            max_seconds=7200,  # До 2 часов на training (увеличено с 1 часа)
            poll_seconds=15.0,
        )

        # Проверяем, что модель действительно создана как LoRA
        model_type = result.raw.get("model_type") or "unknown"
        if model_type.lower() != "lora":
            logger.error(f"⚠️ ВНИМАНИЕ: Astria создал tune {result.tune_id} как '{model_type}', а не как 'lora'!")
            await context.bot.send_message(chat_id=chat_id, text=USER_FRIENDLY_ERROR)
            return

        _bot.store.set_astria_lora_tune(user_id=user_id, tune_id=result.tune_id, class_name=name)
        context.user_data.pop(USERDATA_ASTRIA_LORA_FILE_IDS, None)

        if from_persona:
            context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "done"
            gen_lock.release()
            gen_lock = None
            if await _start_pending_paid_photoset_after_persona(
                context=context,
                chat_id=chat_id,
                user_id=user_id,
            ):
                return
            profile = _bot.store.get_user(user_id)
            credits = profile.persona_credits_remaining
            text = f"Готово! 🎉 Персональная модель обучена\n\nВыберите образ в приложении <b>Персона</b> – у вас {credits} {_fast_credits_word(credits)}"
            await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=_persona_app_keyboard(),
                parse_mode="HTML",
            )
        else:
            context.user_data[USERDATA_MODE] = "normal"
            logger.info(f"✅ LoRA {result.tune_id} успешно создана и сохранена (model_type='{model_type}')")
            profile = _bot.store.get_user(user_id)
            await context.bot.send_message(
                chat_id=chat_id,
                text="✅ Готово! LoRA модель создана на Flux1.dev.\n"
                f"ID модели: {result.tune_id}\n"
                "Теперь нажми «Персона» — я буду генерировать сцены с высоким качеством.",
                reply_markup=_start_keyboard(profile),
            )
    except _bot.AstriaError as e:
        if from_persona:
            context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "error"
        _bot.store.clear_astria_lora_tune_pending(user_id)
        logger.error("Astria LoRA error: %s", e, exc_info=True)
        msg = "Что-то пошло не так, персона не создалась. Загрузите фото заново или напишите в поддержку." if from_persona else USER_FRIENDLY_ERROR
        await context.bot.send_message(chat_id=chat_id, text=msg)
    except Exception as e:
        if from_persona:
            context.user_data[USERDATA_PERSONA_TRAINING_STATUS] = "error"
        _bot.store.clear_astria_lora_tune_pending(user_id)
        logger.error("Astria LoRA error: %s", e, exc_info=True)
        msg = "Что-то пошло не так, персона не создалась. Загрузите фото заново или напишите в поддержку." if from_persona else USER_FRIENDLY_ERROR
        await context.bot.send_message(chat_id=chat_id, text=msg)
    finally:
        if gen_lock is not None:
            gen_lock.release()
