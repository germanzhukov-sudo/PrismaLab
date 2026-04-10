"""Хэндлеры Экспресс-фото: выбор стиля, оплата, генерация."""

from __future__ import annotations

import asyncio
import io
import logging
import secrets
import time
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice, Update, WebAppInfo
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from prismalab.config import (
    MINIAPP_URL,
    USERDATA_FAST_CUSTOM_PROMPT,
    USERDATA_FAST_PERSONA_MSG_ID,
    USERDATA_FAST_SELECTED_STYLE,
    USERDATA_FAST_STYLE_MSG_ID,
    USERDATA_FAST_STYLE_PAGE,
    USERDATA_MODE,
    USERDATA_PERSONA_RECREATING,
    USERDATA_PHOTO_FILE_IDS,
    USERDATA_SUBJECT_GENDER,
    USER_FRIENDLY_ERROR,
)
from prismalab.keyboards import (
    FAST_STYLES_FEMALE,
    FAST_STYLES_MALE,
    _fast_style_choice_keyboard,
    _fast_style_label,
    _fast_tariff_packages_keyboard,
    _fast_tariff_persona_only_keyboard,
    _fast_upload_or_change_keyboard,
    _payment_yookassa_keyboard,
    _persona_rules_keyboard,
)
from prismalab.messages import (
    FAST_TARIFFS_TARIFFS_MESSAGE,
    PERSONA_RULES_MESSAGE,
    STYLE_EXAMPLES_FOOTER,
    _fast_credits_word,
    _format_balance_express,
    _format_balance_persona,
    _generations_count_fast,
)
from prismalab.telegram_utils import _acquire_user_generation_lock, _safe_get_file_bytes, _safe_send_document
from prismalab.alerts import alert_payment_error, alert_slow_generation
from prismalab.payment import (
    INVOICE_AMOUNT_KOPECKS,
    INVOICE_PAYLOAD_PREFIX,
    TELEGRAM_PROVIDER_TOKEN,
    _amount_rub,
    create_payment,
    poll_payment_status,
)

logger = logging.getLogger("prismalab")

import prismalab.bot as _bot  # noqa: E402


async def _update_fast_style_message(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, new_msg: Any
) -> None:
    """Удалить предыдущее сообщение со стилями (если есть) и сохранить ID нового."""
    old_id = context.user_data.get(USERDATA_FAST_STYLE_MSG_ID)
    if old_id is not None and old_id != new_msg.message_id:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=old_id)
        except Exception:
            pass
    context.user_data[USERDATA_FAST_STYLE_MSG_ID] = new_msg.message_id
async def _send_fast_tariffs_two_messages(
    bot: Any, chat_id: int, context: ContextTypes.DEFAULT_TYPE, *, edit_message: Any = None, back_callback: str = "pl_fast_back"
) -> None:
    """Двухсообщенный экран тарифов: призыв к Персоне + тарифы Экспресс. При выборе Персоны второе удаляется, при Назад — первое."""
    if edit_message:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=edit_message.message_id,
            text=_bot.FAST_TARIFFS_PERSONA_MESSAGE,
            reply_markup=_fast_tariff_persona_only_keyboard(),
            parse_mode="HTML",
        )
        persona_msg_id = edit_message.message_id
    else:
        msg1 = await bot.send_message(
            chat_id=chat_id,
            text=_bot.FAST_TARIFFS_PERSONA_MESSAGE,
            reply_markup=_fast_tariff_persona_only_keyboard(),
            parse_mode="HTML",
        )
        persona_msg_id = msg1.message_id
    msg = await bot.send_message(
        chat_id=chat_id,
        text=FAST_TARIFFS_TARIFFS_MESSAGE,
        reply_markup=_fast_tariff_packages_keyboard(back_callback=back_callback),
        parse_mode="HTML",
    )
    context.user_data[USERDATA_FAST_PERSONA_MSG_ID] = persona_msg_id
    context.user_data[USERDATA_FAST_STYLE_MSG_ID] = msg.message_id
def _fast_ready_to_upload_text(credits: int, style_label: str, *, after_payment: bool = False) -> str:
    """Текст экрана «готов к загрузке»: баланс + выбранный стиль."""
    parts = []
    if after_payment:
        parts.append("<b>Оплата получена ✅</b>")
    parts.append(_format_balance_express(credits))
    parts.append(f"Выбранный стиль: <b>{style_label}</b>")
    parts.append("Загрузите фото или поменяйте стиль, если хотите")
    return "\n\n".join(parts)
def _fast_style_screen_text(credits: int, credits_word: str, *, has_photo: bool = False) -> str:
    """Единый текст экрана выбора стиля: с фото (после генерации/возврат) или без."""
    balance = _format_balance_express(credits)
    credit_line = f"{balance}\n\n<b>1 кредит = 1 фото</b>"
    if has_photo:
        base = f"<b>Можете не загружать фото заново</b> – просто выберите другой стиль для этого же снимка\n\nЕсли хотите – <b>загрузите новое</b> 👇\n\n{credit_line}"
    else:
        base = f"{credit_line}\n\n<b>Выберите стиль</b> или введите <b>свою идею</b> 👇"
    return f"{base}\n\n{STYLE_EXAMPLES_FOOTER}"
def _fast_after_gender_content(profile: Any, gender: str | None = None, *, has_photo: bool = False) -> tuple[str | None, InlineKeyboardMarkup | None]:
    """Текст и клавиатура после выбора пола. При 0 кредитов — (None, None), вызывающий делает двухсообщенный экран."""
    g = gender or getattr(profile, "subject_gender", None) or "female"
    credits = _generations_count_fast(profile)
    credits_word = _fast_credits_word(credits)
    has_generations = credits > 0
    if has_generations:
        text = _fast_style_screen_text(credits, credits_word, has_photo=has_photo)
        return text, _fast_style_choice_keyboard(g, include_tariffs=True)
    return None, None
async def handle_fast_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Переключение страницы стилей Экспресс-фото (← Пред / След →)."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    parts = (query.data or "").split(":")
    if len(parts) < 2 or parts[1] == "noop":
        return
    try:
        page = int(parts[1])
    except ValueError:
        return
    ctx = int(parts[2]) if len(parts) > 2 else 0  # 0=main, 1=back_to_ready, 2=from_profile

    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "fast_page", {"page": page})
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or profile.subject_gender or "female"
    credits = _generations_count_fast(profile)
    credits_word = _fast_credits_word(credits)
    has_photo = bool(context.user_data.get(USERDATA_PHOTO_FILE_IDS))
    text = _fast_style_screen_text(credits, credits_word, has_photo=has_photo)

    include_tariffs = credits > 0
    back_to_ready = ctx == 1
    from_profile = ctx == 2
    context.user_data[USERDATA_FAST_STYLE_PAGE] = page
    reply_markup = _fast_style_choice_keyboard(
        gender, include_tariffs=include_tariffs, back_to_ready=back_to_ready, from_profile=from_profile, page=page
    )
    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML", disable_web_page_preview=True)
async def handle_fast_gender_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """После выбора пола в «Быстрое фото»: сохраняем пол в профиль, затем загрузка фото или тарифы."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    # pl_fast_gender:female или pl_fast_gender:male
    _, gender = query.data.split(":", 1)
    context.user_data[USERDATA_SUBJECT_GENDER] = gender
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "fast_gender_select", {"gender": gender})
    except Exception:
        pass
    _bot.store.set_subject_gender(user_id, gender)  # запоминаем пол один раз (смена потом в профиле)
    profile = _bot.store.get_user(user_id)
    context.user_data[USERDATA_MODE] = "fast"
    has_photo = bool(context.user_data.get(USERDATA_PHOTO_FILE_IDS))
    text, reply_markup = _fast_after_gender_content(profile, gender=gender, has_photo=has_photo)
    if text is not None:
        extra = {"parse_mode": "HTML", "disable_web_page_preview": True}
        await query.edit_message_text(text, reply_markup=reply_markup, **extra)
        context.user_data[USERDATA_FAST_STYLE_MSG_ID] = query.message.message_id
    else:
        await _send_fast_tariffs_two_messages(context.bot, query.message.chat_id, context, edit_message=query.message)
async def handle_fast_show_tariffs_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Назад» с экрана оплаты Экспресс: возврат к выбору тарифа. Редактируем текущее сообщение — первое (Персона) уже в чате."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "fast_show_tariffs")
    except Exception:
        pass
    await query.edit_message_text(
        FAST_TARIFFS_TARIFFS_MESSAGE,
        reply_markup=_fast_tariff_packages_keyboard(),
        parse_mode="HTML",
    )
    context.user_data[USERDATA_FAST_STYLE_MSG_ID] = query.message.message_id
async def handle_fast_buy_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Нажатие на пакет генераций (5/10/30). Telegram Payments (инвойс) или ЮKassa или симуляция."""
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    _, count_str = query.data.split(":", 1)
    try:
        count = int(count_str)
    except ValueError:
        count = 5
    if count not in (5, 10, 30):
        count = 5
    user_id = int(query.from_user.id) if query.from_user else 0
    chat_id = query.message.chat_id if query.message else 0
    try:
        _bot.store.log_event(user_id, "fast_buy_init", {"credits": count})
    except Exception:
        pass

    if _bot.use_yookassa():
        amount = _amount_rub(_bot.store, "fast", count)
        me = await context.bot.get_me()
        return_url = f"https://t.me/{me.username}" if me and me.username else None
        url, payment_id = create_payment(
            amount_rub=amount,
            description=f"Экспресс-фото: {count} кредитов",
            metadata={
                "user_id": str(user_id),
                "chat_id": str(chat_id),
                "credits": str(count),
                "product_type": "fast",
            },
            return_url=return_url,
        )
        if url and payment_id:
            await query.edit_message_text(
                f"<b>{count} кредитов Экспресс за {amount:.0f} ₽</b>\n\nОплатите удобным способом, кредиты зачислим моментально",
                parse_mode="HTML",
                reply_markup=_payment_yookassa_keyboard(url, "pl_fast_show_tariffs"),
            )
            # Запускаем поллинг статуса в фоне
            asyncio.create_task(poll_payment_status(
                payment_id=payment_id,
                bot=context.bot,
                store=_bot.store,
                user_id=user_id,
                chat_id=chat_id,
                credits=count,
                product_type="fast",
                amount_rub=amount,
            ))
            return
        else:
            logger.warning("Ошибка создания платежа (fast): %s", payment_id)
            asyncio.create_task(alert_payment_error(user_id, "fast", str(payment_id)))
            await query.edit_message_text("❌ Ошибка оплаты. Попробуйте еще раз.")
            return

    if _bot.use_telegram_payments():
        payload = f"pl:fast:{count}:{user_id}"
        try:
            await context.bot.send_invoice(
                chat_id=chat_id,
                title="Экспресс-фото",
                description=f"Экспресс-фото: {count} кредитов",
                payload=payload,
                provider_token=TELEGRAM_PROVIDER_TOKEN,
                currency="RUB",
                prices=[LabeledPrice(label="Оплата", amount=INVOICE_AMOUNT_KOPECKS)],
            )
        except BadRequest as e:
            logger.exception("send_invoice (fast) BadRequest: %s", e)
            await query.edit_message_text("❌ Не удалось отправить счёт. Проверьте в BotFather → Payments, что подключена платёжная система.")
        return

    profile = _bot.store.get_user(user_id)
    new_total = profile.paid_generations_remaining + count
    _bot.store.set_paid_generations_remaining(user_id, new_total)
    context.user_data[USERDATA_MODE] = "fast"
    profile = _bot.store.get_user(user_id)
    gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or profile.subject_gender or "female"
    credits = _generations_count_fast(profile)
    photo_file_ids = list(context.user_data.get(USERDATA_PHOTO_FILE_IDS, []))
    selected_style = context.user_data.pop(USERDATA_FAST_SELECTED_STYLE, None)
    context.user_data.pop(USERDATA_FAST_CUSTOM_PROMPT, None)
    if photo_file_ids:
        # Если был выбран стиль при 0 кредитов — сразу генерируем
        if selected_style and selected_style != "custom":
            style_label = _fast_style_label(selected_style)
            prompt = _bot._persona_style_prompt(selected_style, style_label)
            chat_id = query.message.chat_id
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=query.message.message_id)
            except Exception:
                pass
            status_msg = await context.bot.send_message(chat_id=chat_id, text="🎨 <i>Создаю изображение...</i>", parse_mode="HTML")
            context.application.create_task(
                _run_fast_generation_impl(
                    context=context,
                    chat_id=chat_id,
                    user_id=user_id,
                    style_id=selected_style,
                    style_label=style_label,
                    prompt=prompt,
                    photo_file_ids=photo_file_ids,
                    profile=profile,
                    status_msg=status_msg,
                )
            )
            return
        page = context.user_data.get(USERDATA_FAST_STYLE_PAGE, 0)
        text = f"Оплата получена ✅\n\n{_format_balance_express(credits)}\n\n<b>1 кредит = 1 фото</b>\n\n<b>Выберите стиль</b> или введите <b>свою идею</b> 👇\n\n{STYLE_EXAMPLES_FOOTER}"
        await query.edit_message_text(
            text,
            reply_markup=_fast_style_choice_keyboard(gender, include_tariffs=True, back_to_ready=True, page=page),
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        context.user_data[USERDATA_FAST_STYLE_MSG_ID] = query.message.message_id
    else:
        styles = FAST_STYLES_FEMALE if gender == "female" else FAST_STYLES_MALE
        style_label, style_id = styles[0]
        context.user_data[USERDATA_FAST_SELECTED_STYLE] = style_id
        text = _fast_ready_to_upload_text(credits, style_label, after_payment=True)
        await query.edit_message_text(
            text,
            reply_markup=_fast_upload_or_change_keyboard(),
            parse_mode="HTML",
        )
async def handle_pre_checkout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Подтверждение перед списанием (Telegram Payments)."""
    try:
        query = update.pre_checkout_query
        if not query:
            return
        payload = (query.invoice_payload or "").strip()
        if payload.startswith(INVOICE_PAYLOAD_PREFIX):
            await query.answer(ok=True)
            try:
                pre_uid = int(query.from_user.id) if query.from_user else 0
                _bot.store.log_event(pre_uid, "pre_checkout", {"payload": payload})
            except Exception:
                pass
        else:
            await query.answer(ok=False, error_message="Неизвестный тип платежа")
    except Exception as e:
        logger.exception("PreCheckout ошибка: %s", e)
        if update.pre_checkout_query:
            try:
                await update.pre_checkout_query.answer(ok=False, error_message="Ошибка. Попробуйте позже.")
            except Exception:
                pass
        raise
async def handle_successful_payment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Успешная оплата через Telegram Payments: начисляем кредиты."""
    try:
        msg = update.message
        if not msg or not msg.successful_payment:
            return
        payload = (msg.successful_payment.invoice_payload or "").strip()
        if not payload.startswith(INVOICE_PAYLOAD_PREFIX):
            return
        parts = payload.split(":")
        if len(parts) != 4:  # pl, product_type, credits, user_id
            logger.warning("Неверный payload успешной оплаты: %s", payload)
            return
        _, product_type, credits_str, user_id_str = parts
        try:
            credits = int(credits_str)
            user_id = int(user_id_str)
        except ValueError:
            logger.warning("Не удалось распарсить payload: %s", payload)
            return

        # Логируем событие успешной оплаты
        try:
            _bot.store.log_event(user_id, "payment_success", {"product_type": product_type, "credits": credits, "method": "telegram"})
        except Exception:
            pass

        # Логируем платёж для аналитики
        try:
            amount_rub = float(msg.successful_payment.total_amount) / 100
            payment_log_id = _bot.store.log_payment(
                user_id=user_id,
                payment_id=msg.successful_payment.telegram_payment_charge_id,
                payment_method="telegram",
                product_type=product_type,
                credits=credits,
                amount_rub=amount_rub,
            )
            if payment_log_id is None:
                logger.info(
                    "Telegram payment %s уже обработан, пропускаем",
                    msg.successful_payment.telegram_payment_charge_id,
                )
                return
            logger.info("Платёж записан: user=%s, amount=%.2f, type=%s", user_id, amount_rub, product_type)
        except Exception as e:
            logger.exception("Ошибка записи платежа в БД: %s", e)
            return

        if product_type == "fast":
            context.user_data[USERDATA_MODE] = "fast"
            profile = _bot.store.get_user(user_id)
            new_total = profile.paid_generations_remaining + credits
            _bot.store.set_paid_generations_remaining(user_id, new_total)
            text = f"Оплата получена ✅\n\n{_format_balance_express(_generations_count_fast(_bot.store.get_user(user_id)))}\n\n<b>1 кредит = 1 фото</b>\n\n<b>Выберите стиль</b> или введите <b>свою идею</b> 👇\n\n{STYLE_EXAMPLES_FOOTER}"
            gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or _bot.store.get_user(user_id).subject_gender or "female"
            reply_msg = await msg.reply_text(
                text,
                reply_markup=_fast_style_choice_keyboard(gender, include_tariffs=True, back_to_ready=True, page=0),
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            context.user_data[USERDATA_FAST_STYLE_MSG_ID] = reply_msg.message_id
            context.user_data[USERDATA_FAST_STYLE_PAGE] = 0
        elif product_type == "persona_topup":
            context.user_data[USERDATA_MODE] = "persona"
            profile = _bot.store.get_user(user_id)
            new_total = profile.persona_credits_remaining + credits
            _bot.store.set_persona_credits(user_id, new_total)
            gender = profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
            text = f"<b>Оплата получена</b> ✅\n\nВыберите образ в приложении <b>Персона</b> 👇\n\n{_format_balance_persona(new_total)}"
            kb_rows = []
            if MINIAPP_URL:
                kb_rows.append([InlineKeyboardButton("Персона", web_app=WebAppInfo(url=MINIAPP_URL), api_kwargs={"style": "primary", "icon_custom_emoji_id": "5235702276424737428"})])
            kb_rows.append([InlineKeyboardButton("Главное меню", callback_data="pl_fast_back")])
            await msg.reply_text(
                text,
                reply_markup=InlineKeyboardMarkup(kb_rows),
                parse_mode="HTML",
            )
        elif product_type == "persona_create":
            context.user_data[USERDATA_MODE] = "persona"
            _bot.store.set_persona_credits(user_id, credits)
            if context.user_data.pop(USERDATA_PERSONA_RECREATING, None):
                _bot.store.set_astria_lora_tune(user_id=user_id, tune_id=None)
            await msg.reply_text(
                PERSONA_RULES_MESSAGE,
                reply_markup=_persona_rules_keyboard(),
                parse_mode="HTML",
            )
        else:
            logger.warning("Неизвестный product_type в оплате: %s", product_type)
    except Exception as e:
        logger.exception("handle_successful_payment ошибка: %s", e)
        raise
async def handle_fast_upload_photo_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Загрузить фото»: показываем правила загрузки."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "fast_upload_rules")
    except Exception:
        pass
    await query.edit_message_text(
        _bot.FAST_PHOTO_RULES_MESSAGE,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="pl_fast_show_ready")]]),
        parse_mode="HTML",
    )
async def handle_fast_change_style_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Поменять стиль»: показываем выбор стиля."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "fast_change_style")
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or profile.subject_gender or "female"
    credits = _generations_count_fast(profile)
    credits_word = _fast_credits_word(credits)
    has_photo = bool(context.user_data.get(USERDATA_PHOTO_FILE_IDS))
    page = context.user_data.get(USERDATA_FAST_STYLE_PAGE, 0)
    text = _fast_style_screen_text(credits, credits_word, has_photo=has_photo)
    await query.edit_message_text(
        text,
        reply_markup=_fast_style_choice_keyboard(gender, include_tariffs=True, back_to_ready=True, page=page),
        parse_mode="HTML",
        disable_web_page_preview=True,
    )
    context.user_data[USERDATA_FAST_STYLE_MSG_ID] = query.message.message_id
async def handle_fast_show_ready_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Возврат на экран «Загрузить фото / Поменять стиль» (из правил)."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "fast_show_ready")
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or profile.subject_gender or "female"
    style_id = context.user_data.get(USERDATA_FAST_SELECTED_STYLE)
    if not style_id:
        styles = FAST_STYLES_FEMALE if gender == "female" else FAST_STYLES_MALE
        style_label, style_id = styles[0]
        context.user_data[USERDATA_FAST_SELECTED_STYLE] = style_id
    else:
        style_label = _fast_style_label(style_id)
    credits = _generations_count_fast(profile)
    text = _fast_ready_to_upload_text(credits, style_label, after_payment=False)
    await query.edit_message_text(
        text,
        reply_markup=_fast_upload_or_change_keyboard(),
        parse_mode="HTML",
    )
async def _run_fast_generation_impl(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    style_id: str,
    style_label: str,
    prompt: str,
    photo_file_ids: list[str],
    profile: Any,
    status_msg: Any,
    status_prefix: str = "",
) -> None:
    """Выполняет генерацию Быстрое фото и отправляет результат + сообщение с кнопками стилей."""
    status_message_id = status_msg.message_id
    bot = context.bot
    settings = _bot.load_settings()
    total_timeout = settings.kie_max_seconds + 120  # KIE + буфер на загрузку/скачивание
    prefix = status_prefix or ""

    async def _do_generation() -> None:
        if not settings.kie_api_key:
            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=f"{prefix}{USER_FRIENDLY_ERROR}")
            return
        status_text = f"{prefix}🎨 <i>Создаю изображение...</i>"
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=status_text, parse_mode="HTML")
        except BadRequest as e:
            if "message is not modified" not in str(e).lower():
                raise
            photo_bytes = await _safe_get_file_bytes(bot, photo_file_ids[-1])
            if not photo_bytes:
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=f"{prefix}{USER_FRIENDLY_ERROR}")
                return
            random_id = secrets.token_hex(8)
            uploaded_url = None
            for upload_attempt in range(3):
                try:
                    uploaded_url = await asyncio.to_thread(
                        _bot.kie_upload_file_base64,
                        api_key=settings.kie_api_key,
                        image_bytes=photo_bytes,
                        file_name=f"fast_{random_id}.jpg",
                        upload_path="user-uploads",
                        timeout_s=90.0,
                    )
                    break
                except Exception as up_e:
                    err_str = str(up_e).lower()
                    if "connection" in err_str or "remote" in err_str or "disconnected" in err_str:
                        logger.warning("Быстрое фото: загрузка в KIE попытка %s/3: %s", upload_attempt + 1, up_e)
                        if upload_attempt < 2:
                            await asyncio.sleep(4.0)
                        else:
                            raise
                    else:
                        raise
            if not uploaded_url:
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=f"{prefix}{USER_FRIENDLY_ERROR}")
                return
            kie_result = None
            for gen_attempt in range(2):
                try:
                    kie_result = await _bot.kie_run_task_and_wait(
                        api_key=settings.kie_api_key,
                        model="seedream/4.5-edit",
                        prompt=prompt,
                        image_input=[uploaded_url],
                        aspect_ratio="1:1",
                    quality="basic",
                    output_format="jpg",
                    max_seconds=settings.kie_max_seconds,
                        poll_seconds=3.0,
                    )
                    break
                except Exception as gen_e:
                    err_str = str(gen_e).lower()
                    if "connection" in err_str or "remote" in err_str or "disconnected" in err_str:
                        logger.warning("Быстрое фото: вызов Seedream попытка %s/2: %s", gen_attempt + 1, gen_e)
                        if gen_attempt < 1:
                            await asyncio.sleep(5.0)
                        else:
                            raise
                    else:
                        raise
            if not kie_result.image_url:
                await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=f"{prefix}{USER_FRIENDLY_ERROR}")
                return
            image_bytes = await asyncio.to_thread(_bot.kie_download_image_bytes, kie_result.image_url, timeout_s=30.0)
            bio = io.BytesIO(image_bytes)
            bio.name = f"fast_{style_id}.jpg"
            await bot.delete_message(chat_id=chat_id, message_id=status_message_id)
        await _safe_send_document(bot=bot, chat_id=chat_id, document=bio, caption=f"Экспресс-фото: {style_label}")
        # Логируем успешную генерацию для аналитики
        try:
            _bot.store.log_event(user_id, "generation", {"mode": "fast", "style": style_id, "provider": "kie"})
        except Exception:
            pass
        if not profile.free_generation_used:
            _bot.store.spend_free_generation(user_id)
        elif profile.paid_generations_remaining > 0:
            _bot.store.set_paid_generations_remaining(user_id, profile.paid_generations_remaining - 1)
        updated_profile = _bot.store.get_user(user_id)
        credits = _generations_count_fast(updated_profile)
        gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or "female"
        context.user_data.pop(USERDATA_FAST_SELECTED_STYLE, None)  # чтобы загрузка нового фото → выбор стиля, а не авто-генерация
        if credits <= 0:
            await _send_fast_tariffs_two_messages(bot, chat_id, context)
        else:
            credits_word = _fast_credits_word(credits)
            page = context.user_data.get(USERDATA_FAST_STYLE_PAGE, 0)
            text = f"<b>Можете не загружать фото заново</b> – просто выберите другой стиль для этого же снимка\n\nЕсли хотите – <b>загрузите новое</b> 👇\n\n{_format_balance_express(credits)}\n\n<b>1 кредит = 1 фото</b>\n\n{STYLE_EXAMPLES_FOOTER}"
            reply_markup = _fast_style_choice_keyboard(gender, include_tariffs=True, back_to_ready=True, page=page)
            msg = await bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            await _update_fast_style_message(context, chat_id, msg)

    try:
        start_time = time.time()
        await asyncio.wait_for(_do_generation(), timeout=float(total_timeout))
        duration = time.time() - start_time
        # Алерт о медленной генерации (> 5 минут)
        if duration > 300:
            await alert_slow_generation(user_id, duration, "express")
    except asyncio.TimeoutError:
        logger.warning("Быстрое фото: таймаут %sс (стиль %s)", total_timeout, style_id)
        await _bot.alert_generation_error(user_id, f"Таймаут {total_timeout}с", "express")
        try:
            await bot.edit_message_text(
                chat_id=chat_id, message_id=status_message_id,
                text=f"{prefix}{USER_FRIENDLY_ERROR}",
            )
        except Exception:
            await bot.send_message(chat_id=chat_id, text=USER_FRIENDLY_ERROR)
    except _bot.KieError as e:
        logger.error("Быстрое фото KIE: %s", e)
        await _bot.alert_generation_error(user_id, str(e), "express")
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=f"{prefix}{USER_FRIENDLY_ERROR}")
        except Exception:
            await bot.send_message(chat_id=chat_id, text=USER_FRIENDLY_ERROR)
    except Exception as e:
        logger.exception("Быстрое фото (стиль %s): %s", style_id, e)
        await _bot.alert_generation_error(user_id, str(e), "express")
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=status_message_id, text=f"{prefix}{USER_FRIENDLY_ERROR}")
        except Exception:
            try:
                await bot.send_message(chat_id=chat_id, text=USER_FRIENDLY_ERROR)
            except Exception as send_err:
                logger.error("Быстрое фото: не удалось отправить сообщение об ошибке: %s", send_err)
async def handle_fast_style_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Выбор стиля: либо «загрузите фото» (если фото нет), либо генерация с имеющимся фото. Для «Своя идея» — сначала ввод текста."""
    query = update.callback_query
    if not query or not query.data or "pl_fast_style:" not in query.data:
        return
    await query.answer()
    _, style_id = query.data.split(":", 1)
    style_label = _fast_style_label(style_id)
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "fast_style_select", {"style_id": style_id})
    except Exception:
        pass
    profile = _bot.store.get_user(user_id)
    if _generations_count_fast(profile) <= 0:
        context.user_data[USERDATA_FAST_SELECTED_STYLE] = style_id
        if style_id == "custom":
            context.user_data.pop(USERDATA_FAST_CUSTOM_PROMPT, None)  # не использовать старый промпт после пополнения
        await _send_fast_tariffs_two_messages(context.bot, query.message.chat_id, context, edit_message=query.message)
        return
    photo_file_ids = list(context.user_data.get(USERDATA_PHOTO_FILE_IDS, []))
    custom_prompt = (context.user_data.get(USERDATA_FAST_CUSTOM_PROMPT) or "").strip()

    if style_id == "custom":
        context.user_data[USERDATA_FAST_SELECTED_STYLE] = style_id
        context.user_data.pop(USERDATA_FAST_CUSTOM_PROMPT, None)  # каждый раз запрашиваем новый запрос
        custom_prompt = (context.user_data.get(USERDATA_FAST_CUSTOM_PROMPT) or "").strip()
        if not custom_prompt:
            await query.edit_message_text(
                _bot.FAST_CUSTOM_PROMPT_REQUEST_MESSAGE,
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="pl_fast_change_style")]]),
                parse_mode="HTML",
            )
            return
        if not photo_file_ids:
            await query.edit_message_text(
                _bot.FAST_PHOTO_RULES_MESSAGE,
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("Назад", callback_data="pl_fast_change_style"),
                ]]),
                parse_mode="HTML",
            )
            return
        gen_lock = await _acquire_user_generation_lock(user_id)
        if gen_lock is None:
            await query.edit_message_text("Пожалуйста, немного подождите. Ещё обрабатываю прошлый запрос")
            return
        prompt = custom_prompt
        chat_id = query.message.chat_id
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=query.message.message_id)
        except Exception:
            pass
        status_msg = await context.bot.send_message(chat_id=chat_id, text="🎨 <i>Создаю изображение...</i>", parse_mode="HTML")

        async def _run_and_release() -> None:
            try:
                await _run_fast_generation_impl(
                    context=context,
                    chat_id=chat_id,
                    user_id=user_id,
                    style_id=style_id,
                    style_label=style_label,
                    prompt=prompt,
                    photo_file_ids=photo_file_ids,
                    profile=profile,
                    status_msg=status_msg,
                )
            finally:
                gen_lock.release()

        context.application.create_task(_run_and_release())
        return

    if not photo_file_ids:
        context.user_data[USERDATA_FAST_SELECTED_STYLE] = style_id
        await query.edit_message_text(
            _bot.FAST_PHOTO_RULES_MESSAGE,
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("Назад", callback_data="pl_fast_change_style"),
            ]]),
            parse_mode="HTML",
        )
        return
    gen_lock = await _acquire_user_generation_lock(user_id)
    if gen_lock is None:
        await query.edit_message_text("Пожалуйста, немного подождите. Ещё обрабатываю прошлый запрос")
        return
    prompt = _bot._persona_style_prompt(style_id, style_label)
    chat_id = query.message.chat_id
    gender = context.user_data.get(USERDATA_SUBJECT_GENDER) or profile.subject_gender or "female"
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=query.message.message_id)
    except Exception:
        pass
    status_msg = await context.bot.send_message(chat_id=chat_id, text="🎨 <i>Создаю изображение...</i>", parse_mode="HTML")

    async def _run_and_release() -> None:
        try:
            await _run_fast_generation_impl(
                context=context,
                chat_id=chat_id,
                user_id=user_id,
                style_id=style_id,
                style_label=style_label,
                prompt=prompt,
                photo_file_ids=photo_file_ids,
                profile=profile,
                status_msg=status_msg,
            )
        finally:
            gen_lock.release()

    context.application.create_task(_run_and_release())
