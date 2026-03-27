"""Хэндлеры фотосетов (паков): покупка, генерация, recovery."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import uuid
from typing import Any

import requests

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from prismalab.config import (
    USERDATA_MODE,
    USERDATA_PERSONA_PACK_GIFT_APPLIED,
    USERDATA_PERSONA_PACK_IN_PROGRESS,
    USERDATA_PERSONA_PACK_PHOTOS,
    USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS,
    USERDATA_PERSONA_PACK_WAITING_UPLOAD,
    USERDATA_PERSONA_SELECTED_PACK_ID,
    USERDATA_PERSONA_TRAINING_MSG_ID,
    USERDATA_PERSONA_WAITING_UPLOAD,
    USERDATA_SUBJECT_GENDER,
    _dev_pack_train_from_images,
    _dev_skip_pack_payment,
    _use_unified_pack_persona_flow,
)
from prismalab.keyboards import (
    _persona_packs_keyboard,
    _persona_training_keyboard,
    _persona_pack_upload_keyboard,
    _photoset_done_keyboard,
    _photoset_retry_keyboard,
    _payment_yookassa_keyboard,
)
from prismalab.messages import (
    PERSONA_PACK_UPLOAD_WAIT_MESSAGE,
    PHOTOSET_PROGRESS_ALERT,
    _photoset_done_message,
)
from prismalab.keyboards import _persona_app_keyboard
from prismalab.pack_offers import _find_pack_offer, _pack_offers
from prismalab.telegram_utils import _acquire_user_generation_lock, _safe_edit_status, _safe_get_file_bytes, _safe_send_document
from prismalab.image_utils import _prepare_image_for_photomaker
from prismalab.alerts import alert_pack_error, alert_payment_error
from prismalab.astria_client import (
    create_tune_from_pack as astria_create_tune_from_pack,
    get_pack as astria_get_pack,
    get_tune_prompt_ids as astria_get_tune_prompt_ids,
    wait_pack_images as astria_wait_pack_images,
)
from prismalab.payment import (
    apply_test_amount,
    build_pack_callback_url,
    create_payment,
    get_payment_status,
    pack_delivered_set,
    pack_in_progress_set,
)

logger = logging.getLogger("prismalab")

import prismalab.bot as _bot  # noqa: E402


def _persona_pack_class_name(gender: str | None) -> str:
    return "man" if gender == "male" else "woman"

def _persona_lora_name(gender: str | None) -> str:
    """
    Имя класса для обучения персональной LoRA.
    По умолчанию сохраняем историческое поведение: person.
    Для dev-экспериментов можно включить man/woman через env.
    """
    mode = (os.getenv("PRISMALAB_PERSONA_LORA_NAME_MODE") or "person").strip().lower()
    if mode in {"gender", "class", "man_woman"}:
        return _persona_pack_class_name(gender)
    return "person"

def _resolve_pack_class_name(offer: dict[str, Any], gender: str | None) -> str:
    custom = str(offer.get("class_name") or "").strip().lower()
    if custom in {"man", "woman", "boy", "girl", "dog", "cat"}:
        return custom
    return _persona_pack_class_name(gender)

def _pack_classes_text(classes: list[str]) -> str:
    mapping = {"man": "man (мужчина)", "woman": "woman (женщина)", "boy": "boy (мальчик)", "girl": "girl (девочка)", "dog": "dog (собакой)", "cat": "cat (кошкой)"}
    labels: list[str] = []
    for cls in classes:
        labels.append(mapping.get(cls, cls))
    return ", ".join(labels)

def _extract_pack_cost_info(class_cost: Any) -> tuple[str, str]:
    if not isinstance(class_cost, dict):
        return "", ""
    for key in ("cost", "cost_mc", "price", "amount"):
        value = class_cost.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if not value_str:
            continue
        return str(key), value_str
    return "", ""

def _pack_wait_timeout_seconds(expected_images: int) -> int:
    """
    Таймаут ожидания паков:
    - можно переопределить через PRISMALAB_ASTRIA_PACK_MAX_SECONDS
    - по умолчанию масштабируем от количества изображений
    """
    raw = (os.getenv("PRISMALAB_ASTRIA_PACK_MAX_SECONDS") or "").strip()
    if raw:
        try:
            return max(900, int(raw))
        except Exception:
            pass
    expected = max(1, int(expected_images))
    # 30 минут минимум, дальше ~75 секунд на изображение.
    return max(1800, expected * 75)

async def _fallback_to_pack_photo_upload(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    pack_id: int,
    status_message_id: int,
) -> None:
    """Fallback: если auto-prepare pack tune невозможен, просим 10 фото."""
    context.user_data[USERDATA_PERSONA_SELECTED_PACK_ID] = pack_id
    context.user_data[USERDATA_PERSONA_PACK_WAITING_UPLOAD] = True
    context.user_data[USERDATA_PERSONA_PACK_PHOTOS] = []
    context.user_data[USERDATA_PERSONA_PACK_UPLOAD_MSG_IDS] = []
    context.user_data[USERDATA_MODE] = "persona_pack_upload"
    try:
        _bot.store.set_pending_pack_upload(user_id=user_id, pack_id=pack_id)
    except Exception as e:
        logger.warning("pack auto-prepare fallback: cannot set pending_pack_id: %s", e)
    text = (
        "⚠️ Не удалось автоматически подготовить модель для фотосетов.\n\n"
        + PERSONA_PACK_UPLOAD_WAIT_MESSAGE
    )
    try:
        await _safe_edit_status(
            context.bot, chat_id, status_message_id,
            text=text,
            reply_markup=_persona_pack_upload_keyboard(),
            parse_mode="HTML",
        )
    except Exception:
        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=_persona_pack_upload_keyboard(),
            parse_mode="HTML",
        )

async def _start_pending_paid_photoset_after_persona(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
) -> bool:
    """
    Если пользователь пришёл из оплаты фотосета без Персоны:
    - после обучения Персоны автоматически запускаем фотосет;
    - дарим +1 кредит Персоны (однократно в этом flow).
    """
    if not _use_unified_pack_persona_flow():
        return False
    pack_id = _bot.store.get_pending_pack_upload(user_id)
    if pack_id is None:
        return False

    offer = _find_pack_offer(int(pack_id))
    if not offer:
        _bot.store.clear_pending_pack_upload(user_id)
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Не удалось найти оплаченный фотосет. Напишите в поддержку.",
        )
        return False

    profile = _bot.store.get_user(user_id)
    gifted = False
    credits_now = int(getattr(profile, "persona_credits_remaining", 0) or 0)
    if credits_now <= 0:
        _bot.store.set_persona_credits(user_id, 1)
        gifted = True

    _bot.store.clear_pending_pack_upload(user_id)
    context.user_data[USERDATA_MODE] = "persona"
    context.user_data[USERDATA_PERSONA_WAITING_UPLOAD] = False
    context.user_data[USERDATA_PERSONA_PACK_GIFT_APPLIED] = gifted
    context.user_data[USERDATA_PERSONA_PACK_IN_PROGRESS] = True

    old_msg_id = context.user_data.pop(USERDATA_PERSONA_TRAINING_MSG_ID, None)
    if old_msg_id:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=old_msg_id)
        except Exception:
            pass

    from html import escape
    msg = await context.bot.send_message(
        chat_id=chat_id,
        text=(
            "Готово! 🎉 Персональная модель обучена\n\n"
            f"Приступаю к созданию фотосета <b>«{escape(offer['title'])}»</b>\n\n"
            "Кстати, вам больше не нужно будет загружать фото заново, мы сохраним модель на <b>30 дней</b>"
        ),
        parse_mode="HTML",
        reply_markup=_persona_training_keyboard(),
    )

    coro = _run_persona_pack_generation(
        context=context,
        chat_id=chat_id,
        user_id=user_id,
        pack_id=int(pack_id),
        offer=offer,
        run_id=f"paid_{user_id}_{int(pack_id)}_{int(time.time())}",
        status_message_id=msg.message_id,
    )
    app = getattr(context, "application", None)
    if app:
        app.create_task(coro)
    else:
        asyncio.create_task(coro)
    return True

async def _ensure_pack_lora_tune_id(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    class_name: str,
    status_message_id: int,
) -> tuple[int | None, str | None]:
    """
    Возвращает tune_id для pack-flow.
    Для классов man/woman используем отдельный pack tune:
    - если уже есть → берём его;
    - если нет → создаём из orig_images persona tune.
    reason:
    - None: всё ок
    - "need_upload": нужен fallback на загрузку 10 фото
    - "pending": pack tune уже обучается, ждём завершения
    """
    if class_name not in {"man", "woman"}:
        return None, "need_upload"

    profile = _bot.store.get_user(user_id)
    existing_pack_tune_raw = getattr(profile, "astria_lora_pack_tune_id", None)
    if existing_pack_tune_raw:
        try:
            return int(str(existing_pack_tune_raw)), None
        except Exception:
            logger.warning("pack tune id is invalid for user %s: %s", user_id, existing_pack_tune_raw)

    settings = _bot.load_settings()
    if not settings.astria_api_key:
        return None, "need_upload"

    pending_pack_tune_raw = getattr(profile, "astria_lora_pack_tune_id_pending", None)
    if pending_pack_tune_raw:
        try:
            pending_tune_id = str(int(str(pending_pack_tune_raw)))
            from prismalab.astria_client import _get_tune, _timeout_s
            pending_obj = await asyncio.to_thread(
                _get_tune,
                api_key=settings.astria_api_key,
                tune_id=pending_tune_id,
                timeout_s=_timeout_s(30.0),
            )
            pending_status = str(pending_obj.get("status") or pending_obj.get("state") or "").lower()
            pending_trained_at = pending_obj.get("trained_at")
            if pending_status in {"completed", "succeeded", "ready", "trained", "finished"} or pending_trained_at:
                _bot.store.set_astria_lora_pack_tune(user_id=user_id, tune_id=pending_tune_id)
                return int(pending_tune_id), None
            if pending_status in {"failed", "error", "cancelled", "canceled"}:
                _bot.store.clear_astria_lora_pack_tune_pending(user_id)
            else:
                logger.info(
                    "pack tune pending and not ready (user=%s tune=%s status=%s), waiting",
                    user_id,
                    pending_tune_id,
                    pending_status,
                )
                return None, "pending"
        except Exception as e:
            logger.warning("cannot resolve pending pack tune for user %s: %s", user_id, e)
            try:
                _bot.store.clear_astria_lora_pack_tune_pending(user_id)
            except Exception:
                pass

    persona_tune_raw = getattr(profile, "astria_lora_tune_id", None)
    if not persona_tune_raw:
        return None, "need_upload"
    try:
        persona_tune_id = str(int(str(persona_tune_raw)))
    except Exception:
        return None, "need_upload"

    try:
        from prismalab.astria_client import _get_tune, _timeout_s, create_lora_tune_and_wait
        persona_tune_obj = await asyncio.to_thread(
            _get_tune,
            api_key=settings.astria_api_key,
            tune_id=persona_tune_id,
            timeout_s=_timeout_s(30.0),
        )
        raw_orig_images = persona_tune_obj.get("orig_images")
        orig_images = [str(x) for x in (raw_orig_images or []) if isinstance(x, str) and x.startswith("http")]
        if len(orig_images) < 4:
            logger.warning(
                "pack tune auto-create: orig_images unavailable (user=%s tune=%s size=%s)",
                user_id,
                persona_tune_id,
                len(orig_images),
            )
            return None, "need_upload"

        pack_tune_title = f"Pack LoRA user {user_id}"
        pack_result = await create_lora_tune_and_wait(
            api_key=settings.astria_api_key,
            name=class_name,
            title=pack_tune_title,
            image_urls=orig_images,
            base_tune_id="1504944",
            preset="flux-lora-portrait",
            on_created=lambda tid: _bot.store.set_astria_lora_pack_tune_pending(user_id=user_id, tune_id=tid),
            max_seconds=7200,
            poll_seconds=15.0,
        )
        _bot.store.set_astria_lora_pack_tune(user_id=user_id, tune_id=pack_result.tune_id)
        return int(str(pack_result.tune_id)), None
    except Exception as e:
        logger.exception("pack tune auto-create failed (user=%s): %s", user_id, e)
        try:
            _bot.store.clear_astria_lora_pack_tune_pending(user_id)
        except Exception:
            pass
        return None, "need_upload"

async def _recover_pending_pack_runs(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Периодическая задача: восстанавливает pack runs, прерванные рестартом бота
    во время ожидания обучения pack tune (4–5 мин).
    """
    rows = _bot.store.get_pending_pack_runs_to_recover()
    if not rows:
        return
    settings = _bot.load_settings()
    if not settings.astria_api_key:
        return
    from prismalab.astria_client import _get_tune, _timeout_s
    for row in rows:
        user_id = int(row["user_id"])
        pack_id = int(row["pack_id"])
        chat_id = int(row["chat_id"])
        run_id = str(row["run_id"] or "")
        expected = int(row["expected"] or 1)
        class_name = str(row["class_name"] or "woman")
        offer_title = str(row.get("offer_title") or "")
        tune_id = str(row.get("tune_id") or "").strip()
        if not tune_id or not run_id:
            continue
        if run_id in pack_delivered_set:
            logger.info("pack recovery: уже доставлено run_id=%s", run_id)
            _bot.store.clear_pending_pack_run(user_id)
            try:
                _bot.store.clear_astria_lora_pack_tune_pending(user_id)
            except Exception:
                pass
            continue
        if run_id in pack_in_progress_set:
            logger.info("pack recovery: доставка уже в процессе run_id=%s", run_id)
            continue
        if run_id in _bot._pack_polling_active:
            logger.info("pack recovery: polling уже выполняется run_id=%s", run_id)
            continue
        if run_id in _bot._pack_processing_active:
            logger.info("pack recovery: основной flow ещё работает run_id=%s", run_id)
            continue
        try:
            pending_obj = await asyncio.to_thread(
                _get_tune,
                api_key=settings.astria_api_key,
                tune_id=tune_id,
                timeout_s=_timeout_s(30.0),
            )
            status = str(pending_obj.get("status") or pending_obj.get("state") or "").lower()
            trained_at = pending_obj.get("trained_at")
            if status in {"failed", "error", "cancelled", "canceled"}:
                _bot.store.clear_astria_lora_pack_tune_pending(user_id)
                _bot.store.clear_pending_pack_run(user_id)
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"❌ Обучение модели для фотосета «{offer_title or str(pack_id)}» завершилось с ошибкой. Попробуйте запустить фотосет снова.",
                )
                continue
            if status not in {"completed", "succeeded", "ready", "trained", "finished"} and not trained_at:
                continue
            _bot.store.set_astria_lora_pack_tune(user_id=user_id, tune_id=tune_id)
            _bot.store.clear_astria_lora_pack_tune_pending(user_id)
            offer = {"title": offer_title or f"Фотосет {pack_id}", "expected_images": expected}
            title = f"{offer['title']} user:{user_id} ts:{int(time.time())}"
            known_prompt_ids: set[str] | None = None
            try:
                known_prompt_ids = await astria_get_tune_prompt_ids(
                    api_key=settings.astria_api_key,
                    tune_id=tune_id,
                )
            except Exception as e:
                logger.warning("pack recovery: не удалось получить prompt_ids для tune %s: %s", tune_id, e)
            callback_url = build_pack_callback_url(user_id, chat_id, pack_id, run_id)
            await astria_create_tune_from_pack(
                api_key=settings.astria_api_key,
                pack_id=pack_id,
                title=title[:120],
                name=class_name,
                tune_ids=[int(tune_id)],
                prompts_callback=callback_url or None,
            )
            # Сразу чистим pending чтобы при крэше recovery не создал промпты повторно
            _bot.store.clear_pending_pack_run(user_id)
            from html import escape
            status_msg = await context.bot.send_message(
                chat_id=chat_id,
                text=f"Приступаю к созданию фотосета <b>«{escape(str(offer.get('title', pack_id)))}»</b>.",
                parse_mode="HTML",
                reply_markup=_persona_training_keyboard(),
            )
            wait_timeout = _pack_wait_timeout_seconds(expected)
            logger.info("pack recovery: начинаю polling tune_id=%s run_id=%s timeout=%ss", tune_id, run_id, wait_timeout)
            _bot._pack_polling_active.add(run_id)
            try:
                urls = await astria_wait_pack_images(
                    api_key=settings.astria_api_key,
                    tune_id=tune_id,
                    expected_images=expected,
                    known_prompt_ids=known_prompt_ids,
                    max_seconds=wait_timeout,
                    poll_seconds=8.0,
                )
            finally:
                _bot._pack_polling_active.discard(run_id)
            if not urls:
                await _safe_edit_status(
                    context.bot, chat_id, status_msg.message_id,
                    text="❌ Фотосет завершился без изображений. Попробуйте еще раз.",
                )
                continue
            if run_id in pack_delivered_set:
                logger.info("pack recovery: уже доставлено через callback run_id=%s", run_id)
                continue
            if run_id in pack_in_progress_set:
                logger.info("pack recovery: доставка уже идёт (основной flow?) run_id=%s", run_id)
                continue
            total = len(urls)
            sent_count = 0
            pack_title = offer.get("title", "") or str(pack_id)
            pack_in_progress_set.add(run_id)
            for i, url in enumerate(urls, start=1):
                try:
                    out_bytes = await _bot.astria_download_first_image_bytes(
                        [url],
                        api_key=settings.astria_api_key,
                        timeout_s=90.0,
                    )
                    if out_bytes:
                        bio = io.BytesIO(out_bytes)
                        bio.name = f"pack_{pack_id}_{sent_count + 1}.png"
                        caption = f"Фотосет «{pack_title}» ({sent_count + 1}/{total})" if sent_count == 0 else ""
                        await _safe_send_document(
                            bot=context.bot,
                            chat_id=chat_id,
                            document=bio,
                            caption=caption,
                        )
                        sent_count += 1
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.warning("pack recovery: download/send failed %s: %s", url[:50], e)
            if sent_count <= 0:
                pack_in_progress_set.discard(run_id)
                asyncio.create_task(
                    alert_pack_error(
                        user_id,
                        pack_id=pack_id,
                        pack_title=offer_title or str(pack_id),
                        stage="recovery",
                        error="no images delivered",
                    )
                )
                await _safe_edit_status(
                    context.bot,
                    chat_id,
                    status_msg.message_id,
                    text="❌ Ошибка генерации фотосета. Попробуйте еще раз.",
                    reply_markup=_photoset_retry_keyboard(pack_id),
                )
                continue
            pack_in_progress_set.discard(run_id)
            if sent_count > 0:
                pack_delivered_set.add(run_id)
            _bot.store.log_event(
                user_id,
                "pack_generation",
                {"pack_id": pack_id, "pack_title": pack_title, "images_sent": sent_count, "recovered": True},
            )
            try:
                await _safe_edit_status(
                    context.bot, chat_id, status_msg.message_id,
                    text=f"✅ Фотосет «{pack_title}» обработан. Отправлено {sent_count} фото.",
                )
            except Exception:
                pass
            await context.bot.send_message(
                chat_id=chat_id,
                text=_photoset_done_message(include_gift=False),
                reply_markup=_photoset_done_keyboard(),
                parse_mode="HTML",
            )
            logger.info("pack recovery: успешно user=%s pack=%s sent=%s (источник: recovery)", user_id, pack_id, sent_count)
        except Exception as e:
            logger.exception("pack recovery error (user=%s pack=%s): %s", user_id, pack_id, e)
            pack_in_progress_set.discard(run_id)
            asyncio.create_task(
                alert_pack_error(
                    user_id,
                    pack_id=pack_id,
                    pack_title=offer_title,
                    stage="recovery",
                    error=str(e),
                )
            )
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="❌ Ошибка восстановления фотосета. Попробуйте запустить фотосет снова.",
                )
            except Exception:
                pass

async def _pack_fallback_polling(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    chat_id: int,
    pack_id: int,
    offer: dict[str, Any],
    lora_tune_id: int,
    class_name: str,
    expected: int,
    run_id: str,
    known_prompt_ids: set[str] | None = None,
    delay_min: int | None = None,
) -> None:
    """Fallback: опрашиваем prompts фотосета (резерв, если callback/polling не доставили)."""
    if run_id in pack_delivered_set:
        logger.info("pack fallback: уже доставлено run_id=%s", run_id)
        return
    if run_id in _bot._pack_polling_active:
        logger.info("pack fallback: основной polling ещё активен run_id=%s", run_id)
        return
    if run_id in pack_in_progress_set:
        logger.info("pack fallback: доставка уже в процессе run_id=%s", run_id)
        return
    logger.info(
        "pack fallback: запуск run_id=%s tune_id=%s delay_min=%s",
        run_id, lora_tune_id, delay_min,
    )
    settings = _bot.load_settings()
    if not settings.astria_api_key:
        return
    try:
        expected = max(1, int(expected))
        fallback_wait_timeout = max(900, min(3600, _pack_wait_timeout_seconds(expected)))
        _bot._pack_polling_active.add(run_id)
        try:
            urls = await astria_wait_pack_images(
                api_key=settings.astria_api_key,
                tune_id=str(lora_tune_id),
                expected_images=expected,
                known_prompt_ids=known_prompt_ids,
                max_seconds=fallback_wait_timeout,
                poll_seconds=8.0,
            )
        finally:
            _bot._pack_polling_active.discard(run_id)
        if not urls:
            logger.info("pack fallback: prompts не найдены (user=%s pack=%s)", user_id, pack_id)
            return
        if run_id in pack_delivered_set:
            logger.info("pack fallback: уже доставлено во время ожидания run_id=%s", run_id)
            return
        if run_id in pack_in_progress_set:
            logger.info("pack fallback: доставка уже стартовала run_id=%s", run_id)
            return
        pack_title = offer.get("title", "") or str(pack_id)
        sent_count = 0
        pack_in_progress_set.add(run_id)
        try:
            for i, url in enumerate(urls):
                try:
                    out_bytes = await _bot.astria_download_first_image_bytes([url], api_key=settings.astria_api_key, timeout_s=90.0)
                    if out_bytes:
                        bio = io.BytesIO(out_bytes)
                        bio.name = f"pack_{pack_id}_{sent_count + 1}.png"
                        caption = f"Фотосет «{pack_title}» ({sent_count + 1}/{len(urls)})" if sent_count == 0 else ""
                        await _safe_send_document(bot=context.bot, chat_id=chat_id, document=bio, caption=caption)
                        sent_count += 1
                        await asyncio.sleep(0.1)
                except Exception as e:
                    logger.warning("pack fallback: download/send failed %s: %s", url[:50], e)
        finally:
            pack_in_progress_set.discard(run_id)
        if sent_count > 0:
            pack_delivered_set.add(run_id)
        try:
            _bot.store.log_event(user_id, "pack_fallback", {"pack_id": pack_id, "images_sent": sent_count})
        except Exception:
            pass
        if sent_count <= 0:
            asyncio.create_task(
                alert_pack_error(
                    user_id,
                    pack_id=pack_id,
                    pack_title=pack_title,
                    stage="fallback",
                    error="no images delivered",
                )
            )
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ Ошибка генерации фотосета. Попробуйте еще раз.",
                reply_markup=_photoset_retry_keyboard(pack_id),
            )
            return
        await context.bot.send_message(
            chat_id=chat_id,
            text=_photoset_done_message(include_gift=False),
            reply_markup=_photoset_done_keyboard(),
            parse_mode="HTML",
        )
        logger.info(
            "pack fallback: доставлено %s фото (user=%s pack=%s) (источник: fallback)",
            sent_count, user_id, pack_id,
        )
    except Exception as e:
        logger.exception("pack fallback error (user=%s pack=%s): %s", user_id, pack_id, e)
        asyncio.create_task(
            alert_pack_error(
                user_id,
                pack_id=pack_id,
                pack_title=str(offer.get("title") or ""),
                stage="fallback",
                error=str(e),
            )
        )

async def _run_persona_pack_generation(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    pack_id: int,
    offer: dict[str, Any],
    train_file_ids: list[str] | None = None,
    run_id: str | None = None,
    status_message_id: int | None = None,
) -> None:
    run_id = run_id or str(uuid.uuid4())
    gen_lock = await _acquire_user_generation_lock(user_id)
    if gen_lock is None:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Сейчас уже идет генерация. Повторите запуск фотосета через минуту.",
        )
        return

    context.user_data[USERDATA_PERSONA_PACK_IN_PROGRESS] = True
    if status_message_id is not None:
        status_msg = type("StatusMsg", (), {"message_id": status_message_id})()
    else:
        from html import escape
        status_msg = await context.bot.send_message(
            chat_id=chat_id,
            text=f"Приступаю к созданию фотосета <b>«{escape(offer['title'])}»</b>.",
            parse_mode="HTML",
            reply_markup=_persona_training_keyboard(),
        )
    try:
        settings = _bot.load_settings()
        if not settings.astria_api_key:
            await _safe_edit_status(
                context.bot, chat_id, status_msg.message_id,
                text="❌ Сервис генерации не настроен.",
            )
            return
        profile = _bot.store.get_user(user_id)
        class_name = _resolve_pack_class_name(offer, profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER))
        pack = await astria_get_pack(api_key=settings.astria_api_key, pack_id=pack_id)
        configured_expected = int(offer.get("expected_images") or 0)
        expected = configured_expected
        try:
            if isinstance(pack.costs, dict) and pack.costs and class_name not in pack.costs:
                available_classes = [k for k, v in pack.costs.items() if isinstance(v, dict)]
                if available_classes:
                    await _safe_edit_status(
                        context.bot, chat_id, status_msg.message_id,
                        text=(
                            "❌ Этот фотосет не поддерживает тип вашей Персоны.\n"
                            f"Доступные классы: {_pack_classes_text(available_classes)}."
                        ),
                    )
                    return
            class_cost = pack.costs.get(class_name) if isinstance(pack.costs, dict) else None
            if isinstance(class_cost, dict):
                astria_num = int(class_cost.get("num_images") or 0)
                if astria_num > 0:
                    expected = max(configured_expected, astria_num)
        except Exception:
            pass
        expected = max(1, expected)

        title = f"{offer['title']} user:{user_id} ts:{int(time.time())}"
        known_prompt_ids: set[str] = set()
        poll_tune_id: str
        if train_file_ids:
            image_bytes_list: list[bytes] = []
            for fid in train_file_ids[:10]:
                img_bytes = await _safe_get_file_bytes(context.bot, fid)
                image_bytes_list.append(_prepare_image_for_photomaker(img_bytes))
            tune = await astria_create_tune_from_pack(
                api_key=settings.astria_api_key,
                pack_id=pack_id,
                title=title[:120],
                name=class_name,
                image_bytes_list=image_bytes_list,
            )
            poll_tune_id = str(tune.tune_id)
        else:
            active_tune_id: int | None = None
            keep_pending_pack_run = False
            _bot._pack_processing_active.add(run_id)
            if class_name in {"man", "woman"}:
                _bot.store.set_pending_pack_run(
                    user_id=user_id,
                    pack_id=pack_id,
                    chat_id=chat_id,
                    run_id=run_id,
                    expected=expected,
                    class_name=class_name,
                    offer_title=offer.get("title", "") or "",
                )
            try:
                if class_name in {"man", "woman"}:
                    active_tune_id, reason = await _ensure_pack_lora_tune_id(
                        context=context,
                        chat_id=chat_id,
                        user_id=user_id,
                        class_name=class_name,
                        status_message_id=status_msg.message_id,
                    )
                    if reason == "need_upload" or active_tune_id is None:
                        if reason == "pending":
                            keep_pending_pack_run = True
                            _bot._pack_processing_active.discard(run_id)
                            await _safe_edit_status(
                                context.bot,
                                chat_id,
                                status_msg.message_id,
                                text=PHOTOSET_PROGRESS_ALERT,
                                reply_markup=_persona_training_keyboard(),
                            )
                            return
                        _bot._pack_processing_active.discard(run_id)
                        await _fallback_to_pack_photo_upload(
                            context=context,
                            chat_id=chat_id,
                            user_id=user_id,
                            pack_id=pack_id,
                            status_message_id=status_msg.message_id,
                        )
                        return
                else:
                    lora_tune_id_raw = profile.astria_lora_tune_id
                    if not lora_tune_id_raw:
                        _bot._pack_processing_active.discard(run_id)
                        await _safe_edit_status(
                            context.bot, chat_id, status_msg.message_id,
                            text="❌ У вас еще нет обученной Персоны. Сначала обучите Персону (10 фото).",
                        )
                        return
                    try:
                        active_tune_id = int(str(lora_tune_id_raw))
                    except ValueError:
                        _bot._pack_processing_active.discard(run_id)
                        await _safe_edit_status(
                            context.bot, chat_id, status_msg.message_id,
                            text="❌ Не удалось определить ID вашей Персоны. Напишите в поддержку.",
                        )
                        return
                assert active_tune_id is not None
                known_prompt_ids: set[str] | None = None
                try:
                    known_prompt_ids = await astria_get_tune_prompt_ids(
                        api_key=settings.astria_api_key,
                        tune_id=str(active_tune_id),
                    )
                except Exception as e:
                    logger.warning("Не удалось получить существующие prompts для tune %s: %s", active_tune_id, e)
                # Защита от дубля: recovery мог уже создать промпты
                if run_id in _bot._pack_polling_active or run_id in pack_delivered_set:
                    logger.info("pack: конфликт перед созданием промптов run_id=%s (polling=%s delivered=%s), пропускаем",
                                run_id, run_id in _bot._pack_polling_active, run_id in pack_delivered_set)
                    _bot._pack_processing_active.discard(run_id)
                    return
                callback_url = build_pack_callback_url(user_id, chat_id, pack_id, run_id)
                tune = await astria_create_tune_from_pack(
                    api_key=settings.astria_api_key,
                    pack_id=pack_id,
                    title=title[:120],
                    name=class_name,
                    tune_ids=[active_tune_id],
                    prompts_callback=callback_url or None,
                )
                # Astria может вернуть новый tune_id для generation — prompts там. Иначе используем active_tune_id.
                poll_tune_id = str(tune.tune_id) if tune.tune_id else str(active_tune_id)
                if str(tune.tune_id) != str(active_tune_id):
                    logger.info(
                        "pack: tune_id из API (%s) != active_tune_id (%s), опрашиваем tune_id из API",
                        tune.tune_id, active_tune_id,
                    )
            finally:
                if class_name in {"man", "woman"} and not keep_pending_pack_run:
                    _bot.store.clear_pending_pack_run(user_id)
        logger.info(
            "pack: astria create_tune_from_pack response: tune_id=%s status=%s raw=%s",
            tune.tune_id,
            tune.status,
            tune.raw,
        )
        app = getattr(context, "application", None)
        job_queue = app.job_queue if app else None
        if job_queue:
            def _make_fallback_job(delay_min: int):
                u, c, p, o, lid, cn, exp, rid, kp = (
                    user_id,
                    chat_id,
                    pack_id,
                    offer,
                    int(poll_tune_id),
                    class_name,
                    expected,
                    run_id,
                    set(known_prompt_ids or set()),
                )
                async def job(ctx):
                    await _pack_fallback_polling(
                        context=ctx,
                        user_id=u,
                        chat_id=c,
                        pack_id=p,
                        offer=o,
                        lora_tune_id=lid,
                        class_name=cn,
                        expected=exp,
                        run_id=rid,
                        known_prompt_ids=kp,
                        delay_min=delay_min,
                    )
                return job
            fallback_delay_min = 15
            job_queue.run_once(
                _make_fallback_job(fallback_delay_min),
                when=fallback_delay_min * 60,
                name=f"pack_fallback_{fallback_delay_min}min_{user_id}_{pack_id}_{run_id}",
            )
            logger.info("pack: fallback polling запланирован через %s мин (резерв)", fallback_delay_min)

        # Защита от дублирования: recovery мог уже начать polling
        if run_id in _bot._pack_polling_active:
            _bot._pack_processing_active.discard(run_id)
            logger.info("pack: polling уже запущен (recovery?) для run_id=%s, пропускаем", run_id)
            return
        if run_id in pack_delivered_set:
            _bot._pack_processing_active.discard(run_id)
            logger.info("pack: уже доставлено для run_id=%s, пропускаем", run_id)
            return

        wait_timeout = _pack_wait_timeout_seconds(expected)
        logger.info(
            "pack: начинаю polling tune_id=%s run_id=%s expected=%s timeout=%ss",
            poll_tune_id, run_id, expected, wait_timeout,
        )
        _bot._pack_polling_active.add(run_id)
        try:
            urls = await astria_wait_pack_images(
                api_key=settings.astria_api_key,
                tune_id=poll_tune_id,
                expected_images=expected,
                known_prompt_ids=known_prompt_ids,
                max_seconds=wait_timeout,
                poll_seconds=8.0,
            )
        finally:
            _bot._pack_polling_active.discard(run_id)
            _bot._pack_processing_active.discard(run_id)
        if not urls:
            await _safe_edit_status(
                context.bot, chat_id, status_msg.message_id,
                text="❌ Фотосет завершился без изображений. Попробуйте еще раз.",
            )
            return

        if run_id in pack_delivered_set:
            logger.info("pack: уже доставлено через callback run_id=%s, пропускаем отправку", run_id)
            return
        if run_id in pack_in_progress_set:
            logger.info("pack: доставка уже идёт (recovery?) run_id=%s, пропускаем", run_id)
            return

        logger.info("pack: доставлено через polling run_id=%s urls=%s", run_id, len(urls) if urls else 0)
        pack_in_progress_set.add(run_id)

        # Сохраняем tune_id как Persona, если у пользователя её ещё нет (пак создал tune из загруженных фото)
        if train_file_ids and tune.tune_id and not profile.astria_lora_tune_id:
            try:
                if _use_unified_pack_persona_flow():
                    logger.info(
                        "pack: в unified-flow не сохраняем tune_id %s как Persona (user=%s class=%s)",
                        tune.tune_id,
                        user_id,
                        class_name,
                    )
                else:
                    _bot.store.set_astria_lora_tune(user_id=user_id, tune_id=tune.tune_id, class_name=class_name)
                    logger.info("pack: сохранён tune_id %s как Persona для user %s (class=%s)", tune.tune_id, user_id, class_name)
            except Exception as e:
                logger.warning("pack: не удалось сохранить tune_id для user %s: %s", user_id, e)

        total = len(urls)
        sent_count = 0
        pack_download_timeout = 90.0
        for i, url in enumerate(urls, start=1):
            out_bytes = None
            for attempt in range(3):
                try:
                    out_bytes = await _bot.astria_download_first_image_bytes(
                        [url],
                        api_key=settings.astria_api_key,
                        timeout_s=pack_download_timeout,
                    )
                    break
                except (_bot.AstriaError, requests.RequestException) as e:
                    if attempt < 2:
                        logger.warning(
                            "Pack image %s/%s download attempt %s failed: %s, retrying...",
                            i,
                            total,
                            attempt + 1,
                            e,
                        )
                        await asyncio.sleep(2.0)
                    else:
                        logger.warning(
                            "Pack image %s/%s download failed after 3 attempts, skipping: %s",
                            i,
                            total,
                            e,
                        )
            if out_bytes:
                bio = io.BytesIO(out_bytes)
                bio.name = f"pack_{pack_id}_{sent_count + 1}.png"
                caption = f"Фотосет «{offer['title']}» ({sent_count + 1}/{total})" if sent_count == 0 else ""
                try:
                    await _safe_send_document(
                        bot=context.bot,
                        chat_id=chat_id,
                        document=bio,
                        caption=caption,
                    )
                    sent_count += 1
                except Exception as e:
                    logger.warning("Pack image %s/%s send failed, skipping: %s", i, total, e)
                await asyncio.sleep(0.1)

        if sent_count <= 0:
            asyncio.create_task(
                alert_pack_error(
                    user_id,
                    pack_id=pack_id,
                    pack_title=str(offer.get("title") or ""),
                    stage="generation",
                    error="no images delivered",
                )
            )
            await _safe_edit_status(
                context.bot,
                chat_id,
                status_msg.message_id,
                text="❌ Ошибка генерации фотосета. Попробуйте еще раз.",
                reply_markup=_photoset_retry_keyboard(pack_id),
            )
            return

        if sent_count > 0:
            pack_delivered_set.add(run_id)
        _bot.store.log_event(
            user_id,
            "pack_generation",
            {"pack_id": pack_id, "pack_title": offer["title"], "images_sent": sent_count},
        )
        await _safe_edit_status(
            context.bot, chat_id, status_msg.message_id,
            text=f"✅ Фотосет «{offer['title']}» обработан. Отправлено {sent_count} фото.",
        )
        include_gift = bool(context.user_data.pop(USERDATA_PERSONA_PACK_GIFT_APPLIED, False))
        await context.bot.send_message(
            chat_id=chat_id,
            text=_photoset_done_message(include_gift=include_gift),
            reply_markup=_photoset_done_keyboard(),
            parse_mode="HTML",
        )
    except _bot.AstriaError as e:
        logger.exception("Ошибка генерации фотосета %s (Astria): %s", pack_id, e)
        asyncio.create_task(
            alert_pack_error(
                user_id,
                pack_id=pack_id,
                pack_title=str(offer.get("title") or ""),
                stage="generation",
                error=str(e),
            )
        )
        try:
            await _safe_edit_status(
                context.bot, chat_id, status_msg.message_id,
                text="❌ Ошибка генерации фотосета. Попробуйте еще раз.",
                reply_markup=_photoset_retry_keyboard(pack_id),
            )
        except Exception:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ Ошибка генерации фотосета. Попробуйте еще раз.",
                reply_markup=_photoset_retry_keyboard(pack_id),
            )
    except Exception as e:
        logger.exception("Ошибка генерации фотосета %s: %s", pack_id, e)
        asyncio.create_task(
            alert_pack_error(
                user_id,
                pack_id=pack_id,
                pack_title=str(offer.get("title") or ""),
                stage="generation",
                error=str(e),
            )
        )
        try:
            await _safe_edit_status(
                context.bot, chat_id, status_msg.message_id,
                text="❌ Ошибка генерации фотосета. Попробуйте еще раз.",
                reply_markup=_photoset_retry_keyboard(pack_id),
            )
        except Exception:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ Ошибка генерации фотосета. Попробуйте еще раз.",
                reply_markup=_photoset_retry_keyboard(pack_id),
            )
    finally:
        pack_in_progress_set.discard(run_id)
        context.user_data[USERDATA_PERSONA_PACK_IN_PROGRESS] = False
        gen_lock.release()

async def _poll_persona_pack_payment_and_run(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    payment_id: str,
    user_id: int,
    chat_id: int,
    pack_id: int,
    offer: dict[str, Any],
    timeout_seconds: int = 900,
    poll_interval: int = 5,
) -> None:
    started = time.monotonic()
    while True:
        if time.monotonic() - started > timeout_seconds:
            logger.info("Таймаут поллинга оплаты пака %s", payment_id)
            return
        status = await asyncio.to_thread(get_payment_status, payment_id)
        if status == "succeeded":
            if _bot.store.is_payment_processed(payment_id):
                return
            amount_rub = apply_test_amount(float(offer["price_rub"]))
            expected_images = int(offer.get("expected_images") or 0)
            try:
                payment_log_id = _bot.store.log_payment(
                    user_id=user_id,
                    payment_id=payment_id,
                    payment_method="yookassa",
                    product_type="persona_pack",
                    credits=max(1, expected_images),
                    amount_rub=amount_rub,
                )
            except Exception as e:
                logger.exception("Не удалось записать платеж %s: %s", payment_id, e)
                await asyncio.sleep(poll_interval)
                continue
            if payment_log_id is None:
                logger.info("Платёж %s уже обработан параллельным обработчиком", payment_id)
                return
            from html import escape
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Оплата получена ✅\n\nПриступаю к созданию фотосета <b>«{escape(offer['title'])}»</b>",
                parse_mode="HTML",
                reply_markup=_persona_training_keyboard(),
            )
            await _run_persona_pack_generation(
                context=context,
                chat_id=chat_id,
                user_id=user_id,
                pack_id=pack_id,
                offer=offer,
                run_id=payment_id,
            )
            return
        if status == "canceled":
            await context.bot.send_message(chat_id=chat_id, text="❌ Платеж отменен или истек.")
            return
        await asyncio.sleep(poll_interval)

async def handle_persona_packs_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "nav_packs")
    except Exception:
        pass
    offers = _pack_offers()
    if not offers:
        await query.edit_message_text(
            "Фотосеты пока не настроены. Добавьте PRISMALAB_ASTRIA_PACK_OFFERS в .env dev-бота.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="pl_persona_back")]]),
        )
        return
    user_id = int(query.from_user.id) if query.from_user else 0
    profile = _bot.store.get_user(user_id)
    if not (profile.astria_lora_tune_id or profile.astria_lora_pack_tune_id) and not _dev_pack_train_from_images():
        await query.edit_message_text(
            "Фотосеты доступны после обучения Персоны.\n\nСначала оплатите «Персона» и загрузите 10 фото.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="pl_persona_back")]]),
        )
        return
    await query.edit_message_text(
        _bot.PERSONA_PACKS_MESSAGE,
        reply_markup=_persona_packs_keyboard(),
        parse_mode="HTML",
    )

async def handle_persona_pack_buy_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    user_id = int(query.from_user.id) if query.from_user else 0
    logger.info("persona_pack_buy click: user_id=%s data=%s", user_id, query.data)
    try:
        _bot.store.log_event(user_id, "pack_buy_init", {"raw_data": query.data})
    except Exception:
        pass

    try:
        parts = query.data.split(":", 1)
        if len(parts) != 2:
            await query.edit_message_text("Некорректная команда фотосета.", reply_markup=_persona_packs_keyboard())
            return
        pack_id = int(parts[1])
    except Exception:
        await query.edit_message_text("Некорректный ID фотосета.", reply_markup=_persona_packs_keyboard())
        return

    offer = _find_pack_offer(pack_id)
    logger.info("persona_pack_buy: parsed pack_id=%s offer_found=%s", pack_id, bool(offer))
    if not offer:
        await query.edit_message_text("Фотосет не найден в конфиге.", reply_markup=_persona_packs_keyboard())
        return
    if not _bot.use_yookassa():
        logger.info("persona_pack_buy: yookassa disabled")
        await query.edit_message_text(
            "Для фотосетов в dev включена только оплата по ссылке.",
            reply_markup=_persona_packs_keyboard(),
        )
        return

    chat_id = query.message.chat_id if query.message else 0
    logger.info("persona_pack_buy: loading profile user_id=%s", user_id)
    profile = _bot.store.get_user(user_id)
    logger.info("persona_pack_buy: profile loaded lora_tune_id=%s", getattr(profile, "astria_lora_tune_id", None))
    if not (profile.astria_lora_tune_id or profile.astria_lora_pack_tune_id) and not _dev_pack_train_from_images():
        await query.edit_message_text("Сначала обучите Персону (10 фото).", reply_markup=_persona_app_keyboard())
        return

    settings = _bot.load_settings()
    logger.info("persona_pack_buy: settings loaded astria_key=%s", bool(settings.astria_api_key))
    if not settings.astria_api_key:
        await query.edit_message_text("❌ Сервис генерации не настроен.", reply_markup=_persona_packs_keyboard())
        return

    logger.info("persona_pack_buy: edit progress message")
    await query.edit_message_text("⏳ Проверяю фотосет и создаю ссылку оплаты…")

    class_name = _resolve_pack_class_name(offer, profile.subject_gender or context.user_data.get(USERDATA_SUBJECT_GENDER))
    logger.info("persona_pack_buy: fetching pack from astria pack_id=%s class=%s", pack_id, class_name)
    try:
        pack = await asyncio.wait_for(
            astria_get_pack(api_key=settings.astria_api_key, pack_id=pack_id),
            timeout=35.0,
        )
        logger.info("persona_pack_buy: astria pack fetched pack_id=%s", pack_id)
    except Exception as e:
        logger.warning("Не удалось получить pack %s перед оплатой: %s", pack_id, e)
        await query.edit_message_text(
            "❌ Не удалось проверить конфигурацию фотосета. Попробуйте еще раз.",
            reply_markup=_persona_packs_keyboard(),
        )
        return
    if isinstance(pack.costs, dict) and pack.costs and class_name not in pack.costs:
        available_classes = [k for k, v in pack.costs.items() if isinstance(v, dict)]
        available_text = _pack_classes_text(available_classes) if available_classes else "не определены"
        await query.edit_message_text(
            (
                "Этот фотосет не подходит для вашей Персоны.\n"
                f"Доступные классы: {available_text}."
            ),
            reply_markup=_persona_packs_keyboard(),
        )
        return

    expected_images_for_payment = max(1, int(offer.get("expected_images") or 1))
    class_cost = pack.costs.get(class_name) if isinstance(pack.costs, dict) else None
    if isinstance(class_cost, dict):
        try:
            expected_images_for_payment = max(1, int(class_cost.get("num_images") or expected_images_for_payment))
        except Exception:
            pass
    pack_cost_field, pack_cost_value = _extract_pack_cost_info(class_cost)

    if _dev_pack_train_from_images():
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
        return

    if _dev_skip_pack_payment():
        logger.info("persona_pack_buy: dev skip payment enabled, run pack directly user_id=%s pack_id=%s", user_id, pack_id)
        await query.edit_message_text(
            "🧪 Dev-режим: оплату пропускаю, запускаю фотосет сразу.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Назад", callback_data="pl_persona_packs")]]),
        )
        await _run_persona_pack_generation(
            context=context,
            chat_id=chat_id,
            user_id=user_id,
            pack_id=pack_id,
            offer=offer,
            run_id=f"dev_{user_id}_{pack_id}_{int(time.time())}",
        )
        return

    amount = apply_test_amount(float(offer["price_rub"]))
    me = await context.bot.get_me()
    return_url = f"https://t.me/{me.username}" if me and me.username else None
    logger.info("persona_pack_buy: creating payment amount=%s", amount)
    try:
        url, payment_id = await asyncio.wait_for(
            asyncio.to_thread(
                create_payment,
                amount_rub=amount,
                description=f"Фотосет: {offer['title']}",
                metadata={
                    "user_id": str(user_id),
                    "chat_id": str(chat_id),
                    "credits": str(expected_images_for_payment),
                    "product_type": "persona_pack",
                    "pack_id": str(pack_id),
                    "pack_title": str(offer.get("title") or "")[:100],
                    "pack_class": class_name[:24],
                    "pack_num_images": str(expected_images_for_payment),
                    "pack_cost_field": pack_cost_field[:24],
                    "pack_cost_value": pack_cost_value[:64],
                },
                return_url=return_url,
            ),
            timeout=35.0,
        )
        logger.info("persona_pack_buy: payment created payment_id=%s has_url=%s", payment_id, bool(url))
    except asyncio.TimeoutError:
        await query.edit_message_text(
            "❌ Таймаут создания платежа. Попробуйте еще раз.",
            reply_markup=_persona_packs_keyboard(),
        )
        return
    except Exception as e:
        logger.warning("Ошибка create_payment для pack %s: %s", pack_id, e)
        asyncio.create_task(alert_payment_error(user_id, "persona_pack", str(e)))
        await query.edit_message_text(
            "❌ Ошибка оплаты. Попробуйте еще раз.",
            reply_markup=_persona_packs_keyboard(),
        )
        return
    if not (url and payment_id):
        err = str(payment_id or "unknown")
        asyncio.create_task(alert_payment_error(user_id, "persona_pack", err))
        if "network error" in err.lower() or "readtimeout" in err.lower() or "connecttimeout" in err.lower():
            await query.edit_message_text(
                "❌ Не удалось связаться с платёжной системой (сеть/таймаут). Попробуйте еще раз через 1-2 минуты.",
                reply_markup=_persona_packs_keyboard(),
            )
        else:
            await query.edit_message_text("❌ Ошибка оплаты. Попробуйте еще раз.", reply_markup=_persona_packs_keyboard())
        return

    context.user_data[USERDATA_PERSONA_SELECTED_PACK_ID] = pack_id
    await query.edit_message_text(
        (
            f"<b>{offer['title']}</b>\n"
            f"Цена: <b>{int(amount)} ₽</b>\n\n"
            "Оплатите, после этого бот автоматически запустит фотосет и пришлет готовые фото."
        ),
        reply_markup=_payment_yookassa_keyboard(url, "pl_persona_packs"),
        parse_mode="HTML",
    )
    asyncio.create_task(
        _poll_persona_pack_payment_and_run(
            context=context,
            payment_id=payment_id,
            user_id=user_id,
            chat_id=chat_id,
            pack_id=pack_id,
            offer=offer,
        )
    )

async def handle_persona_pack_retry_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()

    try:
        _, pack_id_str = query.data.split(":", 1)
        pack_id = int(pack_id_str)
    except Exception:
        await query.edit_message_text("Некорректный ID фотосета.", reply_markup=_persona_packs_keyboard())
        return

    offer = _find_pack_offer(pack_id)
    if not offer:
        await query.edit_message_text("Фотосет не найден.", reply_markup=_persona_packs_keyboard())
        return

    user_id = int(query.from_user.id) if query.from_user else 0
    try:
        _bot.store.log_event(user_id, "pack_retry", {"pack_id": pack_id})
    except Exception:
        pass
    chat_id = query.message.chat_id if query.message else 0
    profile = _bot.store.get_user(user_id)
    if not (profile.astria_lora_tune_id or profile.astria_lora_pack_tune_id):
        await query.edit_message_text("Сначала обучите Персону (10 фото).", reply_markup=_persona_app_keyboard())
        return

    await query.edit_message_text(
        f"Понял. Перезапускаю фотосет «{offer['title']}».",
        reply_markup=_persona_training_keyboard(),
    )
    context.application.create_task(
        _run_persona_pack_generation(
            context=context,
            chat_id=chat_id,
            user_id=user_id,
            pack_id=pack_id,
            offer=offer,
            run_id=f"retry_{user_id}_{pack_id}_{int(time.time())}",
        )
    )
