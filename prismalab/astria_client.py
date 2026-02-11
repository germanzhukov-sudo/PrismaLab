from __future__ import annotations

import asyncio
import os
import time
import json
from dataclasses import dataclass
from typing import Any

import requests


class AstriaError(RuntimeError):
    pass


class AstriaValidationError(AstriaError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


@dataclass(frozen=True)
class AstriaPromptResult:
    prompt_id: str
    images: list[str]
    raw: dict[str, Any]


@dataclass(frozen=True)
class AstriaTuneResult:
    tune_id: str
    status: str
    raw: dict[str, Any]


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _timeout_s(default: float) -> float:
    raw = (os.getenv("PRISMALAB_ASTRIA_HTTP_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except Exception:
        return default
    return max(5.0, min(120.0, v))


def _extract_image_urls(prompt_obj: Any) -> list[str]:
    """
    Astria docs иногда показывают prompt без images в примерах.
    Делаем максимально терпимую распаковку: images может быть списком строк или объектов с url.
    """
    if not isinstance(prompt_obj, dict):
        return []
    imgs = prompt_obj.get("images")
    candidates: list[str] = []
    if isinstance(imgs, list):
        for it in imgs:
            if isinstance(it, str) and it:
                candidates.append(it)
            elif isinstance(it, dict):
                u = it.get("url") or it.get("image_url") or it.get("src")
                if isinstance(u, str) and u:
                    candidates.append(u)

    # fallback: иногда кладут одиночный URL результата
    # ВАЖНО: НЕ используем поле "url" (обычно это endpoint JSON промпта, он требует auth и не является картинкой).
    for k in ("image_url", "output_url"):
        v = prompt_obj.get(k)
        if isinstance(v, str) and v.startswith("http"):
            candidates.append(v)

    # Фильтр должен быть мягким: Astria часто отдаёт ссылки без расширения (например mp.astria.ai/xxxx).
    # Исключаем только явные JSON endpoints/страницы API.
    filtered: list[str] = []
    for u in candidates:
        lu = u.lower()
        if lu.endswith(".json"):
            continue
        if "api.astria.ai" in lu and "/prompts/" in lu:
            continue
        filtered.append(u)

    # uniq preserve order
    seen = set()
    uniq: list[str] = []
    for u in filtered:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    return uniq


def _post_prompt(
    *,
    api_key: str,
    tune_id: str,
    text: str,
    negative_prompt: str | None,
    input_image_bytes: bytes | None,
    cfg_scale: float | None,
    steps: int | None,
    denoising_strength: float | None,
    super_resolution: bool | None,
    hires_fix: bool | None,
    face_correct: bool | None,
    face_swap: bool | None,
    inpaint_faces: bool | None = None,
    style: str | None = None,
    color_grading: str | None = None,
    film_grain: bool | None = None,
    seed: int | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    if not api_key:
        raise AstriaError("PRISMALAB_ASTRIA_API_KEY не задан")
    # tune_id может быть базовой моделью из галереи (для FaceID) или LoRA tune (для LoRA)
    if not tune_id:
        raise AstriaError("tune_id обязателен (базовая модель из галереи или LoRA tune)")

    url = f"https://api.astria.ai/tunes/{tune_id}/prompts"
    data: dict[str, Any] = {"prompt[text]": text}
    if negative_prompt:
        data["prompt[negative_prompt]"] = negative_prompt
    data["prompt[num_images]"] = "1"

    if cfg_scale is not None:
        data["prompt[cfg_scale]"] = str(float(cfg_scale))
    if steps is not None:
        data["prompt[steps]"] = str(int(steps))
    if denoising_strength is not None:
        data["prompt[denoising_strength]"] = str(float(denoising_strength))
    if super_resolution is not None:
        data["prompt[super_resolution]"] = "true" if super_resolution else "false"
    if hires_fix is not None:
        data["prompt[hires_fix]"] = "true" if hires_fix else "false"
    if face_correct is not None:
        data["prompt[face_correct]"] = "true" if face_correct else "false"
    if face_swap is not None:
        data["prompt[face_swap]"] = "true" if face_swap else "false"
    if inpaint_faces is not None:
        data["prompt[inpaint_faces]"] = "true" if inpaint_faces else "false"
    if style is not None:
        data["prompt[style]"] = style
    if color_grading is not None:
        data["prompt[color_grading]"] = color_grading
    if film_grain is not None:
        data["prompt[film_grain]"] = "true" if film_grain else "false"
    if seed is not None:
        data["prompt[seed]"] = str(int(seed))

    files = None
    if input_image_bytes:
        files = {"prompt[input_image]": ("input.png", input_image_bytes, "image/png")}

    # Используем более длинный таймаут для POST - генерация может занимать время
    # requests.post timeout - это общий таймаут на соединение + чтение ответа
    r = requests.post(url, headers=_headers(api_key), data=data, files=files, timeout=(10.0, timeout_s))
    if r.status_code >= 400:
        # Частый кейс: 422 с валидацией параметров (например, для Flux нельзя negative_prompt и cfg_scale < 5)
        if r.status_code == 422:
            details: dict[str, Any] | None = None
            try:
                details = r.json()
            except Exception:
                try:
                    details = json.loads(r.text)
                except Exception:
                    details = None
            raise AstriaValidationError(f"Astria HTTP 422: {r.text}", details=details if isinstance(details, dict) else None)
        raise AstriaError(f"Astria HTTP {r.status_code}: {r.text}")
    try:
        return r.json()
    except Exception as e:
        raise AstriaError("Astria вернул не-JSON ответ") from e


def _get_prompt(*, api_key: str, tune_id: str, prompt_id: str, timeout_s: float) -> dict[str, Any]:
    url = f"https://api.astria.ai/tunes/{tune_id}/prompts/{prompt_id}"
    # Используем более длинный таймаут для GET - polling может быть медленным
    # requests.get timeout - это общий таймаут на соединение + чтение ответа
    r = requests.get(url, headers=_headers(api_key), timeout=(10.0, timeout_s))
    if r.status_code >= 400:
        raise AstriaError(f"Astria HTTP {r.status_code}: {r.text}")
    try:
        return r.json()
    except Exception as e:
        raise AstriaError("Astria вернул не-JSON ответ") from e


async def run_prompt_and_wait(
    *,
    api_key: str,
    tune_id: str,
    text: str,
    negative_prompt: str | None = None,
    input_image_bytes: bytes | None = None,
    cfg_scale: float | None = None,
    steps: int | None = None,
    denoising_strength: float | None = None,
    super_resolution: bool | None = True,
    hires_fix: bool | None = True,
    face_correct: bool | None = True,
    face_swap: bool | None = False,
    inpaint_faces: bool | None = None,
    style: str | None = None,
    color_grading: str | None = None,
    film_grain: bool | None = None,
    seed: int | None = None,
    max_seconds: int = 300,
    poll_seconds: float = 2.0,
) -> AstriaPromptResult:
    # Увеличиваем таймаут для POST запроса - генерация может занимать время
    timeout_s = _timeout_s(90.0)  # Увеличено до 90 секунд для медленных запросов
    # Один авто-ретрай для частых ограничений конкретных бэкендов (Flux).
    attempt_cfg = cfg_scale
    attempt_neg = negative_prompt
    for attempt in range(2):
        try:
            created = await asyncio.to_thread(
                _post_prompt,
                api_key=api_key,
                tune_id=tune_id,
                text=text,
                negative_prompt=attempt_neg,
                input_image_bytes=input_image_bytes,
                cfg_scale=attempt_cfg,
                steps=steps,
                denoising_strength=denoising_strength,
                super_resolution=super_resolution,
                hires_fix=hires_fix,
                face_correct=face_correct,
                face_swap=face_swap,
                inpaint_faces=inpaint_faces,
                style=style,
                color_grading=color_grading,
                film_grain=film_grain,
                seed=seed,
                timeout_s=timeout_s,
            )
            break
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            # Retry при таймауте POST запроса
            import logging
            logger = logging.getLogger("prismalab")
            if attempt == 0:
                logger.warning(f"Astria POST запрос - таймаут (попытка {attempt + 1}/2), жду 5с и повторяю...")
                await asyncio.sleep(5.0)
                continue
            else:
                logger.error(f"Astria POST запрос - таймаут после retry: {e}")
                raise AstriaError(f"Astria API не отвечает (таймаут при создании prompt): {e}") from e
        except AstriaValidationError as ve:
            if attempt == 1:
                raise
            d = getattr(ve, "details", {}) or {}
            # Приводим ошибки к тексту
            cfg_msgs = d.get("cfg_scale")
            neg_msgs = d.get("negative_prompt")
            cfg_s = " ".join(cfg_msgs) if isinstance(cfg_msgs, list) else str(cfg_msgs or "")
            neg_s = " ".join(neg_msgs) if isinstance(neg_msgs, list) else str(neg_msgs or "")

            # Fix 1: cfg_scale must be less than 5
            if "less than 5" in cfg_s.lower():
                attempt_cfg = 4.5 if attempt_cfg is None else min(float(attempt_cfg), 4.5)
            # Fix 2: negative_prompt not supported on Flux
            if "not supported" in neg_s.lower() and "flux" in neg_s.lower():
                attempt_neg = None
            # Если ничего не поправили — не мучаемся
            if attempt_cfg == cfg_scale and attempt_neg == negative_prompt:
                raise
    else:
        # for mypy
        raise AstriaError("Astria prompt create failed")

    prompt_id = str(created.get("id") or "")
    if not prompt_id:
        raise AstriaError(f"Неожиданный ответ Astria (нет id): {created}")

    import logging
    logger = logging.getLogger("prismalab")
    
    deadline = time.monotonic() + int(max_seconds)
    last = created
    poll_count = 0
    logger.info(f"Astria prompt {prompt_id} создан, начинаю опрос статуса (таймаут: {max_seconds}с)")
    
    while True:
        elapsed = time.monotonic() - (deadline - int(max_seconds))
        if time.monotonic() > deadline:
            logger.error(f"Astria prompt {prompt_id} - таймаут после {max_seconds}с, последний статус: {last.get('status')}")
            raise AstriaError("Таймаут ожидания результата Astria")

        poll_count += 1
        if poll_count % 5 == 0:  # Логируем каждые 5 попыток
            logger.info(f"Astria prompt {prompt_id} - опрос #{poll_count}, прошло {elapsed:.0f}с, статус: {last.get('status')}")
        
        # Таймаут для GET при polling: не меньше 45с, иначе при PRISMALAB_ASTRIA_HTTP_TIMEOUT_SECONDS=10 опрос обрывается до появления images
        polling_timeout = max(45.0, _timeout_s(60.0))
        try:
            last = await asyncio.to_thread(_get_prompt, api_key=api_key, tune_id=tune_id, prompt_id=prompt_id, timeout_s=polling_timeout)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            # Retry при таймауте - Astria может быть медленным
            logger.warning(f"Astria prompt {prompt_id} - таймаут при polling (попытка {poll_count}), жду 5с и повторяю...")
            await asyncio.sleep(5.0)
            try:
                last = await asyncio.to_thread(_get_prompt, api_key=api_key, tune_id=tune_id, prompt_id=prompt_id, timeout_s=polling_timeout)
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e2:
                raise AstriaError(f"Astria API не отвечает (таймаут при polling): {e2}") from e2
        
        # Проверяем статус и ошибки
        status = str(last.get("status") or "").lower()
        error_msg = last.get("error") or last.get("user_error") or last.get("message")
        
        # Если есть ошибка - сразу падаем
        if error_msg:
            logger.error(f"Astria prompt {prompt_id} - ошибка: {error_msg}")
            raise AstriaError(f"Astria вернул ошибку: {error_msg}")
        
        # Проверяем статусы ошибок
        if status in {"failed", "error", "cancelled"}:
            error_text = error_msg or str(last)
            logger.error(f"Astria prompt {prompt_id} - статус ошибки: {status}, детали: {error_text}")
            raise AstriaError(f"Astria prompt завершился с ошибкой (status={status}): {error_text}")
        
        images = _extract_image_urls(last)
        
        # Логируем полный ответ каждые 10 попыток для отладки
        if poll_count % 10 == 0:
            logger.info(f"Astria prompt {prompt_id} - полный ответ API (ключи: {list(last.keys())}): {last}")
            logger.info(f"Astria prompt {prompt_id} - статус: '{status}', найденные images: {images}")

        # В реальных ответах Astria у completed prompt обычно появляются images.
        if images:
            logger.info(f"Astria prompt {prompt_id} - готов! Найдено {len(images)} изображений за {elapsed:.0f}с")
            return AstriaPromptResult(prompt_id=prompt_id, images=images, raw=last)

        # fallback: некоторые поля времени
        done_at = last.get("trained_at") or last.get("completed_at") or last.get("finished_at")
        if done_at:
            # если “готово”, но images не нашли — всё равно считаем ошибкой
            raise AstriaError(f"Astria завершил prompt, но не вернул images: {last}")

        await asyncio.sleep(poll_seconds)


def _download_bytes(url: str, timeout_s: float, api_key: str | None = None) -> bytes:
    # 1) пробуем как есть (часто URLs публичные/пресайненные)
    # Таймаут для скачивания: 10 секунд на соединение, timeout_s на чтение
    r = requests.get(url, timeout=(10.0, timeout_s))
    if r.status_code == 401 and api_key:
        # 2) иногда storage защищён и требует Bearer token
        r = requests.get(url, headers=_headers(api_key), timeout=(10.0, timeout_s))
    if r.status_code >= 400:
        raise AstriaError(f"Не удалось скачать результат Astria (HTTP {r.status_code})")
    ct = (r.headers.get("Content-Type") or "").lower()
    # Иногда URL указывает на JSON endpoint — тогда извлекаем images и скачиваем уже их.
    if "application/json" in ct:
        try:
            obj = r.json()
        except Exception:
            raise AstriaError("Astria вернул JSON, но не удалось распарсить")
        urls = _extract_image_urls(obj)
        if not urls:
            raise AstriaError("Astria вернул JSON без ссылок на изображения")
        # скачиваем первую картинку из найденных
        return _download_bytes(urls[0], timeout_s, api_key=api_key)
    return r.content


async def download_first_image_bytes(urls: list[str], *, api_key: str | None = None) -> bytes:
    if not urls:
        raise AstriaError("Astria не вернул URLs изображений")
    return await asyncio.to_thread(_download_bytes, urls[0], _timeout_s(40.0), api_key)


def _create_faceid_tune(
    *,
    api_key: str,
    name: str,
    title: str,
    image_bytes: bytes,
    base_tune_id: str | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    if not api_key:
        raise AstriaError("PRISMALAB_ASTRIA_API_KEY не задан")
    url = "https://api.astria.ai/tunes"
    
    # Определяем тип изображения по первым байтам
    import imghdr
    image_type = "png"  # по умолчанию
    if image_bytes:
        detected = imghdr.what(None, h=image_bytes[:32])
        if detected:
            image_type = detected
        elif image_bytes[:2] == b"\xff\xd8":
            image_type = "jpeg"
        elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            image_type = "png"
    
    # Используем стандартный способ requests для multipart/form-data
    # Это автоматически правильно формирует boundary и кодировку
    files = {
        "tune[images][]": (f"face.{image_type}", image_bytes, f"image/{image_type}")
    }
    
    data = {
        "tune[name]": str(name),
        "tune[title]": str(title),
        "tune[model_type]": "faceid",
        "tune[face_crop]": "true",  # Детектирует и обрезает лица - важно для FaceID!
        "tune[training_face_correct]": "true",  # Улучшает качество входных изображений
    }
    if base_tune_id:
        data["tune[base_tune_id]"] = str(base_tune_id)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    r = requests.post(url, headers=headers, files=files, data=data, timeout=timeout_s)
    if r.status_code >= 400:
        error_text = r.text[:500] if r.text else "Нет текста ошибки"
        raise AstriaError(f"Astria HTTP {r.status_code} при создании tune: {error_text}")
    return r.json()


def _get_tune(*, api_key: str, tune_id: str, timeout_s: float) -> dict[str, Any]:
    url = f"https://api.astria.ai/tunes/{tune_id}"
    # Используем более длинный таймаут для GET
    r = requests.get(url, headers=_headers(api_key), timeout=(10.0, timeout_s))
    if r.status_code >= 400:
        raise AstriaError(f"Astria HTTP {r.status_code} при получении tune: {r.text}")
    try:
        return r.json()
    except Exception as e:
        raise AstriaError("Astria вернул не-JSON ответ при получении tune") from e


async def create_faceid_tune_and_wait(
    *,
    api_key: str,
    name: str,
    title: str,
    image_bytes: bytes,
    base_tune_id: str | None = None,
    max_seconds: int = 120,
    poll_seconds: float = 2.0,
) -> AstriaTuneResult:
    """
    Создаёт FaceID tune по одному фото и ждёт готовности.
    FaceID обычно готовится быстро (секунды), но может быть очередь.
    """
    timeout_s = _timeout_s(30.0)
    created = await asyncio.to_thread(
        _create_faceid_tune,
        api_key=api_key,
        name=name,
        title=title,
        image_bytes=image_bytes,
        base_tune_id=base_tune_id,
        timeout_s=timeout_s,
    )
    import logging
    logger = logging.getLogger("prismalab")
    
    tune_id = str(created.get("id") or "")
    if not tune_id:
        raise AstriaError(f"Неожиданный ответ Astria при создании tune (нет id): {created}")
    
    # Проверяем разные возможные поля статуса
    status = str(created.get("status") or created.get("state") or "").lower()
    trained_at = created.get("trained_at")
    eta = created.get("eta")
    
    logger.info(f"Astria FaceID tune {tune_id} создан, статус: '{status}', trained_at: {trained_at}, eta: {eta}")
    logger.info(f"Полный ответ API при создании: {json.dumps(created, indent=2)}")
    
    # Для FaceID: по документации Astria, FaceID готов сразу (instant)
    # Проверяем все возможные признаки готовности
    if status in {"completed", "succeeded", "ready", "created", "trained", "finished", "pending"}:
        # Для FaceID "pending" или "created" может означать готовность
        # Проверяем trained_at или другие поля
        if trained_at:
            logger.info(f"Astria FaceID tune {tune_id} готов сразу (trained_at: {trained_at})")
            return AstriaTuneResult(tune_id=tune_id, status=status or "trained", raw=created)
        # Если статус "ready" или "completed", значит точно готов
        if status in {"completed", "succeeded", "ready", "trained", "finished"}:
            logger.info(f"Astria FaceID tune {tune_id} готов сразу по статусу: {status}")
            return AstriaTuneResult(tune_id=tune_id, status=status, raw=created)
    
    # Если не готов сразу, опрашиваем статус
    deadline = time.monotonic() + int(max_seconds)
    last = created
    poll_count = 0
    logger.info(f"Astria FaceID tune {tune_id} не готов сразу, начинаю опрос (таймаут: {max_seconds}с)")
    
    while True:
        if time.monotonic() > deadline:
            logger.error(f"Astria FaceID tune {tune_id} - таймаут после {max_seconds}с, последний статус: {status}")
            raise AstriaError(f"Таймаут ожидания готовности Astria FaceID tune {tune_id} (последний статус: {status})")
        
        await asyncio.sleep(poll_seconds)
        poll_count += 1
        
        try:
            last = await asyncio.to_thread(_get_tune, api_key=api_key, tune_id=tune_id, timeout_s=timeout_s)
            status = str(last.get("status") or last.get("state") or "").lower()
            trained_at = last.get("trained_at")
            
            if poll_count % 5 == 0 or status in {"completed", "succeeded", "ready", "trained", "failed", "error"}:
                logger.info(f"Astria FaceID tune {tune_id} - опрос #{poll_count}, статус: '{status}', trained_at: {trained_at}")
            
            if status in {"completed", "succeeded", "ready", "trained", "finished"} or trained_at:
                logger.info(f"Astria FaceID tune {tune_id} готов! Статус: {status}, trained_at: {trained_at}")
                return AstriaTuneResult(tune_id=tune_id, status=status or "trained", raw=last)
            
            if status in {"failed", "error", "cancelled"}:
                error_msg = last.get("error") or last.get("user_error") or str(last)
                logger.error(f"Astria FaceID tune {tune_id} завершился с ошибкой: {error_msg}")
                raise AstriaError(f"Astria FaceID tune {tune_id} завершился с ошибкой: {error_msg}")
        except Exception as e:
            logger.error(f"Ошибка при опросе Astria tune {tune_id}: {e}", exc_info=True)
            # Продолжаем опрос, если это не критическая ошибка
            if poll_count > 10:
                raise


def _create_lora_tune(
    *,
    api_key: str,
    name: str,
    title: str,
    image_bytes_list: list[bytes],
    base_tune_id: str,  # 1504944 для Flux1.dev (проверенная конфигурация для LoRA)
    preset: str = "flux-lora-portrait",  # flux-lora-portrait для людей
    callback: str | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    """
    Создаёт LoRA tune для 10+ фото через Astria API.
    По документации: model_type="lora", base_tune_id=1504944 (Flux1.dev)
    preset="flux-lora-portrait" для людей
    """
    if not api_key:
        raise AstriaError("PRISMALAB_ASTRIA_API_KEY не задан")
    if not base_tune_id:
        raise AstriaError("base_tune_id обязателен для LoRA (1504944 для Flux1.dev)")
    
    url = "https://api.astria.ai/tunes"
    
    # Подготавливаем файлы для multipart/form-data
    files = []
    for idx, image_bytes in enumerate(image_bytes_list):
        import imghdr
        image_type = "png"
        if image_bytes:
            detected = imghdr.what(None, h=image_bytes[:32])
            if detected:
                image_type = detected
            elif image_bytes[:2] == b"\xff\xd8":
                image_type = "jpeg"
            elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
                image_type = "png"
        files.append(("tune[images][]", (f"image_{idx}.{image_type}", image_bytes, f"image/{image_type}")))
    
    data = {
        "tune[name]": str(name),  # man, woman, boy, girl
        "tune[title]": str(title),
        "tune[model_type]": "lora",
        "tune[base_tune_id]": str(base_tune_id),  # 1504944 для Flux1.dev (проверенная конфигурация для LoRA)
        "tune[preset]": preset,  # flux-lora-portrait для людей
    }
    # Токен нужен для Flux1.dev (1504944) - дефолтный "ohwx"
    import logging
    logger = logging.getLogger("prismalab")
    
    # Явно удаляем token, если он был добавлен ранее
    if "tune[token]" in data:
        del data["tune[token]"]
    
    # Добавляем token для Flux1.dev (обязательно для LoRA)
    data["tune[token]"] = "ohwx"  # Дефолтный токен для Flux1.dev
    logger.info(f"[LoRA] Передаю token='ohwx' для base_tune_id={base_tune_id}")
    if callback:
        data["tune[callback]"] = callback
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Логируем, что именно отправляем (без файлов)
    data_for_log = {k: v for k, v in data.items()}
    logger.info(f"[LoRA] Отправляю запрос на создание LoRA: base_tune_id={base_tune_id}, model_type={data.get('tune[model_type]')}, data keys={list(data_for_log.keys())}, token в data: {'tune[token]' in data_for_log}")
    
    r = requests.post(url, headers=headers, files=files, data=data, timeout=timeout_s)
    if r.status_code >= 400:
        error_text = r.text[:500] if r.text else "Нет текста ошибки"
        raise AstriaError(f"Astria HTTP {r.status_code} при создании LoRA tune: {error_text}")
    return r.json()


async def create_lora_tune_and_wait(
    *,
    api_key: str,
    name: str,
    title: str,
    image_bytes_list: list[bytes],
    base_tune_id: str = "1504944",  # Flux1.dev из галереи
    preset: str = "flux-lora-portrait",
    callback: str | None = None,
    max_seconds: int = 7200,  # LoRA training может занять до 2 часов (увеличено с 1 часа)
    poll_seconds: float = 15.0,
) -> AstriaTuneResult:
    """
    Создаёт LoRA tune и ждёт завершения training через polling.
    LoRA training занимает время (минуты-часы), поэтому нужен callback или polling.
    POST с 10 фото может долго уходить — таймаут на создание 600с (10 мин) по умолчанию.
    """
    # Отдельный таймаут для POST создания LoRA (загрузка 10 фото), иначе read timeout
    lora_create_timeout_raw = (os.getenv("PRISMALAB_ASTRIA_LORA_CREATE_TIMEOUT_SECONDS") or "600").strip()
    try:
        timeout_s = float(lora_create_timeout_raw)
    except ValueError:
        timeout_s = 600.0
    timeout_s = max(120.0, min(900.0, timeout_s))  # 2–15 мин
    import logging
    _log = logging.getLogger("prismalab")
    _log.info(f"[LoRA] Таймаут POST создания LoRA (загрузка фото): {timeout_s:.0f}с")
    created = await asyncio.to_thread(
        _create_lora_tune,
        api_key=api_key,
        name=name,
        title=title,
        image_bytes_list=image_bytes_list,
        base_tune_id=base_tune_id,
        preset=preset,
        callback=callback,
        timeout_s=timeout_s,
    )
    import logging
    logger = logging.getLogger("prismalab")
    
    tune_id = str(created.get("id") or "")
    if not tune_id:
        raise AstriaError(f"Неожиданный ответ Astria при создании LoRA tune (нет id): {created}")
    
    # Логируем model_type из ответа для диагностики
    model_type = created.get("model_type") or created.get("type") or "unknown"
    logger.info(f"Astria tune {tune_id} создан, model_type в ответе: '{model_type}'")
    
    status = str(created.get("status") or created.get("state") or "").lower()
    trained_at = created.get("trained_at")
    eta = created.get("eta")
    
    logger.info(f"Astria LoRA tune {tune_id} создан, статус: '{status}', trained_at: {trained_at}, eta: {eta}")
    
    # Если уже готов
    if status in {"completed", "succeeded", "ready", "trained", "finished"} or trained_at:
        logger.info(f"Astria LoRA tune {tune_id} готов сразу")
        return AstriaTuneResult(tune_id=tune_id, status=status or "trained", raw=created)
    
    # Polling для LoRA training
    deadline = time.monotonic() + int(max_seconds)
    last = created
    poll_count = 0
    logger.info(f"Astria LoRA tune {tune_id} - начинаю опрос статуса training (таймаут: {max_seconds}с)")
    
    while True:
        if time.monotonic() > deadline:
            logger.error(f"Astria LoRA tune {tune_id} - таймаут после {max_seconds}с, последний статус: {last.get('status')}")
            raise AstriaError(f"Таймаут ожидания готовности Astria LoRA tune {tune_id}")
        
        await asyncio.sleep(poll_seconds)
        poll_count += 1
        
        last = await asyncio.to_thread(_get_tune, api_key=api_key, tune_id=tune_id, timeout_s=timeout_s)
        status = str(last.get("status") or last.get("state") or "").lower()
        trained_at = last.get("trained_at")
        
        if poll_count % 4 == 0:  # Логируем каждые 4 попытки
            elapsed = time.monotonic() - (deadline - int(max_seconds))
            logger.info(f"Astria LoRA tune {tune_id} - опрос #{poll_count}, прошло {elapsed:.0f}с, статус: {status}")
        
        if status in {"completed", "succeeded", "ready", "trained", "finished"} or trained_at:
            logger.info(f"Astria LoRA tune {tune_id} готов! Найдено trained_at: {trained_at}")
            return AstriaTuneResult(tune_id=tune_id, status=status or "trained", raw=last)
        
        if status in {"failed", "error", "cancelled"}:
            error_msg = last.get("error") or last.get("user_error") or "Unknown error"
            raise AstriaError(f"Astria LoRA tune {tune_id} завершился с ошибкой: {error_msg}")
    
    # unreachable, но для mypy
    raise AstriaError("Unexpected end of polling loop")

