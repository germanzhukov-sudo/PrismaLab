from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import requests


class KieError(RuntimeError):
    pass


@dataclass(frozen=True)
class KieTaskResult:
    task_id: str
    image_url: str | None = None


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _create_task(
    *,
    api_key: str,
    model: str,
    prompt: str,
    image_input: list[str] | None = None,
    aspect_ratio: str = "1:1",
    resolution: str | None = "2K",
    quality: str | None = None,  # Для Seedream: "basic" (2K) или "high" (4K)
    negative_prompt: str | None = None,  # Negative prompt (если поддерживается моделью)
    output_format: str = "png",
    upscale_factor: str | None = None,  # Для Topaz upscale: "1", "2", "4", "8"
    callback_url: str | None = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """
    Создаёт задачу генерации через KIE API.
    
    Args:
        api_key: API ключ KIE
        model: Название модели (например, "nano-banana-pro")
        prompt: Текстовое описание изображения
        image_input: Список URL изображений для image-to-image (до 8 штук)
        aspect_ratio: Соотношение сторон (1:1, 16:9, 9:16, 4:3, 3:4, 4:5, 5:4, 21:9, auto)
        resolution: Разрешение (1K, 2K, 4K)
        output_format: Формат вывода (png, jpg)
        callback_url: URL для callback уведомлений (опционально)
        timeout_s: Таймаут запроса в секундах
    
    Returns:
        Ответ API с taskId
    """
    if not api_key:
        raise KieError("KIE_API_KEY не задан")
    
    url = "https://api.kie.ai/api/v1/jobs/createTask"
    
    input_data: dict[str, Any] = {}
    
    # Промпт добавляем только если он не пустой (для upscale моделей промпт не нужен)
    # Проверяем и на None, и на пустую строку
    if prompt and prompt.strip():
        input_data["prompt"] = prompt
    
    # Aspect ratio и output_format добавляем только если они нужны
    if aspect_ratio:
        input_data["aspect_ratio"] = aspect_ratio
    if output_format:
        input_data["output_format"] = output_format
    
    # Разные модели используют разные параметры качества
    if "seedream" in model.lower():
        # Seedream использует quality вместо resolution
        if quality:
            input_data["quality"] = quality
        elif resolution:
            # Конвертируем resolution в quality: 2K -> basic, 4K -> high
            use_quality = "high" if resolution in ("4K", "4k") else "basic"
            input_data["quality"] = use_quality
        # Seedream использует image_urls для image-to-image
        if image_input:
            input_data["image_urls"] = image_input
    elif "flux-2" in model.lower():
        # Flux-2 модели используют input_urls для image-to-image
        if "image-to-image" in model.lower() and image_input:
            input_data["input_urls"] = image_input
        # Flux-2 использует resolution
        if resolution:
            input_data["resolution"] = resolution
        # Flux-2 ТРЕБУЕТ aspect_ratio (обязательный параметр)
        # Если aspect_ratio не передан, используем дефолт "1:1"
        if aspect_ratio:
            input_data["aspect_ratio"] = aspect_ratio
        else:
            input_data["aspect_ratio"] = "1:1"  # Дефолт для Flux-2
    elif "ideogram/v3" in model.lower():
        # Ideogram V3 модели
        if "edit" in model.lower() or "remix" in model.lower():
            # Edit и Remix требуют image_url (один URL, не массив)
            if image_input:
                input_data["image_url"] = image_input[0] if image_input else None
        # Для всех Ideogram V3 добавляем дефолтные параметры
        if "rendering_speed" not in input_data:
            input_data["rendering_speed"] = "BALANCED"
        if "expand_prompt" not in input_data:
            input_data["expand_prompt"] = True
        if "image_size" not in input_data and "text-to-image" in model.lower():
            input_data["image_size"] = "square_hd"
        # Убираем aspect_ratio для Ideogram V3 (используется image_size)
        if "aspect_ratio" in input_data:
            del input_data["aspect_ratio"]
    elif "ideogram/character" in model.lower():
        # Ideogram Character модели используют reference_image_urls
        if image_input:
            input_data["reference_image_urls"] = image_input
        # Для ideogram/character добавляем дефолтные параметры
        if "rendering_speed" not in input_data:
            input_data["rendering_speed"] = "BALANCED"
        if "style" not in input_data:
            input_data["style"] = "REALISTIC"  # Используем REALISTIC для лучшего сохранения лица
        if "expand_prompt" not in input_data:
            input_data["expand_prompt"] = True
        if "num_images" not in input_data:
            input_data["num_images"] = "1"
        if "image_size" not in input_data:
            input_data["image_size"] = "square_hd"
        # Убираем aspect_ratio для Ideogram Character (используется image_size)
        if "aspect_ratio" in input_data:
            del input_data["aspect_ratio"]
    elif "upscale" in model.lower() or "topaz" in model.lower() or "recraft" in model.lower():
        # Upscale модели используют image_url или image
        # ВАЖНО: для Topaz upscale_factor ОБЯЗАТЕЛЕН
        # Сначала очищаем input_data от всех параметров, которые не нужны для upscale
        input_data = {}  # Начинаем с чистого словаря для upscale
        
        if "topaz" in model.lower():
            # Topaz требует image_url и upscale_factor (оба обязательны)
            if not image_input or not image_input[0]:
                raise KieError("Topaz upscale требует image_url")
            input_data["image_url"] = image_input[0]
            # Topaz требует upscale_factor (обязательный параметр)
            # Всегда устанавливаем upscale_factor для Topaz (даже если None - используем "2")
            final_upscale_factor = str(upscale_factor) if upscale_factor else "2"  # Дефолт 2x
            input_data["upscale_factor"] = final_upscale_factor
            # Логируем для отладки
            import logging
            logger = logging.getLogger("prismalab")
            logger.info(f"KIE _create_task: Topaz upscale, upscale_factor={final_upscale_factor}, input_data={input_data}")
        elif "recraft" in model.lower():
            if image_input:
                input_data["image"] = image_input[0] if image_input else None
    elif "google/imagen4-ultra" in model.lower():
        # Imagen4 Ultra использует только prompt, negative_prompt, aspect_ratio, seed
        # Убираем ненужные параметры
        for key in ["resolution", "quality", "output_format"]:
            if key in input_data:
                del input_data[key]
        # Imagen4 Ultra не поддерживает image-to-image
        if "image_input" in input_data:
            del input_data["image_input"]
    else:
        # Nano Banana и другие используют resolution
        if resolution:
            input_data["resolution"] = resolution
        if image_input:
            input_data["image_input"] = image_input
    
    # Negative prompt (если поддерживается)
    if negative_prompt:
        # Проверяем, поддерживает ли модель negative_prompt
        # Seedream может не поддерживать, но попробуем
        input_data["negative_prompt"] = negative_prompt
    
    data: dict[str, Any] = {
        "model": model,
        "input": input_data,
    }
    
    if callback_url:
        data["callBackUrl"] = callback_url
    
    r = requests.post(
        url,
        headers=_headers(api_key),
        json=data,
        timeout=(10.0, timeout_s),
    )
    
    if r.status_code >= 400:
        error_text = r.text[:500] if r.text else "Нет текста ошибки"
        raise KieError(f"KIE HTTP {r.status_code} при создании задачи: {error_text}")
    
    try:
        result = r.json()
    except Exception as e:
        raise KieError(f"Не удалось распарсить ответ KIE: {e}") from e
    
    if result.get("code") != 200:
        msg = result.get("msg", "Неизвестная ошибка")
        raise KieError(f"KIE вернул ошибку: {msg}")
    
    return result.get("data", {})


def _get_task_detail(
    *,
    api_key: str,
    task_id: str,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """
    Получает детали задачи по task_id.
    
    Args:
        api_key: API ключ KIE
        task_id: ID задачи
        timeout_s: Таймаут запроса в секундах
    
    Returns:
        Детали задачи со статусом и результатами
    """
    if not api_key:
        raise KieError("KIE_API_KEY не задан")
    
    url = "https://api.kie.ai/api/v1/jobs/recordInfo"
    
    params = {"taskId": task_id}
    
    r = requests.get(
        url,
        headers=_headers(api_key),
        params=params,
        timeout=(10.0, timeout_s),
    )
    
    if r.status_code >= 400:
        error_text = r.text[:500] if r.text else "Нет текста ошибки"
        raise KieError(f"KIE HTTP {r.status_code} при получении задачи: {error_text}")
    
    try:
        result = r.json()
    except Exception as e:
        raise KieError(f"Не удалось распарсить ответ KIE: {e}") from e
    
    if result.get("code") != 200:
        msg = result.get("msg", "Неизвестная ошибка")
        raise KieError(f"KIE вернул ошибку: {msg}")
    
    return result.get("data", {})


def _extract_image_urls(data: dict[str, Any]) -> list[str]:
    """
    Извлекает URLs изображений из ответа KIE.
    
    Args:
        data: Данные из ответа recordInfo
    
    Returns:
        Список URLs изображений
    """
    urls: list[str] = []
    
    # KIE API использует resultJson - это JSON строка с resultUrls
    result_json_str = data.get("resultJson") or ""
    if result_json_str:
        try:
            result_json = json.loads(result_json_str)
            result_urls = result_json.get("resultUrls") or result_json.get("result_urls")
            if result_urls:
                if isinstance(result_urls, list):
                    urls.extend(str(url) for url in result_urls if url)
                elif isinstance(result_urls, str):
                    urls.append(result_urls)
        except (json.JSONDecodeError, TypeError) as e:
            import logging
            logger = logging.getLogger("prismalab")
            logger.warning(f"Не удалось распарсить resultJson: {e}, строка: {result_json_str[:100]}")
    
    # Fallback: пробуем другие возможные пути
    if not urls:
        result_urls = data.get("resultUrls") or data.get("result_urls") or data.get("resultUrl")
        if result_urls:
            if isinstance(result_urls, list):
                urls.extend(str(url) for url in result_urls if url)
            elif isinstance(result_urls, str):
                urls.append(result_urls)
    
    return urls


async def run_task_and_wait(
    *,
    api_key: str,
    model: str,
    prompt: str,
    image_input: list[str] | None = None,
    aspect_ratio: str = "1:1",
    resolution: str | None = "2K",
    quality: str | None = None,  # Для Seedream: "basic" или "high"
    negative_prompt: str | None = None,  # Negative prompt
    output_format: str = "png",
    upscale_factor: str | None = None,  # Для Topaz upscale: "1", "2", "4", "8"
    max_seconds: int = 300,
    poll_seconds: float = 3.0,
) -> KieTaskResult:
    """
    Создаёт задачу генерации и ждёт её завершения.
    
    Args:
        api_key: API ключ KIE
        model: Название модели
        prompt: Текстовое описание
        image_input: Список URL изображений (опционально)
        aspect_ratio: Соотношение сторон
        resolution: Разрешение
        output_format: Формат вывода
        max_seconds: Максимальное время ожидания в секундах
        poll_seconds: Интервал опроса статуса в секундах
    
    Returns:
        KieTaskResult с task_id и image_url
    """
    import logging
    logger = logging.getLogger("prismalab")
    
    # Создаём задачу
    timeout_s = 60.0
    created = await asyncio.to_thread(
        _create_task,
        api_key=api_key,
        model=model,
        prompt=prompt,
        image_input=image_input,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        quality=quality,
        negative_prompt=negative_prompt,
        output_format=output_format,
        upscale_factor=upscale_factor,
        callback_url=None,
        timeout_s=timeout_s,
    )
    
    task_id = str(created.get("taskId") or "")
    if not task_id:
        raise KieError(f"Неожиданный ответ KIE (нет taskId): {created}")
    
    logger.info(f"KIE task {task_id} создан, начинаю опрос статуса (таймаут: {max_seconds}с)")
    
    deadline = time.monotonic() + int(max_seconds)
    last_data = created
    poll_count = 0
    
    while True:
        elapsed = time.monotonic() - (deadline - int(max_seconds))
        if time.monotonic() > deadline:
            state = last_data.get("state") or "unknown"
            logger.error(f"KIE task {task_id} - таймаут после {max_seconds}с, последний статус: {state}")
            raise KieError("Таймаут ожидания результата KIE")
        
        poll_count += 1
        if poll_count % 5 == 0:  # Логируем каждые 5 попыток
            state = last_data.get("state") or "unknown"
            logger.info(f"KIE task {task_id} - опрос #{poll_count}, прошло {elapsed:.0f}с, статус: {state}")
        
        # Получаем статус задачи
        polling_timeout = 30.0
        try:
            last_data = await asyncio.to_thread(
                _get_task_detail,
                api_key=api_key,
                task_id=task_id,
                timeout_s=polling_timeout,
            )
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            # Retry при таймауте
            logger.warning(f"KIE task {task_id} - таймаут при polling (попытка {poll_count}), жду 5с и повторяю...")
            await asyncio.sleep(5.0)
            try:
                last_data = await asyncio.to_thread(
                    _get_task_detail,
                    api_key=api_key,
                    task_id=task_id,
                    timeout_s=polling_timeout,
                )
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e2:
                raise KieError(f"KIE API не отвечает (таймаут при polling): {e2}") from e2
        
        # Проверяем статус (KIE использует state: waiting, queuing, generating, success, fail)
        state = last_data.get("state") or ""
        fail_msg = last_data.get("failMsg") or ""
        fail_code = last_data.get("failCode") or ""
        
        if state == "success":
            # Успех - извлекаем URLs
            urls = _extract_image_urls(last_data)
            if urls:
                logger.info(f"KIE task {task_id} - успешно завершён, найдено {len(urls)} изображений")
                return KieTaskResult(task_id=task_id, image_url=urls[0])
            else:
                logger.warning(f"KIE task {task_id} - статус success, но нет URLs в ответе: {last_data}")
                # Продолжаем ждать, возможно данные ещё не готовы
                await asyncio.sleep(poll_seconds)
                continue
        
        if state == "fail":
            error_text = fail_msg or fail_code or str(last_data)
            logger.error(f"KIE task {task_id} - задача завершилась с ошибкой (code={fail_code}): {error_text}")
            raise KieError(f"KIE задача завершилась с ошибкой: {error_text}")
        
        # Статус 0 или другой - продолжаем ждать
        await asyncio.sleep(poll_seconds)


def upload_file_base64(
    *,
    api_key: str,
    image_bytes: bytes,
    file_name: str | None = None,
    upload_path: str = "images",
    timeout_s: float = 60.0,
) -> str:
    """
    Загружает файл в KIE через Base64 upload и возвращает публичный URL.
    
    Args:
        api_key: API ключ KIE
        image_bytes: Байты изображения
        file_name: Имя файла (опционально)
        upload_path: Путь для загрузки
        timeout_s: Таймаут в секундах
    
    Returns:
        Публичный URL загруженного файла
    """
    if not api_key:
        raise KieError("KIE_API_KEY не задан")
    
    import base64
    
    # Конвертируем в base64
    base64_data = base64.b64encode(image_bytes).decode("utf-8")
    # Добавляем data URL prefix
    data_url = f"data:image/jpeg;base64,{base64_data}"
    
    url = "https://kieai.redpandaai.co/api/file-base64-upload"
    
    payload: dict[str, Any] = {
        "base64Data": data_url,
        "uploadPath": upload_path,
    }
    if file_name:
        payload["fileName"] = file_name
    
    r = requests.post(
        url,
        headers=_headers(api_key),
        json=payload,
        timeout=(10.0, timeout_s),
    )
    
    if r.status_code >= 400:
        error_text = r.text[:500] if r.text else "Нет текста ошибки"
        raise KieError(f"KIE HTTP {r.status_code} при загрузке файла: {error_text}")
    
    try:
        result = r.json()
    except Exception as e:
        raise KieError(f"Не удалось распарсить ответ KIE: {e}") from e
    
    if result.get("code") != 200:
        msg = result.get("msg", "Неизвестная ошибка")
        raise KieError(f"KIE вернул ошибку при загрузке: {msg}")
    
    data = result.get("data", {})
    file_url = data.get("fileUrl") or data.get("downloadUrl")
    if not file_url:
        raise KieError(f"KIE не вернул URL файла: {result}")
    
    return str(file_url)


def download_image_bytes(url: str, timeout_s: float = 30.0) -> bytes:
    """
    Скачивает изображение по URL.
    
    Args:
        url: URL изображения
        timeout_s: Таймаут в секундах
    
    Returns:
        Байты изображения
    """
    r = requests.get(url, timeout=(10.0, timeout_s))
    if r.status_code >= 400:
        raise KieError(f"Не удалось скачать изображение (HTTP {r.status_code})")
    
    ct = (r.headers.get("Content-Type") or "").lower()
    if not ct.startswith("image/"):
        raise KieError(f"KIE вернул не изображение (Content-Type: {ct})")
    
    return r.content
