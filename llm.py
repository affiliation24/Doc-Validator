"""Общий слой работы с Groq: клиент, ретраи и разбор ответа.

Раньше клиент, чистка markdown и разбор JSON дублировались в extractor.py и
validator.py. Держим это в одном месте, чтобы правки применялись к обоим вызовам.
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# На free tier Groq лимит в 8000 токенов в минуту выбирается быстро, а промпты
# здесь длинные. 429 — это «подожди секунду», а не отказ, поэтому повторяем.
MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))


def with_rate_limit_retry(call, max_retries: int = MAX_RETRIES):
    """Повторяет вызов при 429, выжидая время, которое называет сам Groq."""
    for attempt in range(max_retries + 1):
        try:
            return call()
        except RateLimitError as e:
            if attempt == max_retries:
                raise
            time.sleep(retry_delay(e, attempt))


def retry_delay(error: RateLimitError, attempt: int) -> float:
    """Пауза перед повтором: из ответа Groq, иначе экспоненциальная."""
    message = str(error)
    match = re.search(r"try again in ([\d.]+)s", message)
    if match:
        # Небольшой запас: лимит освобождается не мгновенно.
        return float(match.group(1)) + 0.5
    return 2.0 ** attempt


def clean_json_response(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return raw


def parse_json_object(raw: str, source: str) -> dict:
    """Разбирает ответ модели в словарь, объясняя в ошибке что именно пришло."""
    cleaned = clean_json_response(raw)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"{source} вернула невалидный JSON: {e}\nОтвет: {cleaned}")

    if not isinstance(result, dict):
        raise ValueError(f"{source} вернула не объект, а {type(result).__name__}: {cleaned}")

    return result
