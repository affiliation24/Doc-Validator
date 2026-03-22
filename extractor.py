import json
import re
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EXTRACT_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """
You are a financial document processing expert.

TASK: Extract fields from the document strictly according to the provided JSON schema.

OUTPUT RULES:
- Return ONLY valid JSON, no explanations, no markdown, no ```
- If a field is not found, set it to null
- All string values must be in Russian
- For "total" field: return only the final number, no currency symbols

CATEGORY CLASSIFICATION RULES (category field):
- "ПО" — software licenses, SaaS subscriptions, ERP/CRM systems, antivirus, OS
  Examples: MS Office, 1С, Windows, Adobe, AutoCAD, SAP
- "оборудование" — physical devices and hardware
  Examples: computers, servers, printers, routers, phones, monitors
- "услуги" — work performed, consulting, support as a standalone service
  Examples: website development, audit, legal advice, cleaning, delivery
- "прочее" — anything that doesn't fit the above categories

IMPORTANT: Software technical support (годовая техподдержка ПО) = "ПО", not "услуги"
"""

SCHEMA = {
    "number": "номер документа",
    "date": "дата в формате YYYY-MM-DD",
    "doc_type": "счёт|договор|акт|накладная|прочее",
    "supplier_name": "название поставщика",
    "supplier_inn": "ИНН поставщика",
    "total": "итоговая сумма числом",
    "currency": "валюта: RUB|USD|EUR",
    "category": "категория: ПО|оборудование|услуги|прочее"
}

def clean_json_response(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return raw

def extract(text: str) -> dict:
    response = client.chat.completions.create(
        model=EXTRACT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Извлеки данные по схеме:\n{json.dumps(SCHEMA, ensure_ascii=False, indent=2)}\n\nДокумент:\n{text}"
            }
        ]
    )

    raw = response.choices[0].message.content
    cleaned = clean_json_response(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Модель вернула невалидный JSON: {e}\nОтвет модели: {cleaned}")


if __name__ == "__main__":
    sample = """
    СЧЁТ-ФАКТУРА №СФ-2024-0891 от 15.03.2024

    Поставщик: ООО "ТехноСервис"
    ИНН: 7712345678

    Позиции:
    1. Лицензия MS Office 365 — 10 шт. * 4 500 руб. = 45 000 руб.
    2. Техподдержка (годовая) — 1 шт. * 12 000 руб. = 12 000 руб.

    Итого без НДС: 57 000 руб.
    НДС 20%: 11 400 руб.
    ИТОГО: 68 400 руб.
    """

    result = extract(sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))