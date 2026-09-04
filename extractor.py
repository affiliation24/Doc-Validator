import json
import os
import re

from groq import BadRequestError

from llm import client, parse_json_object, with_rate_limit_retry

EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "openai/gpt-oss-20b")
EXTRACT_FALLBACK_MODEL = os.getenv("EXTRACT_FALLBACK_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """
You are a financial document processing expert.

TASK: Extract fields from the document strictly according to the provided JSON schema.

OUTPUT RULES:
- Return ONLY valid JSON, no explanations, no markdown, no ```
- If a field is not found, set it to null
- Free-text values (names, purpose, period) must be in Russian, copied from the document
- For "total" field: return only the final number, no currency symbols

ENUM FIELDS — these are CODE VALUES, never translate them to Russian:
- "doc_type" is exactly one of: invoice, contract, act, payment_order, approval, other
  An акт is "act", a платёжное поручение is "payment_order", a счёт is "invoice",
  a договор or госконтракт is "contract", a служебная записка is "approval".
  Returning "акт" or "платёжное поручение" here is an error — the value is
  compared against these Latin identifiers by code, not read by a human.
- "currency" is exactly one of: RUB, USD, EUR
- "category" is exactly one of: ПО, оборудование, услуги, прочее (these four
  Russian words are themselves the codes — use them verbatim)

CATEGORY CLASSIFICATION RULES (category field):
- "ПО" — software licenses, SaaS subscriptions, ERP/CRM systems, antivirus, OS
  Examples: MS Office, 1С, Windows, Adobe, AutoCAD, SAP
- "оборудование" — physical devices and hardware
  Examples: computers, servers, printers, routers, phones, monitors
- "услуги" — work performed, consulting, support as a standalone service
  Examples: website development, audit, legal advice, cleaning, delivery
- "прочее" — anything that doesn't fit the above categories

IMPORTANT: Software technical support (годовая техподдержка ПО) = "ПО", not "услуги"

PAYMENT ORDER (платёжное поручение, форма 0401060) — FIELD DISAMBIGUATION:
The form lists FOUR different parties in a fixed order. Do not mix them up.
1. Плательщик        — the payer. Its ИНН/КПП are printed ABOVE the block,
                       its "Сч. №" is the payer's own account.
                       → payer_name, payer_inn, payer_kpp, payer_account
2. Банк плательщика   — the payer's BANK. This is a bank, never the supplier.
                       Its "Сч. №" is a correspondent account.
                       → payer_bank_name, payer_bank_bik
3. Банк получателя    — the recipient's BANK, also never the supplier.
                       Its "Сч. №" is the recipient bank's correspondent account.
                       → bank_name, bank_bik, bank_corr_account
4. Получатель         — the recipient of the money. THIS is the supplier.
                       The "Сч. №" printed next to the recipient's ИНН/КПП is
                       the recipient's settlement account.
                       → supplier_name, supplier_inn, supplier_kpp, bank_account

CRITICAL — LABEL POSITION: on form 0401060 the label sits in the left margin of
the block it names, so text extracted from a PDF prints the VALUES FIRST and the
label AFTER them. Read every block bottom-up: the label belongs to the lines
ABOVE it, not below. Example of real extracted text:

    ИНН 7724742237 КПП 874874833 Сумма 326 001,00
    ООО «Фельдъегерь»
    Сч. № 40817810238007330404
    Плательщик                     ← labels the THREE lines above
    АО «ТБанк» БИК 044525225
    Сч. № 40817810238118330404
    Банк плательщика               ← labels the TWO lines above

Here payer_name = "ООО «Фельдъегерь»", payer_inn = "7724742237",
payer_account = "40817810238007330404", payer_bank_name = "АО «ТБанк»",
payer_bank_bik = "044525225". Attaching a label to the lines below it shifts
every party by one block and is the single most common mistake — do not make it.

Rule of thumb: a party whose name is a bank (Сбер, ТБанк, ВТБ, Альфа, Райффайзен)
and which is followed by "БИК" is a BANK — put it in payer_bank_name or
bank_name, never in supplier_name or payer_name.

Also extract the numbered fields of the form when present:
- "Вид оп." → operation_type      - "Очер. плат." → payment_priority (1-5)
- "Наз. пл." → payment_type_code   - "Код" → uin (УИН, 0 if literally "0")
- "Назначение платежа" → payment_purpose (full text)
- Budget payments: статус плательщика (101) → payer_status, КБК → kbk,
  ОКТМО → oktmo, "Основание" → tax_basis, "Налоговый период" → tax_period
"""

SCHEMA = {
    "number": "номер документа",
    "date": "дата в формате YYYY-MM-DD",
    "doc_type": "invoice|contract|act|payment_order|approval|other",

    # Получатель средств. В платёжном поручении это блок «Получатель»,
    # в счёте и договоре — поставщик или исполнитель.
    "supplier_name": "наименование получателя средств (в счёте — поставщика)",
    "supplier_inn": "ИНН получателя средств",
    "supplier_kpp": "КПП получателя средств (9 цифр, только для юр. лиц)",
    "bank_account": "расчётный счёт получателя (20 цифр)",
    "bank_name": "наименование банка получателя",
    "bank_bik": "БИК банка получателя (9 цифр)",
    "bank_corr_account": "корреспондентский счёт банка получателя (20 цифр)",

    # Плательщик — заполняется в основном для платёжных поручений.
    "payer_name": "наименование плательщика",
    "payer_inn": "ИНН плательщика",
    "payer_kpp": "КПП плательщика (9 цифр)",
    "payer_account": "расчётный счёт плательщика (20 цифр)",
    "payer_bank_name": "наименование банка плательщика",
    "payer_bank_bik": "БИК банка плательщика (9 цифр)",

    # Поля бланка 0401060.
    "operation_type": "вид операции (для платёжного поручения — 01)",
    "payment_priority": "очерёдность платежа, цифра от 1 до 5",
    "payment_type_code": "назначение платежа кодом (поле «Наз. пл.»)",
    "uin": "код УИН (поле «Код»): 20 или 25 цифр либо 0",
    "payment_purpose": "назначение платежа текстом",

    # Реквизиты бюджетного платежа.
    "payer_status": "статус плательщика (поле 101, две цифры)",
    "kbk": "КБК (20 цифр)",
    "oktmo": "ОКТМО (8 или 11 цифр)",
    "tax_basis": "основание платежа",
    "tax_period": "налоговый период",

    "total": "итоговая сумма числом",
    "total_words": "сумма прописью если указана в документе",
    "currency": "валюта: RUB|USD|EUR",
    "category": "категория: ПО|оборудование|услуги|прочее"
}

def call_model(model: str, text: str) -> str:
    response = with_rate_limit_retry(lambda: client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Извлеки данные по схеме:\n{json.dumps(SCHEMA, ensure_ascii=False, indent=2)}\n\nДокумент:\n{text}"}
        ]
    ))
    return response.choices[0].message.content


def extract(text: str) -> dict:
    """Извлекает поля документа, при отказе лёгкой модели — тяжёлой.

    На плотных документах вроде платёжного поручения лёгкая модель иногда не
    добирает схему до конца и Groq отклоняет ответ в JSON-режиме. Это не повод
    ронять запрос: fallback-модель справляется, а лишний вызов происходит редко.
    """
    models = [EXTRACT_MODEL]
    if EXTRACT_FALLBACK_MODEL and EXTRACT_FALLBACK_MODEL != EXTRACT_MODEL:
        models.append(EXTRACT_FALLBACK_MODEL)

    last_error = None
    for model in models:
        try:
            result = parse_json_object(call_model(model, text), f"Модель {model}")
        except BadRequestError as e:
            last_error = f"модель {model} не смогла собрать JSON по схеме: {e}"
            continue
        except ValueError as e:
            last_error = str(e)
            continue

        result["total"] = parse_amount(result.get("total"))
        result["doc_type"] = normalize_doc_type(result.get("doc_type"))
        return result

    raise ValueError(f"Не удалось извлечь данные документа. {last_error}")


DOC_TYPES = {"invoice", "contract", "act", "payment_order", "approval", "other"}

# Модель периодически переводит enum на русский, несмотря на инструкцию.
# Проверки в validator сравнивают doc_type с латинскими идентификаторами, и
# перевод молча выключал бы их — поэтому приводим значение обратно.
DOC_TYPE_ALIASES = {
    "счёт": "invoice", "счет": "invoice",
    "счёт-фактура": "invoice", "счет-фактура": "invoice",
    "договор": "contract", "контракт": "contract",
    "государственный контракт": "contract", "госконтракт": "contract",
    "акт": "act", "акт выполненных работ": "act", "акт приёмки": "act",
    "платёжное поручение": "payment_order", "платежное поручение": "payment_order",
    "платёжка": "payment_order", "платежка": "payment_order",
    "служебная записка": "approval", "согласование": "approval", "докладная": "approval",
    "прочее": "other", "другое": "other",
}


def normalize_doc_type(value) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in DOC_TYPES:
        return key
    return DOC_TYPE_ALIASES.get(key, str(value).strip())


def parse_amount(value) -> float | None:
    """Приводит сумму к float: '68 400', '1 500 000,00', '28000.5' → число.

    Если распарсить нельзя — None, а не исключение: пусть решает валидатор,
    для которого отсутствующая сумма это отдельное нарушение.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    # Неразрывные и тонкие пробелы — частый артефакт OCR и вёрстки PDF.
    cleaned = re.sub(r"[\s  ]", "", str(value))
    cleaned = re.sub(r"[^\d,.\-]", "", cleaned)
    if not cleaned:
        return None

    if "," in cleaned and "." in cleaned:
        # '1.500.000,75' — точка группирует тысячи, запятая десятичная.
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(,\d{3})+", cleaned):
        # '5,000' в англоязычной записи — это тысячи, а не 5 рублей.
        cleaned = cleaned.replace(",", "")
    else:
        cleaned = cleaned.replace(",", ".")

    try:
        return float(cleaned)
    except ValueError:
        return None


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