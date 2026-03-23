import json
import re
import os
from groq import Groq
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from extractor import extract

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VALIDATE_MODEL = "llama-3.3-70b-versatile"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="./rules_db")
rules_col = db.get_or_create_collection("rules")

RULES = [
    "Закупка любого ПО требует согласования IT-отдела.",
    "Счета свыше 50 000 руб. требуют визы финансового директора.",
    "Новый поставщик должен пройти проверку службы безопасности.",
    "Оплата производится в течение 30 дней с даты счёта.",
    "Закупки оборудования свыше 500 000 руб. выносятся на тендер.",
    "Счета в иностранной валюте требуют курс ЦБ на дату документа.",
    "Договоры свыше 1 000 000 руб. требуют согласования юридического отдела.",
]

SYSTEM_PROMPT = """
You are a compliance officer at a company.

TASK: Verify the document against the company regulations provided.

OUTPUT RULES:
- Return ONLY valid JSON, no explanations, no markdown, no ```
- All string values must be in Russian
- Be strict: if a regulation applies — flag it as an issue

RESPONSE SCHEMA:
{
  "approved": true or false,
  "issues": ["list of violations found"],
  "approvals_needed": ["list of people/departments who must sign"],
  "comment": "brief conclusion in one sentence"
}

If no violations found — return approved: true and empty lists.
"""

def load_rules():
    if rules_col.count() == 0:
        rules_col.add(
            documents=RULES,
            embeddings=embedder.encode(RULES).tolist(),
            ids=[f"rule_{i}" for i in range(len(RULES))]
        )
        print(f"Загружено правил: {len(RULES)}")
    else:
        print(f"Правила уже загружены: {rules_col.count()} шт.")

def find_relevant_rules(extracted: dict, top_k: int = 3) -> list[str]:
    query = (
        f"{extracted.get('category', '')} "
        f"сумма {extracted.get('total', 0)} {extracted.get('currency', 'RUB')} "
        f"поставщик {extracted.get('supplier_name', '')}"
    )
    results = rules_col.query(
        query_embeddings=embedder.encode([query]).tolist(),
        n_results=top_k
    )
    return results["documents"][0]

def clean_json_response(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return raw

def validate(extracted: dict) -> dict:
    relevant_rules = find_relevant_rules(extracted)

    response = client.chat.completions.create(
        model=VALIDATE_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
Verify the document against the regulations.

Document data:
{json.dumps(extracted, ensure_ascii=False, indent=2)}

Applicable regulations:
{chr(10).join(f"- {r}" for r in relevant_rules)}
"""
            }
        ]
    )

    raw = response.choices[0].message.content
    cleaned = clean_json_response(raw)

    try:
        result = json.loads(cleaned)
        result["rules_checked"] = relevant_rules
        return result
    except json.JSONDecodeError as e:
        raise ValueError(f"Модель вернула невалидный JSON: {e}\nОтвет: {cleaned}")


if __name__ == "__main__":

    load_rules()

    sample = """
    СЧЁТ-ФАКТУРА №СФ-2024-0891 от 15.03.2024

    Поставщик: ООО "ТехноСервис"
    ИНН: 7712345678

    Позиции:
    1. Лицензия MS Office 365 — 10 шт. * 4 500 руб. = 45 000 руб.
    2. Техподдержка (годовая) — 1 шт.  * 12 000 руб. = 12 000 руб.

    Итого без НДС: 57 000 руб.
    НДС 20%: 11 400 руб.
    ИТОГО: 68 400 руб.
    """

    extracted = extract(sample)
    print("Извлечённые данные:")
    print(json.dumps(extracted, ensure_ascii=False, indent=2))

    print("\nРезультат валидации:")
    result = validate(extracted)
    print(json.dumps(result, ensure_ascii=False, indent=2))