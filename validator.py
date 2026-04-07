import json
import re
import os
from groq import Groq
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VALIDATE_MODEL = "llama-3.3-70b-versatile"
RULES_FILE = "rules.txt"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="./rules_db")
rules_col = db.get_or_create_collection("rules")

SYSTEM_PROMPT = """
You are a strict compliance officer at a government ministry.

CRITICAL RULES — apply these unconditionally before anything else:
- ANY software, license, SaaS, or IT subscription → IT department approval REQUIRED, mark as "ошибка"
- ANY invoice or payment over 50,000 RUB → CFO (финансовый директор) approval REQUIRED
- ANY contract over 1,000,000 RUB → Legal department (юридический отдел) approval REQUIRED
- ANY purchase over 500,000 RUB → tender (тендер) REQUIRED
- ANY foreign currency (USD, EUR) → CBR exchange rate REQUIRED, mark as "ошибка"
- ANY new supplier not in registry → security check REQUIRED, mark as "предупреждение"

These rules are absolute. They override everything else.

Your task is to verify financial and administrative documents against ministry regulations.


---

SUPPORTED DOCUMENT TYPES:
- invoice          (счёт, счёт-фактура)
- contract         (государственный контракт, договор)
- payment_order    (платёжное поручение)
- act              (акт выполненных работ, акт приёмки)
- approval         (служебная записка, согласование, докладная)
- other

---

VERIFICATION DEPTH — check ALL fields if present:
- Document number and date (presence, valid format)
- Supplier/contractor name and INN:
    - Legal entity: exactly 10 digits
    - Individual: exactly 12 digits
- Total amount and currency (RUB/USD/EUR)
- VAT: 20% standard rate, 0% only for legally exempt categories
- Payment terms: must not exceed 30 calendar days
- Tender: required for purchases over 500,000 RUB
- IT approval: required for any software or license purchase
- Legal approval: required for contracts over 1,000,000 RUB
- CFO approval: required for invoices over 50,000 RUB
- Security check: required for any new supplier not in approved registry
- Foreign currency: CBR exchange rate on document date must be present

---

SEVERITY LEVELS:
- "ошибка" — document CANNOT be approved, processing is blocked
  Use when: missing required fields, INN invalid, limits exceeded without approval, forged data suspected
- "предупреждение" — document CAN proceed but requires an authorising signature
  Use when: borderline amounts, optional fields missing, supplier not yet verified

---

PRIORITY LEVELS:
- "БЛОКИРУЮЩИЙ"        — stops processing entirely, escalate immediately
- "ТРЕБУЕТ_ВНИМАНИЯ"   — requires signature or correction before approval
- "ИНФОРМАЦИОННЫЙ"     — logged for audit, does not block processing

---

UNCERTAINTY HANDLING — STRICT RULE:
If you are not 100% confident about any field:
- Do NOT guess, do NOT skip, do NOT approve
- Set approved = false
- Add the field to "uncertain_fields" with:
    - what exactly is missing
    - why you are uncertain
    - what document or data would resolve the doubt
- Example: "ИНН содержит 9 символов вместо 10 — документ может быть повреждён или данные введены с ошибкой"

---

OUTPUT RULES:
- Return ONLY valid JSON — no markdown, no text outside JSON
- All string values must be in Russian
- "human_comment" — one or two sentences in plain Russian for a non-technical ministry employee

---

RESPONSE SCHEMA:
{
  "approved": true or false,
  "doc_type": "invoice | contract | payment_order | act | approval | other",
  "severity": "ошибка | предупреждение | ок",
  "issues": [
    {
      "rule": "название нарушенного регламента",
      "description": "что именно нарушено и почему",
      "severity": "ошибка | предупреждение",
      "priority": "БЛОКИРУЮЩИЙ | ТРЕБУЕТ_ВНИМАНИЯ | ИНФОРМАЦИОННЫЙ"
    }
  ],
  "uncertain_fields": [
    {
      "field": "название поля",
      "reason": "почему модель сомневается",
      "missing_data": "какие именно данные нужны для проверки"
    }
  ],
  "approvals_needed": ["список отделов или должностей"],
  "rules_checked": ["список регламентов которые были применены"],
  "human_comment": "краткий вывод на русском языке для сотрудника министерства"
}
"""


def read_rules_from_file(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл правил не найден: {path}")

    rules = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rules.append(line)
    return rules


def get_rules_hash(rules: list[str]) -> str:
    import hashlib
    content = "\n".join(rules)
    return hashlib.md5(content.encode()).hexdigest()


def load_rules():
    rules = read_rules_from_file(RULES_FILE)
    current_hash = get_rules_hash(rules)

    stored_hash = None
    try:
        meta = rules_col.get(ids=["__meta__"])
        if meta["documents"]:
            stored_hash = meta["documents"][0]
    except Exception:
        pass

    if stored_hash == current_hash:
        print(f"Правила актуальны: {rules_col.count() - 1} шт.")
        return

    print("Обновляю базу правил...")
    existing = rules_col.get()
    rule_ids = [i for i in existing["ids"] if i != "__meta__"]
    if rule_ids:
        rules_col.delete(ids=rule_ids)

    rules_col.upsert(
        documents=[current_hash],
        embeddings=[[0.0] * 384],
        ids=["__meta__"]
    )

    rules_col.add(
        documents=rules,
        embeddings=embedder.encode(rules).tolist(),
        ids=[f"rule_{i}" for i in range(len(rules))]
    )
    print(f"Загружено правил: {len(rules)}")


def load_rules():
    rules = read_rules_from_file(RULES_FILE)
    current_hash = get_rules_hash(rules)

    stored_hash = None
    try:
        meta = rules_col.get(ids=["__meta__"])
        if meta["documents"]:
            stored_hash = meta["documents"][0]
    except Exception:
        pass

    if stored_hash == current_hash:
        print(f"Правила актуальны: {rules_col.count() - 1} шт.")
        return

    print("Обновляю базу правил...")
    existing = rules_col.get()
    rule_ids = [i for i in existing["ids"] if i != "__meta__"]
    if rule_ids:
        rules_col.delete(ids=rule_ids)

    rules_col.upsert(
        documents=[current_hash],
        embeddings=[[0.0] * 384],
        ids=["__meta__"],
        metadatas=[{"type": "meta"}]
    )

    rules_col.add(
        documents=rules,
        embeddings=embedder.encode(rules).tolist(),
        ids=[f"rule_{i}" for i in range(len(rules))],
        metadatas=[{"type": "rule"} for _ in rules]
    )
    print(f"Загружено правил: {len(rules)}")


def find_relevant_rules(extracted: dict, top_k: int = 5) -> list[str]:
    category = extracted.get("category", "")
    total = extracted.get("total", 0)
    currency = extracted.get("currency", "RUB")

    queries = [
        f"закупка {category} лицензия программное обеспечение согласование",
        f"счёт {total} рублей превышает лимит виза подпись",
        f"валюта {currency} курс поставщик новый проверка",
    ]

    seen = set()
    all_rules = []

    for query in queries:
        results = rules_col.query(
            query_embeddings=embedder.encode([query]).tolist(),
            n_results=3,
            where={"type": {"$eq": "rule"}}
        )
        for doc in results["documents"][0]:
            if doc not in seen:
                seen.add(doc)
                all_rules.append(doc)
        if len(all_rules) >= top_k:
            break

    return all_rules[:top_k]


def clean_json_response(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return raw

def validate_inn(inn: str) -> tuple[bool, str]:
    if not inn:
        return False, "ИНН отсутствует"
    inn = str(inn).strip()
    if not inn.isdigit():
        return False, f"ИНН содержит нецифровые символы: {inn}"
    if len(set(inn)) == 1:
        return False, f"ИНН состоит из одинаковых цифр: {inn}"
    if len(inn) == 10:
        return True, ""
    if len(inn) == 12:
        return True, ""
    return False, f"ИНН содержит {len(inn)} цифр вместо 10 (юр. лицо) или 12 (ИП)"


def validate_kpp(kpp: str) -> tuple[bool, str]:
    if not kpp:
        return True, ""
    kpp = str(kpp).strip()
    if not kpp.isdigit():
        return False, f"КПП содержит нецифровые символы: {kpp}"
    if len(kpp) != 9:
        return False, f"КПП содержит {len(kpp)} цифр вместо 9"
    if len(set(kpp)) == 1:
        return False, f"КПП состоит из одинаковых цифр: {kpp}"
    return True, ""


def validate_bik(bik: str) -> tuple[bool, str]:
    if not bik:
        return True, ""
    bik = str(bik).strip()
    if not bik.isdigit():
        return False, f"БИК содержит нецифровые символы: {bik}"
    if len(bik) != 9:
        return False, f"БИК содержит {len(bik)} цифр вместо 9"
    if not bik.startswith("04"):
        return False, f"БИК не начинается с '04' — возможно, некорректный российский БИК: {bik}"
    return True, ""


def validate_account(account: str) -> tuple[bool, str]:
    if not account:
        return True, ""
    account = str(account).strip()
    if not account.isdigit():
        return False, f"Номер счёта содержит нецифровые символы: {account}"
    if len(account) != 20:
        return False, f"Номер счёта содержит {len(account)} цифр вместо 20"
    if len(set(account)) == 1:
        return False, f"Номер счёта состоит из одинаковых цифр: {account}"
    return True, ""


def build_mandatory_checks(extracted: dict) -> list[str]:
    checks = []
    total = float(extracted.get("total") or 0)
    category = extracted.get("category", "")
    currency = extracted.get("currency", "RUB")
    doc_type = extracted.get("doc_type", "")

    if category == "ПО":
        checks.append("ОБЯЗАТЕЛЬНО: Категория 'ПО' — требуется согласование IT-отдела. Это блокирующее нарушение.")
    if total > 50000:
        checks.append(f"ОБЯЗАТЕЛЬНО: Сумма {total} руб. превышает 50 000 — требуется виза финансового директора.")
    if total > 150000:
        checks.append(f"ОБЯЗАТЕЛЬНО: Сумма {total} руб. превышает 150 000 — требуется виза руководителя подразделения.")
    if total > 500000:
        checks.append(f"ОБЯЗАТЕЛЬНО: Сумма {total} руб. превышает 500 000 — требуется тендер.")
    if doc_type == "contract" and total > 1000000:
        checks.append(f"ОБЯЗАТЕЛЬНО: Договор на {total} руб. превышает 1 000 000 — требуется согласование юридического отдела.")
    if currency in ("USD", "EUR"):
        checks.append(f"ОБЯЗАТЕЛЬНО: Валюта {currency} — требуется курс ЦБ на дату документа.")

    inn = extracted.get("supplier_inn", "")
    inn_ok, inn_err = validate_inn(inn)
    if not inn_ok:
        checks.append(f"ОБЯЗАТЕЛЬНО: Некорректный ИНН поставщика — {inn_err}.")

    kpp = extracted.get("supplier_kpp", "")
    kpp_ok, kpp_err = validate_kpp(kpp)
    if not kpp_ok:
        checks.append(f"ОБЯЗАТЕЛЬНО: Некорректный КПП поставщика — {kpp_err}.")

    bik = extracted.get("bank_bik", "")
    bik_ok, bik_err = validate_bik(bik)
    if not bik_ok:
        checks.append(f"ОБЯЗАТЕЛЬНО: Некорректный БИК банка — {bik_err}.")

    account = extracted.get("bank_account", "")
    acc_ok, acc_err = validate_account(account)
    if not acc_ok:
        checks.append(f"ОБЯЗАТЕЛЬНО: Некорректный расчётный счёт — {acc_err}.")

    payer_inn = extracted.get("payer_inn", "")
    if payer_inn and inn and payer_inn == inn:
        checks.append("ОБЯЗАТЕЛЬНО: ИНН плательщика и получателя совпадают — возможна ошибка или мошенничество.")

    amount_words = extracted.get("total_words", "")
    amount_digits = extracted.get("total")
    if amount_words and amount_digits:
        checks.append(
            f"ПРОВЕРЬ: Сумма цифрами {amount_digits} — убедись что сумма прописью '{amount_words}' соответствует ей точно."
        )

    return checks

def validate(extracted: dict) -> dict:
    relevant_rules = find_relevant_rules(extracted)
    mandatory_checks = build_mandatory_checks(extracted)

    mandatory_section = ""
    if mandatory_checks:
        mandatory_section = "\n\nMANDATORY VIOLATIONS DETECTED (must appear in issues):\n" + \
            "\n".join(f"- {c}" for c in mandatory_checks)

    response = client.chat.completions.create(
        model=VALIDATE_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
Verify the document against the regulations.

Document data:
{json.dumps(extracted, ensure_ascii=False, indent=2)}

Applicable regulations:
{chr(10).join(f"- {r}" for r in relevant_rules)}
{mandatory_section}
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