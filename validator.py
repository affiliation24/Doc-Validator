import hashlib
import json
import re
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from llm import client, parse_json_object, with_rate_limit_retry

VALIDATE_MODEL = os.getenv("VALIDATE_MODEL", "openai/gpt-oss-120b")

BASE_DIR = Path(__file__).resolve().parent
RULES_FILE = os.getenv("RULES_FILE", str(BASE_DIR / "rules.txt"))
RULES_DB_PATH = os.getenv("RULES_DB_PATH", str(BASE_DIR / "rules_db"))

EMBEDDING_DIM = 384

# Порог отсечения нерелевантных правил: 1.0 соответствует косинусной близости
# 0.5 (см. embed() — расстояние считается по нормированным векторам). Подобрано
# по rules.txt: строже — теряются применимые правила, мягче — выдача не растёт.
MAX_DISTANCE = 1.0

# Сколько правил уходит в промпт валидации. По платёжным поручениям в rules.txt
# больше полусотни правил, и при пяти до модели доходила лишь их малая часть.
TOP_K = int(os.getenv("RULES_TOP_K", "8"))

# Единственный источник истины по порогам: и код (build_mandatory_checks),
# и системный промпт собираются из этих констант.
CFO_LIMIT = 50_000
HEAD_LIMIT = 150_000
TENDER_LIMIT = 500_000
LEGAL_CONTRACT_LIMIT = 1_000_000

BLOCKING_PREFIX = "ОБЯЗАТЕЛЬНО:"
ADVISORY_PREFIX = "ПРОВЕРЬ:"

# Мультиязычная модель обязательна: регламенты и запросы на русском,
# англоязычная all-MiniLM-L6-v2 давала фактически случайную выдачу.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

embedder = SentenceTransformer(EMBEDDING_MODEL)
db = chromadb.PersistentClient(path=RULES_DB_PATH)
rules_col = db.get_or_create_collection("rules")


def embed(texts: list[str]) -> list[list[float]]:
    """Эмбеддинги единичной длины.

    Chroma считает квадрат L2. На нормированных векторах он равен 2 - 2*cos,
    то есть превращается в косинусное расстояние — только с ним имеет смысл
    фиксированный порог релевантности MAX_DISTANCE.
    """
    return embedder.encode(texts, normalize_embeddings=True).tolist()

CRITICAL_RULES = f"""
CRITICAL RULES — apply these unconditionally before anything else:
- ANY software, license, SaaS, or IT subscription → IT department approval REQUIRED, mark as "ошибка"
- ANY invoice or payment over {CFO_LIMIT:,} RUB → CFO (финансовый директор) approval REQUIRED
- ANY invoice or payment over {HEAD_LIMIT:,} RUB → head of department (руководитель подразделения) approval REQUIRED
- ANY contract over {LEGAL_CONTRACT_LIMIT:,} RUB → Legal department (юридический отдел) approval REQUIRED
- ANY purchase over {TENDER_LIMIT:,} RUB → tender (тендер) REQUIRED
- ANY foreign currency (USD, EUR) → CBR exchange rate REQUIRED, mark as "ошибка"
- ANY new supplier not in registry → security check REQUIRED, mark as "предупреждение"
"""

SYSTEM_PROMPT = "\nYou are a strict compliance officer at a government ministry.\n" + CRITICAL_RULES + """
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
- LANGUAGE: every string value in the response MUST be written in Russian.
  This includes "rule", "description", "approvals_needed" and "human_comment".
  These strings are shown to a Russian ministry employee as-is.
  Writing them in English is an error, even though this instruction is in English.
  Wrong: "rule": "CFO approval required for invoices over 50 000 RUB"
  Right: "rule": "Требуется виза финансового директора для счетов свыше 50 000 руб."
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


# rules.txt разбит на секции-заголовки, и это готовая разметка по типам
# документов. Секция, которой здесь нет, считается применимой ко всем
# документам — новый заголовок в файле не ломает поиск, а лишь не сужает его.
SECTION_DOC_TYPES = {
    "СЧЕТА И СЧЕТА-ФАКТУРЫ": "invoice",
    "ГОСУДАРСТВЕННЫЕ КОНТРАКТЫ И ДОГОВОРЫ": "contract",
    "ПЛАТЁЖНЫЕ ПОРУЧЕНИЯ": "payment_order",
    "БЮДЖЕТНЫЕ ПЛАТЕЖИ": "payment_order",
    "АКТЫ ВЫПОЛНЕННЫХ РАБОТ": "act",
    "СЛУЖЕБНЫЕ ЗАПИСКИ И СОГЛАСОВАНИЯ": "approval",
}

ANY_DOC_TYPE = "any"


def read_rules_from_file(path: str) -> list[tuple[str, str]]:
    """Читает правила вместе с типом документа, к которому они относятся.

    Возвращает пары (текст правила, doc_type). doc_type берётся из заголовка
    секции — благодаря этому поиск по платёжному поручению не забивается
    правилами про счета и акты.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл правил не найден: {path}")

    rules = []
    doc_type = ANY_DOC_TYPE
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                title = line.lstrip("#").strip().strip("═").strip()
                if title:
                    doc_type = SECTION_DOC_TYPES.get(title.upper(), ANY_DOC_TYPE)
                continue
            if line:
                rules.append((line, doc_type))
    return rules


def get_rules_hash(rules: list[tuple[str, str]]) -> str:
    """Отпечаток индекса: правила с их секциями + модель эмбеддингов.

    Модель входит в хэш намеренно — при её смене старые векторы становятся
    несравнимы с новыми запросами, индекс обязан пересобраться. Секция тоже:
    перенос правила в другую секцию меняет то, когда оно находится.
    """
    content = f"{EMBEDDING_MODEL}|normalized|v2|" + "\n".join(f"{d}|{r}" for r, d in rules)
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
        embeddings=[[0.0] * EMBEDDING_DIM],
        ids=["__meta__"],
        metadatas=[{"type": "meta"}]
    )

    texts = [text for text, _ in rules]
    rules_col.add(
        documents=texts,
        embeddings=embed(texts),
        ids=[f"rule_{i}" for i in range(len(rules))],
        metadatas=[{"type": "rule", "doc_type": doc_type} for _, doc_type in rules]
    )
    print(f"Загружено правил: {len(rules)}")


DOC_TYPE_NAMES = {
    "invoice": "счёт",
    "contract": "договор",
    "payment_order": "платёжное поручение",
    "act": "акт выполненных работ",
    "approval": "служебная записка",
}

# Категория из извлечения — одно слово, для эмбеддинга этого мало.
# Развёрнутая формулировка заметно поднимает близость к нужному регламенту.
CATEGORY_TERMS = {
    "ПО": "программное обеспечение, лицензия, подписка, IT-система",
    "оборудование": "оборудование, техника, компьютеры, серверы, приборы",
    "услуги": "услуги, работы, подряд, консультации",
    "прочее": "товары и материалы",
}


def build_queries(extracted: dict) -> list[str]:
    """Запросы под конкретный документ: спрашиваем только о том, что в нём есть.

    Безусловный запрос про курс ЦБ поднимал валютное правило и для рублёвых
    документов, вытесняя действительно применимые регламенты.
    """
    category = extracted.get("category") or "прочее"
    total = float(extracted.get("total") or 0)
    currency = extracted.get("currency") or "RUB"
    doc_type = DOC_TYPE_NAMES.get(extracted.get("doc_type"), "документ")

    terms = CATEGORY_TERMS.get(category, category)
    queries = [
        f"закупка: {terms} — требуется согласование профильного отдела",
        f"обязательные реквизиты, которые должен содержать {doc_type}",
    ]

    if total > CFO_LIMIT:
        queries.append(f"{doc_type} на сумму {total:,.0f} руб. — виза, тендер, согласование по лимиту")
    if currency != "RUB":
        queries.append(f"документ в иностранной валюте {currency}: курс ЦБ, валютный контроль")
    if extracted.get("supplier_inn") or extracted.get("bank_account"):
        queries.append("проверка ИНН, КПП, БИК и расчётного счёта поставщика")

    return queries


def rules_filter(doc_type: str | None) -> dict:
    """Ограничивает поиск правилами своей секции плюс общими для всех типов.

    Если тип документа не распознан, сужать нельзя: ищем по всей базе. Иначе
    сбой извлечения молча отрезал бы валидатор от правил конкретных типов —
    лучше лишние правила в промпте, чем пропущенное нарушение.
    """
    if doc_type not in SECTION_DOC_TYPES.values():
        return {"type": {"$eq": "rule"}}

    return {"$and": [
        {"type": {"$eq": "rule"}},
        {"$or": [
            {"doc_type": {"$eq": ANY_DOC_TYPE}},
            {"doc_type": {"$eq": doc_type}},
        ]},
    ]}


def find_relevant_rules(extracted: dict, top_k: int = TOP_K, max_distance: float = MAX_DISTANCE) -> list[str]:
    queries = build_queries(extracted)
    where = rules_filter(extracted.get("doc_type"))

    # Расстояние по каждому правилу — минимальное среди всех запросов:
    # правило релевантно, если его нашёл хотя бы один аспект документа.
    best: dict[str, float] = {}
    for query in queries:
        results = rules_col.query(
            query_embeddings=embed([query]),
            n_results=top_k,
            where=where
        )
        for doc, distance in zip(results["documents"][0], results["distances"][0]):
            if distance <= max_distance and distance < best.get(doc, float("inf")):
                best[doc] = distance

    ranked = sorted(best, key=best.get)
    return ranked[:top_k]


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
    """Проверяет БИК по длине и по разрядам, кодирующим страну и регион.

    Требовать префикс '04' нельзя: первый разряд БИК кодирует вид участия в
    платёжной системе, и у подразделений Казначейства он другой — например,
    БИК для единого налогового платежа 017003983. Прежняя проверка отклоняла
    каждую бюджетную платёжку.
    """
    if not bik:
        return True, ""
    bik = str(bik).strip()
    if not bik.isdigit():
        return False, f"БИК содержит нецифровые символы: {bik}"
    if len(bik) != 9:
        return False, f"БИК содержит {len(bik)} цифр вместо 9"
    if bik[:2] != "04" and bik[:2] != "01":
        return False, (
            f"БИК начинается с '{bik[:2]}': у российских банков первые два разряда 04, "
            f"у подразделений Казначейства — 01. Проверьте реквизит: {bik}"
        )
    return True, ""


def validate_digits(value, name: str, lengths: tuple[int, ...]) -> tuple[bool, str]:
    """Общая проверка цифрового реквизита фиксированной длины.

    Пустое значение пропускается: «поля нет» — это забота регламентов и LLM,
    здесь проверяется только формат того, что заполнено.
    """
    if value in (None, ""):
        return True, ""
    value = str(value).strip()
    if not value.isdigit():
        return False, f"{name} содержит нецифровые символы: {value}"
    if len(value) not in lengths:
        expected = " или ".join(str(n) for n in lengths)
        return False, f"{name} содержит {len(value)} цифр вместо {expected}"
    return True, ""


def build_payment_order_checks(extracted: dict) -> list[str]:
    """Проверки реквизитов бланка 0401060 — только для платёжных поручений."""
    checks = []

    corr = extracted.get("bank_corr_account")
    ok, err = validate_digits(corr, "Корреспондентский счёт банка получателя", (20,))
    if not ok:
        checks.append(f"{BLOCKING_PREFIX} {err}.")
    elif corr and not str(corr).startswith("301"):
        checks.append(
            f"{BLOCKING_PREFIX} Корреспондентский счёт банка получателя должен начинаться с 301, "
            f"а начинается с {str(corr)[:3]}: {corr}."
        )

    for field, name, lengths in (
        ("payer_account", "Расчётный счёт плательщика", (20,)),
        ("kbk", "КБК", (20,)),
        ("oktmo", "ОКТМО", (8, 11)),
        ("payer_status", "Статус плательщика (поле 101)", (2,)),
    ):
        ok, err = validate_digits(extracted.get(field), name, lengths)
        if not ok:
            checks.append(f"{BLOCKING_PREFIX} {err}.")

    ok, err = validate_bik(extracted.get("payer_bank_bik"))
    if not ok:
        checks.append(f"{BLOCKING_PREFIX} Некорректный БИК банка плательщика — {err}.")

    ok, err = validate_kpp(extracted.get("payer_kpp"))
    if not ok:
        checks.append(f"{BLOCKING_PREFIX} Некорректный КПП плательщика — {err}.")

    priority = extracted.get("payment_priority")
    if priority in (None, ""):
        checks.append(f"{BLOCKING_PREFIX} Не указана очерёдность платежа — поле обязательно, допустимы цифры от 1 до 5.")
    elif str(priority).strip() not in {"1", "2", "3", "4", "5"}:
        checks.append(f"{BLOCKING_PREFIX} Очерёдность платежа должна быть цифрой от 1 до 5, указано: {priority}.")

    uin = extracted.get("uin")
    if uin not in (None, "") and str(uin).strip() != "0":
        ok, err = validate_digits(uin, "Код УИН", (20, 25))
        if not ok:
            checks.append(f"{BLOCKING_PREFIX} {err} (либо должен быть указан 0, если УИН не присвоен).")

    operation = extracted.get("operation_type")
    if operation not in (None, "") and str(operation).strip() != "01":
        checks.append(
            f"{BLOCKING_PREFIX} Вид операции для платёжного поручения должен быть 01, указано: {operation}."
        )

    purpose = extracted.get("payment_purpose")
    if not purpose:
        checks.append(f"{BLOCKING_PREFIX} Не заполнено назначение платежа.")
    elif not re.search(r"НДС", str(purpose), re.IGNORECASE):
        checks.append(
            f"{ADVISORY_PREFIX} В назначении платежа не упомянут НДС — проверь, "
            f"должно быть указано «в т.ч. НДС» или «Без НДС»: «{purpose}»."
        )

    payer_account = extracted.get("payer_account")
    account = extracted.get("bank_account")
    if payer_account and account and payer_account == account:
        checks.append(f"{BLOCKING_PREFIX} Счёт плательщика и счёт получателя совпадают — платёж самому себе.")

    return checks


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
        checks.append(f"{BLOCKING_PREFIX} Категория 'ПО' — требуется согласование IT-отдела. Это блокирующее нарушение.")
    if total > CFO_LIMIT:
        checks.append(f"{BLOCKING_PREFIX} Сумма {total} руб. превышает {CFO_LIMIT:,} — требуется виза финансового директора.")
    if total > HEAD_LIMIT:
        checks.append(f"{BLOCKING_PREFIX} Сумма {total} руб. превышает {HEAD_LIMIT:,} — требуется виза руководителя подразделения.")
    if total > TENDER_LIMIT:
        checks.append(f"{BLOCKING_PREFIX} Сумма {total} руб. превышает {TENDER_LIMIT:,} — требуется тендер.")
    if doc_type == "contract" and total > LEGAL_CONTRACT_LIMIT:
        checks.append(f"{BLOCKING_PREFIX} Договор на {total} руб. превышает {LEGAL_CONTRACT_LIMIT:,} — требуется согласование юридического отдела.")
    if currency in ("USD", "EUR"):
        checks.append(f"{BLOCKING_PREFIX} Валюта {currency} — требуется курс ЦБ на дату документа.")

    inn = extracted.get("supplier_inn", "")
    inn_ok, inn_err = validate_inn(inn)
    if not inn_ok:
        checks.append(f"{BLOCKING_PREFIX} Некорректный ИНН поставщика — {inn_err}.")

    kpp = extracted.get("supplier_kpp", "")
    kpp_ok, kpp_err = validate_kpp(kpp)
    if not kpp_ok:
        checks.append(f"{BLOCKING_PREFIX} Некорректный КПП поставщика — {kpp_err}.")

    bik = extracted.get("bank_bik", "")
    bik_ok, bik_err = validate_bik(bik)
    if not bik_ok:
        checks.append(f"{BLOCKING_PREFIX} Некорректный БИК банка — {bik_err}.")

    account = extracted.get("bank_account", "")
    acc_ok, acc_err = validate_account(account)
    if not acc_ok:
        checks.append(f"{BLOCKING_PREFIX} Некорректный расчётный счёт — {acc_err}.")

    payer_inn = extracted.get("payer_inn", "")
    if payer_inn and inn and payer_inn == inn:
        checks.append(f"{BLOCKING_PREFIX} ИНН плательщика и получателя совпадают — возможна ошибка или мошенничество.")

    amount_words = extracted.get("total_words", "")
    amount_digits = extracted.get("total")
    if amount_words and amount_digits:
        checks.append(
            f"{ADVISORY_PREFIX} Сумма цифрами {amount_digits} — убедись что сумма прописью '{amount_words}' соответствует ей точно."
        )

    if doc_type == "payment_order":
        checks.extend(build_payment_order_checks(extracted))

    return checks

def validate(extracted: dict) -> dict:
    relevant_rules = find_relevant_rules(extracted)
    mandatory_checks = build_mandatory_checks(extracted)

    mandatory_section = ""
    if mandatory_checks:
        mandatory_section = "\n\nMANDATORY VIOLATIONS DETECTED (must appear in issues):\n" + \
            "\n".join(f"- {c}" for c in mandatory_checks)

    response = with_rate_limit_retry(lambda: client.chat.completions.create(
        model=VALIDATE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
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
    ))

    result = parse_json_object(response.choices[0].message.content, f"Модель {VALIDATE_MODEL}")
    result = normalize_unicode(result)
    result["rules_checked"] = relevant_rules
    return enforce_blocking_checks(result, mandatory_checks)


# Модель охотно ставит типографские дефисы и неразрывные пробелы: 'IT‑отдел'
# с U+2011 выглядит как 'IT-отдел', но не равен ему ни при сравнении, ни при
# поиске. Тире (– —) не трогаем — это осмысленная пунктуация.
LOOKALIKE_CHARS = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "−": "-",
    " ": " ", " ": " ", " ": " ",
})


def normalize_unicode(value):
    """Рекурсивно приводит строки ответа к обычным дефисам и пробелам."""
    if isinstance(value, str):
        return value.translate(LOOKALIKE_CHARS)
    if isinstance(value, dict):
        return {k: normalize_unicode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_unicode(v) for v in value]
    return value


def _already_described(check_text: str, described: str) -> bool:
    """Совпадают ли ключевые слова hard check с тем, что уже написала модель."""
    keywords = {w for w in re.findall(r"[\w\-]{5,}", check_text.lower()) if not w.isdigit()}
    if not keywords:
        return False
    hits = sum(1 for w in keywords if w in described)
    return hits >= max(2, len(keywords) // 3)


def enforce_blocking_checks(result: dict, mandatory_checks: list[str]) -> dict:
    """Детерминированные нарушения решают вердикт, а не модель.

    LLM отвечает за формулировки, но если hard checks нашли блокирующее
    нарушение, документ не может быть одобрен — даже если модель так решила.
    """
    blocking = [c for c in mandatory_checks if c.startswith(BLOCKING_PREFIX)]
    if not blocking:
        return result

    result["approved"] = False
    result["severity"] = "ошибка"

    issues = result.get("issues") or []
    described = " ".join(
        f"{i.get('rule', '')} {i.get('description', '')}" for i in issues if isinstance(i, dict)
    ).lower()

    for check in blocking:
        text = check[len(BLOCKING_PREFIX):].strip()
        # Модель обычно уже описала это нарушение своими словами — не дублируем.
        if _already_described(text, described):
            continue
        issues.append({
            "rule": "Автоматическая проверка реквизитов и лимитов",
            "description": text,
            "severity": "ошибка",
            "priority": "БЛОКИРУЮЩИЙ",
        })

    result["issues"] = issues
    return result