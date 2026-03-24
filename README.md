# Doc Validator

Сервис автоматической проверки финансовых документов на соответствие регламентам компании.

Загружаешь счёт или акт — система извлекает данные и сверяет с правилами. Без ручной проверки.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-Groq-orange?style=flat-square)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple?style=flat-square)

---

## Как это работает

```
Документ (текст)
      ↓
LLM Extract — извлекает поля в JSON          [llama-3.1-8b-instant]
      ↓
RAG Validate — ищет релевантные регламенты   [ChromaDB + embeddings]
      ↓
LLM Validate — сверяет данные с регламентами [llama-3.3-70b-versatile]
      ↓
Результат: APPROVED / REJECTED + список нарушений
```

**Стек:** Python · FastAPI · Groq API · ChromaDB · Sentence Transformers

---

## Возможности

- Извлечение структурированных данных из любого формата документа
- Семантический поиск релевантных регламентов через векторную базу данных
- Проверка документа по правилам компании с объяснением нарушений
- REST API с автодокументацией (Swagger UI)
- Веб-интерфейс для ручного тестирования

---

## Быстрый старт

### 1. Клонируй репозиторий

```bash
git clone https://github.com/твой-username/doc-validator.git
cd doc-validator
```

### 2. Установи зависимости

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Создай .env файл

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

Получить бесплатный ключ: [console.groq.com](https://console.groq.com)

### 4. Запусти сервис

```bash
uvicorn main:app --reload --port 8000
```

### 5. Открой интерфейс

Веб UI: открой `doc_validator_ui.html` в браузере

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Пример запроса

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "СЧЁТ №47 от 10.04.2024. Поставщик: ООО Альфа-Софт, ИНН 7701234567. Лицензия 1С:Бухгалтерия — 28 000 руб."
  }'
```

### Ответ

```json
{
  "status": "REJECTED",
  "extracted": {
    "number": "47",
    "date": "2024-04-10",
    "supplier_name": "ООО Альфа-Софт",
    "supplier_inn": "7701234567",
    "total": 28000,
    "currency": "RUB",
    "category": "ПО"
  },
  "validation": {
    "approved": false,
    "issues": ["Закупка ПО требует согласования IT-отдела"],
    "approvals_needed": ["IT-отдел"],
    "comment": "Счёт может быть проведён после получения визы IT-отдела"
  }
}
```

---

## Структура проекта

```
doc-validator/
├── main.py                 — FastAPI сервис, эндпоинты
├── extractor.py            — LLM извлечение данных из документа
├── validator.py            — RAG поиск + LLM валидация
├── doc_validator_ui.html   — веб-интерфейс
├── requirements.txt        — зависимости
└── .env                    — GROQ_API_KEY (не коммитится)
```

---

## API эндпоинты

| Метод | URL | Описание |
|---|---|---|
| GET | `/health` | Статус сервиса |
| POST | `/process` | Обработка документа |

---

## Архитектурные решения

**Две модели вместо одной** — лёгкая модель (`llama-3.1-8b-instant`) для извлечения данных, тяжёлая (`llama-3.3-70b-versatile`) для валидации. Баланс между скоростью и точностью.

**RAG вместо hardcode правил** — регламенты хранятся в ChromaDB как векторы. Обновить правила = добавить строку в список, без изменения кода.

**`temperature=0`** — детерминированные ответы. Один документ всегда даёт одинаковый результат.

---

## Автор

Создан как пет-проект для изучения RAG архитектуры и agentic программирования.
