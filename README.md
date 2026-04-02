# Doc Validator

Сервис автоматической проверки финансовых документов на соответствие регламентам компании.

Загружаешь счёт, договор, акт или платёжное поручение — система извлекает данные, сверяет с регламентами и выносит вердикт с объяснением.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-Groq-orange?style=flat-square)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple?style=flat-square)
![pytest](https://img.shields.io/badge/Tests-pytest-green?style=flat-square)

---

## Как это работает

```
PDF / скан / текст
      ↓
OCR (pdfplumber / tesseract) — извлекает текст из файла
      ↓
LLM Extract — структурирует данные в JSON     [llama-3.1-8b-instant]
      ↓
Hard Checks — принудительные проверки по сумме, категории, валюте
      ↓
RAG Retrieve — семантический поиск регламентов [ChromaDB + embeddings]
      ↓
LLM Validate — сверяет данные с регламентами  [llama-3.3-70b-versatile]
      ↓
Результат: APPROVED / REJECTED
+ список нарушений с приоритетом
+ uncertain_fields (что модель не смогла проверить)
+ human_comment (вывод на русском для сотрудника)
```

**Стек:** Python · FastAPI · Groq API · ChromaDB · Sentence Transformers · pdfplumber · Tesseract OCR

---

## Возможности

- Приём документов в виде текста, PDF или изображения (JPG, PNG)
- Извлечение структурированных данных через LLM с постобработкой типов
- Принудительные проверки по числовым порогам и категориям до вызова LLM
- Семантический поиск релевантных регламентов через мульти-запросную стратегию
- Валидация с уровнями серьёзности: `ошибка` / `предупреждение`
- Приоритеты нарушений: `БЛОКИРУЮЩИЙ` / `ТРЕБУЕТ_ВНИМАНИЯ` / `ИНФОРМАЦИОННЫЙ`
- Обработка неопределённости: `uncertain_fields` с объяснением сомнений
- Веб-интерфейс в стиле министерства с историей и статистикой
- REST API с автодокументацией (Swagger UI)
- Журнал обработанных документов с эндпоинтами `/history` и `/stats`
- Управление регламентами через `rules.txt` без изменения кода
- Покрытие тестами: extractor, retrieval, pipeline (pytest)

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

Для OCR сканов дополнительно установи Tesseract:

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-rus
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

При первом запуске регламенты из `rules.txt` автоматически загрузятся в ChromaDB.

### 5. Открой интерфейс

Веб UI: открой `doc_validator_ui.html` в браузере

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Примеры запросов

### Текстовый документ

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "СЧЁТ №47 от 10.04.2024. Поставщик: ООО Альфа-Софт, ИНН 7701234567. Лицензия 1С:Бухгалтерия — 28 000 руб."
  }'
```

### PDF файл

```bash
curl -X POST http://localhost:8000/process-file \
  -F "file=@invoice.pdf"
```

### Ответ

```json
{
  "status": "REJECTED",
  "extracted": {
    "number": "47",
    "date": "2024-04-10",
    "doc_type": "invoice",
    "supplier_name": "ООО Альфа-Софт",
    "supplier_inn": "7701234567",
    "total": 28000.0,
    "currency": "RUB",
    "category": "ПО"
  },
  "validation": {
    "approved": false,
    "doc_type": "invoice",
    "severity": "ошибка",
    "issues": [
      {
        "rule": "Закупка ПО требует согласования IT-отдела",
        "description": "Позиция является программным обеспечением — виза IT-отдела обязательна",
        "severity": "ошибка",
        "priority": "БЛОКИРУЮЩИЙ"
      }
    ],
    "uncertain_fields": [],
    "approvals_needed": ["IT-отдел"],
    "rules_checked": ["Закупка любого ПО и лицензий требует согласования IT-отдела."],
    "human_comment": "Счёт не может быть проведён. Направьте на визирование в IT-отдел."
  }
}
```

---

## API эндпоинты

| Метод | URL | Описание |
|---|---|---|
| GET | `/health` | Статус сервиса |
| POST | `/process` | Обработка текстового документа |
| POST | `/process-file` | Обработка PDF или изображения |
| GET | `/history` | История обработанных документов |
| GET | `/stats` | Статистика: всего / одобрено / отклонено |

---

## Управление регламентами

Правила хранятся в `rules.txt` — по одному на строку. Строки начинающиеся с `#` — комментарии.

```
# Счета и счета-фактуры
Закупка любого ПО и лицензий требует согласования IT-отдела.
Счета свыше 50 000 руб. требуют визы финансового директора.

# Государственные контракты
Договоры свыше 1 000 000 руб. требуют согласования юридического отдела.
```

При изменении файла и перезапуске сервиса база обновляется автоматически. Пересоздавать вручную не нужно.

---

## Структура проекта

```
doc-validator/
├── main.py                  — FastAPI сервис, эндпоинты, pipeline
├── extractor.py             — LLM извлечение данных из документа
├── validator.py             — Hard checks + RAG поиск + LLM валидация
├── ocr.py                   — извлечение текста из PDF и изображений
├── rules.txt                — регламенты компании (редактируется без кода)
├── history/
│   └── history.py           — журнал обработанных документов
├── test/
│   ├── test_extractor.py    — тесты извлечения данных
│   ├── test_retrieval.py    — тесты семантического поиска
│   └── test_pipeline.py     — end-to-end тесты API
├── doc_validator_ui.html    — веб-интерфейс
├── requirements.txt
└── .env                     — GROQ_API_KEY (не коммитится)
```

---

## Запуск тестов

```bash
pytest test/ -v
```

```
test/test_extractor.py::test_extract_invoice_basic        PASSED
test/test_extractor.py::test_extract_contract             PASSED
test/test_extractor.py::test_extract_foreign_currency     PASSED
test/test_pipeline.py::test_pipeline_status[software]     PASSED
test/test_pipeline.py::test_pipeline_status[contract]     PASSED
test/test_retrieval.py::test_retrieval_software_purchase  PASSED
...
====== 16 passed ======
```

---

## Архитектурные решения

**Две модели с разными ролями** — лёгкая `llama-3.1-8b-instant` для извлечения данных (скорость), тяжёлая `llama-3.3-70b-versatile` для валидации (точность). Разделение ответственности снижает стоимость и повышает качество.

**Hard checks до LLM** — числовые пороги и категории проверяются детерминированно в коде. LLM только форматирует и объясняет уже известные нарушения. Это устраняет ошибки модели на критичных правилах.

**Мульти-запросный RAG** — для поиска регламентов используется три семантических запроса вместо одного: по категории, по сумме, по валюте. Объединение результатов повышает полноту retrieval.

**rules.txt вместо hardcode** — регламенты живут в текстовом файле. База ChromaDB пересоздаётся только при изменении файла (хэш-сравнение). Добавить правило = одна строка в файле.

**`temperature=0`** — детерминированные ответы. Один и тот же документ всегда даёт одинаковый результат. Важно для аудита и воспроизводимости.

**uncertain_fields** — если модель не уверена в поле, она не пропускает документ, а явно указывает что именно вызывает сомнение и какие данные нужны для проверки.

---

## Автор

Создан как пет-проект для изучения RAG архитектуры, prompt engineering и agentic программирования.
