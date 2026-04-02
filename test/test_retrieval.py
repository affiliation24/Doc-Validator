import pytest
from validator import load_rules, find_relevant_rules

@pytest.fixture(autouse=True)
def setup_rules():
    load_rules()

def test_retrieval_software_purchase():
    extracted = {"category": "ПО", "total": 68400, "currency": "RUB", "supplier_name": "ООО ТехноСервис"}
    rules = find_relevant_rules(extracted)

    assert len(rules) > 0
    combined = " ".join(rules).lower()
    assert "по" in combined or "программ" in combined or "лицензи" in combined


def test_retrieval_large_equipment():
    extracted = {"category": "оборудование", "total": 650000, "currency": "RUB", "supplier_name": "ИП Иванов"}
    rules = find_relevant_rules(extracted)

    assert len(rules) > 0
    combined = " ".join(rules).lower()
    assert "тендер" in combined or "500" in combined


def test_retrieval_foreign_currency():
    extracted = {"category": "услуги", "total": 5000, "currency": "USD", "supplier_name": "ООО Форвард"}
    rules = find_relevant_rules(extracted)

    assert len(rules) > 0
    combined = " ".join(rules).lower()
    assert "валют" in combined or "цб" in combined or "курс" in combined


def test_retrieval_returns_correct_count():
    extracted = {"category": "ПО", "total": 30000, "currency": "RUB", "supplier_name": "ООО Тест"}
    rules = find_relevant_rules(extracted, top_k=3)

    assert len(rules) == 3


def test_retrieval_no_meta_in_results():
    extracted = {"category": "ПО", "total": 68400, "currency": "RUB", "supplier_name": "ООО Тест"}
    rules = find_relevant_rules(extracted)

    for rule in rules:
        assert len(rule) > 20
        assert rule != rules[0] or rules.count(rule) == 1