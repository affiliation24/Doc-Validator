import pytest
from extractor import extract

def test_extract_invoice_basic():
    text = """СЧЁТ-ФАКТУРА №СФ-2024-0891 от 15.03.2024
    Поставщик: ООО ТехноСервис, ИНН 7712345678
    Лицензия MS Office 365 — 68 400 руб."""

    result = extract(text)

    assert result["number"] == "СФ-2024-0891"
    assert result["supplier_inn"] == "7712345678"
    assert result["total"] == 68400
    assert result["currency"] == "RUB"
    assert result["category"] == "ПО"


def test_extract_contract():
    text = """ДОГОВОР №88 от 01.04.2024
    Исполнитель: ООО СтройГрупп, ИНН 7709876543
    Строительные работы — 1 500 000 руб."""

    result = extract(text)

    assert float(result["total"]) == 1500000
    assert result["supplier_inn"] == "7709876543"
    assert result["doc_type"] == "contract" 


def test_extract_foreign_currency():
    text = """СЧЁТ №5 от 05.04.2024
    Поставщик: ООО Форвард, ИНН 7701111222
    Консалтинг — 5 000 USD."""

    result = extract(text)

    assert result["currency"] == "USD"
    assert float(result["total"]) == 5000


def test_extract_returns_null_for_missing_fields():
    text = "Какой-то документ без реквизитов"

    result = extract(text)

    assert isinstance(result, dict)
    assert "total" in result
    assert "number" in result


def test_extract_individual_inn():
    text = """АКТ №12 от 20.03.2024
    Исполнитель: ИП Петров Иван, ИНН 123456789012
    Курьерская доставка — 3 000 руб."""

    result = extract(text)

    assert result["supplier_inn"] == "123456789012"
    assert len(result["supplier_inn"]) == 12