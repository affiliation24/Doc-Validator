import pytest
import requests

BASE = "http://localhost:8000"

CASES = [
    {
        "id": "software_over_limit",
        "name": "ПО > 50к → REJECTED (IT-отдел + ФД)",
        "text": "СЧЁТ №СФ-2024-0891 от 15.03.2024. Поставщик: ООО ТехноСервис, ИНН 7712345678. Лицензия MS Office 365 — 68 400 руб.",
        "expected_status": "REJECTED",
        "expected_approvals": ["IT-отдел", "Финансовый директор"],
    },
    {
    "id": "small_act_approved",
    "name": "Маленький акт — сумма ОК, новый поставщик",
    "text": "АКТ №12 от 20.03.2024. Поставщик: ИП Петров, ИНН 123456789012. Курьерская доставка — 3 000 руб.",
    "expected_status": "APPROVED",   
    "expected_approvals": [],
    }
]


@pytest.fixture(scope="session", autouse=True)
def check_service():
    try:
        r = requests.get(f"{BASE}/health", timeout=3)
        assert r.status_code == 200
    except Exception:
        pytest.skip("Сервис недоступен — запусти: uvicorn main:app --reload --port 8000")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_pipeline_status(case):
    r = requests.post(f"{BASE}/process", json={"text": case["text"]})
    assert r.status_code == 200
    assert r.json()["status"] == case["expected_status"]


@pytest.mark.parametrize("case", [c for c in CASES if c["expected_approvals"]], ids=[c["id"] for c in CASES if c["expected_approvals"]])
def test_pipeline_approvals(case):
    r = requests.post(f"{BASE}/process", json={"text": case["text"]})
    data = r.json()
    approvals = data["validation"].get("approvals_needed", [])

    for expected in case["expected_approvals"]:
        assert any(expected.lower() in a.lower() for a in approvals), \
            f"Ожидали '{expected}' в согласованиях, получили: {approvals}"


def test_pipeline_response_schema():
    r = requests.post(f"{BASE}/process", json={"text": "СЧЁТ №1. Поставщик: ООО Тест, ИНН 7701234567. Сумма: 10 000 руб."})
    data = r.json()

    assert "status" in data
    assert "extracted" in data
    assert "validation" in data
    assert "approved" in data["validation"]
    assert "issues" in data["validation"]
    assert "human_comment" in data["validation"]
    assert "rules_checked" in data["validation"]


def test_pipeline_empty_text_returns_400():
    r = requests.post(f"{BASE}/process", json={"text": ""})
    assert r.status_code == 400


def test_pipeline_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"