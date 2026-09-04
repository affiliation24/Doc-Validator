import json
import os
from datetime import datetime

HISTORY_FILE = "history/processed.json"

def _load() -> list:
    if not os.path.exists(HISTORY_FILE):
        os.makedirs("history", exist_ok=True)
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save(records: list):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def add_record(extracted: dict, validation: dict, source: str = "text"):
    records = _load()
    records.append({
        "id": len(records) + 1,
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "status": "APPROVED" if validation.get("approved") else "REJECTED",
        "extracted": extracted,
        "validation": validation
    })
    _save(records)

def get_history() -> list:
    return _load()

def get_stats() -> dict:
    records = _load()
    if not records:
        return {"total": 0, "approved": 0, "rejected": 0}
    approved = sum(1 for r in records if r["status"] == "APPROVED")
    return {
        "total": len(records),
        "approved": approved,
        "rejected": len(records) - approved,
    }