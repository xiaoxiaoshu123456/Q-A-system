"""Lightweight JSONL conversation logger.

- Appends per-turn chat records to logs/conversations.jsonl for analytics.
- Each entry includes question/answer, provider/model/base_url, timing, etc.
"""
import json
import time
from pathlib import Path
from typing import Any


def log_path(root: Path) -> Path:
    """Return the path to the JSONL file used for conversation logging."""
    return root / "logs" / "conversations.jsonl"


def append(root: Path, entry: dict[str, Any]) -> None:
    """Append a single conversation entry as one JSONL line, adding ts if missing."""
    p = log_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    if "ts" not in entry:
        entry["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
