"""Upload manifest helpers for persisting ingested files.

- Stores records in logs/uploads.json with fields: 文件, 状态, grandpa_id, 时间, hash.
- Provides update utilities for renaming entries by grandpa_id.
"""
import json
import hashlib
from pathlib import Path


def manifest_path(root: Path) -> Path:
    """Return the path to the uploads.json manifest under logs/"""
    return root / "logs" / "uploads.json"


def load_manifest(root: Path) -> list[dict]:
    """Load manifest entries as a list of dicts; returns [] if missing or invalid."""
    path = manifest_path(root)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except Exception:
        pass
    return []


def save_manifest(root: Path, items: list[dict]) -> None:
    """Write manifest entries to disk as pretty-printed UTF-8 JSON."""
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def update_manifest_name(root: Path, grandpa_id: str, new_name: str) -> None:
    """Update all entries matching grandpa_id to use the provided display name."""
    grandpa_id = (grandpa_id or "").strip()
    if not grandpa_id:
        return
    uploads = load_manifest(root)
    changed = False
    for item in uploads:
        if item.get("grandpa_id") == grandpa_id:
            item["文件"] = new_name
            changed = True
    if changed:
        save_manifest(root, uploads)


def md5_hex(data: bytes) -> str:
    """Compute an MD5 hex digest for the given bytes."""
    return hashlib.md5(data).hexdigest()


def find_by_hash(items: list[dict], file_hash: str) -> dict | None:
    """Find a manifest entry by its file hash; returns None if not found."""
    file_hash = (file_hash or "").strip()
    if not file_hash:
        return None
    for item in items:
        if item.get("hash") == file_hash:
            return item
    return None
