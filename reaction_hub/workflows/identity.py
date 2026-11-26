from __future__ import annotations

import re
import uuid
from pathlib import Path

_WINDOWS_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:/")


def normalize_run_path(path_value: str | Path) -> str:
    text = str(path_value or "").strip()
    if not text:
        return ""
    normalized = text.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    if len(normalized) > 1 and normalized.endswith("/"):
        normalized = normalized.rstrip("/")
    return normalized


def canonical_run_path(path_value: str | Path) -> str:
    normalized = normalize_run_path(path_value)
    if _WINDOWS_DRIVE_PREFIX_RE.match(normalized):
        normalized = normalized[2:]
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
    return normalized


def deterministic_run_uuid(path_value: str | Path) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, normalize_run_path(path_value))
