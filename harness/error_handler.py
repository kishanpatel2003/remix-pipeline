from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class ErrorCategory(Enum):
    NETWORK = "network"
    PARSE = "parse"
    VALIDATION = "validation"
    GENERATION = "generation"
    DEPENDENCY = "dependency"


class ErrorDecision(Enum):
    RETRY = "retry"
    ESCALATE = "escalate"
    SKIP = "skip"


# category -> max retries (None = never retry, always escalate)
_MAX_RETRIES = {
    ErrorCategory.NETWORK: 3,
    ErrorCategory.PARSE: None,
    ErrorCategory.VALIDATION: 3,
    ErrorCategory.GENERATION: 2,
    ErrorCategory.DEPENDENCY: None,
}


class ErrorHandler:
    def __init__(self, log_file: Path):
        self._log_path = Path(log_file)
        self._retry_counts: dict[tuple[str, str], int] = {}

    def log_error(self, track_id: str, stage: str, category: ErrorCategory, message: str):
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "track_id": track_id,
            "stage": stage,
            "category": category.value,
            "message": message,
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_retry_count(self, track_id: str, stage: str) -> int:
        return self._retry_counts.get((track_id, stage), 0)

    def record_retry(self, track_id: str, stage: str):
        key = (track_id, stage)
        self._retry_counts[key] = self._retry_counts.get(key, 0) + 1

    def decide(self, track_id: str, stage: str, category: ErrorCategory) -> ErrorDecision:
        max_retries = _MAX_RETRIES[category]
        if max_retries is None:
            return ErrorDecision.ESCALATE
        if self.get_retry_count(track_id, stage) < max_retries:
            return ErrorDecision.RETRY
        return ErrorDecision.SKIP
