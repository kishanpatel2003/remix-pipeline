from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


# checkpoint_state -> (approved_state, rejected_state)
CHECKPOINTS = {
    "stems_separated": ("stems_approved", "rejected"),
    "rough_mixed": ("rough_approved", "rejected_with_loopback"),
    "mastered": ("approved", "rejected_final"),
}


class CheckpointHandler:
    def __init__(self, review_file: Path):
        self._path = Path(review_file)

    def _load(self) -> dict:
        if not self._path.exists():
            return {"pending": []}
        return json.loads(self._path.read_text())

    def _save(self, data: dict):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2) + "\n")

    def is_checkpoint_state(self, state: str) -> bool:
        return state in CHECKPOINTS

    def request_review(self, track_id: str, checkpoint: str):
        data = self._load()
        already = any(
            p["track_id"] == track_id and p["checkpoint"] == checkpoint
            for p in data["pending"]
        )
        if already:
            return
        data["pending"].append({
            "track_id": track_id,
            "checkpoint": checkpoint,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save(data)

    def is_pending_review(self, track_id: str) -> bool:
        data = self._load()
        return any(p["track_id"] == track_id for p in data["pending"])

    def resolve(self, track_id: str, approved: bool, reason: str = "") -> dict:
        data = self._load()
        entry = None
        for p in data["pending"]:
            if p["track_id"] == track_id:
                entry = p
                break
        if entry is None:
            raise KeyError(f"No pending review for track: {track_id}")

        data["pending"] = [p for p in data["pending"] if p["track_id"] != track_id]
        self._save(data)

        result = {
            "track_id": track_id,
            "checkpoint": entry["checkpoint"],
            "approved": approved,
        }
        if not approved:
            result["reason"] = reason
        return result

    def get_approved_state(self, checkpoint: str) -> str:
        return CHECKPOINTS[checkpoint][0]

    def get_rejected_state(self, checkpoint: str) -> str:
        return CHECKPOINTS[checkpoint][1]

    def list_pending(self) -> list[dict]:
        return self._load()["pending"]
