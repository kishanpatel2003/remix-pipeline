from __future__ import annotations

import json
from pathlib import Path


class InvalidTransition(Exception):
    pass


# Maps current_status -> set of valid next statuses.
# failed_{stage} is always valid from any non-terminal state (handled separately).
VALID_TRANSITIONS = {
    "pending": {"downloaded"},
    "downloaded": {"analyzed"},
    "analyzed": {"stems_separated"},
    "stems_separated": {"stems_approved", "rejected"},
    "stems_approved": {"vocals_processed"},
    "vocals_processed": {"instrumentation_generated"},
    "instrumentation_generated": {"rough_mixed"},
    "rough_mixed": {"rough_approved", "rejected_with_loopback"},
    "rough_approved": {"mixed"},
    "mixed": {"mastered"},
    "mastered": {"approved", "rejected_final"},
    "approved": {"packaged"},
}

TERMINAL_STATES = {"packaged", "rejected", "rejected_final"}


class StateManager:
    def __init__(self, state_file: Path):
        self._path = Path(state_file)

    def load(self) -> dict:
        if not self._path.exists():
            self._write({})
            return {}
        return json.loads(self._path.read_text())

    def _write(self, state: dict):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(state, indent=2) + "\n")

    def add_track(self, track_id: str):
        state = self.load()
        if track_id not in state:
            state[track_id] = {"status": "pending"}
            self._write(state)

    def get_status(self, track_id: str) -> str:
        state = self.load()
        if track_id not in state:
            raise KeyError(f"Unknown track: {track_id}")
        return state[track_id]["status"]

    def transition(self, track_id: str, target: str):
        state = self.load()
        if track_id not in state:
            raise KeyError(f"Unknown track: {track_id}")

        current = state[track_id]["status"]

        # failed_{stage} transitions are always valid from non-terminal states
        if target.startswith("failed_"):
            if current in TERMINAL_STATES:
                raise InvalidTransition(f"{current} -> {target}")
        elif current not in VALID_TRANSITIONS:
            raise InvalidTransition(f"{current} is terminal, cannot transition to {target}")
        elif target not in VALID_TRANSITIONS[current]:
            raise InvalidTransition(f"{current} -> {target}")

        state[track_id]["status"] = target
        self._write(state)

    def get_tracks_by_status(self, status: str) -> list[str]:
        state = self.load()
        return [tid for tid, data in state.items() if data["status"] == status]
