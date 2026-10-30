from __future__ import annotations

import logging
from harness.state_manager import StateManager, TERMINAL_STATES
from harness.error_handler import ErrorHandler, ErrorCategory
from harness.dod_validator import DODValidator
from harness.checkpoint_handler import CheckpointHandler, CHECKPOINTS

logger = logging.getLogger(__name__)

# Maps current_status -> (stage_name, next_status_on_success)
_STAGE_MAP = {
    "pending": ("scrape", "downloaded"),
    "downloaded": ("download", "analyzed"),
    "analyzed": ("separate_stems", "stems_separated"),
    "stems_approved": ("process_vocals", "vocals_processed"),
    "vocals_processed": ("generate_instrumentation", "instrumentation_generated"),
    "instrumentation_generated": ("arrange_remix", "rough_mixed"),
    "rough_approved": ("mix", "mixed"),
    "mixed": ("master", "mastered"),
    "approved": ("package", "packaged"),
}


class StageRegistry:
    def __init__(self):
        self._stages: dict[str, callable] = {}

    def register(self, name: str, fn):
        self._stages[name] = fn

    def get(self, name: str):
        return self._stages.get(name)


class Orchestrator:
    def __init__(
        self,
        state_manager: StateManager,
        error_handler: ErrorHandler,
        dod_validator: DODValidator,
        checkpoint_handler: CheckpointHandler,
        stage_registry: StageRegistry,
    ):
        self._sm = state_manager
        self._eh = error_handler
        self._dod = dod_validator
        self._ch = checkpoint_handler
        self._registry = stage_registry

    def get_next_stage(self, status: str) -> str | None:
        entry = _STAGE_MAP.get(status)
        return entry[0] if entry else None

    def select_next_work(self) -> dict | None:
        state = self._sm.load()
        for track_id, data in state.items():
            status = data["status"]

            if status in TERMINAL_STATES:
                continue
            if status.startswith("failed_"):
                continue
            if self._ch.is_pending_review(track_id):
                continue
            if status in CHECKPOINTS:
                continue

            stage_name = self.get_next_stage(status)
            if stage_name is None:
                continue
            if self._registry.get(stage_name) is None:
                continue

            return {"track_id": track_id, "stage": stage_name, "current_status": status}

        return None

    def run_one(self) -> dict | None:
        work = self.select_next_work()
        if work is None:
            return None

        track_id = work["track_id"]
        stage_name = work["stage"]
        current_status = work["current_status"]
        next_status = _STAGE_MAP[current_status][1]
        stage_fn = self._registry.get(stage_name)

        report = {"track_id": track_id, "stage": stage_name, "success": False}

        try:
            result = stage_fn(track_id)
            kwargs = result if isinstance(result, dict) else {}
            validation = self._dod.validate(track_id, stage_name, **kwargs)

            if validation.passed:
                self._sm.transition(track_id, next_status)
                report["success"] = True

                if self._ch.is_checkpoint_state(next_status):
                    self._ch.request_review(track_id, next_status)

                logger.info(f"{track_id}: {stage_name} -> {next_status}")
            else:
                self._eh.log_error(
                    track_id, stage_name, ErrorCategory.VALIDATION,
                    f"DOD failed: {validation.reasons}"
                )
                decision = self._eh.decide(track_id, stage_name, ErrorCategory.VALIDATION)
                self._eh.record_retry(track_id, stage_name)
                report["reasons"] = validation.reasons
                logger.warning(f"{track_id}: {stage_name} DOD failed ({decision.value})")

        except Exception as e:
            self._eh.log_error(
                track_id, stage_name, ErrorCategory.DEPENDENCY, str(e)
            )
            logger.error(f"{track_id}: {stage_name} raised {e}")
            report["error"] = str(e)

        return report
