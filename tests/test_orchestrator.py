import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from harness.orchestrator import Orchestrator, StageRegistry
from harness.state_manager import StateManager
from harness.error_handler import ErrorHandler, ErrorCategory, ErrorDecision
from harness.dod_validator import DODValidator, ValidationResult
from harness.checkpoint_handler import CheckpointHandler


@pytest.fixture
def project_root(tmp_path):
    for d in ["data", "raw", "analysis", "stems", "generated", "mixes", "masters", "output"]:
        (tmp_path / d).mkdir()
    return tmp_path


@pytest.fixture
def sm(project_root):
    return StateManager(project_root / "data" / "pipeline_state.json")


@pytest.fixture
def eh(project_root):
    return ErrorHandler(project_root / "data" / "pipeline_log.jsonl")


@pytest.fixture
def dod(project_root):
    return DODValidator(project_root)


@pytest.fixture
def ch(project_root):
    return CheckpointHandler(project_root / "data" / "pending_review.json")


@pytest.fixture
def registry():
    return StageRegistry()


@pytest.fixture
def orch(sm, eh, dod, ch, registry):
    return Orchestrator(sm, eh, dod, ch, registry)


# Maps state -> (stage_name, next_state)
STAGE_MAP = {
    "pending": ("scrape", "downloaded"),
    "downloaded": ("download", "analyzed"),
    "analyzed": ("analyze", "stems_separated"),
    "stems_approved": ("process_vocals", "vocals_processed"),
    "vocals_processed": ("generate_instrumentation", "instrumentation_generated"),
    "instrumentation_generated": ("arrange_remix", "rough_mixed"),
    "rough_approved": ("mix", "mixed"),
    "mixed": ("master", "mastered"),
    "approved": ("package", "packaged"),
}


class TestStageRegistry:
    def test_register_and_get(self, registry):
        fn = MagicMock()
        registry.register("scrape", fn)
        assert registry.get("scrape") is fn

    def test_get_unknown_returns_none(self, registry):
        assert registry.get("nonexistent") is None


class TestNextStageMapping:
    def test_next_stage_for_pending(self, orch):
        assert orch.get_next_stage("pending") == "scrape"

    def test_next_stage_for_downloaded(self, orch):
        assert orch.get_next_stage("downloaded") == "download"

    def test_checkpoint_states_return_none(self, orch):
        assert orch.get_next_stage("stems_separated") is None
        assert orch.get_next_stage("rough_mixed") is None
        assert orch.get_next_stage("mastered") is None

    def test_terminal_states_return_none(self, orch):
        assert orch.get_next_stage("packaged") is None
        assert orch.get_next_stage("rejected") is None


class TestSelectNextWork:
    def test_selects_actionable_track(self, orch, sm, registry):
        registry.register("scrape", MagicMock())
        sm.add_track("t1")
        work = orch.select_next_work()
        assert work is not None
        assert work["track_id"] == "t1"
        assert work["stage"] == "scrape"

    def test_skips_checkpoint_blocked(self, orch, sm, ch, registry):
        registry.register("scrape", MagicMock())
        sm.add_track("t1")
        sm.transition("t1", "downloaded")
        sm.transition("t1", "analyzed")
        sm.transition("t1", "stems_separated")
        ch.request_review("t1", "stems_separated")
        # t1 is at checkpoint, should be skipped
        sm.add_track("t2")
        registry.register("scrape", MagicMock())
        work = orch.select_next_work()
        assert work["track_id"] == "t2"

    def test_returns_none_when_no_work(self, orch):
        assert orch.select_next_work() is None

    def test_skips_terminal_states(self, orch, sm, registry):
        registry.register("scrape", MagicMock())
        sm.add_track("t1")
        # Force to packaged
        for s in ["downloaded", "analyzed", "stems_separated", "stems_approved",
                   "vocals_processed", "instrumentation_generated", "rough_mixed",
                   "rough_approved", "mixed", "mastered", "approved", "packaged"]:
            sm.transition("t1", s)
        assert orch.select_next_work() is None


class TestRunStage:
    def test_successful_stage_transitions_state(self, orch, sm, eh, registry):
        stage_fn = MagicMock(return_value={"duration_seconds": 120})
        registry.register("scrape", stage_fn)
        sm.add_track("t1")

        # Mock DOD to pass
        orch._dod.validate = MagicMock(return_value=ValidationResult(passed=True))

        orch.run_one()
        stage_fn.assert_called_once_with("t1")
        assert sm.get_status("t1") == "downloaded"

    def test_failed_dod_triggers_error_handler(self, orch, sm, eh, registry):
        stage_fn = MagicMock(return_value={})
        registry.register("scrape", stage_fn)
        sm.add_track("t1")

        orch._dod.validate = MagicMock(
            return_value=ValidationResult(passed=False, reasons=["bad output"])
        )

        orch.run_one()
        # Should still be pending (retry, not transitioned)
        assert sm.get_status("t1") == "pending"

    def test_stage_exception_logs_error(self, orch, sm, registry):
        stage_fn = MagicMock(side_effect=RuntimeError("boom"))
        registry.register("scrape", stage_fn)
        sm.add_track("t1")

        orch.run_one()
        assert sm.get_status("t1") == "pending"

    def test_checkpoint_requested_after_stems_separated(self, orch, sm, ch, registry):
        stage_fn = MagicMock(return_value={})
        registry.register("analyze", stage_fn)
        sm.add_track("t1")
        sm.transition("t1", "downloaded")
        sm.transition("t1", "analyzed")

        # Register separate_stems stage
        sep_fn = MagicMock(return_value={})
        registry.register("separate_stems", sep_fn)

        orch._dod.validate = MagicMock(return_value=ValidationResult(passed=True))
        orch.run_one()  # should run separate_stems on t1

        assert sm.get_status("t1") == "stems_separated"
        assert ch.is_pending_review("t1")


class TestRunOneReturnsReport:
    def test_returns_none_when_no_work(self, orch):
        assert orch.run_one() is None

    def test_returns_report_on_success(self, orch, sm, registry):
        registry.register("scrape", MagicMock(return_value={}))
        sm.add_track("t1")
        orch._dod.validate = MagicMock(return_value=ValidationResult(passed=True))
        report = orch.run_one()
        assert report["track_id"] == "t1"
        assert report["stage"] == "scrape"
        assert report["success"] is True
