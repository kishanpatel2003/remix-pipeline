import json
import pytest
from pathlib import Path
from harness.state_manager import (
    StateManager,
    InvalidTransition,
    VALID_TRANSITIONS,
)


@pytest.fixture
def state_file(tmp_path):
    return tmp_path / "pipeline_state.json"


@pytest.fixture
def sm(state_file):
    return StateManager(state_file)


class TestStateManagerInit:
    def test_creates_file_on_first_read(self, sm, state_file):
        state = sm.load()
        assert state == {}
        assert state_file.exists()

    def test_loads_existing_state(self, state_file):
        state_file.write_text(json.dumps({"track_01": {"status": "pending"}}))
        sm = StateManager(state_file)
        state = sm.load()
        assert state["track_01"]["status"] == "pending"


class TestAddTrack:
    def test_add_new_track(self, sm):
        sm.add_track("01_drake_hotline_bling")
        state = sm.load()
        assert state["01_drake_hotline_bling"]["status"] == "pending"

    def test_add_duplicate_track_is_noop(self, sm):
        sm.add_track("01_drake_hotline_bling")
        sm.add_track("01_drake_hotline_bling")
        state = sm.load()
        assert state["01_drake_hotline_bling"]["status"] == "pending"


class TestTransitions:
    def test_valid_transition_pending_to_downloaded(self, sm):
        sm.add_track("t1")
        sm.transition("t1", "downloaded")
        assert sm.get_status("t1") == "downloaded"

    def test_full_happy_path(self, sm):
        sm.add_track("t1")
        for target in [
            "downloaded", "analyzed", "stems_separated",
            "stems_approved", "vocals_processed",
            "instrumentation_generated", "rough_mixed",
            "rough_approved", "mixed", "mastered",
            "approved", "packaged",
        ]:
            sm.transition("t1", target)
        assert sm.get_status("t1") == "packaged"

    def test_invalid_transition_raises(self, sm):
        sm.add_track("t1")
        with pytest.raises(InvalidTransition):
            sm.transition("t1", "mastered")

    def test_transition_to_rejected(self, sm):
        sm.add_track("t1")
        sm.transition("t1", "downloaded")
        sm.transition("t1", "analyzed")
        sm.transition("t1", "stems_separated")
        sm.transition("t1", "rejected")
        assert sm.get_status("t1") == "rejected"

    def test_transition_to_failed_state(self, sm):
        sm.add_track("t1")
        sm.transition("t1", "downloaded")
        sm.transition("t1", "failed_analyze")
        assert sm.get_status("t1") == "failed_analyze"

    def test_transition_unknown_track_raises(self, sm):
        with pytest.raises(KeyError):
            sm.transition("nonexistent", "downloaded")


class TestGetStatus:
    def test_get_status(self, sm):
        sm.add_track("t1")
        assert sm.get_status("t1") == "pending"

    def test_get_status_unknown_track(self, sm):
        with pytest.raises(KeyError):
            sm.get_status("nonexistent")


class TestGetTracksByStatus:
    def test_filters_by_status(self, sm):
        sm.add_track("t1")
        sm.add_track("t2")
        sm.transition("t1", "downloaded")
        assert sm.get_tracks_by_status("pending") == ["t2"]
        assert sm.get_tracks_by_status("downloaded") == ["t1"]

    def test_empty_result(self, sm):
        assert sm.get_tracks_by_status("pending") == []


class TestPersistence:
    def test_state_survives_reload(self, state_file):
        sm1 = StateManager(state_file)
        sm1.add_track("t1")
        sm1.transition("t1", "downloaded")

        sm2 = StateManager(state_file)
        assert sm2.get_status("t1") == "downloaded"


class TestValidTransitions:
    def test_all_states_have_transitions(self):
        expected_states = {
            "pending", "downloaded", "analyzed", "stems_separated",
            "stems_approved", "vocals_processed", "instrumentation_generated",
            "rough_mixed", "rough_approved", "mixed", "mastered", "approved",
        }
        assert expected_states.issubset(set(VALID_TRANSITIONS.keys()))
