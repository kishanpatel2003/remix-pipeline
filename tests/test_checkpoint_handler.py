import json
import pytest
from pathlib import Path
from harness.checkpoint_handler import CheckpointHandler, CHECKPOINTS


@pytest.fixture
def review_file(tmp_path):
    return tmp_path / "pending_review.json"


@pytest.fixture
def ch(review_file):
    return CheckpointHandler(review_file)


class TestCheckpointDefinitions:
    def test_three_checkpoints_exist(self):
        assert len(CHECKPOINTS) == 3

    def test_checkpoint_1_after_stems_separated(self):
        assert "stems_separated" in CHECKPOINTS

    def test_checkpoint_2_after_rough_mixed(self):
        assert "rough_mixed" in CHECKPOINTS

    def test_checkpoint_3_after_mastered(self):
        assert "mastered" in CHECKPOINTS

    def test_is_checkpoint_state(self, ch):
        assert ch.is_checkpoint_state("stems_separated")
        assert ch.is_checkpoint_state("rough_mixed")
        assert ch.is_checkpoint_state("mastered")
        assert not ch.is_checkpoint_state("downloaded")
        assert not ch.is_checkpoint_state("pending")


class TestRequestReview:
    def test_adds_to_pending(self, ch, review_file):
        ch.request_review("t1", "stems_separated")
        data = json.loads(review_file.read_text())
        assert len(data["pending"]) == 1
        assert data["pending"][0]["track_id"] == "t1"
        assert data["pending"][0]["checkpoint"] == "stems_separated"
        assert "timestamp" in data["pending"][0]

    def test_multiple_tracks(self, ch, review_file):
        ch.request_review("t1", "stems_separated")
        ch.request_review("t2", "rough_mixed")
        data = json.loads(review_file.read_text())
        assert len(data["pending"]) == 2

    def test_no_duplicate_requests(self, ch, review_file):
        ch.request_review("t1", "stems_separated")
        ch.request_review("t1", "stems_separated")
        data = json.loads(review_file.read_text())
        assert len(data["pending"]) == 1

    def test_creates_file_if_missing(self, ch, review_file):
        assert not review_file.exists()
        ch.request_review("t1", "stems_separated")
        assert review_file.exists()


class TestIsPendingReview:
    def test_true_when_pending(self, ch):
        ch.request_review("t1", "stems_separated")
        assert ch.is_pending_review("t1")

    def test_false_when_not_pending(self, ch):
        assert not ch.is_pending_review("t1")

    def test_false_after_resolved(self, ch):
        ch.request_review("t1", "stems_separated")
        ch.resolve("t1", approved=True)
        assert not ch.is_pending_review("t1")


class TestResolve:
    def test_approve_removes_from_pending(self, ch, review_file):
        ch.request_review("t1", "stems_separated")
        result = ch.resolve("t1", approved=True)
        assert result["approved"] is True
        data = json.loads(review_file.read_text())
        assert len(data["pending"]) == 0

    def test_reject_removes_from_pending(self, ch, review_file):
        ch.request_review("t1", "stems_separated")
        result = ch.resolve("t1", approved=False, reason="too much bleed")
        assert result["approved"] is False
        assert result["reason"] == "too much bleed"
        data = json.loads(review_file.read_text())
        assert len(data["pending"]) == 0

    def test_resolve_unknown_track_raises(self, ch):
        with pytest.raises(KeyError):
            ch.resolve("nonexistent", approved=True)

    def test_resolve_returns_checkpoint_info(self, ch):
        ch.request_review("t1", "mastered")
        result = ch.resolve("t1", approved=True)
        assert result["track_id"] == "t1"
        assert result["checkpoint"] == "mastered"


class TestGetApprovedState:
    def test_stems_separated_approved(self, ch):
        assert ch.get_approved_state("stems_separated") == "stems_approved"

    def test_rough_mixed_approved(self, ch):
        assert ch.get_approved_state("rough_mixed") == "rough_approved"

    def test_mastered_approved(self, ch):
        assert ch.get_approved_state("mastered") == "approved"

    def test_rejected_states(self, ch):
        assert ch.get_rejected_state("stems_separated") == "rejected"
        assert ch.get_rejected_state("rough_mixed") == "rejected_with_loopback"
        assert ch.get_rejected_state("mastered") == "rejected_final"


class TestListPending:
    def test_lists_all_pending(self, ch):
        ch.request_review("t1", "stems_separated")
        ch.request_review("t2", "rough_mixed")
        pending = ch.list_pending()
        assert len(pending) == 2
        ids = [p["track_id"] for p in pending]
        assert "t1" in ids
        assert "t2" in ids

    def test_empty_when_none(self, ch):
        assert ch.list_pending() == []
