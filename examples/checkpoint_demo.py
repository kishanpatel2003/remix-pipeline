"""Exercise persistent job state and human approval without audio or API calls."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.state_manager import StateManager
from harness.checkpoint_handler import CheckpointHandler


def main():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        state = StateManager(root / "state.json")
        review = CheckpointHandler(root / "review.json")
        state.add_track("synthetic_demo")
        for status in ("downloaded", "analyzed", "stems_separated"):
            state.transition("synthetic_demo", status)
        review.request_review("synthetic_demo", "stems_separated")
        print("Before review:", state.get_status("synthetic_demo"))
        result = review.resolve("synthetic_demo", approved=True)
        state.transition("synthetic_demo", review.get_approved_state(result["checkpoint"]))
        reloaded = StateManager(root / "state.json")
        print("After reload:", reloaded.get_status("synthetic_demo"))


if __name__ == "__main__":
    main()
