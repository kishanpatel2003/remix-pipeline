import json
import pytest
from pathlib import Path
from harness.error_handler import ErrorHandler, ErrorCategory, ErrorDecision


@pytest.fixture
def log_file(tmp_path):
    return tmp_path / "pipeline_log.jsonl"


@pytest.fixture
def eh(log_file):
    return ErrorHandler(log_file)


class TestErrorCategories:
    def test_all_categories_exist(self):
        assert set(ErrorCategory) == {
            ErrorCategory.NETWORK,
            ErrorCategory.PARSE,
            ErrorCategory.VALIDATION,
            ErrorCategory.GENERATION,
            ErrorCategory.DEPENDENCY,
        }


class TestLogError:
    def test_appends_to_jsonl(self, eh, log_file):
        eh.log_error("t1", "scrape", ErrorCategory.NETWORK, "timeout")
        eh.log_error("t1", "scrape", ErrorCategory.NETWORK, "timeout again")
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2
        entry = json.loads(lines[0])
        assert entry["track_id"] == "t1"
        assert entry["stage"] == "scrape"
        assert entry["category"] == "network"
        assert entry["message"] == "timeout"
        assert "timestamp" in entry

    def test_creates_file_if_missing(self, eh, log_file):
        assert not log_file.exists()
        eh.log_error("t1", "download", ErrorCategory.NETWORK, "fail")
        assert log_file.exists()


class TestRetryCount:
    def test_starts_at_zero(self, eh):
        assert eh.get_retry_count("t1", "scrape") == 0

    def test_increments(self, eh):
        eh.record_retry("t1", "scrape")
        eh.record_retry("t1", "scrape")
        assert eh.get_retry_count("t1", "scrape") == 2

    def test_independent_per_track_and_stage(self, eh):
        eh.record_retry("t1", "scrape")
        eh.record_retry("t2", "scrape")
        eh.record_retry("t1", "download")
        assert eh.get_retry_count("t1", "scrape") == 1
        assert eh.get_retry_count("t2", "scrape") == 1
        assert eh.get_retry_count("t1", "download") == 1


class TestDecide:
    def test_network_retries_up_to_3(self, eh):
        assert eh.decide("t1", "scrape", ErrorCategory.NETWORK) == ErrorDecision.RETRY
        eh.record_retry("t1", "scrape")
        eh.record_retry("t1", "scrape")
        assert eh.decide("t1", "scrape", ErrorCategory.NETWORK) == ErrorDecision.RETRY
        eh.record_retry("t1", "scrape")
        assert eh.decide("t1", "scrape", ErrorCategory.NETWORK) == ErrorDecision.SKIP

    def test_parse_always_escalates(self, eh):
        assert eh.decide("t1", "scrape", ErrorCategory.PARSE) == ErrorDecision.ESCALATE

    def test_validation_retries_up_to_3(self, eh):
        assert eh.decide("t1", "analyze", ErrorCategory.VALIDATION) == ErrorDecision.RETRY
        for _ in range(3):
            eh.record_retry("t1", "analyze")
        assert eh.decide("t1", "analyze", ErrorCategory.VALIDATION) == ErrorDecision.SKIP

    def test_generation_retries_up_to_2(self, eh):
        assert eh.decide("t1", "generate", ErrorCategory.GENERATION) == ErrorDecision.RETRY
        eh.record_retry("t1", "generate")
        assert eh.decide("t1", "generate", ErrorCategory.GENERATION) == ErrorDecision.RETRY
        eh.record_retry("t1", "generate")
        assert eh.decide("t1", "generate", ErrorCategory.GENERATION) == ErrorDecision.SKIP

    def test_dependency_always_escalates(self, eh):
        assert eh.decide("t1", "stems", ErrorCategory.DEPENDENCY) == ErrorDecision.ESCALATE
