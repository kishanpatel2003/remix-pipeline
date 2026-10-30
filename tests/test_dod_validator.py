import json
import pytest
from pathlib import Path
from harness.dod_validator import DODValidator, ValidationResult


@pytest.fixture
def project_root(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "raw").mkdir()
    (tmp_path / "analysis").mkdir()
    (tmp_path / "stems").mkdir()
    (tmp_path / "generated").mkdir()
    (tmp_path / "mixes").mkdir()
    (tmp_path / "masters").mkdir()
    (tmp_path / "output").mkdir()
    return tmp_path


@pytest.fixture
def dod(project_root):
    return DODValidator(project_root)


class TestValidationResult:
    def test_pass(self):
        r = ValidationResult(passed=True, reasons=[])
        assert r.passed
        assert r.reasons == []

    def test_fail_with_reasons(self):
        r = ValidationResult(passed=False, reasons=["file missing"])
        assert not r.passed
        assert "file missing" in r.reasons


class TestScrapeValidation:
    def test_passes_with_valid_manifest(self, dod, project_root):
        manifest = {
            "tracks": [
                {"rank": i, "title": f"Song {i}", "artist": f"Artist {i}",
                 "track_id": f"{i:02d}_artist_{i}_song_{i}", "status": "pending"}
                for i in range(1, 11)
            ]
        }
        (project_root / "data" / "tracks_manifest.json").write_text(json.dumps(manifest))
        result = dod.validate("any_track", "scrape")
        assert result.passed

    def test_fails_missing_manifest(self, dod):
        result = dod.validate("any_track", "scrape")
        assert not result.passed
        assert any("manifest" in r.lower() for r in result.reasons)

    def test_fails_too_few_entries(self, dod, project_root):
        manifest = {"tracks": [
            {"rank": 1, "title": "Song", "artist": "Art", "track_id": "01_art_song", "status": "pending"}
        ]}
        (project_root / "data" / "tracks_manifest.json").write_text(json.dumps(manifest))
        result = dod.validate("any_track", "scrape")
        assert not result.passed

    def test_fails_missing_fields(self, dod, project_root):
        manifest = {"tracks": [
            {"rank": i, "title": f"Song {i}"} for i in range(1, 11)
        ]}
        (project_root / "data" / "tracks_manifest.json").write_text(json.dumps(manifest))
        result = dod.validate("any_track", "scrape")
        assert not result.passed

    def test_fails_duplicate_track_ids(self, dod, project_root):
        entry = {"rank": 1, "title": "Song", "artist": "Art", "track_id": "01_art_song", "status": "pending"}
        manifest = {"tracks": [entry] * 10}
        (project_root / "data" / "tracks_manifest.json").write_text(json.dumps(manifest))
        result = dod.validate("any_track", "scrape")
        assert not result.passed


class TestDownloadValidation:
    def test_passes_with_valid_mp3(self, dod, project_root):
        mp3 = project_root / "raw" / "01_art_song.mp3"
        # 4MB of data simulates a real MP3
        mp3.write_bytes(b"\x00" * (4 * 1024 * 1024))
        result = dod.validate("01_art_song", "download", duration_seconds=120)
        assert result.passed

    def test_fails_missing_file(self, dod):
        result = dod.validate("01_art_song", "download")
        assert not result.passed

    def test_fails_too_small(self, dod, project_root):
        mp3 = project_root / "raw" / "01_art_song.mp3"
        mp3.write_bytes(b"\x00" * 1000)
        result = dod.validate("01_art_song", "download", duration_seconds=120)
        assert not result.passed

    def test_fails_too_short(self, dod, project_root):
        mp3 = project_root / "raw" / "01_art_song.mp3"
        mp3.write_bytes(b"\x00" * (4 * 1024 * 1024))
        result = dod.validate("01_art_song", "download", duration_seconds=30)
        assert not result.passed


class TestAnalyzeValidation:
    def _valid_analysis(self):
        return {
            "bpm": 122.0,
            "key": "C minor",
            "structure": [
                {"label": "intro", "start": 0, "end": 15},
                {"label": "verse", "start": 15, "end": 60},
                {"label": "chorus", "start": 60, "end": 90},
                {"label": "outro", "start": 90, "end": 120},
            ],
            "energy_curve": [0.2] * 120,
            "chords": ["Cm", "Fm", "G"],
            "duration": 120.0,
        }

    def test_passes_valid(self, dod, project_root):
        analysis = self._valid_analysis()
        (project_root / "analysis" / "t1_analysis.json").write_text(json.dumps(analysis))
        result = dod.validate("t1", "analyze")
        assert result.passed

    def test_fails_bpm_out_of_range(self, dod, project_root):
        analysis = self._valid_analysis()
        analysis["bpm"] = 250
        (project_root / "analysis" / "t1_analysis.json").write_text(json.dumps(analysis))
        result = dod.validate("t1", "analyze")
        assert not result.passed

    def test_fails_too_few_segments(self, dod, project_root):
        analysis = self._valid_analysis()
        analysis["structure"] = [{"label": "intro", "start": 0, "end": 120}]
        (project_root / "analysis" / "t1_analysis.json").write_text(json.dumps(analysis))
        result = dod.validate("t1", "analyze")
        assert not result.passed

    def test_fails_short_duration(self, dod, project_root):
        analysis = self._valid_analysis()
        analysis["duration"] = 60
        (project_root / "analysis" / "t1_analysis.json").write_text(json.dumps(analysis))
        result = dod.validate("t1", "analyze")
        assert not result.passed


class TestUnknownStage:
    def test_unknown_stage_fails(self, dod):
        result = dod.validate("t1", "nonexistent_stage")
        assert not result.passed
        assert any("unknown" in r.lower() for r in result.reasons)
