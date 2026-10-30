from __future__ import annotations

import json
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock

from agents.analyze_track import (
    analyze,
    detect_bpm,
    detect_key,
    detect_structure,
    compute_energy_curve,
    detect_chords,
)


@pytest.fixture
def project_root(tmp_path):
    for d in ("data", "raw", "analysis", "stems", "generated", "mixes", "masters", "output"):
        (tmp_path / d).mkdir()
    return tmp_path


def _make_wav(path: Path, duration: float = 120.0, sr: int = 22050):
    """Generate a synthetic WAV with a tone so analysis has something to detect."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    # 440Hz tone with amplitude envelope to create energy variation
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Add some clicks at regular intervals to simulate beats (~120 BPM = 2Hz)
    beat_interval = int(sr * 0.5)  # 120 BPM
    for i in range(0, n_samples, beat_interval):
        end = min(i + int(sr * 0.01), n_samples)
        signal[i:end] += 0.8
    signal = np.clip(signal, -1.0, 1.0)
    sf.write(str(path), signal, sr)


class TestDetectBpm:
    def test_returns_float_in_range(self, project_root):
        wav_path = project_root / "raw" / "test.wav"
        _make_wav(wav_path, duration=30.0)
        bpm = detect_bpm(wav_path)
        assert isinstance(bpm, float)
        assert 60 <= bpm <= 200

    def test_detects_roughly_correct_bpm(self, project_root):
        wav_path = project_root / "raw" / "test.wav"
        _make_wav(wav_path, duration=30.0)
        bpm = detect_bpm(wav_path)
        # Our synthetic has clicks at 120 BPM; allow wide tolerance
        assert 80 <= bpm <= 160


class TestDetectKey:
    def test_returns_valid_key_string(self, project_root):
        wav_path = project_root / "raw" / "test.wav"
        _make_wav(wav_path, duration=30.0)
        key = detect_key(wav_path)
        assert isinstance(key, str)
        assert len(key) >= 1
        # Should contain a note name
        valid_roots = {"C", "D", "E", "F", "G", "A", "B"}
        assert key[0] in valid_roots


class TestDetectStructure:
    def test_returns_labeled_segments(self, project_root):
        wav_path = project_root / "raw" / "test.wav"
        _make_wav(wav_path, duration=60.0)
        structure = detect_structure(wav_path)
        assert isinstance(structure, list)
        assert len(structure) >= 3
        for seg in structure:
            assert "label" in seg
            assert "start" in seg
            assert "end" in seg
            assert isinstance(seg["start"], float)
            assert isinstance(seg["end"], float)
            assert seg["end"] > seg["start"]

    def test_segments_cover_full_duration(self, project_root):
        wav_path = project_root / "raw" / "test.wav"
        _make_wav(wav_path, duration=60.0)
        structure = detect_structure(wav_path)
        assert structure[0]["start"] == pytest.approx(0.0, abs=1.0)
        # Last segment should end near the track duration
        assert structure[-1]["end"] >= 50.0


class TestComputeEnergyCurve:
    def test_returns_list_of_floats(self, project_root):
        wav_path = project_root / "raw" / "test.wav"
        _make_wav(wav_path, duration=30.0)
        curve = compute_energy_curve(wav_path)
        assert isinstance(curve, list)
        assert len(curve) > 0
        assert all(isinstance(v, float) for v in curve)

    def test_length_roughly_matches_duration(self, project_root):
        wav_path = project_root / "raw" / "test.wav"
        duration = 60.0
        _make_wav(wav_path, duration=duration)
        curve = compute_energy_curve(wav_path)
        # Energy curve sampled ~1 per second
        assert abs(len(curve) - duration) < duration * 0.5


class TestDetectChords:
    def test_returns_non_empty_list(self, project_root):
        wav_path = project_root / "raw" / "test.wav"
        _make_wav(wav_path, duration=30.0)
        chords = detect_chords(wav_path)
        assert isinstance(chords, list)
        assert len(chords) > 0

    def test_chord_entries_have_time_and_label(self, project_root):
        wav_path = project_root / "raw" / "test.wav"
        _make_wav(wav_path, duration=30.0)
        chords = detect_chords(wav_path)
        for chord in chords:
            assert "time" in chord
            assert "chord" in chord
            assert isinstance(chord["time"], float)
            assert isinstance(chord["chord"], str)


class TestAnalyze:
    def test_writes_analysis_json(self, project_root):
        track_id = "01_test_track"
        wav_path = project_root / "raw" / f"{track_id}.mp3"
        _make_wav(wav_path, duration=120.0)

        result = analyze(track_id, project_root)

        analysis_path = project_root / "analysis" / f"{track_id}_analysis.json"
        assert analysis_path.exists()
        data = json.loads(analysis_path.read_text())
        assert data["bpm"] == result["bpm"]
        assert data["key"] == result["key"]

    def test_returns_all_required_fields(self, project_root):
        track_id = "01_test_track"
        wav_path = project_root / "raw" / f"{track_id}.mp3"
        _make_wav(wav_path, duration=120.0)

        result = analyze(track_id, project_root)

        assert "bpm" in result
        assert "key" in result
        assert "structure" in result
        assert "energy_curve" in result
        assert "chords" in result
        assert "duration" in result

    def test_bpm_in_valid_range(self, project_root):
        track_id = "01_test_track"
        wav_path = project_root / "raw" / f"{track_id}.mp3"
        _make_wav(wav_path, duration=120.0)

        result = analyze(track_id, project_root)
        assert 60 <= result["bpm"] <= 200

    def test_raises_if_mp3_missing(self, project_root):
        with pytest.raises(FileNotFoundError):
            analyze("nonexistent_track", project_root)

    def test_skips_if_analysis_already_exists(self, project_root):
        track_id = "01_test_track"
        wav_path = project_root / "raw" / f"{track_id}.mp3"
        _make_wav(wav_path, duration=120.0)

        existing = {"bpm": 128.0, "key": "C minor", "structure": [], "energy_curve": [], "chords": [], "duration": 120.0}
        analysis_path = project_root / "analysis" / f"{track_id}_analysis.json"
        analysis_path.write_text(json.dumps(existing))

        result = analyze(track_id, project_root)
        assert result == existing
