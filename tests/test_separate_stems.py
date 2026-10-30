from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from unittest.mock import patch

from agents.separate_stems import separate, normalize_to_peak_db
from harness.dod_validator import DODValidator


@pytest.fixture
def project_root(tmp_path):
    for d in ("data", "raw", "analysis", "stems", "generated", "mixes", "masters", "output"):
        (tmp_path / d).mkdir()
    return tmp_path


def _make_wav(path: Path, duration: float = 120.0, sr: int = 44100, amplitude: float = 0.5):
    """Generate a synthetic WAV with a tone."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), signal, sr)


def _make_stereo_wav(path: Path, duration: float = 120.0, sr: int = 44100, amplitude: float = 0.5):
    """Generate a stereo WAV."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    left = amplitude * np.sin(2 * np.pi * 440 * t)
    right = amplitude * np.sin(2 * np.pi * 880 * t)
    sf.write(str(path), np.column_stack([left, right]), sr)


def _fake_demucs_output(project_root: Path, track_id: str, duration: float = 120.0, sr: int = 44100):
    """Create fake Demucs output files in the expected htdemucs directory structure."""
    demucs_dir = project_root / "separated" / "htdemucs" / track_id
    demucs_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("vocals", "drums", "bass", "other"):
        amp = 0.3 if stem != "vocals" else 0.5
        _make_stereo_wav(demucs_dir / f"{stem}.wav", duration=duration, sr=sr, amplitude=amp)


class TestNormalizeToPeakDb:
    def test_normalizes_to_target(self):
        sr = 44100
        signal = 0.5 * np.ones(sr)  # peak at 0.5 = -6.02 dBFS
        normalized = normalize_to_peak_db(signal, target_db=-1.0)
        peak_db = 20 * np.log10(np.max(np.abs(normalized)))
        assert abs(peak_db - (-1.0)) < 0.01

    def test_silent_signal_stays_silent(self):
        signal = np.zeros(44100)
        normalized = normalize_to_peak_db(signal, target_db=-1.0)
        assert np.max(np.abs(normalized)) == 0.0

    def test_stereo_normalizes_correctly(self):
        sr = 44100
        stereo = np.column_stack([0.3 * np.ones(sr), 0.2 * np.ones(sr)])
        normalized = normalize_to_peak_db(stereo, target_db=-1.0)
        peak_db = 20 * np.log10(np.max(np.abs(normalized)))
        assert abs(peak_db - (-1.0)) < 0.01


class TestSeparate:
    def test_raises_if_mp3_missing(self, project_root):
        with pytest.raises(FileNotFoundError):
            separate("nonexistent_track", project_root)

    @patch("agents.separate_stems._run_demucs")
    def test_creates_four_wav_stems(self, mock_demucs, project_root):
        track_id = "01_test_artist_test_song"
        _make_wav(project_root / "raw" / f"{track_id}.mp3", duration=10.0)
        _fake_demucs_output(project_root, track_id, duration=10.0)

        result = separate(track_id, project_root)

        stems_dir = project_root / "stems" / track_id
        for stem in ("vocals", "drums", "bass", "other"):
            assert (stems_dir / f"{stem}.wav").exists()

    @patch("agents.separate_stems._run_demucs")
    def test_stems_normalized_to_minus_1_dbfs(self, mock_demucs, project_root):
        track_id = "01_test_artist_test_song"
        _make_wav(project_root / "raw" / f"{track_id}.mp3", duration=10.0)
        _fake_demucs_output(project_root, track_id, duration=10.0)

        separate(track_id, project_root)

        stems_dir = project_root / "stems" / track_id
        for stem in ("vocals", "drums", "bass", "other"):
            data, sr = sf.read(str(stems_dir / f"{stem}.wav"))
            peak_db = 20 * np.log10(np.max(np.abs(data)))
            assert abs(peak_db - (-1.0)) < 0.1

    @patch("agents.separate_stems._run_demucs")
    def test_returns_stems_dict(self, mock_demucs, project_root):
        track_id = "01_test_artist_test_song"
        _make_wav(project_root / "raw" / f"{track_id}.mp3", duration=10.0)
        _fake_demucs_output(project_root, track_id, duration=10.0)

        result = separate(track_id, project_root)

        assert "vocals" in result
        assert "drums" in result
        assert "bass" in result
        assert "other" in result
        for v in result.values():
            assert Path(v).exists()

    @patch("agents.separate_stems._run_demucs")
    def test_idempotent_skips_existing(self, mock_demucs, project_root):
        track_id = "01_test_artist_test_song"
        _make_wav(project_root / "raw" / f"{track_id}.mp3", duration=10.0)

        stems_dir = project_root / "stems" / track_id
        stems_dir.mkdir(parents=True)
        for stem in ("vocals", "drums", "bass", "other"):
            _make_wav(stems_dir / f"{stem}.wav", duration=10.0)

        result = separate(track_id, project_root)

        mock_demucs.assert_not_called()
        assert len(result) == 4

    @patch("agents.separate_stems._run_demucs")
    def test_creates_stems_directory(self, mock_demucs, project_root):
        track_id = "01_test_artist_test_song"
        _make_wav(project_root / "raw" / f"{track_id}.mp3", duration=10.0)
        _fake_demucs_output(project_root, track_id, duration=10.0)

        separate(track_id, project_root)

        assert (project_root / "stems" / track_id).is_dir()


class TestDODValidatorStems:
    def test_passes_with_valid_stems(self, project_root):
        track_id = "01_test_artist_test_song"
        _make_wav(project_root / "raw" / f"{track_id}.mp3", duration=120.0)

        stems_dir = project_root / "stems" / track_id
        stems_dir.mkdir(parents=True)
        for stem in ("vocals", "drums", "bass", "other"):
            amp = 0.5 if stem == "vocals" else 0.3
            _make_wav(stems_dir / f"{stem}.wav", duration=120.0, amplitude=amp)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "stems")
        assert result.passed, f"Expected pass, got: {result.reasons}"

    def test_fails_if_stem_missing(self, project_root):
        track_id = "01_test_artist_test_song"
        _make_wav(project_root / "raw" / f"{track_id}.mp3", duration=120.0)

        stems_dir = project_root / "stems" / track_id
        stems_dir.mkdir(parents=True)
        for stem in ("vocals", "drums", "bass"):
            _make_wav(stems_dir / f"{stem}.wav", duration=120.0)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "stems")
        assert not result.passed

    def test_fails_if_duration_mismatch(self, project_root):
        track_id = "01_test_artist_test_song"
        _make_wav(project_root / "raw" / f"{track_id}.mp3", duration=120.0)

        stems_dir = project_root / "stems" / track_id
        stems_dir.mkdir(parents=True)
        for stem in ("vocals", "drums", "bass", "other"):
            _make_wav(stems_dir / f"{stem}.wav", duration=10.0)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "stems")
        assert not result.passed

    def test_fails_if_vocals_rms_too_low(self, project_root):
        track_id = "01_test_artist_test_song"
        _make_wav(project_root / "raw" / f"{track_id}.mp3", duration=120.0)

        stems_dir = project_root / "stems" / track_id
        stems_dir.mkdir(parents=True)
        # Very quiet vocals
        _make_wav(stems_dir / "vocals.wav", duration=120.0, amplitude=0.001)
        for stem in ("drums", "bass", "other"):
            _make_wav(stems_dir / f"{stem}.wav", duration=120.0, amplitude=0.3)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "stems")
        assert not result.passed
        assert any("vocals" in r.lower() or "RMS" in r for r in result.reasons)
