from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from harness.dod_validator import DODValidator


@pytest.fixture
def project_root(tmp_path):
    for d in ("data", "raw", "analysis", "stems", "generated", "mixes", "masters", "output"):
        (tmp_path / d).mkdir()
    return tmp_path


def _make_vocal_wav(path: Path, duration: float = 120.0, sr: int = 44100, amplitude: float = 0.5):
    """Vocal-like signal: 440Hz tone with a low-freq rumble component at 40Hz."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * 440 * t)
    # Add sub-80Hz rumble that should be removed by HP filter
    signal += 0.1 * amplitude * np.sin(2 * np.pi * 40 * t)
    sf.write(str(path), signal.astype(np.float32), sr)


def _make_analysis_json(path: Path, bpm: float = 120.0, duration: float = 120.0):
    """Create an analysis JSON with a chorus section long enough for hook extraction."""
    analysis = {
        "bpm": bpm,
        "key": "C major",
        "duration": duration,
        "structure": [
            {"label": "intro", "start": 0.0, "end": 15.0},
            {"label": "verse", "start": 15.0, "end": 45.0},
            {"label": "chorus", "start": 45.0, "end": 75.0},
            {"label": "bridge", "start": 75.0, "end": 95.0},
            {"label": "chorus", "start": 95.0, "end": 115.0},
            {"label": "outro", "start": 115.0, "end": duration},
        ],
        "energy_curve": [0.5] * int(duration),
        "chords": [{"time": 0.0, "chord": "Cmaj"}],
    }
    path.write_text(json.dumps(analysis, indent=2) + "\n")


def _setup_track(project_root: Path, track_id: str, duration: float = 120.0, bpm: float = 120.0):
    """Set up vocals stem and analysis JSON for a track."""
    stems_dir = project_root / "stems" / track_id
    stems_dir.mkdir(parents=True, exist_ok=True)
    _make_vocal_wav(stems_dir / "vocals.wav", duration=duration)
    _make_analysis_json(
        project_root / "analysis" / f"{track_id}_analysis.json",
        bpm=bpm,
        duration=duration,
    )


class TestProcessVocals:
    def test_raises_if_vocals_missing(self, project_root):
        from agents.process_vocals import process_vocals

        track_id = "01_test_artist_test_song"
        _make_analysis_json(
            project_root / "analysis" / f"{track_id}_analysis.json"
        )
        with pytest.raises(FileNotFoundError):
            process_vocals(track_id, project_root)

    def test_raises_if_analysis_missing(self, project_root):
        from agents.process_vocals import process_vocals

        track_id = "01_test_artist_test_song"
        stems_dir = project_root / "stems" / track_id
        stems_dir.mkdir(parents=True)
        _make_vocal_wav(stems_dir / "vocals.wav", duration=30.0)
        with pytest.raises(FileNotFoundError):
            process_vocals(track_id, project_root)

    def test_creates_processed_and_hook_files(self, project_root):
        from agents.process_vocals import process_vocals

        track_id = "01_test_artist_test_song"
        _setup_track(project_root, track_id)

        process_vocals(track_id, project_root)

        stems_dir = project_root / "stems" / track_id
        assert (stems_dir / "vocals_processed.wav").exists()
        assert (stems_dir / "vocals_hook.wav").exists()

    def test_processed_rms_in_range(self, project_root):
        from agents.process_vocals import process_vocals

        track_id = "01_test_artist_test_song"
        _setup_track(project_root, track_id)

        process_vocals(track_id, project_root)

        data, sr = sf.read(str(project_root / "stems" / track_id / "vocals_processed.wav"))
        rms_linear = np.sqrt(np.mean(data ** 2))
        rms_db = 20 * np.log10(rms_linear) if rms_linear > 0 else -100.0
        assert -18 <= rms_db <= -12, f"RMS {rms_db:.1f} dBFS outside -18 to -12 range"

    def test_hp_filter_removes_sub_80hz(self, project_root):
        from agents.process_vocals import process_vocals

        track_id = "01_test_artist_test_song"
        _setup_track(project_root, track_id)

        process_vocals(track_id, project_root)

        data, sr = sf.read(str(project_root / "stems" / track_id / "vocals_processed.wav"))
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Check spectral energy below 80Hz is negligible
        fft = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1.0 / sr)
        sub_80_energy = np.sum(np.abs(fft[freqs < 80]) ** 2)
        total_energy = np.sum(np.abs(fft) ** 2)
        sub_80_ratio = sub_80_energy / total_energy if total_energy > 0 else 0
        # After HP at 80Hz, sub-80Hz should be < 1% of total energy
        assert sub_80_ratio < 0.01, f"Sub-80Hz energy ratio {sub_80_ratio:.4f} too high"

    def test_hook_duration_at_least_10s(self, project_root):
        from agents.process_vocals import process_vocals

        track_id = "01_test_artist_test_song"
        _setup_track(project_root, track_id)

        process_vocals(track_id, project_root)

        data, sr = sf.read(str(project_root / "stems" / track_id / "vocals_hook.wav"))
        duration = len(data) / sr
        assert duration >= 10.0, f"Hook duration {duration:.1f}s < 10s"

    def test_hook_has_fades(self, project_root):
        from agents.process_vocals import process_vocals

        track_id = "01_test_artist_test_song"
        _setup_track(project_root, track_id)

        process_vocals(track_id, project_root)

        data, sr = sf.read(str(project_root / "stems" / track_id / "vocals_hook.wav"))
        if data.ndim > 1:
            data = data.mean(axis=1)

        # First 50ms should be quieter than the body (fade in)
        fade_samples = int(0.05 * sr)
        head_rms = np.sqrt(np.mean(data[:fade_samples] ** 2))
        body_rms = np.sqrt(np.mean(data[fade_samples:fade_samples * 10] ** 2))
        assert head_rms < body_rms, "No fade-in detected"

        # Last 50ms should be quieter than the body (fade out)
        tail_rms = np.sqrt(np.mean(data[-fade_samples:] ** 2))
        body_end_rms = np.sqrt(np.mean(data[-fade_samples * 10:-fade_samples] ** 2))
        assert tail_rms < body_end_rms, "No fade-out detected"

    def test_idempotent_skips_existing(self, project_root):
        from agents.process_vocals import process_vocals

        track_id = "01_test_artist_test_song"
        _setup_track(project_root, track_id)

        stems_dir = project_root / "stems" / track_id
        # Create dummy outputs
        _make_vocal_wav(stems_dir / "vocals_processed.wav", duration=120.0)
        _make_vocal_wav(stems_dir / "vocals_hook.wav", duration=20.0)

        # Should return without modifying
        result = process_vocals(track_id, project_root)
        assert "vocals_processed" in result
        assert "vocals_hook" in result

    def test_returns_output_paths(self, project_root):
        from agents.process_vocals import process_vocals

        track_id = "01_test_artist_test_song"
        _setup_track(project_root, track_id)

        result = process_vocals(track_id, project_root)

        assert "vocals_processed" in result
        assert "vocals_hook" in result
        assert Path(result["vocals_processed"]).exists()
        assert Path(result["vocals_hook"]).exists()

    def test_works_with_different_bpm(self, project_root):
        from agents.process_vocals import process_vocals

        track_id = "01_test_artist_test_song"
        _setup_track(project_root, track_id, bpm=140.0)

        result = process_vocals(track_id, project_root)
        assert Path(result["vocals_processed"]).exists()

    def test_hook_from_short_chorus_merges_sections(self, project_root):
        """When first chorus is < 10s, should extend to meet minimum."""
        from agents.process_vocals import process_vocals

        track_id = "01_test_artist_test_song"
        stems_dir = project_root / "stems" / track_id
        stems_dir.mkdir(parents=True, exist_ok=True)
        _make_vocal_wav(stems_dir / "vocals.wav", duration=120.0)

        analysis = {
            "bpm": 120.0,
            "key": "C major",
            "duration": 120.0,
            "structure": [
                {"label": "intro", "start": 0.0, "end": 15.0},
                {"label": "verse", "start": 15.0, "end": 45.0},
                {"label": "chorus", "start": 45.0, "end": 53.0},  # only 8s
                {"label": "bridge", "start": 53.0, "end": 80.0},
                {"label": "chorus", "start": 80.0, "end": 110.0},
                {"label": "outro", "start": 110.0, "end": 120.0},
            ],
            "energy_curve": [0.5] * 120,
            "chords": [{"time": 0.0, "chord": "Cmaj"}],
        }
        analysis_path = project_root / "analysis" / f"{track_id}_analysis.json"
        analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")

        process_vocals(track_id, project_root)

        data, sr = sf.read(str(stems_dir / "vocals_hook.wav"))
        duration = len(data) / sr
        assert duration >= 10.0, f"Hook {duration:.1f}s < 10s even after extending"


class TestDODValidatorVocals:
    def test_passes_with_valid_outputs(self, project_root):
        from agents.process_vocals import process_vocals

        track_id = "01_test_artist_test_song"
        _setup_track(project_root, track_id)
        process_vocals(track_id, project_root)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "process_vocals")
        assert result.passed, f"Expected pass, got: {result.reasons}"

    def test_fails_if_processed_missing(self, project_root):
        track_id = "01_test_artist_test_song"
        stems_dir = project_root / "stems" / track_id
        stems_dir.mkdir(parents=True)
        _make_vocal_wav(stems_dir / "vocals_hook.wav", duration=20.0)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "process_vocals")
        assert not result.passed
        assert any("vocals_processed" in r for r in result.reasons)

    def test_fails_if_hook_missing(self, project_root):
        track_id = "01_test_artist_test_song"
        stems_dir = project_root / "stems" / track_id
        stems_dir.mkdir(parents=True)
        _make_vocal_wav(stems_dir / "vocals_processed.wav", duration=120.0)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "process_vocals")
        assert not result.passed
        assert any("vocals_hook" in r for r in result.reasons)

    def test_fails_if_rms_too_low(self, project_root):
        track_id = "01_test_artist_test_song"
        stems_dir = project_root / "stems" / track_id
        stems_dir.mkdir(parents=True)
        # Very quiet signal -> RMS well below -18
        _make_vocal_wav(stems_dir / "vocals_processed.wav", duration=120.0, amplitude=0.001)
        _make_vocal_wav(stems_dir / "vocals_hook.wav", duration=20.0)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "process_vocals")
        assert not result.passed
        assert any("RMS" in r for r in result.reasons)

    def test_fails_if_rms_too_high(self, project_root):
        track_id = "01_test_artist_test_song"
        stems_dir = project_root / "stems" / track_id
        stems_dir.mkdir(parents=True)
        # Loud signal -> RMS above -12
        _make_vocal_wav(stems_dir / "vocals_processed.wav", duration=120.0, amplitude=0.9)
        _make_vocal_wav(stems_dir / "vocals_hook.wav", duration=20.0)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "process_vocals")
        assert not result.passed
        assert any("RMS" in r for r in result.reasons)

    def test_fails_if_hook_too_short(self, project_root):
        track_id = "01_test_artist_test_song"
        stems_dir = project_root / "stems" / track_id
        stems_dir.mkdir(parents=True)
        _make_vocal_wav(stems_dir / "vocals_processed.wav", duration=120.0, amplitude=0.2)
        _make_vocal_wav(stems_dir / "vocals_hook.wav", duration=5.0)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "process_vocals")
        assert not result.passed
        assert any("hook" in r.lower() or "10s" in r for r in result.reasons)

    def test_fails_if_sub_80hz_energy(self, project_root):
        track_id = "01_test_artist_test_song"
        stems_dir = project_root / "stems" / track_id
        stems_dir.mkdir(parents=True)

        # Signal dominated by 30Hz — should fail the sub-80Hz check
        sr = 44100
        duration = 120.0
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        signal = 0.2 * np.sin(2 * np.pi * 30 * t)
        sf.write(str(stems_dir / "vocals_processed.wav"), signal.astype(np.float32), sr)
        _make_vocal_wav(stems_dir / "vocals_hook.wav", duration=20.0)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "process_vocals")
        assert not result.passed
        assert any("80" in r.lower() or "sub" in r.lower() for r in result.reasons)
