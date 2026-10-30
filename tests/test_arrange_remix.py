from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from harness.dod_validator import DODValidator

TRACK_ID = "01_test_artist_test_song"


@pytest.fixture
def project_root(tmp_path):
    for d in ("data", "raw", "analysis", "stems", "generated", "mixes", "masters", "output"):
        (tmp_path / d).mkdir()
    return tmp_path


def _make_analysis(path: Path, bpm=126.0, key="A major", duration=252.0):
    analysis = {
        "bpm": bpm,
        "key": key,
        "duration": duration,
        "structure": [
            {"label": "intro", "start": 0.0, "end": 30.0},
            {"label": "verse", "start": 30.0, "end": 90.0},
            {"label": "chorus", "start": 90.0, "end": 150.0},
            {"label": "bridge", "start": 150.0, "end": 200.0},
            {"label": "chorus", "start": 200.0, "end": 240.0},
            {"label": "outro", "start": 240.0, "end": duration},
        ],
        "energy_curve": [0.5] * int(duration),
        "chords": [{"time": 0.0, "chord": "Amaj"}],
    }
    path.write_text(json.dumps(analysis))


def _make_wav(path: Path, duration=240.0, sr=44100, amplitude=0.5, stereo=True, freq=440.0):
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    sig = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if stereo:
        sig = np.column_stack([sig, sig])
    sf.write(str(path), sig, sr)


def _setup_track(
    project_root: Path,
    track_id=TRACK_ID,
    bpm=126.0,
    key="A major",
    source_duration=252.0,
    instrumental_duration=240.0,
):
    _make_analysis(
        project_root / "analysis" / f"{track_id}_analysis.json",
        bpm=bpm, key=key, duration=source_duration,
    )
    stems_dir = project_root / "stems" / track_id
    stems_dir.mkdir(parents=True, exist_ok=True)
    _make_wav(stems_dir / "vocals_processed.wav", duration=source_duration, amplitude=0.3)
    _make_wav(stems_dir / "vocals_hook.wav", duration=30.0, amplitude=0.3)
    _make_wav(
        project_root / "generated" / f"{track_id}_instrumental.wav",
        duration=instrumental_duration, amplitude=0.5,
    )


class TestArrangeRemix:
    def test_produces_rough_mix(self, project_root):
        from agents.arrange_remix import arrange_remix

        _setup_track(project_root)
        result = arrange_remix(TRACK_ID, project_root)
        assert "rough_mix" in result
        assert (project_root / "mixes" / f"{TRACK_ID}_rough.wav").exists()

    def test_output_is_24bit_stereo(self, project_root):
        from agents.arrange_remix import arrange_remix

        _setup_track(project_root)
        arrange_remix(TRACK_ID, project_root)
        info = sf.info(str(project_root / "mixes" / f"{TRACK_ID}_rough.wav"))
        assert info.subtype == "PCM_24"
        assert info.channels == 2

    def test_duration_matches_instrumental(self, project_root):
        from agents.arrange_remix import arrange_remix

        inst_dur = 240.0
        _setup_track(project_root, instrumental_duration=inst_dur)
        arrange_remix(TRACK_ID, project_root)

        data, sr = sf.read(str(project_root / "mixes" / f"{TRACK_ID}_rough.wav"))
        mix_dur = len(data) / sr
        assert abs(mix_dur - inst_dur) < 0.5

    def test_peak_in_dod_range(self, project_root):
        from agents.arrange_remix import arrange_remix

        _setup_track(project_root)
        arrange_remix(TRACK_ID, project_root)

        data, sr = sf.read(str(project_root / "mixes" / f"{TRACK_ID}_rough.wav"))
        peak = np.max(np.abs(data))
        peak_db = 20 * np.log10(peak) if peak > 0 else -100.0
        assert -3.5 <= peak_db <= -0.5, f"Peak {peak_db:.1f} dBFS outside DOD range"

    def test_idempotent_skips_existing(self, project_root):
        from agents.arrange_remix import arrange_remix

        _setup_track(project_root)
        mix_path = project_root / "mixes" / f"{TRACK_ID}_rough.wav"
        _make_wav(mix_path, duration=240.0)

        result = arrange_remix(TRACK_ID, project_root)
        assert "rough_mix" in result

    def test_raises_if_instrumental_missing(self, project_root):
        from agents.arrange_remix import arrange_remix

        _make_analysis(project_root / "analysis" / f"{TRACK_ID}_analysis.json")
        stems_dir = project_root / "stems" / TRACK_ID
        stems_dir.mkdir(parents=True, exist_ok=True)
        _make_wav(stems_dir / "vocals_processed.wav", duration=252.0)
        _make_wav(stems_dir / "vocals_hook.wav", duration=30.0)

        with pytest.raises(FileNotFoundError):
            arrange_remix(TRACK_ID, project_root)

    def test_raises_if_vocals_missing(self, project_root):
        from agents.arrange_remix import arrange_remix

        _make_analysis(project_root / "analysis" / f"{TRACK_ID}_analysis.json")
        _make_wav(project_root / "generated" / f"{TRACK_ID}_instrumental.wav", duration=240.0)

        with pytest.raises(FileNotFoundError):
            arrange_remix(TRACK_ID, project_root)

    def test_hook_audible_at_drop(self, project_root):
        from agents.arrange_remix import arrange_remix, _calculate_drop_position

        bpm = 126.0
        _setup_track(project_root, bpm=bpm)
        # Distinct hook frequency so we can detect it in the mix
        stems_dir = project_root / "stems" / TRACK_ID
        _make_wav(stems_dir / "vocals_hook.wav", duration=30.0, amplitude=0.5, freq=1000.0)

        arrange_remix(TRACK_ID, project_root)

        data, sr = sf.read(str(project_root / "mixes" / f"{TRACK_ID}_rough.wav"))
        drop_sample = int(_calculate_drop_position(bpm) * sr)
        window_end = min(drop_sample + sr, len(data))
        window = data[drop_sample:window_end]
        if window.ndim > 1:
            window = window.mean(axis=1)

        rms = np.sqrt(np.mean(window ** 2))
        rms_db = 20 * np.log10(rms) if rms > 0 else -100.0
        assert rms_db > -40, f"Hook not audible at drop: RMS {rms_db:.1f} dBFS"

    def test_vocals_layered_at_verse_chorus(self, project_root):
        from agents.arrange_remix import arrange_remix

        _setup_track(project_root, bpm=126.0, source_duration=252.0, instrumental_duration=240.0)
        stems_dir = project_root / "stems" / TRACK_ID
        _make_wav(stems_dir / "vocals_processed.wav", duration=252.0, amplitude=0.3, freq=880.0)

        arrange_remix(TRACK_ID, project_root)

        data, sr = sf.read(str(project_root / "mixes" / f"{TRACK_ID}_rough.wav"))
        scale = 240.0 / 252.0
        # Check 1s into the verse (at ~30s scaled)
        verse_sample = int(31.0 * scale * sr)
        window = data[verse_sample:verse_sample + sr]
        if window.ndim > 1:
            window = window.mean(axis=1)
        rms = np.sqrt(np.mean(window ** 2))
        rms_db = 20 * np.log10(rms) if rms > 0 else -100.0
        assert rms_db > -40, f"No vocal energy at verse: {rms_db:.1f} dBFS"


class TestDropPosition:
    def test_drop_at_126_bpm(self):
        from agents.arrange_remix import _calculate_drop_position

        pos = _calculate_drop_position(126.0)
        assert 60 < pos < 66

    def test_drop_scales_inversely_with_bpm(self):
        from agents.arrange_remix import _calculate_drop_position

        slow = _calculate_drop_position(100.0)
        fast = _calculate_drop_position(150.0)
        assert slow > fast


class TestSemitoneDistance:
    def test_same_key_zero(self):
        from agents.arrange_remix import _semitone_distance

        assert _semitone_distance("A major", "A major") == 0

    def test_a_to_c(self):
        from agents.arrange_remix import _semitone_distance

        assert _semitone_distance("A major", "C major") == 3

    def test_c_to_b_shortest_path(self):
        from agents.arrange_remix import _semitone_distance

        assert _semitone_distance("C major", "B major") == -1


class TestTimeStretchVocals:
    @patch("agents.arrange_remix.librosa")
    def test_calls_librosa_when_bpms_differ(self, mock_librosa):
        from agents.arrange_remix import _time_stretch_vocals

        sr = 44100
        audio = np.sin(np.linspace(0, 10, sr * 10)).astype(np.float32)
        mock_librosa.effects.time_stretch.return_value = audio

        _time_stretch_vocals(audio, sr, source_bpm=161.0, target_bpm=126.0)
        mock_librosa.effects.time_stretch.assert_called_once()

    def test_noop_when_bpms_match(self):
        from agents.arrange_remix import _time_stretch_vocals

        sr = 44100
        audio = np.sin(np.linspace(0, 10, sr * 10)).astype(np.float32)
        result = _time_stretch_vocals(audio, sr, source_bpm=126.0, target_bpm=126.0)
        np.testing.assert_array_equal(result, audio)


class TestPitchShiftVocals:
    @patch("agents.arrange_remix.librosa")
    def test_calls_librosa_when_keys_differ(self, mock_librosa):
        from agents.arrange_remix import _pitch_shift_vocals

        sr = 44100
        audio = np.sin(np.linspace(0, 10, sr * 10)).astype(np.float32)
        mock_librosa.effects.pitch_shift.return_value = audio

        _pitch_shift_vocals(audio, sr, source_key="A major", target_key="C major")
        mock_librosa.effects.pitch_shift.assert_called_once()

    def test_noop_when_keys_match(self):
        from agents.arrange_remix import _pitch_shift_vocals

        sr = 44100
        audio = np.sin(np.linspace(0, 10, sr * 10)).astype(np.float32)
        result = _pitch_shift_vocals(audio, sr, source_key="A major", target_key="A major")
        np.testing.assert_array_equal(result, audio)


class TestDODValidatorArrangeRemix:
    def test_passes_valid_output(self, project_root):
        _setup_track(project_root, instrumental_duration=240.0)
        mix_path = project_root / "mixes" / f"{TRACK_ID}_rough.wav"
        amplitude = 10 ** (-3.0 / 20)
        _make_wav(mix_path, duration=240.0, amplitude=amplitude)
        data, sr = sf.read(str(mix_path))
        sf.write(str(mix_path), data, sr, subtype="PCM_24")

        validator = DODValidator(project_root)
        result = validator.validate(TRACK_ID, "arrange_remix", instrumental_duration=240.0)
        assert result.passed, f"Expected pass: {result.reasons}"

    def test_fails_if_mix_missing(self, project_root):
        validator = DODValidator(project_root)
        result = validator.validate(TRACK_ID, "arrange_remix", instrumental_duration=240.0)
        assert not result.passed

    def test_fails_if_not_24bit(self, project_root):
        mix_path = project_root / "mixes" / f"{TRACK_ID}_rough.wav"
        amplitude = 10 ** (-3.0 / 20)
        _make_wav(mix_path, duration=240.0, amplitude=amplitude)

        validator = DODValidator(project_root)
        result = validator.validate(TRACK_ID, "arrange_remix", instrumental_duration=240.0)
        assert not result.passed
        assert any("24" in r for r in result.reasons)

    def test_fails_if_not_stereo(self, project_root):
        mix_path = project_root / "mixes" / f"{TRACK_ID}_rough.wav"
        amplitude = 10 ** (-3.0 / 20)
        _make_wav(mix_path, duration=240.0, amplitude=amplitude, stereo=False)
        data, sr = sf.read(str(mix_path))
        sf.write(str(mix_path), data, sr, subtype="PCM_24")

        validator = DODValidator(project_root)
        result = validator.validate(TRACK_ID, "arrange_remix", instrumental_duration=240.0)
        assert not result.passed
        assert any("stereo" in r.lower() for r in result.reasons)

    def test_fails_if_duration_mismatch(self, project_root):
        mix_path = project_root / "mixes" / f"{TRACK_ID}_rough.wav"
        amplitude = 10 ** (-3.0 / 20)
        _make_wav(mix_path, duration=200.0, amplitude=amplitude)
        data, sr = sf.read(str(mix_path))
        sf.write(str(mix_path), data, sr, subtype="PCM_24")

        validator = DODValidator(project_root)
        result = validator.validate(TRACK_ID, "arrange_remix", instrumental_duration=240.0)
        assert not result.passed
        assert any("duration" in r.lower() for r in result.reasons)

    def test_fails_if_peak_too_high(self, project_root):
        mix_path = project_root / "mixes" / f"{TRACK_ID}_rough.wav"
        _make_wav(mix_path, duration=240.0, amplitude=0.99)
        data, sr = sf.read(str(mix_path))
        sf.write(str(mix_path), data, sr, subtype="PCM_24")

        validator = DODValidator(project_root)
        result = validator.validate(TRACK_ID, "arrange_remix", instrumental_duration=240.0)
        assert not result.passed
        assert any("peak" in r.lower() for r in result.reasons)

    def test_fails_if_peak_too_low(self, project_root):
        mix_path = project_root / "mixes" / f"{TRACK_ID}_rough.wav"
        _make_wav(mix_path, duration=240.0, amplitude=0.01)
        data, sr = sf.read(str(mix_path))
        sf.write(str(mix_path), data, sr, subtype="PCM_24")

        validator = DODValidator(project_root)
        result = validator.validate(TRACK_ID, "arrange_remix", instrumental_duration=240.0)
        assert not result.passed
        assert any("peak" in r.lower() for r in result.reasons)
