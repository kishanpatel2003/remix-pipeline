from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import soundfile as sf

from harness.dod_validator import DODValidator


@pytest.fixture
def project_root(tmp_path):
    for d in ("data", "raw", "analysis", "stems", "generated", "mixes", "masters", "output"):
        (tmp_path / d).mkdir()
    return tmp_path


def _make_analysis_json(path: Path, bpm: float = 158.0, key: str = "A major", duration: float = 252.0):
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
        "chords": [
            {"time": 0.0, "chord": "Amaj"},
            {"time": 4.0, "chord": "F#min"},
            {"time": 8.0, "chord": "Dmaj"},
            {"time": 12.0, "chord": "Emaj"},
        ],
    }
    path.write_text(json.dumps(analysis, indent=2) + "\n")


def _make_wav(path: Path, duration: float = 240.0, sr: int = 44100, amplitude: float = 0.5, stereo: bool = True):
    """Create a synthetic WAV file."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    if stereo:
        signal = np.column_stack([signal, signal])
    sf.write(str(path), signal, sr)


def _make_silent_wav(path: Path, duration: float = 240.0, sr: int = 44100):
    """Create a near-silent WAV (for Demucs vocal stem mock)."""
    n_samples = int(sr * duration)
    signal = np.random.randn(n_samples).astype(np.float32) * 0.0001
    sf.write(str(path), signal, sr)


def _mock_create_task_response():
    return MagicMock(
        status_code=200,
        json=lambda: {"data": {"taskId": "test-task-id-123"}},
    )


def _mock_poll_queuing_response():
    return MagicMock(
        status_code=200,
        json=lambda: {"data": {"status": "queuing"}},
    )


def _mock_poll_success_response(audio_url="https://example.com/audio.mp3"):
    return MagicMock(
        status_code=200,
        json=lambda: {
            "data": {
                "status": "success",
                "resultJson": {
                    "data": [{"audio_url": audio_url}],
                },
            },
        },
    )


def _mock_poll_fail_response():
    return MagicMock(
        status_code=200,
        json=lambda: {"data": {"status": "fail"}},
    )


def _setup_track(project_root: Path, track_id: str, bpm: float = 158.0):
    _make_analysis_json(
        project_root / "analysis" / f"{track_id}_analysis.json",
        bpm=bpm,
    )


class TestBuildPrompt:
    def test_includes_bpm_and_key(self, project_root):
        from agents.generate_instrumentation import _build_prompt

        analysis = json.loads(
            (project_root / "analysis" or Path("/dev/null")).read_text()
        ) if False else {
            "bpm": 158.0,
            "key": "A major",
            "chords": [{"time": 0.0, "chord": "Amaj"}, {"time": 4.0, "chord": "F#min"}],
        }
        prompt = _build_prompt(analysis)
        assert "158" in prompt
        assert "A major" in prompt

    def test_includes_chord_progression(self):
        from agents.generate_instrumentation import _build_prompt

        analysis = {
            "bpm": 120.0,
            "key": "C major",
            "chords": [
                {"time": 0.0, "chord": "Cmaj"},
                {"time": 4.0, "chord": "Amin"},
            ],
        }
        prompt = _build_prompt(analysis)
        assert "Cmaj" in prompt
        assert "Amin" in prompt

    def test_includes_arrangement_keywords(self):
        from agents.generate_instrumentation import _build_prompt

        analysis = {
            "bpm": 126.0,
            "key": "G minor",
            "chords": [{"time": 0.0, "chord": "Gmin"}],
        }
        prompt = _build_prompt(analysis)
        lower = prompt.lower()
        assert "intro" in lower
        assert "drop" in lower
        assert "breakdown" in lower


class TestCreateTask:
    @patch.dict(os.environ, {"APIPASS_KEY": "test-key"})
    @patch("agents.generate_instrumentation.requests.post")
    def test_creates_task_and_returns_id(self, mock_post):
        from agents.generate_instrumentation import _create_task

        mock_post.return_value = _mock_create_task_response()
        task_id = _create_task(bpm=158, key="A major", style="deep house", prompt="test prompt", title="test")
        assert task_id == "test-task-id-123"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "Bearer test-key" in str(call_kwargs)

    @patch.dict(os.environ, {"APIPASS_KEY": "test-key"})
    @patch("agents.generate_instrumentation.requests.post")
    def test_raises_on_api_error(self, mock_post):
        from agents.generate_instrumentation import _create_task

        mock_post.return_value = MagicMock(
            status_code=500,
            text="Internal Server Error",
            raise_for_status=MagicMock(side_effect=Exception("500 Server Error")),
        )
        with pytest.raises(Exception):
            _create_task(bpm=120, key="C major", style="deep house", prompt="test", title="test")


class TestPollResult:
    @patch.dict(os.environ, {"APIPASS_KEY": "test-key"})
    @patch("agents.generate_instrumentation.requests.get")
    @patch("agents.generate_instrumentation.time.sleep")
    def test_polls_until_success(self, mock_sleep, mock_get):
        from agents.generate_instrumentation import _poll_result

        mock_get.side_effect = [
            _mock_poll_queuing_response(),
            _mock_poll_queuing_response(),
            _mock_poll_success_response("https://example.com/final.mp3"),
        ]
        audio_url = _poll_result("task-123")
        assert audio_url == "https://example.com/final.mp3"
        assert mock_get.call_count == 3

    @patch.dict(os.environ, {"APIPASS_KEY": "test-key"})
    @patch("agents.generate_instrumentation.requests.get")
    @patch("agents.generate_instrumentation.time.sleep")
    def test_raises_on_fail_status(self, mock_sleep, mock_get):
        from agents.generate_instrumentation import _poll_result

        mock_get.return_value = _mock_poll_fail_response()
        with pytest.raises(RuntimeError, match="fail"):
            _poll_result("task-123")

    @patch.dict(os.environ, {"APIPASS_KEY": "test-key"})
    @patch("agents.generate_instrumentation.requests.get")
    @patch("agents.generate_instrumentation.time.sleep")
    def test_raises_on_timeout(self, mock_sleep, mock_get):
        from agents.generate_instrumentation import _poll_result

        mock_get.return_value = _mock_poll_queuing_response()
        with pytest.raises(TimeoutError):
            _poll_result("task-123", max_polls=3)


class TestDownloadAndConvert:
    @patch("agents.generate_instrumentation.requests.get")
    @patch("agents.generate_instrumentation.subprocess.run")
    def test_downloads_mp3_and_converts_to_wav(self, mock_run, mock_get, tmp_path):
        from agents.generate_instrumentation import _download_and_convert

        mp3_path = tmp_path / "test.mp3"
        wav_path = tmp_path / "test.wav"

        # Mock HTTP download
        mock_get.return_value = MagicMock(
            status_code=200,
            content=b"fake mp3 data",
            raise_for_status=MagicMock(),
        )

        # Mock ffmpeg conversion — actually write a WAV so the path exists
        def fake_ffmpeg(*args, **kwargs):
            _make_wav(wav_path, duration=240.0)
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_ffmpeg

        result_path = _download_and_convert("https://example.com/audio.mp3", mp3_path, wav_path)
        assert result_path == wav_path
        assert wav_path.exists()


class TestGenerateInstrumentation:
    @patch("agents.generate_instrumentation._run_demucs_vocal_check")
    @patch("agents.generate_instrumentation._download_and_convert")
    @patch("agents.generate_instrumentation._poll_result")
    @patch("agents.generate_instrumentation._create_task")
    def test_full_pipeline(self, mock_create, mock_poll, mock_download, mock_demucs, project_root):
        from agents.generate_instrumentation import generate_instrumentation

        track_id = "01_lady_gaga_bruno_mars_die_with_a_smile"
        _setup_track(project_root, track_id, bpm=158.0)

        mock_create.return_value = "task-123"
        mock_poll.return_value = "https://example.com/audio.mp3"

        wav_path = project_root / "generated" / f"{track_id}_instrumental.wav"

        def fake_download(url, mp3_path, wav_out):
            _make_wav(wav_out, duration=240.0, amplitude=0.5)
            return wav_out

        mock_download.side_effect = fake_download
        mock_demucs.return_value = -35.0  # below -30 dBFS → no vocals

        result = generate_instrumentation(track_id, project_root)
        assert "instrumental" in result
        assert wav_path.exists()

    @patch("agents.generate_instrumentation._run_demucs_vocal_check")
    @patch("agents.generate_instrumentation._download_and_convert")
    @patch("agents.generate_instrumentation._poll_result")
    @patch("agents.generate_instrumentation._create_task")
    def test_raises_if_analysis_missing(self, mock_create, mock_poll, mock_download, mock_demucs, project_root):
        from agents.generate_instrumentation import generate_instrumentation

        with pytest.raises(FileNotFoundError):
            generate_instrumentation("nonexistent_track", project_root)

    @patch("agents.generate_instrumentation._run_demucs_vocal_check")
    @patch("agents.generate_instrumentation._download_and_convert")
    @patch("agents.generate_instrumentation._poll_result")
    @patch("agents.generate_instrumentation._create_task")
    def test_idempotent_skips_existing(self, mock_create, mock_poll, mock_download, mock_demucs, project_root):
        from agents.generate_instrumentation import generate_instrumentation

        track_id = "01_test_track"
        _setup_track(project_root, track_id)
        wav_path = project_root / "generated" / f"{track_id}_instrumental.wav"
        _make_wav(wav_path, duration=240.0)

        result = generate_instrumentation(track_id, project_root)
        assert "instrumental" in result
        mock_create.assert_not_called()

    @patch("agents.generate_instrumentation._run_demucs_vocal_check")
    @patch("agents.generate_instrumentation._download_and_convert")
    @patch("agents.generate_instrumentation._poll_result")
    @patch("agents.generate_instrumentation._create_task")
    def test_raises_on_vocal_contamination(self, mock_create, mock_poll, mock_download, mock_demucs, project_root):
        from agents.generate_instrumentation import generate_instrumentation

        track_id = "01_test_track"
        _setup_track(project_root, track_id)

        mock_create.return_value = "task-123"
        mock_poll.return_value = "https://example.com/audio.mp3"

        def fake_download(url, mp3_path, wav_out):
            _make_wav(wav_out, duration=240.0, amplitude=0.5)
            return wav_out

        mock_download.side_effect = fake_download
        mock_demucs.return_value = -20.0  # above -30 → vocals detected

        with pytest.raises(ValueError, match="vocal"):
            generate_instrumentation(track_id, project_root)

    @patch("agents.generate_instrumentation._run_demucs_vocal_check")
    @patch("agents.generate_instrumentation._download_and_convert")
    @patch("agents.generate_instrumentation._poll_result")
    @patch("agents.generate_instrumentation._create_task")
    def test_normalizes_peak(self, mock_create, mock_poll, mock_download, mock_demucs, project_root):
        from agents.generate_instrumentation import generate_instrumentation

        track_id = "01_test_track"
        _setup_track(project_root, track_id)

        mock_create.return_value = "task-123"
        mock_poll.return_value = "https://example.com/audio.mp3"

        def fake_download(url, mp3_path, wav_out):
            # Write a loud signal that needs peak normalization
            _make_wav(wav_out, duration=240.0, amplitude=0.95)
            return wav_out

        mock_download.side_effect = fake_download
        mock_demucs.return_value = -35.0

        generate_instrumentation(track_id, project_root)

        wav_path = project_root / "generated" / f"{track_id}_instrumental.wav"
        data, sr = sf.read(str(wav_path))
        peak_db = 20 * np.log10(np.max(np.abs(data)))
        assert peak_db <= -0.5, f"Peak {peak_db:.1f} dBFS exceeds -0.5 dBFS"


class TestDemucsVocalCheck:
    @patch("agents.generate_instrumentation.subprocess.run")
    def test_returns_vocal_rms_db(self, mock_run, tmp_path):
        from agents.generate_instrumentation import _run_demucs_vocal_check

        wav_path = tmp_path / "instrumental.wav"
        _make_wav(wav_path, duration=30.0)

        # Mock demucs to write a near-silent vocal stem
        def fake_demucs(cmd, **kwargs):
            # Demucs writes to htdemucs/ subdir inside output dir
            out_dir = None
            for i, arg in enumerate(cmd):
                if arg == "-o":
                    out_dir = Path(cmd[i + 1])
                    break
            if out_dir:
                stems_dir = out_dir / "htdemucs" / wav_path.stem
                stems_dir.mkdir(parents=True, exist_ok=True)
                _make_silent_wav(stems_dir / "vocals.wav", duration=30.0)
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_demucs

        rms_db = _run_demucs_vocal_check(wav_path, tmp_path)
        assert rms_db < -30.0


class TestDODValidatorInstrumentation:
    def test_passes_with_valid_output(self, project_root):
        track_id = "01_test_track"
        _setup_track(project_root, track_id, bpm=158.0)
        wav_path = project_root / "generated" / f"{track_id}_instrumental.wav"
        # Peak at -1 dBFS
        amplitude = 10 ** (-1.0 / 20)
        _make_wav(wav_path, duration=240.0, amplitude=amplitude)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "generate_instrumentation",
                                    vocal_rms_db=-35.0, measured_bpm=158.5)
        assert result.passed, f"Expected pass, got: {result.reasons}"

    def test_fails_if_wav_missing(self, project_root):
        track_id = "01_test_track"
        _setup_track(project_root, track_id)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "generate_instrumentation",
                                    vocal_rms_db=-35.0, measured_bpm=158.0)
        assert not result.passed
        assert any("missing" in r.lower() or "Missing" in r for r in result.reasons)

    def test_fails_if_too_short(self, project_root):
        track_id = "01_test_track"
        _setup_track(project_root, track_id)
        wav_path = project_root / "generated" / f"{track_id}_instrumental.wav"
        amplitude = 10 ** (-1.0 / 20)
        _make_wav(wav_path, duration=120.0, amplitude=amplitude)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "generate_instrumentation",
                                    vocal_rms_db=-35.0, measured_bpm=158.0)
        assert not result.passed
        assert any("duration" in r.lower() or "180" in r for r in result.reasons)

    def test_fails_if_bpm_off(self, project_root):
        track_id = "01_test_track"
        _setup_track(project_root, track_id, bpm=158.0)
        wav_path = project_root / "generated" / f"{track_id}_instrumental.wav"
        amplitude = 10 ** (-1.0 / 20)
        _make_wav(wav_path, duration=240.0, amplitude=amplitude)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "generate_instrumentation",
                                    vocal_rms_db=-35.0, measured_bpm=165.0)
        assert not result.passed
        assert any("bpm" in r.lower() for r in result.reasons)

    def test_fails_if_vocals_detected(self, project_root):
        track_id = "01_test_track"
        _setup_track(project_root, track_id)
        wav_path = project_root / "generated" / f"{track_id}_instrumental.wav"
        amplitude = 10 ** (-1.0 / 20)
        _make_wav(wav_path, duration=240.0, amplitude=amplitude)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "generate_instrumentation",
                                    vocal_rms_db=-20.0, measured_bpm=158.0)
        assert not result.passed
        assert any("vocal" in r.lower() for r in result.reasons)

    def test_fails_if_peak_too_high(self, project_root):
        track_id = "01_test_track"
        _setup_track(project_root, track_id)
        wav_path = project_root / "generated" / f"{track_id}_instrumental.wav"
        _make_wav(wav_path, duration=240.0, amplitude=0.99)

        validator = DODValidator(project_root)
        result = validator.validate(track_id, "generate_instrumentation",
                                    vocal_rms_db=-35.0, measured_bpm=158.0)
        assert not result.passed
        assert any("peak" in r.lower() for r in result.reasons)
