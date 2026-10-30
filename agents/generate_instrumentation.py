from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

logger = logging.getLogger(__name__)

APIPASS_BASE_URL = "https://api.apipass.dev"


def _load_analysis(project_root: Path, track_id: str) -> dict:
    path = project_root / "analysis" / f"{track_id}_analysis.json"
    if not path.exists():
        raise FileNotFoundError(f"Analysis file not found: {path}")
    return json.loads(path.read_text())


def _build_prompt(analysis: dict) -> str:
    """Build a Suno prompt encoding BPM, key, chords, and arrangement."""
    bpm = analysis["bpm"]
    key = analysis["key"]
    chords = analysis.get("chords", [])
    chord_names = []
    seen = set()
    for c in chords:
        name = c.get("chord", "")
        if name and name not in seen:
            chord_names.append(name)
            seen.add(name)

    progression = " - ".join(chord_names) if chord_names else "standard house progression"

    return (
        f"Deep house instrumental, {int(bpm)} BPM, key of {key}. "
        f"Chord progression: {progression}. "
        f"Arrangement: minimal intro with filtered pad, "
        f"gradual build with rising hi-hats, "
        f"first drop with full kick and bass groove, "
        f"breakdown with atmospheric pads and tension, "
        f"final drop with maximum energy. "
        f"Four-on-the-floor kick, deep rolling bassline, "
        f"warm analog pads, crisp hi-hats. "
        f"No vocals. Pure instrumental. Target 4 minutes."
    )


def _create_task(bpm: int, key: str, style: str, prompt: str, title: str) -> str:
    """POST to APIPASS to create a Suno generation task. Returns taskId."""
    api_key = os.environ.get("APIPASS_KEY", "")
    resp = requests.post(
        f"{APIPASS_BASE_URL}/api/v1/jobs/createTask",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "suno/generate",
            "input": {
                "customMode": True,
                "instrumental": True,
                "style": style,
                "title": title,
                "prompt": prompt,
            },
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    task_id = data["data"]["taskId"]
    logger.info(f"Created APIPASS task: {task_id}")
    return task_id


def _poll_result(task_id: str, max_polls: int = 120, interval: float = 10.0) -> str:
    """Poll APIPASS until task completes. Returns audio URL."""
    api_key = os.environ.get("APIPASS_KEY", "")
    for i in range(max_polls):
        resp = requests.get(
            f"{APIPASS_BASE_URL}/api/v1/jobs/recordInfo",
            params={"taskId": task_id},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        state = data.get("state") or data.get("status")

        if state == "success":
            result_json = data.get("resultJson")
            if isinstance(result_json, str):
                result_json = json.loads(result_json)
            audio_url = result_json["data"][0]["audio_url"]
            logger.info(f"Task {task_id} succeeded: {audio_url}")
            return audio_url
        elif state == "fail":
            fail_msg = data.get("failMsg", "unknown")
            raise RuntimeError(f"APIPASS task {task_id} failed: {fail_msg}")
        else:
            logger.debug(f"Task {task_id} state: {state} (poll {i+1}/{max_polls})")
            time.sleep(interval)

    raise TimeoutError(f"APIPASS task {task_id} did not complete within {max_polls} polls")


def _download_and_convert(audio_url: str, mp3_path: Path, wav_path: Path) -> Path:
    """Download MP3 from URL and convert to WAV via ffmpeg."""
    resp = requests.get(audio_url, timeout=60)
    resp.raise_for_status()
    mp3_path.write_bytes(resp.content)
    logger.info(f"Downloaded {len(resp.content)} bytes to {mp3_path}")

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3_path), "-acodec", "pcm_f32le", "-ar", "44100", str(wav_path)],
        check=True,
        capture_output=True,
    )
    logger.info(f"Converted to WAV: {wav_path}")
    return wav_path


def _normalize_peak(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Normalize audio so peak is at target_db dBFS."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    target_linear = 10 ** (target_db / 20)
    return audio * (target_linear / peak)


def _run_demucs_vocal_check(wav_path: Path, work_dir: Path) -> float:
    """Run Demucs on the WAV and return the vocal stem RMS in dBFS."""
    demucs_out = work_dir / "demucs_check"
    demucs_out.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [sys.executable, "-m", "demucs", "-n", "htdemucs", "-o", str(demucs_out), str(wav_path)],
        check=True,
        capture_output=True,
    )

    vocal_path = demucs_out / "htdemucs" / wav_path.stem / "vocals.wav"
    if not vocal_path.exists():
        raise FileNotFoundError(f"Demucs vocal stem not found: {vocal_path}")

    data, sr = sf.read(str(vocal_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    rms = np.sqrt(np.mean(data ** 2))
    rms_db = 20 * np.log10(rms) if rms > 0 else -100.0
    logger.info(f"Vocal check RMS: {rms_db:.1f} dBFS")
    return rms_db


def generate_instrumentation(track_id: str, project_root: Path) -> dict[str, str]:
    """Generate a deep house instrumental via APIPASS/Suno.

    Returns dict with path to the generated instrumental WAV.
    """
    project_root = Path(project_root)
    generated_dir = project_root / "generated"
    wav_path = generated_dir / f"{track_id}_instrumental.wav"

    # Idempotent
    if wav_path.exists():
        logger.info(f"Instrumental already exists for {track_id}, skipping")
        return {"instrumental": str(wav_path)}

    analysis = _load_analysis(project_root, track_id)
    bpm = analysis["bpm"]
    key = analysis["key"]

    prompt = _build_prompt(analysis)
    style = f"deep house, {int(bpm)} BPM, {key}"
    title = f"{track_id} deep house instrumental"

    # Create generation task
    task_id = _create_task(
        bpm=int(bpm),
        key=key,
        style=style,
        prompt=prompt,
        title=title,
    )

    # Poll until complete
    audio_url = _poll_result(task_id)

    # Download and convert
    mp3_path = generated_dir / f"{track_id}_instrumental.mp3"
    _download_and_convert(audio_url, mp3_path, wav_path)

    # Normalize peak to -1 dBFS (leaves headroom, meets <= -0.5 dBFS DOD)
    data, sr = sf.read(str(wav_path))
    data = _normalize_peak(data, target_db=-1.0)
    sf.write(str(wav_path), data, sr)

    # Vocal contamination check via Demucs
    vocal_rms_db = _run_demucs_vocal_check(wav_path, project_root)
    if vocal_rms_db > -30.0:
        # Clean up the contaminated file
        wav_path.unlink(missing_ok=True)
        raise ValueError(
            f"Vocal contamination detected: vocal stem RMS {vocal_rms_db:.1f} dBFS "
            f"(threshold: -30 dBFS). Regeneration needed."
        )

    # Clean up intermediate MP3
    mp3_path.unlink(missing_ok=True)

    logger.info(f"Generated instrumental for {track_id}: {wav_path}")
    return {
        "instrumental": str(wav_path),
        "vocal_rms_db": vocal_rms_db,
    }
