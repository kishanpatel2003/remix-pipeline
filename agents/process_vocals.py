from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, HighpassFilter, Reverb, Delay

logger = logging.getLogger(__name__)


def _load_analysis(project_root: Path, track_id: str) -> dict:
    path = project_root / "analysis" / f"{track_id}_analysis.json"
    if not path.exists():
        raise FileNotFoundError(f"Analysis file not found: {path}")
    return json.loads(path.read_text())


def _find_hook_region(structure: list[dict], total_duration: float) -> tuple[float, float]:
    """Find the hook region from the analysis structure.

    Uses the first chorus (or pre-chorus). If that section is shorter than 10s,
    extends forward into the next section to meet the minimum.
    """
    chorus_sections = [s for s in structure if s["label"] in ("chorus", "pre-chorus")]
    if not chorus_sections:
        # Fallback: use the middle 20s of the track
        mid = total_duration / 2
        return (max(0, mid - 10), min(total_duration, mid + 10))

    hook = chorus_sections[0]
    start = hook["start"]
    end = hook["end"]

    # Extend if shorter than 10s
    if end - start < 10.0:
        end = min(start + 10.0, total_duration)

    return (start, end)


def _apply_fx_chain(audio: np.ndarray, sr: int, bpm: float) -> np.ndarray:
    """Apply house vocal FX: HP 80Hz, room reverb, 1/8 note delay."""
    delay_seconds = (60.0 / bpm) / 2  # 1/8 note

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        Reverb(room_size=0.3, wet_level=0.15),
        Delay(delay_seconds=delay_seconds, mix=0.2),
    ])

    # pedalboard expects float32, shape (channels, samples) or (samples,)
    if audio.ndim == 1:
        processed = board(audio.astype(np.float32), sr)
    else:
        # Transpose to (channels, samples) for pedalboard
        processed = board(audio.astype(np.float32).T, sr).T

    return processed


def _normalize_rms(audio: np.ndarray, target_db: float = -15.0) -> np.ndarray:
    """Normalize audio to a target RMS level in dBFS."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio
    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)
    result = audio * gain_linear
    # Prevent clipping
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result = result / peak * 0.99
    return result


def _apply_fades(audio: np.ndarray, sr: int, fade_ms: float = 50.0) -> np.ndarray:
    """Apply linear fade-in and fade-out."""
    fade_samples = int(sr * fade_ms / 1000.0)
    if fade_samples == 0 or len(audio) < fade_samples * 2:
        return audio

    result = audio.copy()
    fade_in = np.linspace(0.0, 1.0, fade_samples)
    fade_out = np.linspace(1.0, 0.0, fade_samples)

    if result.ndim == 1:
        result[:fade_samples] *= fade_in
        result[-fade_samples:] *= fade_out
    else:
        result[:fade_samples] *= fade_in[:, np.newaxis]
        result[-fade_samples:] *= fade_out[:, np.newaxis]

    return result


def process_vocals(track_id: str, project_root: Path) -> dict[str, str]:
    """Process separated vocals with house FX and extract hook section.

    Returns dict with paths to vocals_processed and vocals_hook.
    """
    project_root = Path(project_root)
    stems_dir = project_root / "stems" / track_id

    processed_path = stems_dir / "vocals_processed.wav"
    hook_path = stems_dir / "vocals_hook.wav"

    # Idempotent
    if processed_path.exists() and hook_path.exists():
        logger.info(f"Processed vocals already exist for {track_id}, skipping")
        return {
            "vocals_processed": str(processed_path),
            "vocals_hook": str(hook_path),
        }

    vocals_path = stems_dir / "vocals.wav"
    if not vocals_path.exists():
        raise FileNotFoundError(f"Vocals stem not found: {vocals_path}")

    analysis = _load_analysis(project_root, track_id)
    bpm = analysis["bpm"]
    structure = analysis.get("structure", [])
    total_duration = analysis.get("duration", 0)

    # Load vocals
    audio, sr = sf.read(str(vocals_path))

    # Apply FX chain
    processed = _apply_fx_chain(audio, sr, bpm)

    # Normalize RMS to target range (-15 dBFS is center of -18 to -12)
    processed = _normalize_rms(processed, target_db=-15.0)

    # Write processed vocals
    sf.write(str(processed_path), processed, sr)
    logger.info(f"Written processed vocals to {processed_path}")

    # Extract hook
    hook_start, hook_end = _find_hook_region(structure, total_duration)
    start_sample = int(hook_start * sr)
    end_sample = int(hook_end * sr)

    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(processed), end_sample)

    hook_audio = processed[start_sample:end_sample].copy()
    hook_audio = _apply_fades(hook_audio, sr, fade_ms=50.0)

    sf.write(str(hook_path), hook_audio, sr)
    logger.info(f"Written hook ({hook_end - hook_start:.1f}s) to {hook_path}")

    return {
        "vocals_processed": str(processed_path),
        "vocals_hook": str(hook_path),
    }
