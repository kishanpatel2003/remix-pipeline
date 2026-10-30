from __future__ import annotations

import json
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

DROP_BAR = 33

NOTE_MAP = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}


def _load_analysis(project_root: Path, track_id: str) -> dict:
    path = project_root / "analysis" / f"{track_id}_analysis.json"
    if not path.exists():
        raise FileNotFoundError(f"Analysis file not found: {path}")
    return json.loads(path.read_text())


def _calculate_drop_position(bpm: float) -> float:
    """Drop position in seconds at bar 33 (standard deep house arrangement)."""
    beat_duration = 60.0 / bpm
    bar_duration = 4 * beat_duration
    return DROP_BAR * bar_duration


def _semitone_distance(source_key: str, target_key: str) -> int:
    """Shortest semitone distance between two keys (root note only)."""
    source_note = source_key.split()[0]
    target_note = target_key.split()[0]
    source_st = NOTE_MAP.get(source_note, 0)
    target_st = NOTE_MAP.get(target_note, 0)
    diff = target_st - source_st
    if diff > 6:
        diff -= 12
    elif diff < -6:
        diff += 12
    return diff


def _time_stretch_vocals(
    audio: np.ndarray, sr: int, source_bpm: float, target_bpm: float,
) -> np.ndarray:
    """Time-stretch audio to match target BPM. No-op if BPMs are close."""
    if abs(source_bpm - target_bpm) < 0.5:
        return audio
    rate = target_bpm / source_bpm
    if audio.ndim == 2:
        stretched = librosa.effects.time_stretch(audio.T, rate=rate)
        return stretched.T
    return librosa.effects.time_stretch(audio, rate=rate)


def _pitch_shift_vocals(
    audio: np.ndarray, sr: int, source_key: str, target_key: str,
) -> np.ndarray:
    """Pitch-shift audio by the semitone distance between keys. No-op if same."""
    n_steps = _semitone_distance(source_key, target_key)
    if n_steps == 0:
        return audio
    if audio.ndim == 2:
        shifted = librosa.effects.pitch_shift(audio.T, sr=sr, n_steps=n_steps)
        return shifted.T
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def _to_stereo(audio: np.ndarray) -> np.ndarray:
    """Ensure audio is (samples, 2)."""
    if audio.ndim == 1:
        return np.column_stack([audio, audio])
    if audio.shape[1] == 1:
        return np.column_stack([audio[:, 0], audio[:, 0]])
    return audio


def _overlay(base: np.ndarray, layer: np.ndarray, start_sample: int):
    """Add layer onto base at start_sample (in-place)."""
    end = min(start_sample + len(layer), len(base))
    n = end - start_sample
    if n <= 0:
        return
    base[start_sample:end] += layer[:n]


def _normalize_peak(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Normalize audio peak to target_db dBFS."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    target_linear = 10 ** (target_db / 20)
    return audio * (target_linear / peak)


def arrange_remix(track_id: str, project_root: Path) -> dict[str, str]:
    """Combine processed vocals with generated instrumentation into a rough mix.

    Returns dict with path to the rough mix WAV.
    """
    project_root = Path(project_root)
    mix_path = project_root / "mixes" / f"{track_id}_rough.wav"

    if mix_path.exists():
        logger.info(f"Rough mix already exists for {track_id}, skipping")
        return {"rough_mix": str(mix_path)}

    analysis = _load_analysis(project_root, track_id)
    source_bpm = analysis["bpm"]
    source_key = analysis["key"]
    structure = analysis.get("structure", [])
    source_duration = analysis.get("duration", 0)

    # Load instrumental
    instrumental_path = project_root / "generated" / f"{track_id}_instrumental.wav"
    if not instrumental_path.exists():
        raise FileNotFoundError(f"Instrumental not found: {instrumental_path}")
    instrumental, sr = sf.read(str(instrumental_path))
    instrumental_duration = len(instrumental) / sr

    # Load processed vocals and hook
    stems_dir = project_root / "stems" / track_id
    vocals_path = stems_dir / "vocals_processed.wav"
    hook_path = stems_dir / "vocals_hook.wav"
    if not vocals_path.exists():
        raise FileNotFoundError(f"Processed vocals not found: {vocals_path}")
    if not hook_path.exists():
        raise FileNotFoundError(f"Hook vocals not found: {hook_path}")

    vocals, _ = sf.read(str(vocals_path))
    hook, _ = sf.read(str(hook_path))

    # Time-stretch vocals to match instrumental BPM
    target_bpm = source_bpm  # instrumental generated at source BPM
    vocals = _time_stretch_vocals(vocals, sr, source_bpm, target_bpm)
    hook = _time_stretch_vocals(hook, sr, source_bpm, target_bpm)

    # Pitch-shift if keys differ
    target_key = source_key  # instrumental generated in source key
    vocals = _pitch_shift_vocals(vocals, sr, source_key, target_key)
    hook = _pitch_shift_vocals(hook, sr, source_key, target_key)

    # Build stereo mix buffer (float64 for headroom during summing)
    instrumental = _to_stereo(instrumental)
    mix = instrumental.copy().astype(np.float64)

    # Place hook at drop point
    drop_pos = _calculate_drop_position(source_bpm)
    drop_sample = int(drop_pos * sr)
    hook_stereo = _to_stereo(hook).astype(np.float64)
    _overlay(mix, hook_stereo, drop_sample)

    # Layer full vocals across verse/chorus sections, timestamps scaled proportionally
    scale = instrumental_duration / source_duration if source_duration > 0 else 1.0
    for section in structure:
        if section["label"] not in ("verse", "chorus"):
            continue
        src_start = section["start"]
        src_end = section["end"]
        vocal_start_sample = int(src_start * sr)
        vocal_end_sample = min(int(src_end * sr), len(vocals))
        if vocal_start_sample >= len(vocals):
            continue
        segment = vocals[vocal_start_sample:vocal_end_sample]
        target_start_sample = int(src_start * scale * sr)
        segment_stereo = _to_stereo(segment).astype(np.float64)
        _overlay(mix, segment_stereo, target_start_sample)

    # Normalize peak to -3 dBFS
    mix = _normalize_peak(mix, target_db=-3.0)

    # Write 24-bit stereo WAV
    sf.write(str(mix_path), mix, sr, subtype="PCM_24")
    logger.info(f"Written rough mix for {track_id}: {mix_path}")

    return {"rough_mix": str(mix_path)}
