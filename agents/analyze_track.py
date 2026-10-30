from __future__ import annotations

import json
import logging
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Key profiles for Krumhansl-Schmuckler key detection
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_bpm(audio_path: Path) -> float:
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    # Clamp to valid range — double/halve if out of bounds
    while bpm > 200:
        bpm /= 2
    while bpm < 60:
        bpm *= 2
    return round(bpm, 1)


def detect_key(audio_path: Path) -> str:
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    best_corr = -2.0
    best_key = "C major"

    for shift in range(12):
        rolled = np.roll(chroma_mean, -shift)
        major_corr = float(np.corrcoef(rolled, _MAJOR_PROFILE)[0, 1])
        minor_corr = float(np.corrcoef(rolled, _MINOR_PROFILE)[0, 1])
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = f"{_NOTE_NAMES[shift]} major"
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = f"{_NOTE_NAMES[shift]} minor"

    return best_key


def detect_structure(audio_path: Path) -> list[dict]:
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # Use spectral clustering via librosa's structural segmentation
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    bound_frames = librosa.segment.agglomerative(mfcc, k=max(4, int(duration / 30)))
    bound_times = librosa.frames_to_time(bound_frames, sr=sr).tolist()

    # Build boundary list: start with 0, end with duration
    boundaries = sorted(set([0.0] + bound_times + [duration]))

    # Label segments based on position
    labels = _label_segments(boundaries)

    segments = []
    for i in range(len(boundaries) - 1):
        segments.append({
            "label": labels[i],
            "start": round(boundaries[i], 2),
            "end": round(boundaries[i + 1], 2),
        })

    return segments


def _label_segments(boundaries: list[float]) -> list[str]:
    n = len(boundaries) - 1
    if n <= 0:
        return []

    labels = []
    total_dur = boundaries[-1]

    for i in range(n):
        start = boundaries[i]
        frac = start / total_dur if total_dur > 0 else 0

        if i == 0:
            labels.append("intro")
        elif i == n - 1:
            labels.append("outro")
        elif frac < 0.25:
            labels.append("verse")
        elif frac < 0.5:
            labels.append("chorus")
        elif frac < 0.75:
            labels.append("bridge")
        else:
            labels.append("chorus")

    return labels


def compute_energy_curve(audio_path: Path) -> list[float]:
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    # RMS energy with ~1 second hop
    hop_length = sr
    rms = librosa.feature.rms(y=y, frame_length=sr, hop_length=hop_length)[0]
    return [round(float(v), 6) for v in rms]


def detect_chords(audio_path: Path) -> list[dict]:
    """Chord detection using librosa chroma features (essentia fallback)."""
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    # Compute chroma with ~2 second windows for stable chord estimates
    hop_length = sr  # 1 second hops
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    chord_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    chords = []

    for frame_idx in range(chroma.shape[1]):
        frame = chroma[:, frame_idx]
        root = int(np.argmax(frame))
        # Simple major/minor detection using third interval
        major_third = frame[(root + 4) % 12]
        minor_third = frame[(root + 3) % 12]
        quality = "maj" if major_third >= minor_third else "min"
        time_sec = round(float(frame_idx * hop_length / sr), 2)
        chords.append({
            "time": time_sec,
            "chord": f"{chord_names[root]}{quality}",
        })

    # Deduplicate consecutive identical chords
    if not chords:
        return chords
    deduped = [chords[0]]
    for c in chords[1:]:
        if c["chord"] != deduped[-1]["chord"]:
            deduped.append(c)
    return deduped


def analyze(track_id: str, project_root: Path) -> dict:
    project_root = Path(project_root)
    analysis_path = project_root / "analysis" / f"{track_id}_analysis.json"

    # Idempotent: skip if analysis already exists
    if analysis_path.exists():
        return json.loads(analysis_path.read_text())

    # Find the source audio (MP3 in raw/)
    audio_path = project_root / "raw" / f"{track_id}.mp3"
    if not audio_path.exists():
        raise FileNotFoundError(f"Source audio not found: {audio_path}")

    logger.info(f"Analyzing {track_id}")

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    result = {
        "bpm": detect_bpm(audio_path),
        "key": detect_key(audio_path),
        "structure": detect_structure(audio_path),
        "energy_curve": compute_energy_curve(audio_path),
        "chords": detect_chords(audio_path),
        "duration": round(duration, 2),
    }

    analysis_path.write_text(json.dumps(result, indent=2) + "\n")
    logger.info(f"Analysis written to {analysis_path}")

    return result
