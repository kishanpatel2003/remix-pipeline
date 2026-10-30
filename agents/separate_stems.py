from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

STEM_NAMES = ("vocals", "drums", "bass", "other")


def normalize_to_peak_db(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Normalize audio so its peak reaches target_db (dBFS)."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    target_linear = 10 ** (target_db / 20)
    return audio * (target_linear / peak)


def _run_demucs(audio_path: Path, output_dir: Path) -> None:
    """Run Demucs htdemucs model via subprocess."""
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", "htdemucs",
        "-o", str(output_dir),
        str(audio_path),
    ]
    logger.info(f"Running Demucs: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed (exit {result.returncode}): {result.stderr}")


def separate(track_id: str, project_root: Path) -> dict[str, str]:
    """Separate a track into 4 stems using Demucs htdemucs.

    Returns a dict mapping stem name to output WAV path.
    """
    project_root = Path(project_root)
    stems_dir = project_root / "stems" / track_id

    # Idempotent: skip if all stems already exist
    if stems_dir.exists() and all((stems_dir / f"{s}.wav").exists() for s in STEM_NAMES):
        logger.info(f"Stems already exist for {track_id}, skipping")
        return {s: str(stems_dir / f"{s}.wav") for s in STEM_NAMES}

    audio_path = project_root / "raw" / f"{track_id}.mp3"
    if not audio_path.exists():
        raise FileNotFoundError(f"Source audio not found: {audio_path}")

    # Run Demucs — outputs to {output_dir}/htdemucs/{track_id}/
    separated_dir = project_root / "separated"
    _run_demucs(audio_path, separated_dir)

    demucs_out = separated_dir / "htdemucs" / track_id
    if not demucs_out.exists():
        raise RuntimeError(f"Demucs output directory not found: {demucs_out}")

    # Move and normalize stems
    stems_dir.mkdir(parents=True, exist_ok=True)

    for stem in STEM_NAMES:
        src = demucs_out / f"{stem}.wav"
        if not src.exists():
            raise RuntimeError(f"Expected Demucs output missing: {src}")

        data, sr = sf.read(str(src))
        normalized = normalize_to_peak_db(data, target_db=-1.0)
        dst = stems_dir / f"{stem}.wav"
        sf.write(str(dst), normalized, sr)

    # Clean up Demucs intermediate files
    shutil.rmtree(separated_dir, ignore_errors=True)

    logger.info(f"Stems separated and normalized for {track_id}")
    return {s: str(stems_dir / f"{s}.wav") for s in STEM_NAMES}
