from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def get_duration(mp3_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(mp3_path),
        ],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def search_and_download(artist: str, title: str, output_path: Path) -> None:
    query = f"ytsearch1:{artist} {title}"
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", str(output_path.with_suffix(".%(ext)s")),
        query,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed (exit {result.returncode}): {result.stderr}")


def log_failed_download(project_root: Path, track_id: str, reason: str) -> None:
    log_path = Path(project_root) / "data" / "failed_downloads.json"
    entries = []
    if log_path.exists():
        entries = json.loads(log_path.read_text())
    entries.append({
        "track_id": track_id,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
    })
    log_path.write_text(json.dumps(entries, indent=2) + "\n")


def download(track_id: str, project_root: Path) -> dict:
    project_root = Path(project_root)
    manifest_path = project_root / "data" / "tracks_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    tracks_by_id = {t["track_id"]: t for t in manifest.get("tracks", [])}

    if track_id not in tracks_by_id:
        raise KeyError(track_id)

    track = tracks_by_id[track_id]
    output_path = project_root / "raw" / f"{track_id}.mp3"

    if not output_path.exists():
        search_and_download(track["artist"], track["title"], output_path)
        logger.info(f"Downloaded {track_id} to {output_path}")

    duration = get_duration(output_path)
    return {"duration_seconds": duration}
