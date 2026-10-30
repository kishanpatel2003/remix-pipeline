from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from harness.state_manager import StateManager

logger = logging.getLogger(__name__)

BILLBOARD_URL = "https://www.billboard.com/charts/hot-100/"


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text


def parse_chart(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("li.o-chart-results-list-row-container")
    if not rows:
        return []

    tracks = []
    for rank, row in enumerate(rows, start=1):
        title_el = row.select_one("h3.c-title")
        artist_el = row.select_one("span.c-label.a-no-trucate")

        if not title_el or not artist_el:
            continue

        title = title_el.get_text(strip=True)
        artist = artist_el.get_text(strip=True)
        track_id = f"{rank:02d}_{slugify(artist)}_{slugify(title)}"

        tracks.append({
            "rank": rank,
            "title": title,
            "artist": artist,
            "track_id": track_id,
            "status": "pending",
        })

    return tracks


def scrape(project_root: Path, state_manager: StateManager) -> None:
    project_root = Path(project_root)
    manifest_path = project_root / "data" / "tracks_manifest.json"

    resp = requests.get(BILLBOARD_URL, timeout=30, headers={
        "User-Agent": "Mozilla/5.0 (compatible; HouseRemixPipeline/0.1)",
    })
    resp.raise_for_status()

    tracks = parse_chart(resp.text)

    if not tracks:
        debug_path = project_root / "data" / f"billboard_debug_{datetime.now():%Y%m%d_%H%M%S}.html"
        debug_path.write_text(resp.text)
        raise RuntimeError(f"parse: no tracks found in Billboard HTML. Raw HTML saved to {debug_path}")

    # Load existing manifest for deduplication
    existing_by_id: dict[str, dict] = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for t in existing.get("tracks", []):
            existing_by_id[t["track_id"]] = t

    # Merge: existing tracks keep their status, new tracks get added
    for t in tracks:
        if t["track_id"] not in existing_by_id:
            existing_by_id[t["track_id"]] = t
            state_manager.add_track(t["track_id"])

    merged = sorted(existing_by_id.values(), key=lambda t: t["rank"])

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"tracks": merged}, indent=2) + "\n")
    logger.info(f"Wrote {len(merged)} tracks to manifest ({len(tracks)} scraped, {len(existing_by_id) - len(tracks)} existing)")
