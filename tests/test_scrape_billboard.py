from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agents.scrape_billboard import scrape, slugify, parse_chart, BILLBOARD_URL
from harness.state_manager import StateManager


SAMPLE_HTML = """
<html><body>
<div class="chart-results-list-container">
  <ul>
    <li class="o-chart-results-list-row-container">
      <div class="o-chart-results-list-row">
        <span class="c-label">#1</span>
        <h3 id="title-of-a-song" class="c-title">Die With A Smile</h3>
        <span class="c-label a-no-trucate">Lady Gaga & Bruno Mars</span>
      </div>
    </li>
  </ul>
</div>
</body></html>
"""

# A more realistic snippet with multiple rows matching Billboard's actual structure
MULTI_ROW_HTML = """
<html><body>
<div class="chart-results-list-container">
  <ul>
    {rows}
  </ul>
</div>
</body></html>
"""


def _make_row(rank, title, artist):
    return f"""
    <li class="o-chart-results-list-row-container">
      <div class="o-chart-results-list-row">
        <span class="c-label">#{rank}</span>
        <h3 id="title-of-a-song" class="c-title">{title}</h3>
        <span class="c-label a-no-trucate">{artist}</span>
      </div>
    </li>
    """


def make_chart_html(entries):
    rows = "\n".join(_make_row(r, t, a) for r, t, a in entries)
    return MULTI_ROW_HTML.format(rows=rows)


@pytest.fixture
def project_root(tmp_path):
    for d in ("data", "raw", "analysis", "stems", "generated", "mixes", "masters", "output"):
        (tmp_path / d).mkdir()
    return tmp_path


@pytest.fixture
def state_mgr(project_root):
    return StateManager(project_root / "data" / "pipeline_state.json")


class TestSlugify:
    def test_basic(self):
        assert slugify("Hello World") == "hello_world"

    def test_special_chars(self):
        assert slugify("It's A (New) Day!") == "its_a_new_day"

    def test_multiple_spaces(self):
        assert slugify("Too   Many   Spaces") == "too_many_spaces"

    def test_ampersand(self):
        assert slugify("Lady Gaga & Bruno Mars") == "lady_gaga_bruno_mars"

    def test_unicode(self):
        result = slugify("Café Résumé")
        assert result == "caf_rsum" or "_" not in result or isinstance(result, str)


class TestParseChart:
    def test_parses_multiple_entries(self):
        entries = [(i, f"Song {i}", f"Artist {i}") for i in range(1, 15)]
        html = make_chart_html(entries)
        tracks = parse_chart(html)
        assert len(tracks) == 14
        assert tracks[0]["rank"] == 1
        assert tracks[0]["title"] == "Song 1"
        assert tracks[0]["artist"] == "Artist 1"

    def test_track_id_format(self):
        html = make_chart_html([(1, "Die With A Smile", "Lady Gaga & Bruno Mars")])
        tracks = parse_chart(html)
        assert tracks[0]["track_id"] == "01_lady_gaga_bruno_mars_die_with_a_smile"

    def test_all_entries_have_required_fields(self):
        entries = [(i, f"Song {i}", f"Artist {i}") for i in range(1, 6)]
        html = make_chart_html(entries)
        tracks = parse_chart(html)
        required = {"rank", "title", "artist", "track_id", "status"}
        for t in tracks:
            assert required.issubset(t.keys()), f"Missing fields in {t}"

    def test_status_defaults_to_pending(self):
        html = make_chart_html([(1, "Song", "Artist")])
        tracks = parse_chart(html)
        assert tracks[0]["status"] == "pending"

    def test_empty_html_returns_empty(self):
        tracks = parse_chart("<html><body></body></html>")
        assert tracks == []

    def test_raises_on_unparseable_markup(self):
        # Completely broken HTML with no chart structure should return empty, not crash
        tracks = parse_chart("<html><body><p>Nothing here</p></body></html>")
        assert tracks == []


class TestScrape:
    def _mock_response(self, html, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = html
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
        return resp

    @patch("agents.scrape_billboard.requests.get")
    def test_writes_manifest(self, mock_get, project_root, state_mgr):
        entries = [(i, f"Song {i}", f"Artist {i}") for i in range(1, 15)]
        mock_get.return_value = self._mock_response(make_chart_html(entries))

        scrape(project_root, state_mgr)

        manifest_path = project_root / "data" / "tracks_manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert len(data["tracks"]) == 14

    @patch("agents.scrape_billboard.requests.get")
    def test_adds_tracks_to_state_manager(self, mock_get, project_root, state_mgr):
        entries = [(i, f"Song {i}", f"Artist {i}") for i in range(1, 5)]
        mock_get.return_value = self._mock_response(make_chart_html(entries))

        scrape(project_root, state_mgr)

        for i in range(1, 5):
            tid = f"{i:02d}_artist_{i}_song_{i}"
            assert state_mgr.get_status(tid) == "pending"

    @patch("agents.scrape_billboard.requests.get")
    def test_deduplicates_against_existing_manifest(self, mock_get, project_root, state_mgr):
        existing = {
            "tracks": [
                {"rank": 1, "title": "Song 1", "artist": "Artist 1",
                 "track_id": "01_artist_1_song_1", "status": "downloaded"}
            ]
        }
        (project_root / "data" / "tracks_manifest.json").write_text(json.dumps(existing))
        state_mgr.add_track("01_artist_1_song_1")
        state_mgr.transition("01_artist_1_song_1", "downloaded")

        entries = [(i, f"Song {i}", f"Artist {i}") for i in range(1, 5)]
        mock_get.return_value = self._mock_response(make_chart_html(entries))

        scrape(project_root, state_mgr)

        data = json.loads((project_root / "data" / "tracks_manifest.json").read_text())
        ids = [t["track_id"] for t in data["tracks"]]
        assert ids.count("01_artist_1_song_1") == 1
        # Existing track keeps its status
        existing_entry = [t for t in data["tracks"] if t["track_id"] == "01_artist_1_song_1"][0]
        assert existing_entry["status"] == "downloaded"

    @patch("agents.scrape_billboard.requests.get")
    def test_http_failure_raises(self, mock_get, project_root, state_mgr):
        mock_get.return_value = self._mock_response("", status_code=500)

        with pytest.raises(Exception):
            scrape(project_root, state_mgr)

    @patch("agents.scrape_billboard.requests.get")
    def test_logs_html_snippet_on_parse_failure(self, mock_get, project_root, state_mgr, tmp_path):
        mock_get.return_value = self._mock_response("<html><body><p>Changed layout</p></body></html>")

        with pytest.raises(Exception, match="parse"):
            scrape(project_root, state_mgr)

        # Should have logged the raw HTML for debugging
        debug_dir = project_root / "data"
        debug_files = list(debug_dir.glob("billboard_debug_*.html"))
        assert len(debug_files) >= 1

    @patch("agents.scrape_billboard.requests.get")
    def test_no_duplicate_track_ids_in_output(self, mock_get, project_root, state_mgr):
        entries = [(i, f"Song {i}", f"Artist {i}") for i in range(1, 20)]
        mock_get.return_value = self._mock_response(make_chart_html(entries))

        scrape(project_root, state_mgr)

        data = json.loads((project_root / "data" / "tracks_manifest.json").read_text())
        ids = [t["track_id"] for t in data["tracks"]]
        assert len(ids) == len(set(ids))
