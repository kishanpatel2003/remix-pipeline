from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

from agents.download_source import download, search_and_download, log_failed_download


@pytest.fixture
def project_root(tmp_path):
    for d in ("data", "raw", "analysis", "stems", "generated", "mixes", "masters", "output"):
        (tmp_path / d).mkdir()
    return tmp_path


@pytest.fixture
def manifest(project_root):
    data = {
        "tracks": [
            {
                "rank": 1,
                "title": "Die With A Smile",
                "artist": "Lady Gaga & Bruno Mars",
                "track_id": "01_lady_gaga_bruno_mars_die_with_a_smile",
                "status": "pending",
            },
            {
                "rank": 2,
                "title": "APT.",
                "artist": "Rose & Bruno Mars",
                "track_id": "02_rose_bruno_mars_apt",
                "status": "pending",
            },
        ]
    }
    (project_root / "data" / "tracks_manifest.json").write_text(json.dumps(data))
    return data


def _fake_mp3(path: Path, size_mb: float = 4.0):
    path.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * int(size_mb * 1024 * 1024))


class TestSearchAndDownload:
    @patch("agents.download_source.subprocess.run")
    def test_calls_ytdlp_with_search_query(self, mock_run, project_root):
        mock_run.return_value = MagicMock(returncode=0)
        output_path = project_root / "raw" / "01_test_song.mp3"
        _fake_mp3(output_path)

        search_and_download("Test Artist", "Test Song", output_path)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "yt-dlp" in cmd[0]
        assert any("ytsearch1:" in arg for arg in cmd)
        assert any("Test Artist" in arg and "Test Song" in arg for arg in cmd)

    @patch("agents.download_source.subprocess.run")
    def test_raises_on_ytdlp_failure(self, mock_run, project_root):
        mock_run.return_value = MagicMock(returncode=1, stderr="not found")
        output_path = project_root / "raw" / "01_test_song.mp3"

        with pytest.raises(RuntimeError, match="yt-dlp"):
            search_and_download("Artist", "Song", output_path)


class TestDownload:
    @patch("agents.download_source.search_and_download")
    @patch("agents.download_source.get_duration")
    def test_downloads_and_returns_duration(self, mock_dur, mock_dl, project_root, manifest):
        track_id = "01_lady_gaga_bruno_mars_die_with_a_smile"
        output_path = project_root / "raw" / f"{track_id}.mp3"

        def fake_dl(artist, title, path):
            _fake_mp3(path)

        mock_dl.side_effect = fake_dl
        mock_dur.return_value = 210.0

        result = download(track_id, project_root)

        assert result["duration_seconds"] == 210.0
        mock_dl.assert_called_once_with(
            "Lady Gaga & Bruno Mars", "Die With A Smile", output_path
        )

    @patch("agents.download_source.search_and_download")
    @patch("agents.download_source.get_duration")
    def test_skips_if_mp3_already_exists(self, mock_dur, mock_dl, project_root, manifest):
        track_id = "01_lady_gaga_bruno_mars_die_with_a_smile"
        output_path = project_root / "raw" / f"{track_id}.mp3"
        _fake_mp3(output_path)
        mock_dur.return_value = 200.0

        result = download(track_id, project_root)

        mock_dl.assert_not_called()
        assert result["duration_seconds"] == 200.0

    @patch("agents.download_source.search_and_download")
    def test_raises_if_track_not_in_manifest(self, mock_dl, project_root, manifest):
        with pytest.raises(KeyError, match="not_in_manifest"):
            download("not_in_manifest", project_root)

    @patch("agents.download_source.search_and_download")
    def test_raises_if_no_manifest(self, mock_dl, project_root):
        with pytest.raises(FileNotFoundError):
            download("01_test", project_root)


class TestLogFailedDownload:
    def test_logs_failure_to_json(self, project_root):
        log_failed_download(project_root, "01_test_track", "HTTP 403")

        log_path = project_root / "data" / "failed_downloads.json"
        assert log_path.exists()
        data = json.loads(log_path.read_text())
        assert len(data) == 1
        assert data[0]["track_id"] == "01_test_track"
        assert data[0]["reason"] == "HTTP 403"

    def test_appends_to_existing_log(self, project_root):
        log_failed_download(project_root, "01_first", "timeout")
        log_failed_download(project_root, "02_second", "not found")

        data = json.loads((project_root / "data" / "failed_downloads.json").read_text())
        assert len(data) == 2
        assert data[0]["track_id"] == "01_first"
        assert data[1]["track_id"] == "02_second"
