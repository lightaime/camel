# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
from unittest.mock import MagicMock, mock_open, patch

import pytest

from camel.toolkits import VideoDownloaderToolkit


@pytest.fixture
def mock_downloader():
    with (
        patch("builtins.open", mock_open(read_data=b"test")),
        patch("yt_dlp.YoutubeDL") as mock_youtube_dl,
        patch('tempfile.mkdtemp') as mock_mkdtemp,
        patch('pathlib.Path.mkdir'),
        patch('ffmpeg.input') as mock_ffmpeg_input,
        patch('ffmpeg.probe') as mock_ffmpeg_probe,
        patch('PIL.Image.open'),
    ):
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.prepare_filename.return_value = "test.mp4"
        mock_youtube_dl.return_value.__enter__.return_value = mock_ydl_instance

        mock_mkdtemp.return_value = "C:\\temp\\abcdefg"

        mock_ffmpeg = MagicMock()
        mock_ffmpeg_input.return_value = mock_ffmpeg
        mock_ffmpeg.filter.return_value = mock_ffmpeg
        mock_ffmpeg.output.return_value = mock_ffmpeg
        mock_ffmpeg.run.return_value = (b"test", b"test")

        mock_ffmpeg_probe.return_value = {'format': {'duration': 10}}

        yield VideoDownloaderToolkit()


def test_video_bytes_download(mock_downloader):
    video_bytes = mock_downloader.get_video_bytes(
        video_path="https://test_video.mp4",
    )
    assert len(video_bytes) > 0
    assert video_bytes == b"test"


def test_video_screenshots_download(mock_downloader):
    screenshots = mock_downloader.get_video_screenshots(
        "https://test_video.mp4",
        2,
    )
    assert len(screenshots) == 2
