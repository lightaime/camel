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

import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from camel.toolkits.video_analysis_toolkit import VideoAnalysisToolkit


class TestVideoAnalysisToolkit(unittest.TestCase):
    @patch("camel.toolkits.video_analysis_toolkit.OpenAIAudioModels")
    @patch("camel.models.ModelFactory.create")
    @patch("camel.toolkits.VideoAnalysisToolkit._transcribe_audio")
    @patch("camel.toolkits.VideoAnalysisToolkit._extract_audio_from_video")
    @patch("camel.toolkits.VideoDownloaderToolkit")
    def test_ask_question_about_video(
        self,
        mock_video_downloader,
        mock_extract_audio,
        mock_transcribe_audio,
        mock_model_factory,
        mock_audio_models,
    ):
        # Setup mocks
        mock_video_downloader.download_video.return_value = "mock_video.mp4"
        mock_extract_audio.return_value = "mock_audio.mp3"
        mock_transcribe_audio.return_value = "Mocked audio transcription."

        # Mocking ModelFactory.create() to return a fake VL model
        mock_vl_model = MagicMock()
        mock_vl_agent = MagicMock()
        mock_vl_agent.step.return_value.msgs = [
            MagicMock(content="Mocked answer.")
        ]

        mock_model_factory.return_value = mock_vl_model
        mock_vl_agent.return_value = mock_vl_agent

        # Create an instance of the VideoAnalysisToolkit
        toolkit = VideoAnalysisToolkit()

        # Manually assign the mocked agent to the instance
        toolkit.vl_agent = mock_vl_agent

        mock_audio_models.return_value = MagicMock()

        # Mock the _extract_keyframes method to return dummy images
        with patch.object(
            toolkit,
            "_extract_keyframes",
            return_value=[Image.new('RGB', (100, 100))] * 5,
        ):
            result = toolkit.ask_question_about_video(
                "mock_video.mp4", "What is happening in the video?"
            )

        # Assertions
        mock_video_downloader.download_video.assert_not_called()
        mock_extract_audio.assert_called_once_with("mock_video.mp4")
        mock_transcribe_audio.assert_called_once_with("mock_audio.mp3")
        mock_vl_agent.step.assert_called_once()

        self.assertEqual(result, "Mocked answer.")


if __name__ == "__main__":
    unittest.main()
