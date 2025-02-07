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

from camel.toolkits import AudioAnalysisToolkit


class TestAudioAnalysisToolkit(unittest.TestCase):
    @patch("camel.toolkits.audio_analysis_toolkit.openai.OpenAI")
    @patch("camel.toolkits.audio_analysis_toolkit.requests.get")
    @patch("camel.toolkits.audio_analysis_toolkit.base64.b64encode")
    def test_ask_question_about_audio_with(
        self, mock_b64encode, mock_requests_get, mock_openai
    ):
        mock_audio_data = b"fake_audio_data"

        toolkit = AudioAnalysisToolkit()

        with (
            patch("requests.get") as mock_requests_get,
            patch.object(
                toolkit.audio_client.chat.completions, "create"
            ) as mock_create,
        ):
            mock_requests_get.return_value.status_code = 200
            mock_requests_get.return_value.content = mock_audio_data

            mock_create.return_value.choices = [
                MagicMock(message=MagicMock(content="Mocked response"))
            ]

            response = toolkit.ask_question_about_audio(
                "http://example.com/audio.wav", "What is this audio about?"
            )

            mock_requests_get.assert_called_once_with(
                "http://example.com/audio.wav"
            )
            mock_create.assert_called_once()
            self.assertEqual(response, "Mocked response")


if __name__ == "__main__":
    unittest.main()
