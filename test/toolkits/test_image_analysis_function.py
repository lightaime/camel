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

from camel.toolkits import ImageAnalysisToolkit


class TestImageAnalysisToolkit(unittest.TestCase):
    @patch("camel.toolkits.image_analysis_toolkit.OpenAIModel")
    @patch("camel.toolkits.ImageAnalysisToolkit._encode_image")
    def test_ask_question_about_image(
        self, mock_encode_image, mock_openai_model
    ):
        toolkit = ImageAnalysisToolkit()

        # Mock response from the OpenAI model
        mock_openai_instance = MagicMock()
        mock_openai_model.return_value = mock_openai_instance

        # Correct the mock setup for the response structure
        mock_openai_instance.run.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="This is a cat"))]
        )

        # Define test inputs
        question = "What is in this image?"
        image_path = "test_image.jpg"  # This could be a local path or URL

        # Run the function
        response = toolkit.ask_question_about_image(question, image_path)

        # Assertions
        self.assertEqual(response, "This is a cat")


if __name__ == "__main__":
    unittest.main()
