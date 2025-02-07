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
import base64
import logging
from typing import List
from urllib.parse import urlparse

from camel.models import OpenAIModel
from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.types import ModelType

logger = logging.getLogger(__name__)


class ImageAnalysisToolkit(BaseToolkit):
    r"""A class representing a toolkit for image comprehension operations.

    This class provides methods for understanding images, such as identifying
    objects, text in images.
    """

    def _encode_image(self, image_path: str):
        r"""Encode an image by its image path.

        Arg:
            image_path (str): The path to the image file."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def ask_question_about_image(self, question: str, image_path: str) -> str:
        r"""Ask a question about the image based on the image path.

        Args:
            question (str): The question to ask about the image.
            image_path (str): The path to the image file.

        Returns:
            str: The answer to the question based on the image.
        """
        logger.debug(
            f"Calling ask_image_by_path with question: `{question}` and \
            image_path: `{image_path}`"
        )
        parsed_url = urlparse(image_path)
        is_url = all([parsed_url.scheme, parsed_url.netloc])

        _image_url = image_path

        if not is_url:
            _image_url = (
                f"data:image/jpeg;base64,{self._encode_image(image_path)}"
            )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for \
                image relevant tasks.",
            },
            {
                "role": "user",
                "content": [
                    {'type': 'text', 'text': question},
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': _image_url,
                        },
                    },
                ],
            },
        ]

        LLM = OpenAIModel(model_type=ModelType.DEFAULT)
        resp = LLM.run(messages)  # type: ignore[arg-type]

        return str(resp.choices[0].message.content)  # type: ignore[union-attr]

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the functions
        in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects representing the
                functions in the toolkit.
        """
        return [
            FunctionTool(self.ask_question_about_image),
        ]
