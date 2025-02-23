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
"""
please set the below os environment variable:
export GEMINI_API_KEY="xxx"
"""

import re
from unittest.mock import MagicMock, patch

import pytest
from openai import NOT_GIVEN

from camel.configs import GeminiConfig
from camel.models import GeminiModel
from camel.types import ModelType
from camel.utils import OpenAITokenCounter

model_types = [
    ModelType.GEMINI_2_0_FLASH,
    ModelType.GEMINI_1_5_FLASH,
    ModelType.GEMINI_1_5_PRO,
    ModelType.GEMINI_2_0_FLASH_THINKING,
    ModelType.GEMINI_2_0_FLASH_LITE_PREVIEW,
    ModelType.GEMINI_2_0_PRO_EXP,
]

user_role_message = {
    "role": "user",
    "content": "Is cloud run a good solution for a web app ?",
}

empty_content_message = {"role": "user", "content": ""}


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    model_types,
)
def test_gemini_model(model_type: ModelType):
    model_config_dict = GeminiConfig().as_dict()
    model = GeminiModel(model_type, model_config_dict)
    assert model.model_type == model_type
    assert model.model_config_dict == model_config_dict
    assert isinstance(model.token_counter, OpenAITokenCounter)
    assert isinstance(model.model_type.value_for_tiktoken, str)
    assert isinstance(model.model_type.token_limit, int)


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    model_types,
)
@patch("camel.models.gemini_model.OpenAI")
def test_gemini_run(mock_openai, model_type: ModelType):
    # Mock the Gemini create client function
    mock_openai_client = MagicMock()
    mock_openai.return_value = mock_openai_client
    mock_openai_client.chat.completions.create.return_value = None

    model = GeminiModel(model_type)
    model.run([user_role_message, empty_content_message])  # type: ignore[list-item]
    mock_openai_client.chat.completions.create.assert_called_once_with(
        messages=[
            {
                'role': 'user',
                'content': 'Is cloud run a good solution for a web app ?',
            },
            {'role': 'user', 'content': 'null'},
        ],
        model=model_type,
        tools=NOT_GIVEN,
        temperature=0.2,
        top_p=1.0,
        n=1,
        stream=False,
        stop=NOT_GIVEN,
        max_tokens=NOT_GIVEN,
        response_format=NOT_GIVEN,
        tool_choice=NOT_GIVEN,
    )


@pytest.mark.model_backend
def test_gemini_model_unexpected_argument():
    model_type = ModelType.GEMINI_1_5_FLASH
    model_config_dict = {"model_path": "vicuna-7b-v1.5"}

    with pytest.raises(
        ValueError,
        match=re.escape(
            (
                "Unexpected argument `model_path` is "
                "input into Gemini model backend."
            )
        ),
    ):
        _ = GeminiModel(model_type, model_config_dict)
