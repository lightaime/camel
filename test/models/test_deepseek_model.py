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

import os
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from openai import NOT_GIVEN

from camel.configs import DeepSeekConfig
from camel.models import DeepSeekModel, ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.utils import OpenAITokenCounter

"""
please set the below os environment variable:
export DEEPSEEK_API_KEY="xxx"
"""

model_types = [
    ModelType.DEEPSEEK_CHAT,
    ModelType.DEEPSEEK_REASONER,
]

user_role_message = {
    "role": "user",
    "content": "Why is the sky blue ?",
}

deepseek_model_response = SimpleNamespace(
    id="1",
    usage=SimpleNamespace(
        tokens=SimpleNamespace(input_tokens=100, output_tokens=100),
    ),
    model=ModelType.DEEPSEEK_CHAT,
    created=1,
    choices=[
        SimpleNamespace(
            index=1,
            message=SimpleNamespace(
                role="assistant",
                content="Because sunlight is scattered by the atmosphere, and blue waves are dispersed more than other colors.",
                reasoning_content="blue waves are dispersed more than other colors",
            ),
            finish_reason=None,
        )
    ],
)


@pytest.mark.model_backend
@pytest.mark.parametrize("model_type", model_types)
def test_deepseek_model(model_type):
    model_config_dict = DeepSeekConfig().as_dict()
    model = DeepSeekModel(model_type, model_config_dict)
    assert model.model_type == model_type
    assert model.model_config_dict == model_config_dict
    assert isinstance(model.token_counter, OpenAITokenCounter)
    assert isinstance(model.model_type.value_for_tiktoken, str)
    assert isinstance(model.model_type.token_limit, int)


@pytest.mark.model_backend
@pytest.mark.parametrize("model_type", model_types)
def test_deepseek_model_create(model_type: ModelType):
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEEPSEEK,
        model_type=model_type,
        model_config_dict=DeepSeekConfig(temperature=1.3).as_dict(),
    )
    assert model.model_type == model_type


@pytest.mark.model_backend
@pytest.mark.parametrize("model_type", model_types)
@patch("camel.models.deepseek_model.OpenAI")
def test_deepseek_run(mock_openai, model_type: ModelType):
    # Mock the Deepseek create client function
    mock_openai_client = MagicMock()
    mock_openai.return_value = mock_openai_client
    mock_openai_client.chat.completions.create.return_value = (
        deepseek_model_response
    )
    os.environ["GET_REASONING_CONTENT"] = "True"

    model = DeepSeekModel(model_type)
    response = model.run(messages=[user_role_message])  # type: ignore[list-item]
    if model_type in [
        ModelType.DEEPSEEK_REASONER,
    ]:
        mock_openai_client.chat.completions.create.assert_called_once_with(
            messages=[{'role': 'user', 'content': 'Why is the sky blue ?'}],
            model=model_type,
            stream=False,
            stop=NOT_GIVEN,
            max_tokens=NOT_GIVEN,
            response_format=NOT_GIVEN,
            tool_choice=None,
        )
        assert (
            response.choices[0].message.content.strip()  # type: ignore[union-attr]
            == (
                f"<think>\n{deepseek_model_response.choices[0].message.reasoning_content}\n</think>\n"
            )
            + deepseek_model_response.choices[0].message.content
        )
    else:
        mock_openai_client.chat.completions.create.asser_called_once_with(
            messages=[{'role': 'user', 'content': 'Why is the sky blue ?'}],
            model=model_type,
            tools=NOT_GIVEN,
            temperature=1.0,
            top_p=1.0,
            stream=False,
            stop=NOT_GIVEN,
            max_tokens=NOT_GIVEN,
            presence_penalty=0.0,
            response_format=NOT_GIVEN,
            frequency_penalty=0.0,
            tool_choice=None,
            logprobs=False,
            top_logprobs=None,
        )


@pytest.mark.model_backend
def test_deepseek_model_unexpected_argument():
    model_type = ModelType.DEEPSEEK_CHAT
    model_config_dict = {"model_path": "vicuna-7b-v1.5"}

    with pytest.raises(
        ValueError,
        # ruff: noqa: E501
        match=re.escape(
            (
                "Unexpected argument `model_path` is input into DeepSeek model backend."
            )
        ),
    ):
        _ = DeepSeekModel(model_type, model_config_dict)
