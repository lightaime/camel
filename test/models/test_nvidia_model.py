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

import re
from unittest.mock import MagicMock, patch

import pytest
from openai import NOT_GIVEN

from camel.configs import NvidiaConfig
from camel.models import NvidiaModel
from camel.types import ModelType

"""
please set the below os environment:
export NVIDIA_API_KEY="xx"
"""

user_role_message = {
    "role": "user",
    "content": "How fast can you solve a math problem ?",
}

model_types = [
    ModelType.NVIDIA_NEMOTRON_340B_INSTRUCT,
    ModelType.NVIDIA_NEMOTRON_340B_REWARD,
    ModelType.NVIDIA_YI_LARGE,
    ModelType.NVIDIA_MISTRAL_LARGE,
    ModelType.NVIDIA_LLAMA3_70B,
    ModelType.NVIDIA_MIXTRAL_8X7B,
    ModelType.NVIDIA_LLAMA3_1_8B_INSTRUCT,
    ModelType.NVIDIA_LLAMA3_1_70B_INSTRUCT,
    ModelType.NVIDIA_LLAMA3_1_405B_INSTRUCT,
    ModelType.NVIDIA_LLAMA3_2_1B_INSTRUCT,
    ModelType.NVIDIA_LLAMA3_2_3B_INSTRUCT,
]


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    model_types,
)
def test_nvidia_model(model_type: ModelType):
    model = NvidiaModel(model_type)
    assert model.model_type == model_type
    assert model.model_config_dict == NvidiaConfig().as_dict()
    assert isinstance(model.model_type.value_for_tiktoken, str)
    assert isinstance(model.model_type.token_limit, int)


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    model_types,
)
@patch("camel.models.nvidia_model.OpenAI")
def test_nvidia_model_run(mock_openai, model_type: ModelType):
    # Mock the client creation function OpenAI
    mock_openai_client = MagicMock()
    mock_openai.return_value = mock_openai_client
    mock_openai_client.chat.completions.create.return_value = None
    config = NvidiaConfig().as_dict()
    config["tool_choice"] = "google"
    config["tools"] = None
    model = NvidiaModel(model_type, config)
    model.run([user_role_message])  # type: ignore[list-item]
    mock_openai_client.chat.completions.create.assert_called_once_with(
        messages=[
            {
                'role': 'user',
                'content': 'How fast can you solve a math problem ?',
            }
        ],
        model=model_type,
        stream=False,
        temperature=0.7,
        top_p=0.95,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        max_tokens=NOT_GIVEN,
        seed=None,
        stop=None,
    )


@pytest.mark.model_backend
def test_nvidia_model_unexpected_argument():
    model_type = ModelType.NVIDIA_LLAMA3_70B
    model_config_dict = {"model_path": "nvidia-llama3"}

    with pytest.raises(
        ValueError,
        match=re.escape(
            (
                "Unexpected argument `model_path` is "
                "input into NVIDIA model backend."
            )
        ),
    ):
        _ = NvidiaModel(model_type, model_config_dict)
