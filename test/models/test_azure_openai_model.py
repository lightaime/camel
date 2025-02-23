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
please set the below os environment:
export AZURE_OPENAI_BASE_URL=""

# if `AZURE_API_VERSION` is not set, `OPENAI_API_VERSION` will be used as api version
export AZURE_API_VERSION=""
export AZURE_OPENAI_API_KEY=""
export AZURE_DEPLOYMENT_NAME=""
"""

import re
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from camel.configs import ChatGPTConfig
from camel.models import AzureOpenAIModel, ModelFactory
from camel.types import ChatCompletion, ModelPlatformType, ModelType
from camel.utils import OpenAITokenCounter

model_backend_rsp = ChatCompletion(
    id="mock_response_id",
    choices=[
        Choice(
            finish_reason="stop",
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content="This is a mock response content.",
                role="assistant",
                function_call=None,
                tool_calls=None,
            ),
        )
    ],
    created=123456789,
    model="gpt-4o-2024-05-13",
    object="chat.completion",
    usage=CompletionUsage(
        completion_tokens=32,
        prompt_tokens=15,
        total_tokens=47,
    ),
)

model_types = [
    ModelType.GPT_3_5_TURBO,
    ModelType.GPT_4,
    ModelType.GPT_4_TURBO,
    ModelType.GPT_4O,
    ModelType.GPT_4O_MINI,
]


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    model_types,
)
def test_openai_model(model_type):
    model_config_dict = ChatGPTConfig().as_dict()
    model = AzureOpenAIModel(model_type, model_config_dict)
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
def test_openai_model_create(model_type: ModelType):
    model = ModelFactory.create(
        model_platform=ModelPlatformType.AZURE,
        model_type=model_type,
        model_config_dict=ChatGPTConfig(temperature=0.8, n=3).as_dict(),
    )
    assert model.model_type == model_type


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    model_types,
)
@patch("camel.models.azure_openai_model.AzureOpenAI")
def test_openai_model_run(
    mock_azure_openai, model_type: ModelType, max_steps=5
):
    # Mock the client creation function AzureOpenAI
    mock_client = MagicMock()
    mock_azure_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = model_backend_rsp

    model = ModelFactory.create(
        model_platform=ModelPlatformType.AZURE,
        model_type=model_type,
        model_config_dict=ChatGPTConfig(temperature=0.8, n=3).as_dict(),
    )
    for _ in range(max_steps):
        model_inference = model.run(messages=[])
        if isinstance(model_inference, ChatCompletion):
            assert (
                model_inference.choices[0].message.content
                == model_backend_rsp.choices[0].message.content
            )


@pytest.mark.model_backend
def test_openai_model_unexpected_argument():
    model_type = ModelType.GPT_4
    model_config_dict = {"model_path": "vicuna-7b-v1.5"}

    with pytest.raises(
        ValueError,
        # ruff: noqa: E501
        match=re.escape(
            (
                "Unexpected argument `model_path` is input into Azure OpenAI model backend."
            )
        ),
    ):
        _ = AzureOpenAIModel(model_type, model_config_dict)
