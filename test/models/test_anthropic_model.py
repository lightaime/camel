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
from anthropic import NOT_GIVEN

from camel.configs import AnthropicConfig
from camel.models import AnthropicModel
from camel.types import ChatCompletion, ModelType
from camel.utils import AnthropicTokenCounter

anthropic_model_response = SimpleNamespace(
    id="msg_123456",
    type="message",
    role="assistant",
    model="claude-2.1",
    content=[SimpleNamespace(text="The capital city of Morocco is Rabat")],
    stop_reason="stop reason",
)

system_role_message = {
    "role": "system",
    "content": "Answer the user's question with 6 words maximum",
}

user_role_message = {
    "role": "user",
    "content": "What is the capital city of Morocco ?",
}

model_types = [
    ModelType.CLAUDE_INSTANT_1_2,
    ModelType.CLAUDE_2_0,
    ModelType.CLAUDE_2_1,
    ModelType.CLAUDE_3_OPUS,
    ModelType.CLAUDE_3_SONNET,
    ModelType.CLAUDE_3_HAIKU,
    ModelType.CLAUDE_3_5_SONNET,
    ModelType.CLAUDE_3_5_HAIKU,
]

# Test configurations
test_api_key = "test-api-key"
test_url = "https://test-url.com"


@pytest.fixture(autouse=True)
def mock_env_vars():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": test_api_key}):
        yield


@pytest.fixture
def mock_anthropic_client():
    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = anthropic_model_response
        yield mock_client


@pytest.fixture
def mock_async_anthropic_client():
    with patch("anthropic.AsyncAnthropic") as mock_async_anthropic:
        mock_client = MagicMock()
        mock_async_anthropic.return_value = mock_client
        # Create an async mock for messages.create
        mock_messages = MagicMock()

        async def async_create(*args, **kwargs):
            mock_messages.create(*args, **kwargs)
            return anthropic_model_response

        mock_client.messages.create = async_create
        mock_client.messages.create.mock = mock_messages.create
        yield mock_client


@pytest.mark.model_backend
@pytest.mark.parametrize("model_type", model_types)
def test_anthropic_model_initialization(model_type: ModelType):
    r"""Test basic model initialization and properties."""
    model = AnthropicModel(model_type)
    assert model.model_type == model_type
    assert model.model_config_dict == AnthropicConfig().as_dict()
    assert isinstance(model.token_counter, AnthropicTokenCounter)
    assert isinstance(model.model_type.value_for_tiktoken, str)
    assert isinstance(model.model_type.token_limit, int)


@pytest.mark.model_backend
def test_anthropic_model_custom_config():
    r"""Test model initialization with custom configuration."""
    custom_config = {"max_tokens": 100, "temperature": 0.5}
    model = AnthropicModel(
        ModelType.CLAUDE_2_1,
        model_config_dict=custom_config,
        api_key=test_api_key,
        url=test_url,
    )
    assert model.model_config_dict == custom_config
    assert model._api_key == test_api_key
    assert model._url == test_url


@pytest.mark.model_backend
def test_anthropic_model_unexpected_argument():
    r"""Test model initialization with invalid configuration."""
    model_type = ModelType.CLAUDE_2_0
    model_config_dict = {"model_path": "vicuna-7b-v1.5"}

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unexpected argument `model_path` is input into Anthropic "
            "model backend"
        ),
    ):
        _ = AnthropicModel(model_type, model_config_dict)


@pytest.mark.model_backend
@pytest.mark.parametrize("model_type", model_types)
def test_anthropic_run(mock_anthropic_client, model_type: ModelType):
    r"""Test synchronous message generation with system message."""
    model = AnthropicModel(model_type)
    model_inference = model.run(
        messages=[system_role_message, user_role_message]  # type: ignore[list-item]
    )

    mock_anthropic_client.messages.create.assert_called_once_with(
        model=model_type,
        system=system_role_message["content"],
        messages=[user_role_message],
        max_tokens=8192,
        temperature=1,
        top_p=0.7,
        top_k=5,
        stream=False,
    )

    assert isinstance(model_inference, ChatCompletion)
    assert model_inference.choices[0].message.role == "assistant"
    assert (
        model_inference.choices[0].message.content
        == anthropic_model_response.content[0].text
    )


@pytest.mark.model_backend
@pytest.mark.parametrize("model_type", model_types)
def test_anthropic_run_without_sys_msg(
    mock_anthropic_client, model_type: ModelType
):
    r"""Test synchronous message generation without system message."""
    model = AnthropicModel(model_type)
    model_inference = model.run(messages=[user_role_message])  # type: ignore[list-item]

    mock_anthropic_client.messages.create.assert_called_once_with(
        model=model_type,
        system=NOT_GIVEN,
        messages=[user_role_message],
        max_tokens=8192,
        temperature=1,
        top_p=0.7,
        top_k=5,
        stream=False,
    )

    assert isinstance(model_inference, ChatCompletion)
    assert model_inference.choices[0].message.role == "assistant"
    assert (
        model_inference.choices[0].message.content
        == anthropic_model_response.content[0].text
    )


@pytest.mark.model_backend
@pytest.mark.asyncio
async def test_anthropic_arun(mock_async_anthropic_client):
    r"""Test asynchronous message generation."""
    model = AnthropicModel(ModelType.CLAUDE_2_1)

    model_inference = await model.arun(
        messages=[system_role_message, user_role_message]
    )

    mock_async_anthropic_client.messages.create.mock.assert_called_once_with(
        model=ModelType.CLAUDE_2_1,
        system=system_role_message["content"],
        messages=[user_role_message],
        max_tokens=8192,
        temperature=1,
        top_p=0.7,
        top_k=5,
        stream=False,
    )
    assert isinstance(model_inference, ChatCompletion)
    assert model_inference.choices[0].message.role == "assistant"
    assert (
        model_inference.choices[0].message.content
        == anthropic_model_response.content[0].text
    )


@pytest.mark.model_backend
def test_anthropic_error_handling(mock_anthropic_client):
    r"""Test error handling during message generation."""
    mock_anthropic_client.messages.create.side_effect = Exception("API Error")
    model = AnthropicModel(ModelType.CLAUDE_2_1)

    with pytest.raises(Exception, match="API Error"):
        model.run(messages=[user_role_message])


@pytest.mark.model_backend
def test_invalid_message_format():
    r"""Test handling of invalid message format."""
    model = AnthropicModel(ModelType.CLAUDE_2_1)
    invalid_message = {"invalid_key": "content"}
    with pytest.raises(KeyError):
        model.run(messages=[invalid_message])
