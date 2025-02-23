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
export COHERE_API_KEY="xxx"
export AGENTOPS_API_KEY="xxx"
"""

import os
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from cohere.types.chat_message_v2 import (
    AssistantChatMessageV2,
    SystemChatMessageV2,
    UserChatMessageV2,
)

from camel.configs import CohereConfig
from camel.models import CohereModel
from camel.types import ChatCompletion, ModelType
from camel.utils import OpenAITokenCounter

cohere_model_response = SimpleNamespace(
    id="1",
    usage=SimpleNamespace(
        tokens=SimpleNamespace(input_tokens=100, output_tokens=100)
    ),
    finish_reason="finish reason",
    message=SimpleNamespace(
        content=[SimpleNamespace(text="The capital of Morocco is Rabat")],
        tool_plan="The capital of Morocco is Rabat",
        tool_calls=[
            SimpleNamespace(
                id=1,
                function=SimpleNamespace(
                    name="my_function",
                    arguments=["arg1", "arg2"],
                    type="my_type",
                ),
                type="my_type",
            )
        ],
    ),
)

system_role_message = {
    "role": "system",
    "content": "Answer the user's question with 6 words maximum",
}
user_role_message = {
    "role": "user",
    "content": "What is the capital of Morocco ?",
}

assistant_role_message = {
    "role": "assistant",
    "content": "Morocco is a country in North Africa",
}

unsupported_role_message = {
    "role": "unsupported_role",
    "content": "Hello World",
}

model_types = [
    ModelType.COHERE_COMMAND_R,
    ModelType.COHERE_COMMAND_LIGHT,
    ModelType.COHERE_COMMAND,
    ModelType.COHERE_COMMAND_NIGHTLY,
    ModelType.COHERE_COMMAND_R_PLUS,
]


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    model_types,
)
def test_cohere_model(model_type):
    model_config_dict = CohereConfig().as_dict()
    model = CohereModel(model_type, model_config_dict)
    assert model.model_type == model_type
    assert model.model_config_dict == model_config_dict
    assert isinstance(model.token_counter, OpenAITokenCounter)
    assert isinstance(model.model_type.value_for_tiktoken, str)
    assert isinstance(model.model_type.token_limit, int)


@pytest.mark.model_backend
def test_cohere_model_unexpected_argument():
    model_type = ModelType.COHERE_COMMAND_R
    model_config_dict = {"model_path": "vicuna-7b-v1.5"}

    with pytest.raises(
        ValueError,
        match=re.escape(
            (
                "Unexpected argument `model_path` is "
                "input into Cohere model backend."
            )
        ),
    ):
        _ = CohereModel(model_type, model_config_dict)


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    model_types,
)
@patch("cohere.ClientV2")
def test_cohere_model_run(mock_cohere, model_type):
    # Mock the Cohere create client function
    mock_cohere_client = MagicMock()
    mock_cohere.return_value = mock_cohere_client
    mock_cohere_client.chat.return_value = cohere_model_response
    model_config_dict = CohereConfig().as_dict()
    model_config_dict.update(
        {"tools": [{"function": {"strict": True, "another_function": True}}]}
    )
    if os.getenv("AGENTOPS_API_KEY") is not None:
        mock_record = patch("camel.models.cohere_model.record").start()
    model = CohereModel(model_type, model_config_dict)
    model_inferance = model.run(
        [system_role_message, user_role_message, assistant_role_message]
    )
    mock_cohere_client.chat.assert_called_once_with(
        messages=[
            SystemChatMessageV2(
                role='system',
                content="Answer the user's question with 6 words maximum",
            ),
            UserChatMessageV2(
                role='user', content='What is the capital of Morocco ?'
            ),
            AssistantChatMessageV2(
                role='assistant',
                tool_calls=None,
                tool_plan=None,
                content='Morocco is a country in North Africa',
                citations=None,
            ),
        ],
        model=model_type,
        tools=[{'function': {'another_function': True}}],
        temperature=0.2,
        documents=None,
        max_tokens=None,
        stop_sequences=None,
        seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        k=0,
        p=0.75,
    )
    assert isinstance(model_inferance, ChatCompletion)
    assert (
        model_inferance.choices[0].message.content
        == "The capital of Morocco is Rabat"
    )
    if os.getenv("AGENTOPS_API_KEY") is not None:
        mock_record.assert_called()


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    model_types,
)
@patch("cohere.ClientV2")
def test_cohere_model_run_unsupported_role(mock_cohere, model_type):
    # Mock the Cohere create client function
    mock_cohere_client = MagicMock()
    mock_cohere.return_value = mock_cohere_client
    mock_cohere_client.chat.return_value = cohere_model_response
    model = CohereModel(model_type)
    with pytest.raises(
        ValueError,
        match=re.escape(("Unsupported message role: unsupported_role")),
    ):
        _ = model.run([unsupported_role_message])
