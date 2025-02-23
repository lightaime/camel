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

from camel.configs import InternLMConfig
from camel.models import InternLMModel
from camel.types import ModelType

"""
please set the below os environment:
export INTERNLM_API_KEY="xx"
"""

user_role_message = {
    "role": "user",
    "content": "How fast can you solve a math problem ?",
}

model_types = [
    ModelType.INTERNLM3_8B_INSTRUCT,
    ModelType.INTERNLM3_LATEST,
    ModelType.INTERNLM2_5_LATEST,
    ModelType.INTERNLM2_PRO_CHAT,
]


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    model_types,
)
def test_internlm_model(model_type: ModelType):
    model = InternLMModel(model_type)
    assert model.model_type == model_type
    assert model.model_config_dict == InternLMConfig().as_dict()
    assert isinstance(model.model_type.value_for_tiktoken, str)
    assert isinstance(model.model_type.token_limit, int)


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    model_types,
)
@patch("camel.models.internlm_model.OpenAI")
def test_internlm_model_run(mock_openai, model_type: ModelType):
    # Mock the client creation function OpenAI
    mock_openai_client = MagicMock()
    mock_openai.return_value = mock_openai_client
    mock_openai_client.chat.completions.create.return_value = None
    model = InternLMModel(model_type)
    model.run([user_role_message])  # type: ignore[list-item]
    mock_openai_client.chat.completions.create.assert_called_once_with(
        messages=[
            {
                'role': 'user',
                'content': 'How fast can you solve a math problem ?',
            }
        ],
        model=model_type,
        tools=None,
        stream=False,
        temperature=0.8,
        top_p=0.9,
        max_tokens=None,
        tool_choice=None,
    )


@pytest.mark.model_backend
def test_internlm_model_unexpected_argument():
    model_type = ModelType.INTERNLM3_LATEST
    model_config_dict = {"model_path": "internlm-max"}

    with pytest.raises(
        ValueError,
        match=re.escape(
            (
                "Unexpected argument `model_path` is "
                "input into InternLM model backend."
            )
        ),
    ):
        _ = InternLMModel(model_type, model_config_dict)
