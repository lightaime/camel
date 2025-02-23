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
export MOONSHOT_API_KEY="xx"
"""

import re
from unittest.mock import MagicMock, patch

import pytest

from camel.configs import MoonshotConfig
from camel.models import MoonshotModel
from camel.types import ModelType

model_types = [
    ModelType.MOONSHOT_V1_8K,
    ModelType.MOONSHOT_V1_32K,
    ModelType.MOONSHOT_V1_128K,
]


user_role_message = {
    "role": "user",
    "content": "How fast can you solve a math problem ?",
}


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    model_types,
)
def test_moonshot_model(model_type: ModelType):
    model = MoonshotModel(model_type)
    assert model.model_type == model_type
    assert model.model_config_dict == MoonshotConfig().as_dict()
    assert isinstance(model.model_type.value_for_tiktoken, str)
    assert isinstance(model.model_type.token_limit, int)


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    model_types,
)
@patch("camel.models.moonshot_model.OpenAI")
def test_moonshot_model_run(mock_openai, model_type: ModelType):
    # Mock the client creation function OpenAI
    mock_openai_client = MagicMock()
    mock_openai.return_value = mock_openai_client
    mock_openai_client.chat.completions.create.return_value = None
    model = MoonshotModel(model_type)
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
        temperature=0.3,
        max_tokens=None,
        stream=False,
        top_p=1.0,
        n=1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stop=None,
    )


@pytest.mark.model_backend
def test_moonshot_model_unexpected_argument():
    model_type = ModelType.MOONSHOT_V1_8K
    model_config_dict = {"model_path": "moonshot_v1"}

    with pytest.raises(
        ValueError,
        match=re.escape(
            (
                "Unexpected argument `model_path` is "
                "input into Moonshot model backend."
            )
        ),
    ):
        _ = MoonshotModel(model_type, model_config_dict)
