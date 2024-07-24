# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import re

import pytest

from camel.configs import LiteLLMConfig, OpenSourceConfig
from camel.models import LiteLLMModel
from camel.types import ModelType
from camel.utils import LiteLLMTokenCounter


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    [
        ModelType.GPT_3_5_TURBO,
        ModelType.GPT_4,
        ModelType.GPT_4_32K,
        ModelType.GPT_4_TURBO,
        ModelType.GPT_4O,
    ],
)
def test_litellm_model(model_type: ModelType):
    model_config_dict = LiteLLMConfig().model_dump()
    model = LiteLLMModel(model_type.value, model_config_dict)
    assert model.model_type == model_type.value
    assert model.model_config_dict == model_config_dict
    assert isinstance(model.token_counter, LiteLLMTokenCounter)
    assert isinstance(model_type.value_for_tiktoken, str)
    assert isinstance(model_type.token_limit, int)


@pytest.mark.model_backend
def test_litellm_model_unexpected_argument():
    model_type = ModelType.GPT_4.value
    model_config = OpenSourceConfig(
        model_path="vicuna-7b-v1.5",
        server_url="http://localhost:8000/v1",
    )
    model_config_dict = model_config.__dict__

    with pytest.raises(
        ValueError,
        match=re.escape(
            (
                "Unexpected argument `model_path` is "
                "input into LiteLLM model backend."
            )
        ),
    ):
        _ = LiteLLMModel(model_type, model_config_dict)
