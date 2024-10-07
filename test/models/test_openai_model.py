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

from camel.configs import ChatGPTConfig, OpenSourceConfig
from camel.models import OpenAIModel
from camel.types import PredefinedModelType
from camel.types.model_type import ModelType
from camel.utils import OpenAITokenCounter


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    [
        ModelType(PredefinedModelType.GPT_3_5_TURBO),
        ModelType(PredefinedModelType.GPT_4),
        ModelType(PredefinedModelType.GPT_4_TURBO),
        ModelType(PredefinedModelType.GPT_4O),
        ModelType(PredefinedModelType.GPT_4O_MINI),
        ModelType(PredefinedModelType.O1_PREVIEW),
        ModelType(PredefinedModelType.O1_MINI),
    ],
)
def test_openai_model(model_type):
    model_config_dict = ChatGPTConfig().as_dict()
    model = OpenAIModel(model_type, model_config_dict)
    assert model.model_type == model_type
    assert model.model_config_dict == model_config_dict
    assert isinstance(model.token_counter, OpenAITokenCounter)
    assert isinstance(model.model_type.value_for_tiktoken, str)
    assert isinstance(model.model_type.token_limit, int)


@pytest.mark.model_backend
def test_openai_model_unexpected_argument():
    model_type = ModelType(PredefinedModelType.GPT_4)
    model_config = OpenSourceConfig(
        model_path="vicuna-7b-v1.5",
        server_url="http://localhost:8000/v1",
    )
    model_config_dict = model_config.as_dict()

    with pytest.raises(
        ValueError,
        match=re.escape(
            (
                "Unexpected argument `model_path` is "
                "input into OpenAI model backend."
            )
        ),
    ):
        _ = OpenAIModel(model_type, model_config_dict)
