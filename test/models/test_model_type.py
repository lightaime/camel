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
from camel.types import ModelType
from camel.types.augmented_model_type import AugmentedModelType


def test_predefined_model():
    model_type = AugmentedModelType(ModelType.GPT_4O_MINI)
    assert model_type.type == ModelType.GPT_4O_MINI
    assert model_type.value == "gpt-4o-mini"


def test_predefined_model_str():
    model_type = AugmentedModelType("gpt-4o-mini")
    assert model_type.type == ModelType.GPT_4O_MINI
    assert model_type.value == "gpt-4o-mini"


def test_open_source_model():
    model_type = AugmentedModelType("random-open-source")
    assert model_type.type == ModelType.OPEN_SOURCE
    assert model_type.value == "random-open-source"


def test_duplicated_model_types():
    model_type_1 = AugmentedModelType("random-open-source")
    model_type_2 = AugmentedModelType("random-open-source")
    assert model_type_1 == model_type_2

    model_type_3 = AugmentedModelType(ModelType.GPT_4O_MINI)
    model_type_4 = AugmentedModelType("gpt-4o-mini")
    assert model_type_3 == model_type_4
