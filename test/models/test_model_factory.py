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
import pytest

from camel.configs import *
from camel.models import ModelFactory
from camel.models.stub_model import StubTokenCounter
from camel.types import ModelPlatformType, ModelType
from camel.utils import *

parametrize = pytest.mark.parametrize(
    'model_platform, model_type',
    [
        (ModelPlatformType.OPENAI, ModelType.GPT_3_5_TURBO),
        (ModelPlatformType.OPENAI, ModelType.GPT_4_TURBO),
        (ModelPlatformType.OPENSOURCE, ModelType.STUB),
    ],
)

parameterize_token_counter = pytest.mark.parametrize(
    'model_platform, model_type, model_config_dict, token_counter, expected_counter_type, expected_model_type',
    [
        # Test OpenAI model
        (ModelPlatformType.OPENAI, ModelType.GPT_3_5_TURBO, ChatGPTConfig().__dict__, None, OpenAITokenCounter, ModelType.GPT_3_5_TURBO),
        (ModelPlatformType.OPENAI, ModelType.GPT_4, ChatGPTConfig().__dict__, None, OpenAITokenCounter, ModelType.GPT_4),

        # Test Stub model
        # Stub model uses StubTokenCounter as default
        (ModelPlatformType.OPENSOURCE, ModelType.STUB, ChatGPTConfig().__dict__, None, StubTokenCounter, None),
        (ModelPlatformType.OPENSOURCE, ModelType.STUB, ChatGPTConfig().__dict__, OpenAITokenCounter(ModelType.GPT_4), OpenAITokenCounter, ModelType.GPT_4),

        # Test Anthropic model
        # Anthropic model uses AnthropicTokenCounter as default
        (ModelPlatformType.ANTHROPIC, ModelType.CLAUDE_2_0, AnthropicConfig().__dict__, None, AnthropicTokenCounter, ModelType.CLAUDE_2_0),
        (ModelPlatformType.ANTHROPIC, ModelType.CLAUDE_2_0, AnthropicConfig().__dict__, OpenAITokenCounter(ModelType.GPT_3_5_TURBO), OpenAITokenCounter, ModelType.GPT_3_5_TURBO),

        # Test OpenSource model (take VICUNA as an example)
        (ModelPlatformType.OPENSOURCE, ModelType.VICUNA,
         OpenSourceConfig(
            model_path="lmsys/vicuna-7b-v1.5",
            server_url="http://localhost:8000/v1",
         ).__dict__,
         None, OpenSourceTokenCounter, ModelType.VICUNA
        ),

        (ModelPlatformType.OPENSOURCE, ModelType.VICUNA,
         OpenSourceConfig(
             model_path="lmsys/vicuna-7b-v1.5",
             server_url="http://localhost:8000/v1",
         ).__dict__,
         OpenAITokenCounter(ModelType.GPT_4), OpenAITokenCounter, ModelType.GPT_4
        ),

        # Test OpenSource model (take VICUNA as an example)
        (ModelPlatformType.GEMINI, ModelType.GEMINI_1_5_FLASH,
         GeminiConfig().__dict__,
         OpenAITokenCounter(ModelType.GPT_4), OpenAITokenCounter, ModelType.GPT_4
        ),

        # Test Ollama model
        (ModelPlatformType.OLLAMA, "gpt-3.5-turbo", OllamaConfig().__dict__, None, OpenAITokenCounter, ModelType.GPT_3_5_TURBO),
        (ModelPlatformType.OLLAMA, "gpt-3.5-turbo", OllamaConfig().__dict__, OpenAITokenCounter(ModelType.GPT_4), OpenAITokenCounter, ModelType.GPT_4),
    ],
)


@parametrize
def test_model_factory(model_platform, model_type):
    model_config_dict = ChatGPTConfig().__dict__
    model_inst = ModelFactory.create(
        model_platform, model_type, model_config_dict
    )
    messages = [
        {
            "role": "system",
            "content": "Initialize system",
        },
        {
            "role": "user",
            "content": "Hello",
        },
    ]
    response = model_inst.run(messages).model_dump()
    assert isinstance(response, dict)
    assert 'id' in response
    assert isinstance(response['id'], str)
    assert 'usage' in response
    assert isinstance(response['usage'], dict)
    assert 'choices' in response
    assert isinstance(response['choices'], list)
    assert len(response['choices']) == 1
    choice = response['choices'][0]
    assert 'finish_reason' in choice
    assert isinstance(choice['finish_reason'], str)
    assert 'message' in choice
    message = choice['message']
    assert isinstance(message, dict)
    assert 'content' in message
    assert isinstance(message['content'], str)
    assert 'role' in message
    assert isinstance(message['role'], str)
    assert message['role'] == 'assistant'


@parameterize_token_counter
def test_model_factory_self_set_token_counter(
    model_platform,
    model_type,
    model_config_dict,
    token_counter,
    expected_counter_type,
    expected_model_type,
):
    model_inst = ModelFactory.create(
        model_platform=model_platform,
        model_type=model_type,
        token_counter=token_counter,
        model_config_dict=model_config_dict,
    )
    assert isinstance(model_inst.token_counter, expected_counter_type)
    if hasattr(model_inst.token_counter, 'model_type'):
        assert model_inst.token_counter.model_type == expected_model_type
