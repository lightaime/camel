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
from io import BytesIO
from typing import List
from unittest.mock import Mock

import pytest
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from PIL import Image

from camel.agents import ChatAgent
from camel.agents.chat_agent import FunctionCallingRecord
from camel.configs import ChatGPTConfig, FunctionCallingConfig
from camel.functions import MATH_FUNCS
from camel.generators import SystemMessageGenerator
from camel.memories import MemoryRecord
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.terminators import ResponseWordsTerminator
from camel.types import (
    ChatCompletion,
    ModelPlatformType,
    ModelType,
    OpenAIBackendRole,
    RoleType,
    TaskType,
)


def test_chat_agent():
    model_config = ChatGPTConfig()
    llm = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_3_5_TURBO,
        model_config_dict=model_config.__dict__,
    )
    system_msg = SystemMessageGenerator(
        task_type=TaskType.AI_SOCIETY
    ).from_dict(
        dict(assistant_role="doctor"),
        role_tuple=("doctor", RoleType.ASSISTANT),
    )
    assistant = ChatAgent(system_msg, llm=llm)

    assert str(assistant) == (
        "ChatAgent(doctor, " f"RoleType.ASSISTANT, {ModelType.GPT_3_5_TURBO})"
    )

    assistant.reset()
    user_msg = BaseMessage(
        role_name="Patient",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Hello!",
    )
    assistant_response = assistant.step(user_msg)

    assert isinstance(assistant_response.msgs, list)
    assert len(assistant_response.msgs) > 0
    assert isinstance(assistant_response.terminated, bool)
    assert assistant_response.terminated is False
    assert isinstance(assistant_response.info, dict)
    assert assistant_response.info['id'] is not None


def test_chat_agent_stored_messages():
    system_msg = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    assistant = ChatAgent(system_msg)

    expected_context = [system_msg.to_openai_system_message()]
    context, _ = assistant.memory.get_context()
    assert context == expected_context

    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Tell me a joke.",
    )
    assistant.update_memory(user_msg, OpenAIBackendRole.USER)
    expected_context = [
        system_msg.to_openai_system_message(),
        user_msg.to_openai_user_message(),
    ]
    context, _ = assistant.memory.get_context()
    assert context == expected_context


@pytest.mark.model_backend
def test_chat_agent_messages_window():
    system_msg = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    assistant = ChatAgent(
        system_message=system_msg,
        message_window_size=2,
    )

    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Tell me a joke.",
    )

    assistant.memory.write_records(
        [
            MemoryRecord(
                user_msg,
                OpenAIBackendRole.USER,
            )
            for _ in range(5)
        ]
    )
    openai_messages, _ = assistant.memory.get_context()
    assert len(openai_messages) == 2


@pytest.mark.model_backend
def test_chat_agent_step_exceed_token_number():
    system_msg = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    assistant = ChatAgent(
        system_message=system_msg,
        token_limit=1,
    )

    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Tell me a joke.",
    )

    response = assistant.step(user_msg)
    assert len(response.msgs) == 0
    assert response.terminated


@pytest.mark.model_backend
@pytest.mark.parametrize('n', [1, 2, 3])
def test_chat_agent_multiple_return_messages(n):
    model_config = ChatGPTConfig(temperature=1.4, n=n)
    llm = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_3_5_TURBO,
        model_config_dict=model_config.__dict__,
    )
    system_msg = BaseMessage(
        "Assistant",
        RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a helpful assistant.",
    )
    assistant = ChatAgent(system_msg, llm=llm)
    assistant.reset()
    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Tell me a joke.",
    )
    assistant_response = assistant.step(user_msg)
    assert assistant_response.msgs is not None
    assert len(assistant_response.msgs) == n


@pytest.mark.model_backend
@pytest.mark.parametrize('n', [2])
def test_chat_agent_multiple_return_message_error(n):
    model_config = ChatGPTConfig(temperature=1.4, n=n)
    llm = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_3_5_TURBO,
        model_config_dict=model_config.__dict__,
    )
    system_msg = BaseMessage(
        "Assistant",
        RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a helpful assistant.",
    )

    assistant = ChatAgent(system_msg, llm=llm)
    assistant.reset()

    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Tell me a joke.",
    )
    assistant_response = assistant.step(user_msg)

    with pytest.raises(
        RuntimeError,
        match=(
            "Property msg is only available " "for a single message in msgs."
        ),
    ):
        _ = assistant_response.msg


@pytest.mark.model_backend
def test_chat_agent_stream_output():
    system_msg = BaseMessage(
        "Assistant",
        RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a helpful assistant.",
    )
    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Tell me a joke.",
    )

    stream_model_config = ChatGPTConfig(temperature=0, n=2, stream=True)
    llm = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_3_5_TURBO,
        model_config_dict=stream_model_config.__dict__,
    )
    stream_assistant = ChatAgent(system_msg, llm=llm)
    stream_assistant.reset()
    stream_assistant_response = stream_assistant.step(user_msg)

    for msg in stream_assistant_response.msgs:
        assert len(msg.content) > 0

    stream_usage = stream_assistant_response.info["usage"]
    assert stream_usage["completion_tokens"] > 0
    assert stream_usage["prompt_tokens"] > 0
    assert (
        stream_usage["total_tokens"]
        == stream_usage["completion_tokens"] + stream_usage["prompt_tokens"]
    )


@pytest.mark.model_backend
def test_set_output_language():
    system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    agent = ChatAgent(system_message=system_message)
    assert agent.output_language is None

    # Set the output language to "Arabic"
    output_language = "Arabic"
    agent.set_output_language(output_language)

    # Check if the output language is set correctly
    assert agent.output_language == output_language

    # Verify that the system message is updated with the new output language
    updated_system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant."
        "\nRegardless of the input language, you must output text in Arabic.",
    )
    assert agent.system_message.content == updated_system_message.content


@pytest.mark.model_backend
def test_set_multiple_output_language():
    system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    agent = ChatAgent(system_message=system_message)

    # Verify that the length of the system message is kept constant even when
    # multiple set_output_language operations are called
    agent.set_output_language("Chinese")
    agent.set_output_language("English")
    agent.set_output_language("French")
    updated_system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant."
        "\nRegardless of the input language, you must output text in French.",
    )
    assert agent.system_message.content == updated_system_message.content


@pytest.mark.model_backend
def test_token_exceed_return():
    system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    agent = ChatAgent(system_message=system_message)

    expect_info = {
        "id": None,
        "usage": None,
        "termination_reasons": ["max_tokens_exceeded"],
        "num_tokens": 1000,
        "called_functions": [],
    }
    agent.terminated = True
    response = agent.step_token_exceed(1000, [], "max_tokens_exceeded")
    assert response.msgs == []
    assert response.terminated
    assert response.info == expect_info


@pytest.mark.model_backend
def test_function_enabled():
    system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    model_config = FunctionCallingConfig(
        functions=[func.get_openai_function_schema() for func in MATH_FUNCS]
    )
    llm = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_3_5_TURBO,
        model_config_dict=model_config.__dict__,
    )
    agent_no_func = ChatAgent(system_message=system_message)
    agent_with_funcs = ChatAgent(
        system_message=system_message,
        llm=llm,
        function_list=MATH_FUNCS,
    )

    assert not agent_no_func.is_function_calling_enabled()
    assert agent_with_funcs.is_function_calling_enabled()


@pytest.mark.model_backend
def test_function_calling():
    system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    model_config = FunctionCallingConfig(
        functions=[func.get_openai_function_schema() for func in MATH_FUNCS]
    )
    llm = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_3_5_TURBO,
        model_config_dict=model_config.__dict__,
    )
    agent = ChatAgent(
        system_message=system_message,
        llm=llm,
        function_list=MATH_FUNCS,
    )

    ref_funcs = MATH_FUNCS

    assert len(agent.func_dict) == len(ref_funcs)

    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Calculate the result of: 2*8-10.",
    )
    agent_response = agent.step(user_msg)

    called_funcs: List[FunctionCallingRecord] = agent_response.info[
        'called_functions'
    ]
    for called_func in called_funcs:
        print(str(called_func))

    assert len(called_funcs) > 0
    assert str(called_funcs[0]).startswith("Function Execution")

    assert called_funcs[0].func_name == "mul"
    assert called_funcs[0].args == {"a": 2, "b": 8}
    assert called_funcs[0].result == 16


def test_response_words_termination():
    system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    response_terminator = ResponseWordsTerminator(words_dict=dict(goodbye=1))
    agent = ChatAgent(
        system_message=system_message,
        response_terminators=[response_terminator],
    )
    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Just say 'goodbye' once.",
    )
    agent_response = agent.step(user_msg)

    assert agent.terminated
    assert agent_response.terminated
    assert "goodbye" in agent_response.info['termination_reasons'][0]


def test_chat_agent_vision():
    system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    model_config = ChatGPTConfig(temperature=0, max_tokens=200, stop="")
    llm = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_3_5_TURBO,
        model_config_dict=model_config.__dict__,
    )
    agent = ChatAgent(
        system_message=system_message,
        llm=llm,
    )

    # Create an all blue PNG image:
    image = Image.new("RGB", (100, 100), "blue")
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    image = Image.open(img_byte_arr)

    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Is this image blue? Just answer yes or no.",
        image=image,
        image_detail="low",
    )
    # Mock the OpenAI model return value:
    agent.model_backend = Mock()
    agent.model_backend.run.return_value = ChatCompletion(
        id="mock_vision_id",
        choices=[
            Choice(
                finish_reason='stop',
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content='Yes.',
                    role='assistant',
                    function_call=None,
                    tool_calls=None,
                ),
            )
        ],
        created=123456,
        model='gpt-4-turbo-2024-04-09',
        object='chat.completion',
        system_fingerprint='fp_5d12056990',
        usage=CompletionUsage(
            completion_tokens=2, prompt_tokens=113, total_tokens=115
        ),
    )

    agent_response = agent.step(user_msg)
    assert agent_response.msgs[0].content == "Yes."
