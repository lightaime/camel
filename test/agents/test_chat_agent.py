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
import asyncio
import json
from copy import deepcopy
from io import BytesIO
from typing import List
from unittest.mock import MagicMock

import pytest
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage
from PIL import Image
from pydantic import BaseModel, Field

from camel.agents import ChatAgent
from camel.agents.chat_agent import ToolCallingRecord
from camel.configs import ChatGPTConfig
from camel.generators import SystemMessageGenerator
from camel.memories import MemoryRecord
from camel.messages import BaseMessage
from camel.messages.acl_parameter import Content
from camel.models import ModelFactory
from camel.terminators import ResponseWordsTerminator
from camel.toolkits import (
    FunctionTool,
    MathToolkit,
    SearchToolkit,
)
from camel.types import (
    ChatCompletion,
    ModelPlatformType,
    ModelType,
    OpenAIBackendRole,
    RoleType,
    TaskType,
    UnifiedModelType,
)
from camel.utils.async_func import sync_funcs_to_async

model_backend_rsp_base = ChatCompletion(
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

parametrize = pytest.mark.parametrize(
    'model',
    [
        ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
        ),
        pytest.param(None, marks=pytest.mark.model_backend),
    ],
)


@parametrize
def test_chat_agent(model, step_call_count=3):
    model = model
    system_msg = SystemMessageGenerator(
        task_type=TaskType.AI_SOCIETY
    ).from_dict(
        dict(assistant_role="doctor"),
        role_tuple=("doctor", RoleType.ASSISTANT),
    )
    assistant_with_sys_msg = ChatAgent(system_msg, model=model)
    assistant_without_sys_msg = ChatAgent(model=model)

    assert str(assistant_with_sys_msg) == (
        "ChatAgent(doctor, " f"RoleType.ASSISTANT, {ModelType.GPT_4O_MINI})"
    )
    assert str(assistant_without_sys_msg) == (
        "ChatAgent(assistant, "
        f"RoleType.ASSISTANT, {UnifiedModelType(ModelType.GPT_4O_MINI)})"
    )

    for assistant in [assistant_with_sys_msg, assistant_without_sys_msg]:
        assistant.reset()
        assistant.model_backend.run = MagicMock(
            return_value=model_backend_rsp_base
        )

    user_msg_bm = BaseMessage(
        role_name="Patient",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Hello!",
    )

    user_msg_str = "Hello!"

    for assistant in [assistant_with_sys_msg, assistant_without_sys_msg]:
        for i in range(step_call_count):
            for user_msg in [user_msg_bm, user_msg_str]:
                response = assistant.step(user_msg)
                assert isinstance(response.msgs, list), f"Error in round {i+1}"
                assert len(response.msgs) > 0, f"Error in round {i+1}"
                assert isinstance(
                    response.terminated, bool
                ), f"Error in round {i+1}"
                assert response.terminated is False, f"Error in round {i+1}"
                assert isinstance(response.info, dict), f"Error in round {i+1}"
                assert response.info['id'] is not None, f"Error in round {i+1}"


@pytest.mark.model_backend
def test_chat_agent_stored_messages():
    system_msg = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )

    assistant_with_sys_msg = ChatAgent(system_msg)
    assistant_without_sys_msg = ChatAgent()

    expected_context = [system_msg.to_openai_system_message()]

    context_with_sys_msg, _ = assistant_with_sys_msg.memory.get_context()
    assert context_with_sys_msg == expected_context
    context_without_sys_msg, _ = assistant_without_sys_msg.memory.get_context()
    assert context_without_sys_msg == []

    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Tell me a joke.",
    )

    for assistant in [assistant_with_sys_msg, assistant_without_sys_msg]:
        assistant.update_memory(user_msg, OpenAIBackendRole.USER)

    expected_context_with_sys_msg = [
        system_msg.to_openai_system_message(),
        user_msg.to_openai_user_message(),
    ]
    expected_context_without_sys_msg = [
        user_msg.to_openai_user_message(),
    ]

    context_with_sys_msg, _ = assistant_with_sys_msg.memory.get_context()
    assert context_with_sys_msg == expected_context_with_sys_msg
    context_without_sys_msg, _ = assistant_without_sys_msg.memory.get_context()
    assert context_without_sys_msg == expected_context_without_sys_msg


@pytest.mark.model_backend
def test_chat_agent_step_with_structure_response(step_call_count=3):
    system_msg = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    assistant = ChatAgent(
        system_message=system_msg,
    )

    class JokeResponse(BaseModel):
        joke: str = Field(description="a joke")
        funny_level: str = Field(description="Funny level, from 1 to 10")

    user_msg = BaseMessage.make_user_message(
        role_name="User",
        content="Tell a jokes.",
    )

    model_backend_rsp_tool = deepcopy(model_backend_rsp_base)

    model_backend_rsp_tool.choices[0].message.content = '{ \
        "joke":"What do you call fake spaghetti? An impasta!", \
        "funny_level":"6" \
    }'
    model_backend_rsp_tool.choices[0].message.tool_calls = []
    model_backend_rsp_tool.choices[0].message.parsed = JokeResponse(
        joke="What do you call fake spaghetti? An impasta!", funny_level='6'
    )
    assistant.model_backend.run = MagicMock(
        return_value=model_backend_rsp_tool
    )

    for i in range(step_call_count):
        response = assistant.step(user_msg, response_format=JokeResponse)
        response_content_json = json.loads(response.msg.content.text)
        joke_response_keys = set(
            JokeResponse.model_json_schema()["properties"].keys()
        )

        response_content_keys = set(response_content_json.keys())

        assert joke_response_keys.issubset(response_content_keys), (
            f"Error in calling round {i+1}: "
            f"Missing keys: {joke_response_keys - response_content_keys}"
        )

        for key in joke_response_keys:
            assert key in response_content_json, (
                f"Error in calling round {i+1}: "
                f"Key {key} not found in response content"
            )


@pytest.mark.model_backend
def test_chat_agent_step_with_external_tools(step_call_count=3):
    internal_tools = [FunctionTool(SearchToolkit().search_wiki)]
    external_tools = MathToolkit().get_tools()
    tool_list = internal_tools + external_tools

    model_config_dict = ChatGPTConfig(
        tools=tool_list,
        temperature=0.0,
    ).as_dict()

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=model_config_dict,
    )
    model_backend_external1 = ChatCompletion(
        id='mock_id_123456',
        choices=[
            Choice(
                finish_reason='tool_calls',
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content=None,
                    refusal=None,
                    role='assistant',
                    audio=None,
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id='call_mock_123456',
                            function=Function(
                                arguments='{ \
                                    "entity": "Portal game release year" \
                                }',
                                name='search_wiki',
                            ),
                            type='function',
                        ),
                        ChatCompletionMessageToolCall(
                            id='call_mock_123457',
                            function=Function(
                                arguments='{ \
                                    "entity": "United States founding year" \
                                }',
                                name='search_wiki',
                            ),
                            type='function',
                        ),
                    ],
                ),
            )
        ],
        created=1730745899,
        model='gpt-4o-mini-2024-07-18',
        object='chat.completion',
        service_tier=None,
        usage=CompletionUsage(
            completion_tokens=56, prompt_tokens=292, total_tokens=348
        ),
    )

    model_backend_external2 = ChatCompletion(
        id='chatcmpl-mock_id_123456',
        choices=[
            Choice(
                finish_reason='tool_calls',
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content=None,
                    refusal=None,
                    role='assistant',
                    audio=None,
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id='call_mock_123456',
                            function=Function(
                                arguments='{"a":1776,"b":2007}', name='sub'
                            ),
                            type='function',
                        )
                    ],
                ),
            )
        ],
        created=1730745902,
        model='gpt-4o-mini-2024-07-18',
        object='chat.completion',
        service_tier=None,
        usage=CompletionUsage(
            completion_tokens=19, prompt_tokens=991, total_tokens=1010
        ),
    )

    model.run = MagicMock(
        side_effect=[model_backend_external1, model_backend_external2]
        * step_call_count
    )

    # Set external_tools
    external_tool_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Tools calling operator",
            content="You are a helpful assistant",
        ),
        model=model,
        tools=internal_tools,
        external_tools=external_tools,
    )

    usr_msg = BaseMessage.make_user_message(
        role_name="User",
        content="What's the result of the release year of Portal subtracted "
        "from the year that United States was founded?",
    )

    for i in range(step_call_count):
        response = external_tool_agent.step(usr_msg)
        assert not response.msg.content.text

        external_tool_call_request = response.info[
            "external_tool_call_request"
        ]
        assert (
            external_tool_call_request.tool_name == "sub"
        ), f"Error in calling round {i+1}"


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
                message=user_msg,
                role_at_backend=OpenAIBackendRole.USER,
            )
            for _ in range(5)
        ]
    )
    openai_messages, _ = assistant.memory.get_context()
    assert len(openai_messages) == 2


@pytest.mark.model_backend
def test_chat_agent_step_exceed_token_number(step_call_count=3):
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
    assistant.model_backend.run = MagicMock(
        return_value=model_backend_rsp_base
    )

    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Tell me a joke.",
    )

    for i in range(step_call_count):
        response = assistant.step(user_msg)
        assert len(response.msgs) == 0, f"Error in calling round {i+1}"
        assert response.terminated, f"Error in calling round {i+1}"


@pytest.mark.model_backend
@pytest.mark.parametrize('n', [1, 2, 3])
def test_chat_agent_multiple_return_messages(n, step_call_count=3):
    model_config = ChatGPTConfig(temperature=1.4, n=n)
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=model_config.as_dict(),
    )
    model_backend_rsp_tool = ChatCompletion(
        id="mock_response_id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content="What do you call fake spaghetti? An impasta!",
                    role="assistant",
                    function_call=None,
                ),
            )
        ]
        * n,
        created=123456789,
        model="gpt-4o-2024-05-13",
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=32,
            prompt_tokens=15,
            total_tokens=47,
        ),
    )
    model.run = MagicMock(return_value=model_backend_rsp_tool)

    system_msg = BaseMessage(
        "Assistant",
        RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a helpful assistant.",
    )
    assistant_with_sys_msg = ChatAgent(system_msg, model=model)
    assistant_without_sys_msg = ChatAgent(model=model)

    assistant_with_sys_msg.reset()
    assistant_without_sys_msg.reset()

    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Tell me a joke.",
    )
    assistant_with_sys_msg_response = assistant_with_sys_msg.step(user_msg)
    assistant_without_sys_msg_response = assistant_without_sys_msg.step(
        user_msg
    )

    for i in range(step_call_count):
        assert (
            assistant_with_sys_msg_response.msgs is not None
        ), f"Error in calling round {i+1}"
        assert (
            len(assistant_with_sys_msg_response.msgs) == n
        ), f"Error in calling round {i+1}"
        assert (
            assistant_without_sys_msg_response.msgs is not None
        ), f"Error in calling round {i+1}"
        assert (
            len(assistant_without_sys_msg_response.msgs) == n
        ), f"Error in calling round {i+1}"


@pytest.mark.model_backend
@pytest.mark.parametrize('n', [2])
def test_chat_agent_multiple_return_message_error(n, step_call_count=3):
    model_config = ChatGPTConfig(temperature=1.4, n=n)
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=model_config.as_dict(),
    )
    model_backend_multi_messages = deepcopy(model_backend_rsp_base)
    model_backend_multi_messages.choices.append(
        Choice(
            finish_reason='stop',
            index=1,
            logprobs=None,
            message=ChatCompletionMessage(
                content='Why did the scarecrow win an award? '
                'Because he was outstanding in his field!',
                refusal=None,
                role='assistant',
                audio=None,
                function_call=None,
                tool_calls=None,
            ),
        )
    )
    model.run = MagicMock(return_value=model_backend_multi_messages)

    system_msg = BaseMessage(
        "Assistant",
        RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a helpful assistant.",
    )

    assistant = ChatAgent(system_msg, model=model)
    assistant.reset()

    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Tell me a joke.",
    )
    for _ in range(step_call_count):
        assistant_response = assistant.step(user_msg)

        with pytest.raises(
            RuntimeError,
            match=(
                "Property msg is only available "
                "for a single message in msgs."
            ),
        ):
            _ = assistant_response.msg


@pytest.mark.model_backend
def test_chat_agent_stream_output(step_call_count=3):
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
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=stream_model_config.as_dict(),
    )
    model.run = MagicMock(return_value=model_backend_rsp_base)
    stream_assistant = ChatAgent(system_msg, model=model)
    stream_assistant.reset()
    for i in range(step_call_count):
        stream_assistant_response = stream_assistant.step(user_msg)

        for msg in stream_assistant_response.msgs:
            assert len(msg.content.text) > 0, f"Error in calling round {i+1}"

        stream_usage = stream_assistant_response.info["usage"]
        assert (
            stream_usage["completion_tokens"] > 0
        ), f"Error in calling round {i+1}"
        assert (
            stream_usage["prompt_tokens"] > 0
        ), f"Error in calling round {i+1}"
        assert (
            stream_usage["total_tokens"]
            == stream_usage["completion_tokens"]
            + stream_usage["prompt_tokens"]
        ), f"Error in calling round {i+1}"


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
    agent.output_language = output_language

    # Check if the output language is set correctly
    assert agent.output_language == output_language

    # Verify that the system message is updated with the new output language
    updated_system_message = {
        'role': 'system',
        'content': 'You are a help assistant.\nRegardless of the '
        'input language, you must output text in Arabic.',
    }
    memory_content = agent.memory.get_context()
    assert memory_content[0][0] == updated_system_message


@pytest.mark.model_backend
def test_set_multiple_output_language():
    system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    agent_with_sys_msg = ChatAgent(system_message=system_message)
    agent_without_sys_msg = ChatAgent()

    # Verify that the length of the system message is kept constant even when
    # multiple set_output_language operations are called
    agent_with_sys_msg.output_language = "Chinese"
    agent_with_sys_msg.output_language = "English"
    agent_with_sys_msg.output_language = "French"
    agent_without_sys_msg.output_language = "Chinese"
    agent_without_sys_msg.output_language = "English"
    agent_without_sys_msg.output_language = "French"

    updated_system_message_with_sys_msg = {
        'role': 'system',
        'content': 'You are a help assistant.\nRegardless of the '
        'input language, you must output text in French.',
    }
    updated_system_message_without_sys_msg = {
        'role': 'system',
        'content': '\nRegardless of the input language, you must output '
        'text in French.',
    }

    memory_content_with_sys_msg = agent_with_sys_msg.memory.get_context()
    memory_content_without_sys_msg = agent_without_sys_msg.memory.get_context()

    assert (
        memory_content_with_sys_msg[0][0]
        == updated_system_message_with_sys_msg
    )
    assert (
        memory_content_without_sys_msg[0][0]
        == updated_system_message_without_sys_msg
    )


@pytest.mark.model_backend
def test_tool_calling_sync(step_call_count=3):
    system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )
    agent = ChatAgent(
        system_message=system_message,
        model=model,
        tools=MathToolkit().get_tools(),
    )

    ref_funcs = MathToolkit().get_tools()

    assert len(agent.tool_dict) == len(ref_funcs)

    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Calculate the result of: 2*8-10.",
    )
    model_backend_rsp_tool = ChatCompletion(
        id='mock_id_123456',
        choices=[
            Choice(
                finish_reason='tool_calls',
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content=None,
                    refusal=None,
                    role='assistant',
                    audio=None,
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id='call_mock_123456',
                            function=Function(
                                arguments='{ \
                                    "a": 2, \
                                    "b": 8, \
                                    "decimal_places": 0 \
                                }',
                                name='multiply',
                            ),
                            type='function',
                        ),
                        ChatCompletionMessageToolCall(
                            id='call_mock_123457',
                            function=Function(
                                arguments='{"a": 0, "b": 10}', name='sub'
                            ),
                            type='function',
                        ),
                    ],
                ),
            )
        ],
        created=1730752528,
        model='gpt-4o-mini-2024-07-18',
        object='chat.completion',
        service_tier=None,
        usage=CompletionUsage(
            completion_tokens=50, prompt_tokens=152, total_tokens=202
        ),
    )
    model_backend_rsp_tool1 = ChatCompletion(
        id='mock_id_123456',
        choices=[
            Choice(
                finish_reason='tool_calls',
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content=None,
                    refusal=None,
                    role='assistant',
                    audio=None,
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id='call_kIJby7Y6As6gAbWoxDqxD6HG',
                            function=Function(
                                arguments='{"a":16,"b":10}', name='sub'
                            ),
                            type='function',
                        )
                    ],
                ),
            )
        ],
        created=1730752528,
        model='gpt-4o-mini-2024-07-18',
        object='chat.completion',
        service_tier=None,
        usage=CompletionUsage(
            completion_tokens=50, prompt_tokens=152, total_tokens=202
        ),
    )
    model_backend_rsp_tool2 = ChatCompletion(
        id='mock_id_123456',
        choices=[
            Choice(
                finish_reason='tool_calls',
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content='The result of the calculation \
                        (2 times 8 - 10) is (6).',
                    refusal=None,
                    role='assistant',
                    audio=None,
                    function_call=None,
                    tool_calls=None,
                ),
            )
        ],
        created=1730752528,
        model='gpt-4o-mini-2024-07-18',
        object='chat.completion',
        service_tier=None,
        usage=CompletionUsage(
            completion_tokens=50, prompt_tokens=152, total_tokens=202
        ),
    )

    model.run = MagicMock(
        side_effect=[
            model_backend_rsp_tool,
            model_backend_rsp_tool1,
            model_backend_rsp_tool2,
        ]
        * step_call_count
    )

    for i in range(step_call_count):
        agent_response = agent.step(user_msg)

        tool_calls: List[ToolCallingRecord] = [
            call for call in agent_response.info['tool_calls']
        ]

        assert len(tool_calls) > 0, f"Error in calling round {i+1}"
        assert str(tool_calls[0]).startswith(
            "Tool Execution"
        ), f"Error in calling round {i+1}"
        assert (
            tool_calls[0].tool_name == "multiply"
        ), f"Error in calling round {i+1}"
        assert tool_calls[0].args == {
            "a": 2,
            "b": 8,
            'decimal_places': 0,
        }, f"Error in calling round {i+1}"
        assert tool_calls[0].result == 16, f"Error in calling round {i+1}"


@pytest.mark.model_backend
@pytest.mark.asyncio
async def test_tool_calling_math_async(step_call_count=3):
    system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    math_funcs = sync_funcs_to_async([FunctionTool(MathToolkit().multiply)])
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )
    agent = ChatAgent(
        system_message=system_message,
        model=model,
        tools=math_funcs,
    )

    ref_funcs = math_funcs

    assert len(agent.tool_dict) == len(ref_funcs)

    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Calculate the result of: 2*8",
    )

    model_backend_rsp_tool = ChatCompletion(
        id='mock_id_123456',
        choices=[
            Choice(
                finish_reason='tool_calls',
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content=None,
                    refusal=None,
                    role='assistant',
                    audio=None,
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id='call_mock_123456',
                            function=Function(
                                arguments='{ \
                                    "a": 2, \
                                    "b": 8, \
                                    "decimal_places": 0 \
                                }',
                                name='multiply',
                            ),
                            type='function',
                        )
                    ],
                ),
            )
        ],
        created=1730752528,
        model='gpt-4o-mini-2024-07-18',
        object='chat.completion',
        service_tier=None,
        usage=CompletionUsage(
            completion_tokens=50, prompt_tokens=152, total_tokens=202
        ),
    )
    model_backend_rsp_tool1 = ChatCompletion(
        id='mock_id_123456',
        choices=[
            Choice(
                finish_reason='tool_calls',
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content='The result of ( 2 times 8 ) is ( 16 ).',
                    refusal=None,
                    role='assistant',
                    audio=None,
                    function_call=None,
                    tool_calls=None,
                ),
            )
        ],
        created=1730752528,
        model='gpt-4o-mini-2024-07-18',
        object='chat.completion',
        service_tier=None,
        usage=CompletionUsage(
            completion_tokens=50, prompt_tokens=152, total_tokens=202
        ),
    )

    model.run = MagicMock(
        side_effect=[
            model_backend_rsp_tool,
            model_backend_rsp_tool1,
        ]
        * step_call_count
    )

    for i in range(step_call_count):
        agent_response = await agent.astep(user_msg)

        tool_calls = agent_response.info['tool_calls']

        assert (
            tool_calls[0].tool_name == "multiply"
        ), f"Error in calling round {i+1}"
        assert tool_calls[0].args == {
            "a": 2,
            "b": 8,
            'decimal_places': 0,
        }, f"Error in calling round {i+1}"
        assert tool_calls[0].result == 16, f"Error in calling round {i+1}"


@pytest.mark.model_backend
@pytest.mark.asyncio
async def test_tool_calling_async(step_call_count=3):
    system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )

    async def async_sleep(second: int) -> int:
        r"""Async sleep function.

        Args:
            second (int): Number of seconds to sleep.

        Returns:
            integer: Number of seconds to sleep.
        """
        await asyncio.sleep(second)
        return second

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )
    model_backend_rsp_tool_async = deepcopy(model_backend_rsp_base)

    # Mock tool calling
    def mock_run_tool_calling_async(*args, **kwargs):
        # Reset tool_calls at the beginning of each new round of step() call
        if model.run.call_count % 2 == 1:
            model_backend_rsp_tool_async.choices[0].message.tool_calls = [
                ChatCompletionMessageToolCall(
                    id='call_mock_123456',
                    function=Function(
                        arguments='{"second":1}', name='async_sleep'
                    ),
                    type='function',
                )
            ]

        # mock the completion of the tool call
        elif model_backend_rsp_tool_async.choices[0].message.tool_calls:
            model_backend_rsp_tool_async.choices[0].message.tool_calls.pop(0)

        if (
            len(model_backend_rsp_tool_async.choices[0].message.tool_calls)
            == 0
        ):
            model_backend_rsp_tool_async.choices[0].message.tool_calls = None

        return model_backend_rsp_tool_async

    model.run = MagicMock(side_effect=mock_run_tool_calling_async)

    agent = ChatAgent(
        system_message=system_message,
        model=model,
        tools=[FunctionTool(async_sleep)],
    )

    assert len(agent.tool_dict) == 1

    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="Call the async sleep which is specified in function list with"
        " 1 second.",
    )

    for i in range(step_call_count):
        agent_response = await agent.astep(user_msg)

        tool_calls = agent_response.info['tool_calls']

        assert tool_calls, f"Error in calling round {i+1}"
        assert str(tool_calls[0]).startswith(
            "Tool Execution"
        ), f"Error in calling round {i+1}"

        assert (
            tool_calls[0].tool_name == "async_sleep"
        ), f"Error in calling round {i+1}"
        assert tool_calls[0].args == {
            'second': 1
        }, f"Error in calling round {i+1}"
        assert tool_calls[0].result == 1, f"Error in calling round {i+1}"


def test_response_words_termination(step_call_count=3):
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
    model_backend_rsp = deepcopy(model_backend_rsp_base)
    model_backend_rsp.choices[0].message.content = "Goodbye."
    agent.model_backend.run = MagicMock(return_value=model_backend_rsp)

    for i in range(step_call_count):
        agent_response = agent.step(user_msg)

        assert agent.terminated, f"Error in calling round {i+1}"
        assert agent_response.terminated, f"Error in calling round {i+1}"
        assert (
            "goodbye" in agent_response.info['termination_reasons'][0]
        ), f"Error in calling round {i+1}"


def test_chat_agent_vision(step_call_count=3):
    system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a help assistant.",
    )
    model_config = ChatGPTConfig(temperature=0, max_tokens=200, stop="")
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=model_config.as_dict(),
    )
    agent = ChatAgent(
        system_message=system_message,
        model=model,
    )

    # Create an all blue PNG image:
    image = Image.new("RGB", (100, 100), "blue")
    image_list = []
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    image = Image.open(img_byte_arr)
    image_list.append(image)

    content = Content(
        text="Is this image blue? Just answer yes or no.",
        image_list=image_list,
        image_detail="low",
    )
    user_msg = BaseMessage(
        role_name="User",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content=content,
    )
    # Mock the OpenAI model return value:
    agent.model_backend.run = MagicMock(
        return_value=ChatCompletion(
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
    )

    for i in range(step_call_count):
        agent_response = agent.step(user_msg)
        assert (
            agent_response.msgs[0].content.text == "Yes."
        ), f"Error in calling round {i+1}"
