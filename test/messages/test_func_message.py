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
from typing import Any, Dict, List

import pytest

from camel.memories import ContextRecord
from camel.messages import BaseMessage, FunctionCallingMessage
from camel.messages.axolotl.sharegpt.functions.hermes.hermes_function_formatter import (
    HermesFunctionFormatter,
)
from camel.models import ModelFactory
from camel.societies import RolePlaying
from camel.toolkits import MathToolkit
from camel.types import ModelPlatformType, ModelType, RoleType, TaskType


@pytest.fixture
def assistant_func_call_message() -> FunctionCallingMessage:
    role_name = "assistant"
    role_type = RoleType.ASSISTANT
    meta_dict = None
    content = "test function message"

    return FunctionCallingMessage(
        role_name=role_name,
        role_type=role_type,
        meta_dict=meta_dict,
        content=content,
        func_name="add",
        args={"a": "1", "b": "2"},
    )


@pytest.fixture
def function_result_message() -> FunctionCallingMessage:
    role_name = "tool"
    role_type = RoleType.ASSISTANT
    meta_dict = None
    content = "test function message"

    return FunctionCallingMessage(
        role_name=role_name,
        role_type=role_type,
        meta_dict=meta_dict,
        content=content,
        func_name="add",
        result=3,
    )


def test_assistant_func_message(
    assistant_func_call_message: FunctionCallingMessage,
):
    content = "test function message"

    assert assistant_func_call_message.func_name == "add"
    assert assistant_func_call_message.args == {"a": "1", "b": "2"}

    msg_dict: Dict[str, Any]
    msg_dict = {
        "role": "assistant",
        "content": content,
        "function_call": {
            "name": "add",
            "arguments": str({"a": "1", "b": "2"}),
        },
    }
    assert (
        assistant_func_call_message.to_openai_assistant_message() == msg_dict
    )


def test_function_func_message(
    function_result_message: FunctionCallingMessage,
):
    assert function_result_message.func_name == "add"
    assert function_result_message.result == 3

    result_content = {"result": {str(3)}}
    msg_dict: Dict[str, str] = {
        "role": "function",
        "name": "add",
        "content": f'{result_content}',
    }
    assert function_result_message.to_openai_function_message() == msg_dict


def test_assistant_func_message_to_openai_function_message(
    assistant_func_call_message: FunctionCallingMessage,
):
    expected_msg_dict: Dict[str, str] = {
        "role": "function",
        "name": "add",
        "content": "{'result': {'None'}}",
    }

    assert (
        assistant_func_call_message.to_openai_function_message()
        == expected_msg_dict
    )


@pytest.mark.model_backend
def test_roleplay_conversion_with_tools():
    tools = MathToolkit().get_tools()
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    role_playing = RolePlaying(
        assistant_role_name="assistant",
        assistant_agent_kwargs=dict(
            model=model,
            tools=tools,
        ),
        user_role_name="user",
        user_agent_kwargs=dict(model=model),
        task_prompt="Perform the task",
        task_specify_agent_kwargs=dict(model=model),
        task_type=TaskType.AI_SOCIETY,
    )
    input_msg = role_playing.init_chat("What is 2 + 4?")
    [assistant, _] = role_playing.step(input_msg)
    role_playing.step(assistant.msg)

    records: List[ContextRecord] = (
        role_playing.assistant_agent.memory.retrieve()
    )
    original_messages = []
    sharegpt_msgs = []

    for record in records:
        message = record.memory_record.message
        # Remove meta_dict to avoid comparison issues
        message.meta_dict = None
        original_messages.append(message)
        sharegpt_msgs.append(message.to_sharegpt())

    converted_back = []
    for msg in sharegpt_msgs:
        converted_back.append(
            BaseMessage.from_sharegpt(
                msg, function_format=HermesFunctionFormatter()
            )
        )

    assert converted_back == original_messages


def test_convert_function_call_and_response_to_from_sharegpt_hermes(
    assistant_func_call_message: FunctionCallingMessage,
    function_result_message: FunctionCallingMessage,
):
    sharegpt_function_call = assistant_func_call_message.to_sharegpt()

    # Check the function call contains a hermes function call
    # TODO: Consider using code from https://github.com/NousResearch/Hermes-Function-Calling/blob/main/validator.py
    #  and adjacent files
    assert "<tool_call>" in sharegpt_function_call.value

    # Test it converts back
    reconverted_function_call = BaseMessage.from_sharegpt(
        sharegpt_function_call
    )
    assert assistant_func_call_message == reconverted_function_call

    sharegpt_function_result = function_result_message.to_sharegpt()
    reconverted_function_result = BaseMessage.from_sharegpt(
        sharegpt_function_result
    )

    assert function_result_message == reconverted_function_result


def test_function_func_message_to_openai_assistant_message(
    function_result_message: FunctionCallingMessage,
):
    with pytest.raises(
        ValueError,
        match=(
            "Invalid request for converting into assistant message"
            " due to missing function name or arguments."
        ),
    ):
        function_result_message.to_openai_assistant_message()
