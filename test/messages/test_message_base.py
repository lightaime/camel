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
from typing import List

import pytest

from camel.memories import ContextRecord
from camel.messages import BaseMessage
from camel.messages.acl_parameter import Content
from camel.models import ModelFactory
from camel.prompts import CodePrompt, TextPrompt
from camel.societies import RolePlaying
from camel.types import (
    ModelPlatformType,
    ModelType,
    OpenAIBackendRole,
    RoleType,
    TaskType,
)


@pytest.fixture
def base_message() -> BaseMessage:
    return BaseMessage(
        role_name="test_user",
        role_type=RoleType.USER,
        meta_dict={"key": "value"},
        content="test content",
    )


def test_base_message_addition_operator(base_message: BaseMessage):
    new_message = base_message.content.text + "!"
    assert new_message == "test content!"


def test_base_message_multiplication_operator(base_message: BaseMessage):
    new_message = base_message * 3
    assert new_message.content.text == "test contenttest contenttest content"


def test_base_message_length_operator(base_message: BaseMessage):
    assert len(base_message) == 12


def test_base_message_contains_operator(base_message: BaseMessage):
    assert "test" in base_message.content.text
    assert "foo" not in base_message.content.text


def test_extract_text_and_code_prompts():
    base_message = BaseMessage(
        role_name="test_role_name",
        role_type=RoleType.USER,
        meta_dict=dict(),
        content="This is a text prompt.\n\n"
        "```python\nprint('This is a code prompt')\n```\n"
        "This is another text prompt.\n\n"
        "```c\nprintf(\"This is another code prompt\");\n```",
    )
    text_prompts, code_prompts = base_message.extract_text_and_code_prompts()

    assert len(text_prompts) == 2
    assert isinstance(text_prompts[0], TextPrompt)
    assert isinstance(text_prompts[1], TextPrompt)
    assert text_prompts[0] == "This is a text prompt."
    assert text_prompts[1] == "This is another text prompt."

    assert len(code_prompts) == 2
    assert isinstance(code_prompts[0], CodePrompt)
    assert isinstance(code_prompts[1], CodePrompt)
    assert code_prompts[0] == "print('This is a code prompt')"
    assert code_prompts[1] == "printf(\"This is another code prompt\");"
    assert code_prompts[0].code_type == "python"
    assert code_prompts[1].code_type == "c"


def test_base_message_to_dict(base_message: BaseMessage) -> None:
    role_type = RoleType.USER
    text = "test content"

    expected_dict = {
        "role_name": "test_user",
        "role_type": role_type,
        "key": "value",
        "content": text,
    }
    base_message_dict = base_message.to_dict()
    del base_message_dict['acl_parameter']
    assert base_message_dict == expected_dict


def test_base_message():
    role_name = "test_role_name"
    role_type = RoleType.USER
    meta_dict = {"key": "value"}
    backend_role = OpenAIBackendRole.USER
    content = Content(text="test_content")

    message = BaseMessage(
        role_name=role_name,
        role_type=role_type,
        meta_dict=meta_dict,
        content=content,
    )

    assert message.role_name == role_name
    assert message.role_type == role_type
    assert message.meta_dict == meta_dict
    assert message.content.text == content.text

    openai_message = message.to_openai_message(backend_role)
    assert openai_message == {
        "role": backend_role.value,
        "content": "test_content",
    }

    openai_system_message = message.to_openai_system_message()
    assert openai_system_message == {
        "role": "system",
        "content": "test_content",
    }

    openai_user_message = message.to_openai_user_message()
    assert openai_user_message == {"role": "user", "content": "test_content"}

    openai_assistant_message = message.to_openai_assistant_message()
    assert openai_assistant_message == {
        "role": "assistant",
        "content": "test_content",
    }


@pytest.mark.model_backend
def test_roleplay_sharegpt_conversion():
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    role_playing = RolePlaying(
        assistant_role_name="assistant",
        assistant_agent_kwargs=dict(
            model=model,
            tools=[],
        ),
        user_role_name="user",
        user_agent_kwargs=dict(model=model),
        task_prompt="Perform the task",
        task_specify_agent_kwargs=dict(model=model),
        task_type=TaskType.AI_SOCIETY,
    )
    input_msg = role_playing.init_chat()
    role_playing.step(input_msg)

    records: List[ContextRecord] = (
        role_playing.assistant_agent.memory.retrieve()
    )
    original_messages = []
    sharegpt_msgs = []

    for record in records:
        message = record.memory_record.message
        # Remove meta_dict to avoid comparison issues
        message.meta_dict = None
        original_messages.append(message.content.text)
        sharegpt_msgs.append(message.to_sharegpt())

    converted_back = []
    for msg in sharegpt_msgs:
        item = BaseMessage.from_sharegpt(msg)
        converted_back.append(item.content.text)

    assert converted_back == original_messages
