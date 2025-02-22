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

import pytest

from camel.messages import BaseMessage
from camel.messages.acl_parameter import Content
from camel.types import RoleType


@pytest.fixture
def chat_message() -> BaseMessage:
    return BaseMessage(
        role_name="test_role",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="test chat message",
    )


@pytest.fixture
def assistant_chat_message() -> BaseMessage:
    return BaseMessage(
        role_name="test_assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="test assistant chat message",
    )


@pytest.fixture
def user_chat_message() -> BaseMessage:
    return BaseMessage(
        role_name="test_user",
        role_type=RoleType.USER,
        meta_dict=None,
        content="test user chat message",
    )


def test_chat_message(chat_message: BaseMessage) -> None:
    role_name = "test_role"
    role_type = RoleType.ASSISTANT
    meta_dict = None
    content = Content(text="test chat message")

    assert chat_message.role_name == role_name
    assert chat_message.role_type == role_type
    assert chat_message.meta_dict == meta_dict
    assert chat_message.content.text == content.text


def test_assistant_chat_message(assistant_chat_message: BaseMessage) -> None:
    role_name = "test_assistant"
    role_type = RoleType.ASSISTANT
    meta_dict = None
    content = Content(text="test assistant chat message")

    assert assistant_chat_message.role_name == role_name
    assert assistant_chat_message.role_type == role_type
    assert assistant_chat_message.meta_dict == meta_dict
    assert assistant_chat_message.content.text == content.text


def test_user_chat_message(user_chat_message: BaseMessage) -> None:
    role_name = "test_user"
    role_type = RoleType.USER
    meta_dict = None
    content = Content(text="test user chat message")

    assert user_chat_message.role_name == role_name
    assert user_chat_message.role_type == role_type
    assert user_chat_message.meta_dict == meta_dict
    assert user_chat_message.content.text == content.text
