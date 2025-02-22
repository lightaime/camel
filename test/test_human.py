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
from camel.human import Human
from camel.messages import BaseMessage


def test_display_options():
    human = Human()
    msgs = [
        BaseMessage.make_assistant_message(
            role_name="assistant", content="Hello"
        ),
        BaseMessage.make_assistant_message(
            role_name="assistant", content="World"
        ),
    ]
    human.display_options(msgs)


def test_get_input(monkeypatch):
    human = Human()
    msgs = [
        BaseMessage.make_assistant_message(
            role_name="assistant", content="Hello"
        ),
        BaseMessage.make_assistant_message(
            role_name="assistant", content="World"
        ),
    ]
    human.display_options(msgs)
    monkeypatch.setattr('builtins.input', lambda _: str(1))
    assert human.get_input() == str(1)


def test_reduce_step(monkeypatch):
    human = Human()
    msgs = [
        BaseMessage.make_assistant_message(
            role_name="assistant", content="Hello"
        ),
        BaseMessage.make_assistant_message(
            role_name="assistant", content="World"
        ),
    ]

    monkeypatch.setattr('builtins.input', lambda _: str(1))
    human_response = human.reduce_step(msgs)
    assert human_response.msg.content.text == "Hello"
