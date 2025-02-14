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
from unittest.mock import MagicMock

import pytest
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from camel.agents import ChatAgent, CriticAgent
from camel.human import Human
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.societies import RolePlaying
from camel.toolkits import MathToolkit
from camel.types import (
    ChatCompletion,
    ModelPlatformType,
    ModelType,
    RoleType,
    TaskType,
)

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
)

model_backend_rsp = ChatCompletion(
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


@pytest.mark.parametrize("model", [None, model])
@pytest.mark.parametrize("critic_role_name", ["human", "critic agent"])
@pytest.mark.parametrize("with_critic_in_the_loop", [True, False])
def test_role_playing_init(model, critic_role_name, with_critic_in_the_loop):
    if model is not None:
        model.run = MagicMock(return_value=model_backend_rsp)

    role_playing = RolePlaying(
        assistant_role_name="assistant",
        assistant_agent_kwargs=dict(model=model),
        user_role_name="user",
        user_agent_kwargs=dict(model=model),
        model=model,
        critic_role_name=critic_role_name,
        task_prompt="Perform the task",
        with_task_specify=False,
        task_specify_agent_kwargs=dict(model=model),
        with_task_planner=False,
        task_planner_agent_kwargs=dict(model=model),
        with_critic_in_the_loop=with_critic_in_the_loop,
        task_type=TaskType.AI_SOCIETY,
    )
    assert role_playing.with_task_specify is False
    assert role_playing.with_task_planner is False
    assert role_playing.with_critic_in_the_loop is with_critic_in_the_loop
    assert role_playing.task_type == TaskType.AI_SOCIETY
    assert role_playing.task_prompt == "Perform the task"
    assert role_playing.specified_task_prompt is None
    assert role_playing.planned_task_prompt is None

    assert isinstance(role_playing.assistant_sys_msg, BaseMessage)
    assert role_playing.assistant_sys_msg.role_type == RoleType.ASSISTANT
    assert isinstance(role_playing.user_sys_msg, BaseMessage)
    assert role_playing.user_sys_msg.role_type == RoleType.USER

    assistant_agent = role_playing.assistant_agent
    user_agent = role_playing.user_agent
    critic = role_playing.critic

    assert isinstance(assistant_agent, ChatAgent)
    assert isinstance(user_agent, ChatAgent)
    if model is None:
        assert (
            assistant_agent.model_backend.model_type == ModelType.GPT_4O_MINI
        )
        assert user_agent.model_backend.model_type == ModelType.GPT_4O_MINI
    else:
        assert assistant_agent.model_backend.model_type == ModelType.GPT_4O
        assert user_agent.model_backend.model_type == ModelType.GPT_4O

    if not with_critic_in_the_loop:
        assert critic is None
    else:
        assert critic is not None
        if critic_role_name == "human":
            assert isinstance(critic, Human)
        else:
            assert isinstance(critic, CriticAgent)
            assert role_playing.critic_sys_msg is not None
            if model is None:
                assert critic.model_backend.model_type == ModelType.GPT_4O_MINI
            else:
                assert critic.model_backend.model_type == ModelType.GPT_4O


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "task_type, extend_sys_msg_meta_dicts, extend_task_specify_meta_dict",
    [
        (TaskType.AI_SOCIETY, None, None),
        (
            TaskType.CODE,
            [dict(domain="science", language="python")] * 2,
            dict(domain="science", language="python"),
        ),
        (TaskType.MISALIGNMENT, None, None),
    ],
)
def test_role_playing_step(
    task_type,
    extend_sys_msg_meta_dicts,
    extend_task_specify_meta_dict,
    step_call_count=3,
):
    if model is not None:
        model._run = MagicMock(return_value=model_backend_rsp)

    role_playing = RolePlaying(
        assistant_role_name="AI Assistant",
        assistant_agent_kwargs=dict(model=model),
        user_role_name="AI User",
        user_agent_kwargs=dict(model=model),
        task_prompt="Perform the task",
        task_specify_agent_kwargs=dict(model=model),
        task_type=task_type,
        extend_sys_msg_meta_dicts=extend_sys_msg_meta_dicts,
        extend_task_specify_meta_dict=extend_task_specify_meta_dict,
    )
    init_assistant_msg = BaseMessage.make_assistant_message(
        role_name="AI Assistant", content="Hello"
    )
    print(role_playing.assistant_agent.system_message)
    print(role_playing.user_agent.system_message)

    for i in range(step_call_count):
        assistant_response, user_response = role_playing.step(
            init_assistant_msg
        )

        for response in (assistant_response, user_response):
            assert isinstance(
                response.msgs, list
            ), f"Error in round {i+1}: response.msgs is not a list"
            assert (
                len(response.msgs) == 1
            ), f"Error in round {i+1}: len(response.msgs) is not 1"
            assert isinstance(
                response.msgs[0], BaseMessage
            ), f"Error in round {i+1}: response.msgs[0] is not a BaseMessage"
            assert isinstance(
                response.terminated, bool
            ), f"Error in round {i+1}: response.terminated is not a bool"
            assert (
                response.terminated is False
            ), f"Error in round {i+1}: response.terminated is not False"
            assert isinstance(
                response.info, dict
            ), f"Error in round {i+1}: response.info is not a dict"


@pytest.mark.model_backend
def test_role_playing_with_function(step_call_count=3):
    if model is not None:
        model._run = MagicMock(return_value=model_backend_rsp)

    tools = MathToolkit().get_tools()

    role_playing = RolePlaying(
        assistant_role_name="AI Assistant",
        assistant_agent_kwargs=dict(
            model=model,
            tools=tools,
        ),
        user_role_name="AI User",
        user_agent_kwargs=dict(model=model),
        task_prompt="Perform the task",
        task_specify_agent_kwargs=dict(model=model),
        task_type=TaskType.AI_SOCIETY,
    )

    input_msg = role_playing.init_chat()
    for i in range(step_call_count):
        assistant_response, user_response = role_playing.step(input_msg)
        for response in (assistant_response, user_response):
            assert isinstance(
                response.msgs, list
            ), f"Error in round {i+1}: response.msgs is not a list"
            assert (
                len(response.msgs) == 1
            ), f"Error in round {i+1}: len(response.msgs) is not 1"
            assert isinstance(
                response.msgs[0], BaseMessage
            ), f"Error in round {i+1}: response.msgs[0] is not a BaseMessage"
            assert isinstance(
                response.terminated, bool
            ), f"Error in round {i+1}: response.terminated is not a bool"
            assert (
                response.terminated is False
            ), f"Error in round {i+1}: response.terminated is not False"
            assert isinstance(
                response.info, dict
            ), f"Error in round {i+1}: response.info is not a dict"


def test_role_playing_role_sequence(
    model=None,
):
    task_prompt = "Develop a trading bot for the stock market"
    role_playing = RolePlaying(
        assistant_role_name="Python Programmer",
        assistant_agent_kwargs=dict(model=model),
        user_role_name="Stock Trader",
        user_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=True,
        task_specify_agent_kwargs=dict(model=model),
    )
    role_playing.user_agent.model_backend.run = MagicMock(
        return_value=model_backend_rsp
    )
    role_playing.assistant_agent.model_backend.run = MagicMock(
        return_value=model_backend_rsp
    )

    assistant_role_sequence = []
    user_role_sequence = []

    input_msg = role_playing.init_chat()
    assistant_response, _ = role_playing.step(input_msg)
    input_msg = assistant_response.msg
    assistant_response, _ = role_playing.step(input_msg)

    for record in role_playing.user_agent.memory.get_context()[0]:
        user_role_sequence.append(record["role"])
    for record in role_playing.assistant_agent.memory.get_context()[0]:
        assistant_role_sequence.append(record["role"])

    assert user_role_sequence == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert assistant_role_sequence == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
