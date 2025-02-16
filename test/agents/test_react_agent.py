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
from unittest.mock import MagicMock, PropertyMock

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from camel.agents import ReActAgent
from camel.messages import BaseMessage
from camel.types import RoleType


@pytest.fixture
def system_message():
    return BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content=(
            "You are a helpful assistant that uses reasoning and actions "
            "to complete tasks."
        ),
    )


# Mock response for testing
model_backend_rsp = ChatCompletion(
    id="mock_response_id",
    choices=[
        Choice(
            finish_reason="stop",
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content=(
                    "Thought: I need to search for information\n"
                    "Action: Search(query=test query)\n"
                    "Observation: Found test results"
                ),
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


def setup_mocked_agent(system_message, mock_model=None, max_steps=None):
    """Helper function to setup a mocked agent with proper memory mocking."""
    if mock_model is None:
        mock_model = MagicMock()
        mock_model.run = MagicMock(return_value=model_backend_rsp)

    agent = ReActAgent(
        system_message=system_message,
        model=mock_model,
        max_steps=max_steps or 10,
    )

    # Mock the memory components
    context_creator = agent.memory.get_context_creator()
    type(context_creator).token_limit = PropertyMock(return_value=4096)
    token_counter = agent.memory.get_context_creator().token_counter
    token_counter.count_tokens_from_messages = MagicMock(return_value=10)
    agent.step_count = 0

    return agent


@pytest.mark.model_backend
def test_react_agent_init():
    """Test ReActAgent initialization."""
    system_message = BaseMessage(
        role_name="assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict=None,
        content="You are a helpful assistant.",
    )

    tools = [
        lambda x: "Test result",
    ]

    agent = ReActAgent(
        system_message=system_message,
        tools=tools,
        max_steps=5,
    )

    assert agent.tools == tools
    assert agent.max_steps == 5
    assert len(agent.scratchpad) == 0
    assert agent.step_count == 0


@pytest.mark.model_backend
def test_react_agent_parse_components(system_message):
    """Test ReActAgent's ability to parse response components."""
    agent = ReActAgent(system_message=system_message)

    test_content = (
        "Thought: I should search for information\n"
        "Action: Search(query=test)\n"
        "Observation: Found results"
    )

    thought, action, observation = agent._parse_react_components(test_content)

    assert thought == "I should search for information"
    assert action == "Search(query=test)"
    assert observation == "Found results"


@pytest.mark.model_backend
def test_react_agent_execute_action(system_message):
    """Test ReActAgent's action execution."""
    mock_tool = MagicMock()
    mock_tool.can_handle = MagicMock(return_value=True)
    mock_tool.execute = MagicMock(return_value="Test result")

    agent = ReActAgent(system_message=system_message, tools=[mock_tool])

    result = agent._execute_action("Search(query=test)")
    assert result == "Test result"

    # Test Finish action
    result = agent._execute_action("Finish(answer=Done)")
    assert result == "Task completed."


@pytest.mark.model_backend
def test_react_agent_step(system_message, step_call_count=3):
    """Test ReActAgent's step function."""
    mock_tool = MagicMock()
    mock_tool.can_handle = MagicMock(return_value=True)
    mock_tool.execute = MagicMock(return_value="Test result")

    agent = setup_mocked_agent(
        system_message=system_message,
        max_steps=10,
    )
    agent.tools = [mock_tool]

    input_message = BaseMessage(
        role_name="user",
        role_type=RoleType.USER,
        meta_dict={},
        content="Please help me find information.",
    )

    for i in range(step_call_count):
        response = agent.step(input_message)

        assert isinstance(response.msgs, list), f"Error in round {i+1}"
        assert len(response.msgs) == 1, f"Error in round {i+1}"
        assert isinstance(
            response.msgs[0], BaseMessage
        ), f"Error in round {i+1}"
        assert isinstance(response.terminated, bool), f"Error in round {i+1}"
        assert isinstance(response.info, dict), f"Error in round {i+1}"
        assert "thought" in response.info, f"Error in round {i+1}"
        assert "action" in response.info, f"Error in round {i+1}"
        assert "observation" in response.info, f"Error in round {i+1}"


@pytest.mark.model_backend
def test_react_agent_max_steps(system_message):
    """Test ReActAgent's max steps limit."""
    agent = setup_mocked_agent(system_message=system_message, max_steps=1)

    input_message = BaseMessage(
        role_name="user",
        role_type=RoleType.USER,
        meta_dict={},
        content="Please help me find information.",
    )

    # First step should work normally
    response = agent.step(input_message)
    assert not response.terminated, "First step should not terminate"

    # Second step should terminate due to max steps
    response = agent.step(input_message)
    assert response.terminated, "Second step should terminate"
    assert response.info["thought"] == "Maximum steps reached"
    assert response.info["observation"] == "Task terminated due to step limit"
