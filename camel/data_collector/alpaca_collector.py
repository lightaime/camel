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

from typing import Any, Dict, List, Optional, Union

from typing_extensions import Self

from camel.agents import ChatAgent
from camel.data_collector.base import BaseDataCollector
from camel.messages import AlpacaItem, BaseMessage
from camel.schemas import OpenAISchemaConverter

# ruff: noqa: E501
DEFAULT_CONVERTER_PROMPTS = """
    Extract key entities and attributes from the conversations
    and convert them into a structured JSON format.
    For example:
    Instruction: You are a helpful assistant. 
    User: When is the release date of the video game Portal?
    Assistant: The release date of the video game Portal is October 9.
    Your output should be:
    {
        "instruction": "You are a helpful assistant. When is the release date of the video game Portal?",
        "input": "",
        "output": "The release date of the video game Portal is October 9."
    }
"""


class AlpacaDataCollector(BaseDataCollector):
    def __init__(self) -> None:
        super().__init__()
        self.system_message: Optional[BaseMessage] = None
        self.agent_name: Optional[str] = None

    def record(
        self,
        agent: Union[List[ChatAgent], ChatAgent],
    ) -> Self:
        r"""Inject an agent into the data collector.

        Args:
            agent (Union[List[ChatAgent], ChatAgent]):
                The agent to inject.
        """
        if not self.agent_name:
            _agent = agent if isinstance(agent, ChatAgent) else agent[0]
            self.agent_name = _agent.role_name
            self.system_message = _agent._system_message
        super().record(agent)
        return self

    def convert(self) -> Dict[str, Any]:
        r"""Convert the collected data into a dictionary."""
        if self.agent_name is None:
            raise ValueError("No agent injected")

        history = self.get_agent_history(self.agent_name)
        if not history:
            raise ValueError("No data collected.")

        # Validate and process history
        if len(history) == 3 and history[0].role == "system":
            history = history[1:]  # Ignore the system message.
        elif len(history) != 2:
            raise ValueError(
                f"AlpacaDataCollector only supports one message pair, but "
                f"got {len(history)}"
            )

        input_message, output_message = history
        instruction = (
            self.system_message.content.text if self.system_message else ""
        ) + str(input_message.message)

        data = {
            "instruction": instruction,
            "input": "",
            "output": output_message.message,
        }
        self.data.append(data)
        return data

    def llm_convert(
        self,
        converter: Optional[OpenAISchemaConverter] = None,
        prompt: Optional[str] = None,
    ) -> Dict[str, str]:
        r"""Convert collected data using an LLM schema converter.

        Args:
            converter (Optional[OpenAISchemaConverter], optional):
                The converter to use. (default: :obj:`OpenAISchemaConverter`)
            prompt (Optional[str], optional): Prompt to guide the conversion.
                (default: :obj:`DEFAULT_CONVERTER_PROMPTS`)

        Returns:
            Dict[str, str]: The converted data.

        Raises:
            ValueError: If no agent is injected or data cannot be collected.
        """
        prompt = prompt or DEFAULT_CONVERTER_PROMPTS
        converter = converter or OpenAISchemaConverter()

        system = self.system_message.content if self.system_message else ""
        context = [f"Instruction: {system}\n"]

        for message in self.get_agent_history(str(self.agent_name)):
            if message.role == "user":
                context.append(f"User: {message.message}\n")
            else:
                context.append(f"{message.name}: {message.message}\n")
        return converter.convert(
            "\n".join(context), AlpacaItem, prompt=prompt
        ).model_dump()
