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
from typing import Any, List, Optional

from colorama import Fore

from camel.agents import BaseToolAgent, ChatAgent
from camel.interpreters import (
    BaseInterpreter,
    InternalPythonInterpreter,
    SubprocessInterpreter,
)
from camel.messages import BaseMessage
from camel.responses import ChatAgentResponse
from camel.types import ModelType
from camel.utils import print_text_animated


class EmbodiedAgent(ChatAgent):
    r"""Class for managing conversations of CAMEL Embodied Agents.

    Args:
        system_message (BaseMessage): The system message for the chat agent.
        model_type (ModelType, optional): The LLM model to use for generating
            responses. (default :obj:`ModelType.GPT_4`)
        model_config (Any, optional): Configuration options for the LLM model.
            (default: :obj:`None`)
        message_window_size (int, optional): The maximum number of previous
            messages to include in the context window. If `None`, no windowing
            is performed. (default: :obj:`None`)
        tool_agents (List[BaseToolAgent], optinal): The tools agents to use in
            the emboied agent. (default: :obj:`None`)
        code_interpreter (BaseInterpreter, optional): The code interpreter to
            execute codes. If `code_interpreter` and `tool_agent` are both
            `None`, default to `SubProcessInterpreter`. If
            `code_interpreter=None` and `tool_agents!=None`, default to
            `InternalPythonInterpreter`.  (default: :obj:`None`)
        verbose (bool, optional): Whether to print the critic's messages.
        logger_color (Any): The color of the logger displayed to the user.
            (default: :obj:`Fore.MAGENTA`)
    """

    def __init__(
        self,
        system_message: BaseMessage,
        model_type: ModelType = ModelType.GPT_4,
        model_config: Optional[Any] = None,
        message_window_size: Optional[int] = None,
        tool_agents: Optional[List[BaseToolAgent]] = None,
        code_interpreter: Optional[BaseInterpreter] = None,
        verbose: bool = False,
        logger_color: Any = Fore.MAGENTA,
    ) -> None:
        self.tool_agents = tool_agents
        self.code_interpreter: BaseInterpreter
        if code_interpreter is not None:
            self.code_interpreter = code_interpreter
        elif self.tool_agents:
            self.code_interpreter = InternalPythonInterpreter()
        else:
            self.code_interpreter = SubprocessInterpreter()

        if self.tool_agents:
            system_message = self._set_tool_agents(system_message)
        self.verbose = verbose
        self.logger_color = logger_color
        super().__init__(
            system_message=system_message,
            model_type=model_type,
            model_config=model_config,
            message_window_size=message_window_size,
        )

    def _set_tool_agents(self, system_message: BaseMessage) -> BaseMessage:
        action_space_prompt = self._get_tool_agents_prompt()
        system_message.content = system_message.content.format(
            action_space=action_space_prompt)
        if self.tool_agents is not None:
            self.code_interpreter.set_action_space(
                {tool.name: tool
                 for tool in self.tool_agents})
        return system_message

    def _get_tool_agents_prompt(self) -> str:
        r"""Returns the action space prompt.

        Returns:
            str: The action space prompt.
        """
        if self.tool_agents is not None:
            return "\n".join([
                f"*** {tool.name} ***:\n {tool.description}"
                for tool in self.tool_agents
            ])
        else:
            return ""

    def step(
        self,
        input_message: BaseMessage,
    ) -> ChatAgentResponse:
        r"""Performs a step in the conversation.

        Args:
            input_message (BaseMessage): The input message.

        Returns:
            ChatAgentResponse: A struct containing the output messages,
                a boolean indicating whether the chat session has terminated,
                and information about the chat session.
        """
        response = super().step(input_message)

        if response.msgs is None or len(response.msgs) == 0:
            raise RuntimeError("Got None output messages.")
        if response.terminated:
            raise RuntimeError(f"{self.__class__.__name__} step failed.")

        # NOTE: Only single output messages are supported
        explanations, codes = response.msg.extract_text_and_code_prompts()

        if self.verbose:
            for explanation, code in zip(explanations, codes):
                print_text_animated(self.logger_color +
                                    f"> Explanation:\n{explanation}")
                print_text_animated(self.logger_color + f"> Code:\n{code}")

            if len(explanations) > len(codes):
                print_text_animated(self.logger_color +
                                    f"> Explanation:\n{explanations}")

        content = response.msg.content

        if codes is not None:
            try:
                content = "\n> Executed Results:"
                for block_idx, code in enumerate(codes):
                    executed_output = self.code_interpreter.run(
                        code, code.code_type)
                    content += (f"Executing code block {block_idx}:\n" +
                                executed_output + "\n")
            except InterruptedError as e:
                content = (f"\n> Running code fail: {e}\n"
                           "Please regenerate the code.")

        # TODO: Handle errors
        content = input_message.content + f"\n> Embodied Actions:\n{content}"
        message = BaseMessage(input_message.role_name, input_message.role_type,
                              input_message.meta_dict, content)
        return ChatAgentResponse([message], response.terminated, response.info)
