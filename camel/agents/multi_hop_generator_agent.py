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

import textwrap
from typing import Any

from pydantic import ConfigDict

from camel.agents.programmed_agent_instruction import (
    ProgrammableChatAgent,
    ProgrammedAgentInstructionResult,
    programmable_capability,
)
from camel.datagen.source2synth.models import (
    ContextPrompt,
    MultiHopQA,
)
from camel.messages import BaseMessage


class MultiHopGeneratorAgent(ProgrammableChatAgent):
    r"""An agent specialized in generating multi-hop question-answer pairs.

    This agent is designed to create complex questions that require multiple
    steps of reasoning to answer. It analyzes context to identify related
    facts and generates questions that require connecting these facts
    logically.

    Attributes:
        model_config (ConfigDict): Configuration for model behavior.
        system_message (BaseMessage): System message defining agent's role and
            instructions.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs: Any) -> None:
        r"""Initialize the MultiHopGeneratorAgent.

        Args:
            **kwargs (Any): Additional keyword arguments to pass to parent
                class.
        """
        super().__init__(**kwargs)

        system_text: str = textwrap.dedent(
            """\
            You are an expert at generating 
            multi-hop question-answer pairs.
            For each context, you should:
            1. Identify multiple related facts or pieces of information
            2. Create questions that require reasoning across these multiple pieces
            3. Ensure the reasoning chain is clear and logical
            4. Generate questions that require at least 2-3 steps of reasoning
            5. Include the reasoning steps in the answer

            Give your response with this information:
            Question: [Complex question requiring multiple reasoning steps]
            Reasoning Steps:
            1. [First reasoning step]
            2. [Second reasoning step]
            3. [Final reasoning step]
            Answer: [Final answer]
            Supporting Facts: [List of relevant text segments used]
            """  # noqa: E501
        )
        self._system_message = BaseMessage.make_assistant_message(
            role_name='Assistant', content=system_text
        )

    @programmable_capability
    def generate_multi_hop_qa(
        self, context: str
    ) -> ProgrammedAgentInstructionResult[MultiHopQA]:
        r"""Generate a multi-hop question-answer pair from given context.

        Args:
            context (str): The input text context to generate QA from.

        Returns:
            ProgrammedAgentInstructionResult[MultiHopQA]: Result containing the
                generated question, reasoning steps, answer, and supporting
                facts.

        Raises:
            RuntimeError: If the agent fails to generate a response.
        """
        context_prompt = ContextPrompt(
            main_context=context, related_contexts=None
        )

        user_message = BaseMessage.make_user_message(
            content=context_prompt.model_dump_json(), role_name="User"
        )
        response = self.step(
            input_message=user_message, response_format=MultiHopQA
        )
        value = MultiHopQA.model_validate_json(response.msgs[0].content)

        if response.msgs:
            return ProgrammedAgentInstructionResult(
                user_message=user_message,
                agent_message=response.msgs[0],
                value=value,
            )
        raise RuntimeError("No response from agent")
