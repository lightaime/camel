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
from typing import Any, Optional

from camel.agents.chat_agent import ChatAgent
from camel.messages import BaseMessage
from camel.prompts import TextPrompt
from camel.types import ModelType, RoleType
from camel.utils import create_chunks


class SearchAgent(ChatAgent):
    r"""An agent that summarizes text based on a query and evaluates the
    relevance of an answer.

    Attributes:
        role_assignment_prompt (TextPrompt): A prompt for the agent to
            generate role names.

    Args:
        model_type (ModelType, optional): The type of model to use for the
            agent. (default: :obj:`ModelType.GPT_3_5_TURBO`)
        model_config (Any, optional): The configuration for the model.
            (default: :obj:`None`)
    """

    def __init__(
        self,
        model_type: ModelType = ModelType.GPT_3_5_TURBO,
        model_config: Optional[Any] = None,
    ) -> None:
        system_message = BaseMessage(
            role_name="Assistant",
            role_type=RoleType.ASSISTANT,
            meta_dict=None,
            content="You are a helpful assistant.",
        )
        super().__init__(system_message, model_type, model_config)

    def summarize_text(self, text: str, query: str) -> str:
        r"""Summarize the information from the text, base on the query.

        Args:
            text (str): Text to summarize.
            query (str): What information you want.

        Returns:
            str: Strings with information.
        """
        self.reset()

        summary_prompt = TextPrompt(
            '''Gather information from this text that relative to the
            question, but do not directly answer the question.\nquestion:
            {query}\ntext '''
        )
        summary_prompt = summary_prompt.format(query=query)
        # Max length of each chunk
        max_len = 3000
        results = ""
        chunks = create_chunks(text, max_len)
        # Summarize
        for i, chunk in enumerate(chunks, start=1):
            prompt = summary_prompt + str(i) + ": " + chunk
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=prompt,
            )
            result = self.step(user_msg).msg.content
            results += result + "\n"

        # Final summarise
        final_prompt = TextPrompt(
            '''Here are some summarized texts which split from one text. Using
            the information to answer the question. If can't find the answer,
            you must answer "I can not find the answer to the query" and
            explain why.\n Query:\n{query}.\n\nText:\n'''
        )
        final_prompt = final_prompt.format(query=query)
        prompt = final_prompt + results

        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=prompt,
        )
        response = self.step(user_msg).msg.content

        return response

    def continue_search(self, query: str, answer: str) -> bool:
        r"""Ask whether to continue search or not based on the provided answer.

        Args:
            query (str): The question.
            answer (str): The answer to the question.

        Returns:
            bool: `True` if the user want to continue search, `False`
            otherwise.
        """
        from camel.prompts import TextPrompt

        prompt = TextPrompt(
            "Do you think the ANSWER can answer the QUERY? "
            "Use only 'yes' or 'no' to answer.\n"
            "===== QUERY =====\n{query}\n\n"
            "===== ANSWER =====\n{answer}"
        )
        prompt = prompt.format(query=query, answer=answer)
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=prompt,
        )
        response = self.step(user_msg).msg.content
        if "yes" in str(response).lower():
            return False
        return True
