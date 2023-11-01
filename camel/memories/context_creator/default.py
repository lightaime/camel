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
from dataclasses import dataclass
from typing import List, Tuple

from camel.memories.context_creator.base import BaseContextCreator
from camel.memories.memory_record import ContextRecord
from camel.messages import OpenAIMessage
from camel.utils.token_counting import BaseTokenCounter


@dataclass(frozen=True)
class _ContextUnit():
    idx: int
    record: ContextRecord
    token_num: int


class DefaultContextCreator(BaseContextCreator):
    r"""A default implementation of context creation strategy, which inherits from
    :obj:`BaseContextCreator`.

    This class provides a strategy to generate a conversational context from
    a list of chat history records while ensuring the total token count of
    the context does not exceed a specified limit. It prunes messages based
    on their importance if the total token count exceeds the limit.

    Args:
        token_counter (BaseTokenCounter): An instance responsible for counting
            tokens in a message.
        token_limit (int): The maximum number of tokens allowed in the
            generated context.
    """

    def __init__(self, token_counter: BaseTokenCounter,
                 token_limit: int) -> None:
        self._token_counter = token_counter
        self._token_limit = token_limit

    @property
    def token_counter(self) -> BaseTokenCounter:
        return self._token_counter

    @property
    def token_limit(self) -> int:
        return self._token_limit

    def create_context(
            self,
            records: List[ContextRecord]) -> Tuple[List[OpenAIMessage], int]:
        r"""
        Creates conversational context from chat history while respecting token
        limits.

        Constructs the context from provided records and ensures that the total
        token count does not exceed the specified limit by pruning the least
        important messages if necessary.

        Args:
            records (List[ContextRecord]): A list of message records from which
                to generate the context.

        Returns:
            Tuple[List[OpenAIMessage], int]: A tuple containing the constructed
                context in OpenAIMessage format and the total token count.

        Raises:
            RuntimeError: If it's impossible to create a valid context without
                exceeding the token limit.
        """
        context_units = [
            _ContextUnit(
                idx,
                r,
                self.token_counter.count_tokens_from_messages(
                    [r.m_record.to_openai_message()]),
            ) for idx, r in enumerate(records)
        ]
        # TODO: optimize the process, may give information back to memory

        # If not exceed token limit, simply return
        total_tokens = sum([unit.token_num for unit in context_units])
        if total_tokens <= self.token_limit:
            return self._create_output(context_units)

        # Sort by importance
        context_units = sorted(context_units,
                               key=lambda unit: unit.record.importance)

        # Remove least important messages until total token number < token limit
        flag = None
        for i, unit in enumerate(context_units):
            if unit.record.importance == 1:
                raise RuntimeError(
                    "Cannot create context: exceed token limit.", total_tokens)
            total_tokens -= unit.token_num
            if total_tokens <= self.token_limit:
                flag = i
                break
        if flag is None:
            raise RuntimeError("Cannot create context: exceed token limit.",
                               total_tokens)
        return self._create_output(context_units[flag + 1:])

    def _create_output(
            self, context_units: List[_ContextUnit]
    ) -> Tuple[List[OpenAIMessage], int]:
        r"""
        Helper method to generate output from context units.

        This method converts the provided context units into a format suitable
        for output, specifically a list of OpenAIMessages and an integer
        representing the total token count.
        """
        context_units = sorted(context_units, key=lambda unit: unit.idx)
        return [
            unit.record.m_record.to_openai_message() for unit in context_units
        ], sum([unit.token_num for unit in context_units])
