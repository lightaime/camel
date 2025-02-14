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
import time
from typing import Any, Dict, List, Optional, Type, Union

from openai import AsyncStream, Stream
from pydantic import BaseModel

from camel.messages import OpenAIMessage
from camel.models import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
    ModelType,
)
from camel.utils import BaseTokenCounter


class StubTokenCounter(BaseTokenCounter):
    def count_tokens_from_messages(self, messages: List[OpenAIMessage]) -> int:
        r"""Token counting for STUB models, directly returning a constant.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            int: A constant to act as the number of the tokens in the
                messages.
        """
        return 10


class StubModel(BaseModelBackend):
    r"""A dummy model used for unit tests."""

    model_type = ModelType.STUB

    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
    ) -> None:
        r"""All arguments are unused for the dummy model."""
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter
        )

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = StubTokenCounter()
        return self._token_counter

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        r"""Run fake inference by returning a fixed string.
        All arguments are unused for the dummy model.

        Returns:
            Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
                The response from the dummy model.
        """
        ARBITRARY_STRING = "Lorem Ipsum"
        response: ChatCompletion = ChatCompletion(
            id="stub_model_id",
            model="stub",
            object="chat.completion",
            created=int(time.time()),
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content=ARBITRARY_STRING,
                        role="assistant",
                    ),
                    logprobs=None,
                )
            ],
            usage=CompletionUsage(
                completion_tokens=10,
                prompt_tokens=10,
                total_tokens=20,
            ),
        )
        return response

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        r"""Run fake inference by returning a fixed string.
        All arguments are unused for the dummy model.

        Returns:
            Dict[str, Any]: Response in the OpenAI API format.
        """
        ARBITRARY_STRING = "Lorem Ipsum"
        response: ChatCompletion = ChatCompletion(
            id="stub_model_id",
            model="stub",
            object="chat.completion",
            created=int(time.time()),
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content=ARBITRARY_STRING,
                        role="assistant",
                    ),
                    logprobs=None,
                )
            ],
            usage=CompletionUsage(
                completion_tokens=10,
                prompt_tokens=10,
                total_tokens=20,
            ),
        )
        return response

    def check_model_config(self):
        r"""Directly pass the check on arguments to STUB model."""
        pass
