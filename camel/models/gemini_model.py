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
import os
from typing import Any, Dict, List, Optional, Type, Union

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from pydantic import BaseModel

from camel.configs import Gemini_API_PARAMS, GeminiConfig
from camel.messages import OpenAIMessage
from camel.models import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ModelType,
)
from camel.utils import (
    BaseTokenCounter,
    OpenAITokenCounter,
    api_keys_required,
)


class GeminiModel(BaseModelBackend):
    r"""Gemini API in a unified BaseModelBackend interface.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created, one of Gemini series.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into:obj:`openai.ChatCompletion.create()`. If
            :obj:`None`, :obj:`GeminiConfig().as_dict()` will be used.
            (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating with
            the Gemini service. (default: :obj:`None`)
        url (Optional[str], optional): The url to the Gemini service.
            (default: :obj:`https://generativelanguage.googleapis.com/v1beta/
            openai/`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter(
            ModelType.GPT_4O_MINI)` will be used.
            (default: :obj:`None`)
    """

    @api_keys_required(
        [
            ("api_key", 'GEMINI_API_KEY'),
        ]
    )
    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
    ) -> None:
        if model_config_dict is None:
            model_config_dict = GeminiConfig().as_dict()
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        url = url or os.environ.get(
            "GEMINI_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter
        )
        self._client = OpenAI(
            timeout=180,
            max_retries=3,
            api_key=self._api_key,
            base_url=self._url,
        )
        self._async_client = AsyncOpenAI(
            timeout=180,
            max_retries=3,
            api_key=self._api_key,
            base_url=self._url,
        )

    def _process_messages(self, messages) -> List[OpenAIMessage]:
        r"""Process the messages for Gemini API to ensure no empty content,
        which is not accepted by Gemini.
        """
        processed_messages = []
        for msg in messages:
            msg_copy = msg.copy()
            if 'content' in msg_copy and msg_copy['content'] == '':
                msg_copy['content'] = 'null'
            processed_messages.append(msg_copy)
        return processed_messages

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        r"""Runs inference of Gemini chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.
            response_format (Optional[Type[BaseModel]]): The format of the
                response.
            tools (Optional[List[Dict[str, Any]]]): The schema of the tools to
                use for the request.

        Returns:
            Union[ChatCompletion, Stream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `Stream[ChatCompletionChunk]` in the stream mode.
        """
        response_format = response_format or self.model_config_dict.get(
            "response_format", None
        )
        messages = self._process_messages(messages)
        if response_format:
            return self._request_parse(messages, response_format)
        else:
            return self._request_chat_completion(messages, tools)

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        r"""Runs inference of OpenAI chat completion in async mode.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.
            response_format (Optional[Type[BaseModel]]): The format of the
                response.
            tools (Optional[List[Dict[str, Any]]]): The schema of the tools to
                use for the request.

        Returns:
            Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `AsyncStream[ChatCompletionChunk]` in the stream mode.
        """
        response_format = response_format or self.model_config_dict.get(
            "response_format", None
        )
        messages = self._process_messages(messages)
        if response_format:
            return await self._arequest_parse(messages, response_format)
        else:
            return await self._arequest_chat_completion(messages, tools)

    def _request_chat_completion(
        self,
        messages: List[OpenAIMessage],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        request_config = self.model_config_dict.copy()

        if tools:
            for tool in tools:
                function_dict = tool.get('function', {})
                function_dict.pop("strict", None)
            request_config["tools"] = tools

        return self._client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **request_config,
        )

    async def _arequest_chat_completion(
        self,
        messages: List[OpenAIMessage],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        request_config = self.model_config_dict.copy()

        if tools:
            for tool in tools:
                function_dict = tool.get('function', {})
                function_dict.pop("strict", None)
            request_config["tools"] = tools

        return await self._async_client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **request_config,
        )

    def _request_parse(
        self,
        messages: List[OpenAIMessage],
        response_format: Type[BaseModel],
    ) -> ChatCompletion:
        request_config = self.model_config_dict.copy()

        request_config["response_format"] = response_format
        request_config.pop("stream", None)

        return self._client.beta.chat.completions.parse(
            messages=messages,
            model=self.model_type,
            **request_config,
        )

    async def _arequest_parse(
        self,
        messages: List[OpenAIMessage],
        response_format: Type[BaseModel],
    ) -> ChatCompletion:
        request_config = self.model_config_dict.copy()

        request_config["response_format"] = response_format
        request_config.pop("stream", None)

        return await self._async_client.beta.chat.completions.parse(
            messages=messages,
            model=self.model_type,
            **request_config,
        )

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(ModelType.GPT_4O_MINI)
        return self._token_counter

    def check_model_config(self):
        r"""Check whether the model configuration contains any
        unexpected arguments to Gemini API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to Gemini API.
        """
        for param in self.model_config_dict:
            if param not in Gemini_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into Gemini model backend."
                )

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode, which sends partial
        results each time.

        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get('stream', False)
