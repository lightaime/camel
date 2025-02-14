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

from camel.configs import DEEPSEEK_API_PARAMS, DeepSeekConfig
from camel.logger import get_logger
from camel.messages import OpenAIMessage
from camel.models._utils import try_modify_message_with_format
from camel.models.base_model import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ModelType,
)
from camel.utils import BaseTokenCounter, OpenAITokenCounter, api_keys_required

logger = get_logger(__name__)

REASONSER_UNSUPPORTED_PARAMS = [
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    "top_logprobs",
    "tools",
]


class DeepSeekModel(BaseModelBackend):
    r"""DeepSeek API in a unified BaseModelBackend interface.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into:obj:`openai.ChatCompletion.create()`. If
            :obj:`None`, :obj:`DeepSeekConfig().as_dict()` will be used.
            (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating with
            the DeepSeek service. (default: :obj:`None`)
        url (Optional[str], optional): The url to the DeepSeek service.
            (default: :obj:`https://api.deepseek.com`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter`
            will be used. (default: :obj:`None`)

    References:
        https://api-docs.deepseek.com/
    """

    @api_keys_required(
        [
            ("api_key", "DEEPSEEK_API_KEY"),
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
            model_config_dict = DeepSeekConfig().as_dict()
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        url = url or os.environ.get(
            "DEEPSEEK_API_BASE_URL",
            "https://api.deepseek.com",
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

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(
                model=ModelType.GPT_4O_MINI
            )
        return self._token_counter

    def _prepare_request(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        request_config = self.model_config_dict.copy()

        if self.model_type in [
            ModelType.DEEPSEEK_REASONER,
        ]:
            import re

            # Remove thinking content from messages before sending to API
            # This ensures only the final response is sent, excluding
            # intermediate thought processes
            messages = [
                {  # type: ignore[misc]
                    **msg,
                    'content': re.sub(
                        r'<think>.*?</think>',
                        '',
                        msg['content'],  # type: ignore[arg-type]
                        flags=re.DOTALL,
                    ).strip(),
                }
                for msg in messages
            ]

            logger.warning(
                "Warning: You are using an DeepSeek Reasoner model, "
                "which has certain limitations, reference: "
                "`https://api-docs.deepseek.com/guides/reasoning_model"
                "#api-parameters`.",
            )
            request_config = {
                key: value
                for key, value in request_config.items()
                if key not in REASONSER_UNSUPPORTED_PARAMS
            }

        if tools:
            for tool in tools:
                function_dict = tool.get('function', {})
                function_dict.pop("strict", None)
            request_config["tools"] = tools
        elif response_format:
            try_modify_message_with_format(messages[-1], response_format)
            request_config["response_format"] = {"type": "json_object"}

        return request_config

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        r"""Runs inference of DeepSeek chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            Union[ChatCompletion, Stream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `Stream[ChatCompletionChunk]` in the stream mode.
        """
        request_config = self._prepare_request(
            messages, response_format, tools
        )

        response = self._client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **request_config,
        )

        return response

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        r"""Runs inference of DeepSeek chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `AsyncStream[ChatCompletionChunk]` in the stream mode.
        """
        request_config = self._prepare_request(
            messages, response_format, tools
        )
        response = await self._async_client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **request_config,
        )

        # Handle reasoning content with <think> tags at the beginning
        if (
            self.model_type
            in [
                ModelType.DEEPSEEK_REASONER,
            ]
            and os.environ.get("GET_REASONING_CONTENT", "false").lower()
            == "true"
        ):
            reasoning_content = response.choices[0].message.reasoning_content
            combined_content = (
                f"<think>\n{reasoning_content}\n</think>\n"
                if reasoning_content
                else ""
            ) + response.choices[0].message.content

            response = ChatCompletion.construct(
                id=response.id,
                choices=[
                    dict(
                        index=response.choices[0].index,
                        message={
                            "role": response.choices[0].message.role,
                            "content": combined_content,
                            "tool_calls": None,
                        },
                        finish_reason=response.choices[0].finish_reason
                        if response.choices[0].finish_reason
                        else None,
                    )
                ],
                created=response.created,
                model=response.model,
                object="chat.completion",
                usage=response.usage,
            )

        return response

    def check_model_config(self):
        r"""Check whether the model configuration contains any
        unexpected arguments to DeepSeek API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to DeepSeek API.
        """
        for param in self.model_config_dict:
            if param not in DEEPSEEK_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into DeepSeek model backend."
                )

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode, which sends partial
        results each time.

        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get("stream", False)
