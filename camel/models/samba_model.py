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
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Type, Union

import httpx
from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from pydantic import BaseModel

from camel.configs import (
    SAMBA_CLOUD_API_PARAMS,
    SAMBA_VERSE_API_PARAMS,
    SambaCloudAPIConfig,
)
from camel.messages import OpenAIMessage
from camel.models import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionUsage,
    ModelType,
)
from camel.utils import (
    BaseTokenCounter,
    OpenAITokenCounter,
    api_keys_required,
)

try:
    if os.getenv("AGENTOPS_API_KEY") is not None:
        from agentops import LLMEvent, record
    else:
        raise ImportError
except (ImportError, AttributeError):
    LLMEvent = None


class SambaModel(BaseModelBackend):
    r"""SambaNova service interface.

    Args:
        model_type (Union[ModelType, str]): Model for which a SambaNova backend
            is created. Supported models via SambaNova Cloud:
            `https://community.sambanova.ai/t/supported-models/193`.
            Supported models via SambaVerse API is listed in
            `https://sambaverse.sambanova.ai/models`.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into:obj:`openai.ChatCompletion.create()`. If
            :obj:`None`, :obj:`SambaCloudAPIConfig().as_dict()` will be used.
            (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating
            with the SambaNova service. (default: :obj:`None`)
        url (Optional[str], optional): The url to the SambaNova service.
            Current support SambaVerse API:
            :obj:`"https://sambaverse.sambanova.ai/api/predict"` and
            SambaNova Cloud:
            :obj:`"https://api.sambanova.ai/v1"` (default: :obj:`https://api.
            sambanova.ai/v1`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter(
            ModelType.GPT_4O_MINI)` will be used.
    """

    @api_keys_required(
        [
            ("api_key", 'SAMBA_API_KEY'),
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
            model_config_dict = SambaCloudAPIConfig().as_dict()
        api_key = api_key or os.environ.get("SAMBA_API_KEY")
        url = url or os.environ.get(
            "SAMBA_API_BASE_URL",
            "https://api.sambanova.ai/v1",
        )
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter
        )

        if self._url == "https://api.sambanova.ai/v1":
            self._client = OpenAI(
                timeout=180,
                max_retries=3,
                base_url=self._url,
                api_key=self._api_key,
            )
            self._async_client = AsyncOpenAI(
                timeout=180,
                max_retries=3,
                base_url=self._url,
                api_key=self._api_key,
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
        unexpected arguments to SambaNova API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to SambaNova API.
        """
        if self._url == "https://sambaverse.sambanova.ai/api/predict":
            for param in self.model_config_dict:
                if param not in SAMBA_VERSE_API_PARAMS:
                    raise ValueError(
                        f"Unexpected argument `{param}` is "
                        "input into SambaVerse API."
                    )

        elif self._url == "https://api.sambanova.ai/v1":
            for param in self.model_config_dict:
                if param not in SAMBA_CLOUD_API_PARAMS:
                    raise ValueError(
                        f"Unexpected argument `{param}` is "
                        "input into SambaCloud API."
                    )

        else:
            raise ValueError(
                f"{self._url} is not supported, please check the url to the"
                " SambaNova service"
            )

    async def _arun(  # type: ignore[misc]
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        r"""Runs SambaNova's service.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `AsyncStream[ChatCompletionChunk]` in the stream mode.
        """
        if "tools" in self.model_config_dict:
            del self.model_config_dict["tools"]
        if self.model_config_dict.get("stream") is True:
            return await self._arun_streaming(messages)
        else:
            return await self._arun_non_streaming(messages)

    def _run(  # type: ignore[misc]
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        r"""Runs SambaNova's service.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            Union[ChatCompletion, Stream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `Stream[ChatCompletionChunk]` in the stream mode.
        """
        if "tools" in self.model_config_dict:
            del self.model_config_dict["tools"]
        if self.model_config_dict.get("stream") is True:
            return self._run_streaming(messages)
        else:
            return self._run_non_streaming(messages)

    def _run_streaming(
        self, messages: List[OpenAIMessage]
    ) -> Stream[ChatCompletionChunk]:
        r"""Handles streaming inference with SambaNova's API.

        Args:
            messages (List[OpenAIMessage]): A list of messages representing the
                chat history in OpenAI API format.

        Returns:
            Stream[ChatCompletionChunk]: A generator yielding
                `ChatCompletionChunk` objects as they are received from the
                API.

        Raises:
            RuntimeError: If the HTTP request fails.
            ValueError: If the API doesn't support stream mode.
        """
        # Handle SambaNova's Cloud API
        if self._url == "https://api.sambanova.ai/v1":
            response = self._client.chat.completions.create(
                messages=messages,
                model=self.model_type,
                **self.model_config_dict,
            )

            # Add AgentOps LLM Event tracking
            if LLMEvent:
                llm_event = LLMEvent(
                    thread_id=response.id,
                    prompt=" ".join(
                        [message.get("content") for message in messages]  # type: ignore[misc]
                    ),
                    prompt_tokens=response.usage.prompt_tokens,  # type: ignore[union-attr]
                    completion=response.choices[0].message.content,
                    completion_tokens=response.usage.completion_tokens,  # type: ignore[union-attr]
                    model=self.model_type,
                )
                record(llm_event)

            return response

        elif self._url == "https://sambaverse.sambanova.ai/api/predict":
            raise ValueError(
                "https://sambaverse.sambanova.ai/api/predict doesn't support"
                " stream mode"
            )
        raise RuntimeError(f"Unknown URL: {self._url}")

    def _run_non_streaming(
        self, messages: List[OpenAIMessage]
    ) -> ChatCompletion:
        r"""Handles non-streaming inference with SambaNova's API.

        Args:
            messages (List[OpenAIMessage]): A list of messages representing the
                message in OpenAI API format.

        Returns:
            ChatCompletion: A `ChatCompletion` object containing the complete
                response from the API.

        Raises:
            RuntimeError: If the HTTP request fails.
            ValueError: If the JSON response cannot be decoded or is missing
                expected data.
        """
        # Handle SambaNova's Cloud API
        if self._url == "https://api.sambanova.ai/v1":
            response = self._client.chat.completions.create(
                messages=messages,
                model=self.model_type,
                **self.model_config_dict,
            )

            # Add AgentOps LLM Event tracking
            if LLMEvent:
                llm_event = LLMEvent(
                    thread_id=response.id,
                    prompt=" ".join(
                        [message.get("content") for message in messages]  # type: ignore[misc]
                    ),
                    prompt_tokens=response.usage.prompt_tokens,  # type: ignore[union-attr]
                    completion=response.choices[0].message.content,
                    completion_tokens=response.usage.completion_tokens,  # type: ignore[union-attr]
                    model=self.model_type,
                )
                record(llm_event)

            return response

        # Handle SambaNova's Sambaverse API
        else:
            headers = {
                "Content-Type": "application/json",
                "key": str(self._api_key),
                "modelName": self.model_type,
            }

            data = {
                "instance": json.dumps(
                    {
                        "conversation_id": str(uuid.uuid4()),
                        "messages": messages,
                    }
                ),
                "params": {
                    "do_sample": {"type": "bool", "value": "true"},
                    "max_tokens_to_generate": {
                        "type": "int",
                        "value": str(self.model_config_dict.get("max_tokens")),
                    },
                    "process_prompt": {"type": "bool", "value": "true"},
                    "repetition_penalty": {
                        "type": "float",
                        "value": str(
                            self.model_config_dict.get("repetition_penalty")
                        ),
                    },
                    "return_token_count_only": {
                        "type": "bool",
                        "value": "false",
                    },
                    "select_expert": {
                        "type": "str",
                        "value": self.model_type.split('/')[1],
                    },
                    "stop_sequences": {
                        "type": "str",
                        "value": self.model_config_dict.get("stop_sequences"),
                    },
                    "temperature": {
                        "type": "float",
                        "value": str(
                            self.model_config_dict.get("temperature")
                        ),
                    },
                    "top_k": {
                        "type": "int",
                        "value": str(self.model_config_dict.get("top_k")),
                    },
                    "top_p": {
                        "type": "float",
                        "value": str(self.model_config_dict.get("top_p")),
                    },
                },
            }

            try:
                # Send the request and handle the response
                with httpx.Client() as client:
                    response = client.post(
                        self._url,  # type: ignore[arg-type]
                        headers=headers,
                        json=data,
                    )

                raw_text = response.text
                # Split the string into two dictionaries
                dicts = raw_text.split('}\n{')

                # Keep only the last dictionary
                last_dict = '{' + dicts[-1]

                # Parse the dictionary
                last_dict = json.loads(last_dict)
                return self._sambaverse_to_openai_response(last_dict)  # type: ignore[arg-type]

            except httpx.HTTPStatusError:
                raise RuntimeError(f"HTTP request failed: {raw_text}")

    def _sambaverse_to_openai_response(
        self, samba_response: Dict[str, Any]
    ) -> ChatCompletion:
        r"""Converts SambaVerse API response into an OpenAI-compatible
        response.

        Args:
            samba_response (Dict[str, Any]): A dictionary representing
                responses from the SambaVerse API.

        Returns:
            ChatCompletion: A `ChatCompletion` object constructed from the
                aggregated response data.
        """
        choices = [
            dict(
                index=0,
                message={
                    "role": 'assistant',
                    "content": samba_response['result']['responses'][0][
                        'completion'
                    ],
                },
                finish_reason=samba_response['result']['responses'][0][
                    'stop_reason'
                ],
            )
        ]

        obj = ChatCompletion.construct(
            id=None,
            choices=choices,
            created=int(time.time()),
            model=self.model_type,
            object="chat.completion",
            # SambaVerse API only provide `total_tokens`
            usage=CompletionUsage(
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=int(
                    samba_response['result']['responses'][0][
                        'total_tokens_count'
                    ]
                ),
            ),
        )

        return obj

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode, which sends partial
        results each time.

        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get('stream', False)

    async def _arun_streaming(
        self, messages: List[OpenAIMessage]
    ) -> AsyncStream[ChatCompletionChunk]:
        r"""Handles streaming inference with SambaNova's API.

        Args:
            messages (List[OpenAIMessage]): A list of messages representing the
                chat history in OpenAI API format.

        Returns:
            AsyncStream[ChatCompletionChunk]: A generator yielding
                `ChatCompletionChunk` objects as they are received from the
                API.

        Raises:
            RuntimeError: If the HTTP request fails.
            ValueError: If the API doesn't support stream mode.
        """
        # Handle SambaNova's Cloud API
        if self._url == "https://api.sambanova.ai/v1":
            response = await self._async_client.chat.completions.create(
                messages=messages,
                model=self.model_type,
                **self.model_config_dict,
            )

            # Add AgentOps LLM Event tracking
            if LLMEvent:
                llm_event = LLMEvent(
                    thread_id=response.id,
                    prompt=" ".join(
                        [message.get("content") for message in messages]  # type: ignore[misc]
                    ),
                    prompt_tokens=response.usage.prompt_tokens,  # type: ignore[union-attr]
                    completion=response.choices[0].message.content,
                    completion_tokens=response.usage.completion_tokens,  # type: ignore[union-attr]
                    model=self.model_type,
                )
                record(llm_event)

            return response

        elif self._url == "https://sambaverse.sambanova.ai/api/predict":
            raise ValueError(
                "https://sambaverse.sambanova.ai/api/predict doesn't support"
                " stream mode"
            )
        raise RuntimeError(f"Unknown URL: {self._url}")

    async def _arun_non_streaming(
        self, messages: List[OpenAIMessage]
    ) -> ChatCompletion:
        r"""Handles non-streaming inference with SambaNova's API.

        Args:
            messages (List[OpenAIMessage]): A list of messages representing the
                message in OpenAI API format.

        Returns:
            ChatCompletion: A `ChatCompletion` object containing the complete
                response from the API.

        Raises:
            RuntimeError: If the HTTP request fails.
            ValueError: If the JSON response cannot be decoded or is missing
                expected data.
        """
        # Handle SambaNova's Cloud API
        if self._url == "https://api.sambanova.ai/v1":
            response = await self._async_client.chat.completions.create(
                messages=messages,
                model=self.model_type,
                **self.model_config_dict,
            )

            # Add AgentOps LLM Event tracking
            if LLMEvent:
                llm_event = LLMEvent(
                    thread_id=response.id,
                    prompt=" ".join(
                        [message.get("content") for message in messages]  # type: ignore[misc]
                    ),
                    prompt_tokens=response.usage.prompt_tokens,  # type: ignore[union-attr]
                    completion=response.choices[0].message.content,
                    completion_tokens=response.usage.completion_tokens,  # type: ignore[union-attr]
                    model=self.model_type,
                )
                record(llm_event)

            return response

        # Handle SambaNova's Sambaverse API
        else:
            headers = {
                "Content-Type": "application/json",
                "key": str(self._api_key),
                "modelName": self.model_type,
            }

            data = {
                "instance": json.dumps(
                    {
                        "conversation_id": str(uuid.uuid4()),
                        "messages": messages,
                    }
                ),
                "params": {
                    "do_sample": {"type": "bool", "value": "true"},
                    "max_tokens_to_generate": {
                        "type": "int",
                        "value": str(self.model_config_dict.get("max_tokens")),
                    },
                    "process_prompt": {"type": "bool", "value": "true"},
                    "repetition_penalty": {
                        "type": "float",
                        "value": str(
                            self.model_config_dict.get("repetition_penalty")
                        ),
                    },
                    "return_token_count_only": {
                        "type": "bool",
                        "value": "false",
                    },
                    "select_expert": {
                        "type": "str",
                        "value": self.model_type.split("/")[1],
                    },
                    "stop_sequences": {
                        "type": "str",
                        "value": self.model_config_dict.get("stop_sequences"),
                    },
                    "temperature": {
                        "type": "float",
                        "value": str(
                            self.model_config_dict.get("temperature")
                        ),
                    },
                    "top_k": {
                        "type": "int",
                        "value": str(self.model_config_dict.get("top_k")),
                    },
                    "top_p": {
                        "type": "float",
                        "value": str(self.model_config_dict.get("top_p")),
                    },
                },
            }

            try:
                # Send the request and handle the response
                with httpx.Client() as client:
                    response = client.post(
                        self._url,  # type: ignore[arg-type]
                        headers=headers,
                        json=data,
                    )

                raw_text = response.text
                # Split the string into two dictionaries
                dicts = raw_text.split("}\n{")

                # Keep only the last dictionary
                last_dict = "{" + dicts[-1]

                # Parse the dictionary
                last_dict = json.loads(last_dict)
                return self._sambaverse_to_openai_response(last_dict)  # type: ignore[arg-type]

            except httpx.HTTPStatusError:
                raise RuntimeError(f"HTTP request failed: {raw_text}")
