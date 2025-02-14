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
import subprocess
from typing import Any, Dict, List, Optional, Type, Union

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from pydantic import BaseModel

from camel.configs import OLLAMA_API_PARAMS, OllamaConfig
from camel.messages import OpenAIMessage
from camel.models import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ModelType,
)
from camel.utils import BaseTokenCounter, OpenAITokenCounter


class OllamaModel(BaseModelBackend):
    r"""Ollama service interface.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into:obj:`openai.ChatCompletion.create()`.
            If:obj:`None`, :obj:`OllamaConfig().as_dict()` will be used.
            (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating with
            the model service.  Ollama doesn't need API key, it would be
            ignored if set. (default: :obj:`None`)
        url (Optional[str], optional): The url to the model service.
            (default: :obj:`None`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter(
            ModelType.GPT_4O_MINI)` will be used.
            (default: :obj:`None`)

    References:
        https://github.com/ollama/ollama/blob/main/docs/openai.md
    """

    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
    ) -> None:
        if model_config_dict is None:
            model_config_dict = OllamaConfig().as_dict()
        url = url or os.environ.get("OLLAMA_BASE_URL")
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter
        )
        if not self._url:
            self._start_server()
        # Use OpenAI client as interface call Ollama
        self._client = OpenAI(
            timeout=180,
            max_retries=3,
            api_key="Set-but-ignored",  # required but ignored
            base_url=self._url,
        )
        self._async_client = AsyncOpenAI(
            timeout=180,
            max_retries=3,
            api_key="Set-but-ignored",  # required but ignored
            base_url=self._url,
        )

    def _start_server(self) -> None:
        r"""Starts the Ollama server in a subprocess."""
        try:
            subprocess.Popen(
                ["ollama", "server", "--port", "11434"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._url = "http://localhost:11434/v1"
            print(
                f"Ollama server started on {self._url} "
                f"for {self.model_type} model."
            )
        except Exception as e:
            print(f"Failed to start Ollama server: {e}.")

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
        unexpected arguments to Ollama API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to OpenAI API.
        """
        for param in self.model_config_dict:
            if param not in OLLAMA_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into Ollama model backend."
                )

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        r"""Runs inference of OpenAI chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `AsyncStream[ChatCompletionChunk]` in the stream mode.
        """
        if self.model_config_dict.get("response_format"):
            # stream is not supported in beta.chat.completions.parse
            if "stream" in self.model_config_dict:
                del self.model_config_dict["stream"]

            response = await self._async_client.beta.chat.completions.parse(
                messages=messages,
                model=self.model_type,
                **self.model_config_dict,
            )

            return self._to_chat_completion(response)

        response = await self._async_client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **self.model_config_dict,
        )
        return response

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        r"""Runs inference of OpenAI chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            Union[ChatCompletion, Stream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `Stream[ChatCompletionChunk]` in the stream mode.
        """
        if self.model_config_dict.get("response_format"):
            # stream is not supported in beta.chat.completions.parse
            if "stream" in self.model_config_dict:
                del self.model_config_dict["stream"]

            response = self._client.beta.chat.completions.parse(
                messages=messages,
                model=self.model_type,
                **self.model_config_dict,
            )

            return self._to_chat_completion(response)

        response = self._client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **self.model_config_dict,
        )
        return response

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode, which sends partial
        results each time.

        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get('stream', False)
