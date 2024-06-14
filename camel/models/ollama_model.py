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
import os
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI, Stream

from camel.configs import OPENAI_API_PARAMS
from camel.messages import OpenAIMessage
from camel.types import ChatCompletion, ChatCompletionChunk, ModelType
from camel.utils import BaseTokenCounter, OpenAITokenCounter


class OllamaModel:
    r"""Ollama service interface."""

    # NOTE: Current `ModelType and `TokenCounter` desigen is not suitable,
    # stream mode is not supported

    def __init__(
        self,
        model_type: str,
        model_config_dict: Dict[str, Any],
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        r"""Constructor for Ollama backend with OpenAI compatibility.

        Args:
            model_type (str): Model for which a backend is created.
            model_config_dict (Dict[str, Any]): A dictionary that will
                be fed into openai.ChatCompletion.create().
            api_key (Optional[str]): The API key for authenticating with the
                model service. (default: :obj:`None`)
            url (Optional[str]): The url to the model service.
        """
        self.model_type = model_type
        self.model_config_dict = model_config_dict
        self._url = url or os.environ.get('OPENAI_API_BASE_URL')
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = OpenAI(
            timeout=60,
            max_retries=3,
            base_url=self._url,
            api_key=self._api_key,
        )
        self._token_counter: Optional[BaseTokenCounter] = None

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        # NOTE: Use OpenAITokenCounter temporarily
        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(ModelType.GPT_3_5_TURBO)
        return self._token_counter

    def check_model_config(self):
        r"""Check whether the model configuration contains any
        unexpected arguments to OpenAI API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to OpenAI API.
        """
        for param in self.model_config_dict:
            if param not in OPENAI_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into OpenAI model backend."
                )

    def run(
        self,
        messages: List[OpenAIMessage],
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
