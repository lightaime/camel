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

from openai import AsyncAzureOpenAI, AsyncStream, AzureOpenAI, Stream
from pydantic import BaseModel

from camel.configs import OPENAI_API_PARAMS, ChatGPTConfig
from camel.messages import OpenAIMessage
from camel.models.base_model import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ModelType,
)
from camel.utils import BaseTokenCounter, OpenAITokenCounter


class AzureOpenAIModel(BaseModelBackend):
    r"""Azure OpenAI API in a unified BaseModelBackend interface.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created, one of GPT_* series.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into:obj:`openai.ChatCompletion.create()`. If
            :obj:`None`, :obj:`ChatGPTConfig().as_dict()` will be used.
            (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating with
            the OpenAI service. (default: :obj:`None`)
        url (Optional[str], optional): The url to the OpenAI service.
            (default: :obj:`None`)
        api_version (Optional[str], optional): The api version for the model.
            (default: :obj:`None`)
        azure_deployment_name (Optional[str], optional): The deployment name
            you chose when you deployed an azure model. (default: :obj:`None`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter`
            will be used. (default: :obj:`None`)

    References:
        https://learn.microsoft.com/en-us/azure/ai-services/openai/
    """

    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
        api_version: Optional[str] = None,
        azure_deployment_name: Optional[str] = None,
    ) -> None:
        if model_config_dict is None:
            model_config_dict = ChatGPTConfig().as_dict()
        api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        url = url or os.environ.get("AZURE_OPENAI_BASE_URL")
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter
        )

        self.api_version = api_version or os.environ.get("AZURE_API_VERSION")
        self.azure_deployment_name = azure_deployment_name or os.environ.get(
            "AZURE_DEPLOYMENT_NAME"
        )
        if self.api_version is None:
            raise ValueError(
                "Must provide either the `api_version` argument "
                "or `AZURE_API_VERSION` environment variable."
            )
        if self.azure_deployment_name is None:
            raise ValueError(
                "Must provide either the `azure_deployment_name` argument "
                "or `AZURE_DEPLOYMENT_NAME` environment variable."
            )

        self._client = AzureOpenAI(
            azure_endpoint=str(self._url),
            azure_deployment=self.azure_deployment_name,
            api_version=self.api_version,
            api_key=self._api_key,
            timeout=180,
            max_retries=3,
        )

        self._async_client = AsyncAzureOpenAI(
            azure_endpoint=str(self._url),
            azure_deployment=self.azure_deployment_name,
            api_version=self.api_version,
            api_key=self._api_key,
            timeout=180,
            max_retries=3,
        )

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(self.model_type)
        return self._token_counter

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        r"""Runs inference of Azure OpenAI chat completion.

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
            model=self.azure_deployment_name,  # type:ignore[arg-type]
            **self.model_config_dict,
        )
        return response

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        r"""Runs inference of Azure OpenAI chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `AsyncStream[ChatCompletionChunk]` in the stream mode.
        """
        response = await self._async_client.chat.completions.create(
            messages=messages,
            model=self.azure_deployment_name,  # type:ignore[arg-type]
            **self.model_config_dict,
        )
        return response

    def check_model_config(self):
        r"""Check whether the model configuration contains any
        unexpected arguments to Azure OpenAI API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to Azure OpenAI API.
        """
        for param in self.model_config_dict:
            if param not in OPENAI_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into Azure OpenAI model backend."
                )

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode,
            which sends partial results each time.
        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get("stream", False)
