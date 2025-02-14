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
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from camel.configs import LITELLM_API_PARAMS, LiteLLMConfig
from camel.messages import OpenAIMessage
from camel.models import BaseModelBackend
from camel.types import ChatCompletion, ModelType
from camel.utils import (
    BaseTokenCounter,
    LiteLLMTokenCounter,
    dependencies_required,
)


class LiteLLMModel(BaseModelBackend):
    r"""Constructor for LiteLLM backend with OpenAI compatibility.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created, such as GPT-3.5-turbo, Claude-2, etc.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into:obj:`openai.ChatCompletion.create()`.
            If:obj:`None`, :obj:`LiteLLMConfig().as_dict()` will be used.
            (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating with
            the model service. (default: :obj:`None`)
        url (Optional[str], optional): The url to the model service.
            (default: :obj:`None`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`LiteLLMTokenCounter` will
            be used. (default: :obj:`None`)
    """

    # NOTE: Currently stream mode is not supported.

    @dependencies_required('litellm')
    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
    ) -> None:
        from litellm import completion

        if model_config_dict is None:
            model_config_dict = LiteLLMConfig().as_dict()

        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter
        )
        self.client = completion

    def _convert_response_from_litellm_to_openai(
        self, response
    ) -> ChatCompletion:
        r"""Converts a response from the LiteLLM format to the OpenAI format.

        Parameters:
            response (LiteLLMResponse): The response object from LiteLLM.

        Returns:
            ChatCompletion: The response object in OpenAI's format.
        """
        return ChatCompletion.construct(
            id=response.id,
            choices=[
                {
                    "index": response.choices[0].index,
                    "message": {
                        "role": response.choices[0].message.role,
                        "content": response.choices[0].message.content,
                    },
                    "finish_reason": response.choices[0].finish_reason,
                }
            ],
            created=response.created,
            model=response.model,
            object=response.object,
            system_fingerprint=response.system_fingerprint,
            usage=response.usage,
        )

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = LiteLLMTokenCounter(self.model_type)
        return self._token_counter

    async def _arun(self) -> None:  # type: ignore[override]
        raise NotImplementedError

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        r"""Runs inference of LiteLLM chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI format.

        Returns:
            ChatCompletion
        """
        response = self.client(
            api_key=self._api_key,
            base_url=self._url,
            model=self.model_type,
            messages=messages,
            **self.model_config_dict,
        )
        response = self._convert_response_from_litellm_to_openai(response)
        return response

    def check_model_config(self):
        r"""Check whether the model configuration contains any unexpected
        arguments to LiteLLM API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments.
        """
        for param in self.model_config_dict:
            if param not in LITELLM_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into LiteLLM model backend."
                )
