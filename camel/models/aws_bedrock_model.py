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
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from camel.configs import BEDROCK_API_PARAMS, BedrockConfig
from camel.messages import OpenAIMessage
from camel.models.base_model import BaseModelBackend
from camel.types import ChatCompletion, ModelType
from camel.utils import (
    BaseTokenCounter,
    OpenAITokenCounter,
    api_keys_required,
)


class AWSBedrockModel(BaseModelBackend):
    r"""AWS Bedrock API in a unified BaseModelBackend interface.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created.
        model_config_dict (Dict[str, Any], optional): A dictionary
            that will be fed into:obj:`openai.ChatCompletion.create()`.
            If:obj:`None`, :obj:`BedrockConfig().as_dict()` will be used.
            (default: :obj:`None`)
        api_key (str, optional): The API key for authenticating with
            the AWS Bedrock service. (default: :obj:`None`)
        url (str, optional): The url to the AWS Bedrock service.
        token_counter (BaseTokenCounter, optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter(
            ModelType.GPT_4O_MINI)` will be used.
            (default: :obj:`None`)

    References:
        https://docs.aws.amazon.com/bedrock/latest/APIReference/welcome.html
    """

    @api_keys_required(
        [
            ("url", "BEDROCK_API_BASE_URL"),
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
            model_config_dict = BedrockConfig().as_dict()
        api_key = api_key or os.environ.get("BEDROCK_API_KEY")
        url = url or os.environ.get(
            "BEDROCK_API_BASE_URL",
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

    def run(self, messages: List[OpenAIMessage]) -> ChatCompletion:
        r"""Runs the query to the backend model.

        Args:
            message (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            ChatCompletion: The response object in OpenAI's format.
        """
        response = self._client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **self.model_config_dict,
        )
        return response

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
        r"""Check whether the input model configuration contains unexpected
        arguments.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected argument for this model class.
        """
        for param in self.model_config_dict:
            if param not in BEDROCK_API_PARAMS:
                raise ValueError(
                    f"Invalid parameter '{param}' in model_config_dict. "
                    f"Valid parameters are: {BEDROCK_API_PARAMS}"
                )
