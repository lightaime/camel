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

from openai import OpenAI, Stream

from camel.configs import AIML_API_PARAMS, AIMLConfig
from camel.messages import OpenAIMessage
from camel.models.base_model import BaseModelBackend
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


class AIMLModel(BaseModelBackend):
    r"""AIML API in a unified BaseModelBackend interface.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into OpenAI client. If :obj:`None`,
            :obj:`AIMLConfig().as_dict()` will be used.
            (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating with
            the AIML service. (default: :obj:`None`)
        url (Optional[str], optional): The URL to the AIML service. If
            not provided, :obj:`https://api.aimlapi.com/v1` will be used.
            (default: :obj:`None`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter(
            ModelType.GPT_4O_MINI)` will be used.
            (default: :obj:`None`)
    """

    @api_keys_required(
        [
            ("api_key", 'AIML_API_KEY'),
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
            model_config_dict = AIMLConfig().as_dict()
        api_key = api_key or os.environ.get("AIML_API_KEY")
        url = url or os.environ.get(
            "AIML_API_BASE_URL",
            "https://api.aimlapi.com/v1",
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
        # Process model configuration parameters
        model_config = self.model_config_dict.copy()

        # Remove tools parameter if None
        if 'tools' in model_config and model_config['tools'] is None:
            model_config.pop('tools')

        if model_config.get("response_format"):
            # stream is not supported in beta.chat.completions.parse
            if "stream" in model_config:
                del model_config["stream"]

            response = self._client.beta.chat.completions.parse(
                messages=messages, model=self.model_type, **model_config
            )

            return self._to_chat_completion(response)

        response = self._client.chat.completions.create(
            messages=messages, model=self.model_type, **model_config
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
        r"""Check whether the model configuration contains any
        unexpected arguments to AIML API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to AIML API.
        """
        for param in self.model_config_dict:
            if param not in AIML_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into AIML model backend."
                )

    @property
    def stream(self) -> bool:
        """Returns whether the model is in stream mode, which sends partial
        results each time.

        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get('stream', False)
