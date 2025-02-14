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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from camel.configs import REKA_API_PARAMS, RekaConfig
from camel.messages import OpenAIMessage
from camel.models import BaseModelBackend
from camel.types import ChatCompletion, ModelType
from camel.utils import (
    BaseTokenCounter,
    OpenAITokenCounter,
    api_keys_required,
    dependencies_required,
)

if TYPE_CHECKING:
    from reka.types import ChatMessage, ChatResponse

try:
    import os

    if os.getenv("AGENTOPS_API_KEY") is not None:
        from agentops import LLMEvent, record
    else:
        raise ImportError
except (ImportError, AttributeError):
    LLMEvent = None


class RekaModel(BaseModelBackend):
    r"""Reka API in a unified BaseModelBackend interface.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created, one of REKA_* series.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into:obj:`Reka.chat.create()`. If :obj:`None`,
            :obj:`RekaConfig().as_dict()` will be used. (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating with
            the Reka service. (default: :obj:`None`)
        url (Optional[str], optional): The url to the Reka service.
            (default: :obj:`None`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter` will
            be used. (default: :obj:`None`)
    """

    @api_keys_required(
        [
            ("api_key", "REKA_API_KEY"),
        ]
    )
    @dependencies_required('reka')
    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
    ) -> None:
        from reka.client import AsyncReka, Reka

        if model_config_dict is None:
            model_config_dict = RekaConfig().as_dict()
        api_key = api_key or os.environ.get("REKA_API_KEY")
        url = url or os.environ.get("REKA_API_BASE_URL")
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter
        )
        self._client = Reka(api_key=self._api_key, base_url=self._url)
        self._async_client = AsyncReka(
            api_key=self._api_key, base_url=self._url
        )

    def _convert_reka_to_openai_response(
        self, response: 'ChatResponse'
    ) -> ChatCompletion:
        r"""Converts a Reka `ChatResponse` to an OpenAI-style `ChatCompletion`
        response.

        Args:
            response (ChatResponse): The response object from the Reka API.

        Returns:
            ChatCompletion: An OpenAI-compatible chat completion response.
        """
        openai_response = ChatCompletion.construct(
            id=response.id,
            choices=[
                dict(
                    message={
                        "role": response.responses[0].message.role,
                        "content": response.responses[0].message.content,
                    },
                    finish_reason=response.responses[0].finish_reason
                    if response.responses[0].finish_reason
                    else None,
                )
            ],
            created=None,
            model=response.model,
            object="chat.completion",
            usage=response.usage,
        )

        return openai_response

    def _convert_openai_to_reka_messages(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[str]] = None,
    ) -> List["ChatMessage"]:
        r"""Converts OpenAI API messages to Reka API messages.

        Args:
            messages (List[OpenAIMessage]): A list of messages in OpenAI
                format.

        Returns:
            List[ChatMessage]: A list of messages converted to Reka's format.
        """
        from reka.types import ChatMessage

        reka_messages = []
        for msg in messages:
            role = msg.get("role")
            content = str(msg.get("content"))

            if role == "user":
                reka_messages.append(ChatMessage(role="user", content=content))
            elif role == "assistant":
                reka_messages.append(
                    ChatMessage(role="assistant", content=content)
                )
            elif role == "system":
                reka_messages.append(ChatMessage(role="user", content=content))

                # Add one more assistant msg since Reka requires conversation
                # history must alternate between 'user' and 'assistant',
                # starting and ending with 'user'.
                reka_messages.append(
                    ChatMessage(
                        role="assistant",
                        content="",
                    )
                )
            else:
                raise ValueError(f"Unsupported message role: {role}")

        return reka_messages

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        # NOTE: Temporarily using `OpenAITokenCounter`

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(
                model=ModelType.GPT_4O_MINI
            )
        return self._token_counter

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        r"""Runs inference of Mistral chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            ChatCompletion.
        """
        reka_messages = self._convert_openai_to_reka_messages(messages)

        response = await self._async_client.chat.create(
            messages=reka_messages,
            model=self.model_type,
            **self.model_config_dict,
        )

        openai_response = self._convert_reka_to_openai_response(response)

        # Add AgentOps LLM Event tracking
        if LLMEvent:
            llm_event = LLMEvent(
                thread_id=openai_response.id,
                prompt=" ".join(
                    [message.get("content") for message in messages]  # type: ignore[misc]
                ),
                prompt_tokens=openai_response.usage.input_tokens,  # type: ignore[union-attr]
                completion=openai_response.choices[0].message.content,
                completion_tokens=openai_response.usage.output_tokens,  # type: ignore[union-attr]
                model=self.model_type,
            )
            record(llm_event)

        return openai_response

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        r"""Runs inference of Mistral chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            ChatCompletion.
        """
        reka_messages = self._convert_openai_to_reka_messages(messages)

        response = self._client.chat.create(
            messages=reka_messages,
            model=self.model_type,
            **self.model_config_dict,
        )

        openai_response = self._convert_reka_to_openai_response(response)

        # Add AgentOps LLM Event tracking
        if LLMEvent:
            llm_event = LLMEvent(
                thread_id=openai_response.id,
                prompt=" ".join(
                    [message.get("content") for message in messages]  # type: ignore[misc]
                ),
                prompt_tokens=openai_response.usage.input_tokens,  # type: ignore[union-attr]
                completion=openai_response.choices[0].message.content,
                completion_tokens=openai_response.usage.output_tokens,  # type: ignore[union-attr]
                model=self.model_type,
            )
            record(llm_event)

        return openai_response

    def check_model_config(self):
        r"""Check whether the model configuration contains any
        unexpected arguments to Reka API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to Reka API.
        """
        for param in self.model_config_dict:
            if param not in REKA_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into Reka model backend."
                )

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode, which sends partial
        results each time.

        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get('stream', False)
