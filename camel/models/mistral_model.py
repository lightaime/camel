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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

if TYPE_CHECKING:
    from mistralai.models import (
        ChatCompletionResponse,
        Messages,
    )

from openai import AsyncStream

from camel.configs import MISTRAL_API_PARAMS, MistralConfig
from camel.messages import OpenAIMessage
from camel.models import BaseModelBackend
from camel.models._utils import try_modify_message_with_format
from camel.types import ChatCompletion, ChatCompletionChunk, ModelType
from camel.utils import (
    BaseTokenCounter,
    OpenAITokenCounter,
    api_keys_required,
    dependencies_required,
)

try:
    if os.getenv("AGENTOPS_API_KEY") is not None:
        from agentops import LLMEvent, record
    else:
        raise ImportError
except (ImportError, AttributeError):
    LLMEvent = None


class MistralModel(BaseModelBackend):
    r"""Mistral API in a unified BaseModelBackend interface.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created, one of MISTRAL_* series.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into:obj:`Mistral.chat.complete()`.
            If:obj:`None`, :obj:`MistralConfig().as_dict()` will be used.
            (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating with
            the mistral service. (default: :obj:`None`)
        url (Optional[str], optional): The url to the mistral service.
            (default: :obj:`None`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter` will
            be used. (default: :obj:`None`)
    """

    @api_keys_required(
        [
            ("api_key", "MISTRAL_API_KEY"),
        ]
    )
    @dependencies_required('mistralai')
    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
    ) -> None:
        from mistralai import Mistral

        if model_config_dict is None:
            model_config_dict = MistralConfig().as_dict()

        api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        url = url or os.environ.get("MISTRAL_API_BASE_URL")
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter
        )
        self._client = Mistral(api_key=self._api_key, server_url=self._url)

    def _to_openai_response(
        self, response: 'ChatCompletionResponse'
    ) -> ChatCompletion:
        tool_calls = None
        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.tool_calls is not None
        ):
            tool_calls = [
                dict(
                    id=tool_call.id,  # type: ignore[union-attr]
                    function={
                        "name": tool_call.function.name,  # type: ignore[union-attr]
                        "arguments": tool_call.function.arguments,  # type: ignore[union-attr]
                    },
                    type=tool_call.type,  # type: ignore[union-attr]
                )
                for tool_call in response.choices[0].message.tool_calls
            ]

        obj = ChatCompletion.construct(
            id=response.id,
            choices=[
                dict(
                    index=response.choices[0].index,  # type: ignore[index]
                    message={
                        "role": response.choices[0].message.role,  # type: ignore[index,union-attr]
                        "content": response.choices[0].message.content,  # type: ignore[index,union-attr]
                        "tool_calls": tool_calls,
                    },
                    finish_reason=response.choices[0].finish_reason  # type: ignore[index]
                    if response.choices[0].finish_reason  # type: ignore[index]
                    else None,
                )
            ],
            created=response.created,
            model=response.model,
            object="chat.completion",
            usage=response.usage,
        )

        return obj

    def _to_mistral_chatmessage(
        self,
        messages: List[OpenAIMessage],
    ) -> List["Messages"]:
        import uuid

        from mistralai.models import (
            AssistantMessage,
            FunctionCall,
            SystemMessage,
            ToolCall,
            ToolMessage,
            UserMessage,
        )

        new_messages = []
        for msg in messages:
            tool_id = uuid.uuid4().hex[:9]
            tool_call_id = msg.get("tool_call_id") or uuid.uuid4().hex[:9]

            role = msg.get("role")
            tool_calls = msg.get("tool_calls")
            content = msg.get("content")

            mistral_function_call = None
            if tool_calls:
                # Ensure tool_calls is treated as a list
                tool_calls_list = (
                    tool_calls
                    if isinstance(tool_calls, list)
                    else [tool_calls]
                )
                for tool_call in tool_calls_list:
                    mistral_function_call = FunctionCall(
                        name=tool_call["function"].get("name"),  # type: ignore[attr-defined]
                        arguments=tool_call["function"].get("arguments"),  # type: ignore[attr-defined]
                    )

            tool_calls = None
            if mistral_function_call:
                tool_calls = [
                    ToolCall(function=mistral_function_call, id=tool_id)
                ]

            if role == "user":
                new_messages.append(UserMessage(content=content))  # type: ignore[arg-type]
            elif role == "assistant":
                new_messages.append(
                    AssistantMessage(content=content, tool_calls=tool_calls)  # type: ignore[arg-type]
                )
            elif role == "system":
                new_messages.append(SystemMessage(content=content))  # type: ignore[arg-type]
            elif role in {"tool", "function"}:
                new_messages.append(
                    ToolMessage(
                        content=content,  # type: ignore[arg-type]
                        tool_call_id=tool_call_id,  # type: ignore[arg-type]
                        name=msg.get("name"),  # type: ignore[arg-type]
                    )
                )
            else:
                raise ValueError(f"Unsupported message role: {role}")

        return new_messages  # type: ignore[return-value]

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        # NOTE: Temporarily using `OpenAITokenCounter` due to a current issue
        # with installing `mistral-common` alongside `mistralai`.
        # Refer to: https://github.com/mistralai/mistral-common/issues/37

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
        response_format: Optional[Type[BaseModel]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        raise NotImplementedError("Mistral does not support async inference.")

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> ChatCompletion:
        r"""Runs inference of Mistral chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.
            response_format (Optional[Type[BaseModel]]): The format of the
                response for this query.
            tools (Optional[List[Dict[str, Any]]]): The tools to use for this
                query.

        Returns:
            ChatCompletion: The response from the model.
        """
        request_config = self._prepare_request(
            messages, response_format, tools
        )
        mistral_messages = self._to_mistral_chatmessage(messages)

        response = self._client.chat.complete(
            messages=mistral_messages,
            model=self.model_type,
            **request_config,
        )

        openai_response = self._to_openai_response(response)  # type: ignore[arg-type]

        # Add AgentOps LLM Event tracking
        if LLMEvent:
            llm_event = LLMEvent(
                thread_id=openai_response.id,
                prompt=" ".join(
                    [message.get("content") for message in messages]  # type: ignore[misc]
                ),
                prompt_tokens=openai_response.usage.prompt_tokens,  # type: ignore[union-attr]
                completion=openai_response.choices[0].message.content,
                completion_tokens=openai_response.usage.completion_tokens,  # type: ignore[union-attr]
                model=self.model_type,
            )
            record(llm_event)

        return openai_response

    def _prepare_request(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        request_config = self.model_config_dict.copy()
        if tools:
            request_config["tools"] = tools
        elif response_format:
            try_modify_message_with_format(messages[-1], response_format)
            request_config["response_format"] = {"type": "json_object"}

        return request_config

    def check_model_config(self):
        r"""Check whether the model configuration contains any
        unexpected arguments to Mistral API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to Mistral API.
        """
        for param in self.model_config_dict:
            if param not in MISTRAL_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into Mistral model backend."
                )

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode, which sends partial
        results each time. Current it's not supported.

        Returns:
            bool: Whether the model is in stream mode.
        """
        return False
