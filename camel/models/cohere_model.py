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
import ast
import json
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

if TYPE_CHECKING:
    from cohere.types import ChatMessageV2, ChatResponse

from camel.configs import COHERE_API_PARAMS, CohereConfig
from camel.messages import OpenAIMessage
from camel.models import BaseModelBackend
from camel.models._utils import try_modify_message_with_format
from camel.types import ChatCompletion, ModelType
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


class CohereModel(BaseModelBackend):
    r"""Cohere API in a unified BaseModelBackend interface."""

    @api_keys_required(
        [
            ("api_key", 'COHERE_API_KEY'),
        ]
    )
    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
    ):
        import cohere

        if model_config_dict is None:
            model_config_dict = CohereConfig().as_dict()

        api_key = api_key or os.environ.get("COHERE_API_KEY")
        url = url or os.environ.get("COHERE_API_BASE_URL")
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter
        )
        self._client = cohere.ClientV2(api_key=self._api_key)
        self._async_client = cohere.AsyncClientV2(api_key=self._api_key)

    def _to_openai_response(self, response: 'ChatResponse') -> ChatCompletion:
        if response.usage and response.usage.tokens:
            input_tokens = response.usage.tokens.input_tokens or 0
            output_tokens = response.usage.tokens.output_tokens or 0
            usage = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        else:
            usage = {}

        tool_calls = response.message.tool_calls
        choices = []
        if tool_calls:
            for tool_call in tool_calls:
                openai_tool_calls = [
                    dict(
                        id=tool_call.id,
                        function={
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                        if tool_call.function
                        else {},
                        type=tool_call.type,
                    )
                ]

                choice = dict(
                    index=None,
                    message={
                        "role": "assistant",
                        "content": response.message.tool_plan,
                        "tool_calls": openai_tool_calls,
                    },
                    finish_reason=response.finish_reason
                    if response.finish_reason
                    else None,
                )
                choices.append(choice)

        else:
            openai_tool_calls = None

            choice = dict(
                index=None,
                message={
                    "role": "assistant",
                    "content": response.message.content[0].text,  # type: ignore[union-attr,index]
                    "tool_calls": openai_tool_calls,
                },
                finish_reason=response.finish_reason
                if response.finish_reason
                else None,
            )
            choices.append(choice)

        obj = ChatCompletion.construct(
            id=response.id,
            choices=choices,
            created=None,
            model=self.model_type,
            object="chat.completion",
            usage=usage,
        )
        return obj

    def _to_cohere_chatmessage(
        self, messages: List[OpenAIMessage]
    ) -> List["ChatMessageV2"]:
        from cohere.types import ToolCallV2Function
        from cohere.types.chat_message_v2 import (
            AssistantChatMessageV2,
            SystemChatMessageV2,
            ToolCallV2,
            ToolChatMessageV2,
            UserChatMessageV2,
        )

        tool_call_id = None
        new_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            function_call = msg.get("function_call")

            if role == "user":
                new_message = UserChatMessageV2(role="user", content=content)  # type: ignore[arg-type]
            elif role in {"tool", "function"}:
                new_message = ToolChatMessageV2(
                    role="tool",
                    tool_call_id=tool_call_id,  # type: ignore[arg-type]
                    content=content,  # type: ignore[assignment,arg-type]
                )
            elif role == "assistant":
                if not function_call:
                    new_message = AssistantChatMessageV2(  # type: ignore[assignment]
                        role="assistant",
                        content=content,  # type: ignore[arg-type]
                    )
                else:
                    arguments = function_call.get("arguments")  # type: ignore[attr-defined]
                    arguments_dict = ast.literal_eval(arguments)
                    arguments_json = json.dumps(arguments_dict)

                    assis_tool_call_id = str(uuid.uuid4())
                    tool_call_id = assis_tool_call_id
                    new_message = AssistantChatMessageV2(  # type: ignore[assignment]
                        role="assistant",
                        tool_calls=[
                            ToolCallV2(
                                id=assis_tool_call_id,
                                type="function",
                                function=ToolCallV2Function(
                                    name=function_call.get("name"),  # type: ignore[attr-defined]
                                    arguments=arguments_json,  # type: ignore[attr-defined]
                                ),
                            )
                        ],
                        content=content,  # type: ignore[arg-type]
                    )
            elif role == "system":
                new_message = SystemChatMessageV2(  # type: ignore[assignment]
                    role="system",
                    content=content,  # type: ignore[arg-type]
                )
            else:
                raise ValueError(f"Unsupported message role: {role}")

            new_messages.append(new_message)
        return new_messages  # type: ignore[return-value]

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
    ) -> ChatCompletion:
        r"""Runs inference of Cohere chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.
        Returns:
            ChatCompletion.
        """
        from cohere.core.api_error import ApiError

        request_config = self._prepare_request(
            messages, response_format, tools
        )

        cohere_messages = self._to_cohere_chatmessage(messages)

        try:
            response = self._client.chat(
                messages=cohere_messages,
                model=self.model_type,
                **request_config,
            )
        except ApiError as e:
            logging.error(f"Cohere API Error: {e.status_code}")
            logging.error(f"Error body: {e.body}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error when calling Cohere API: {e!s}")
            raise

        openai_response = self._to_openai_response(response)

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

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        r"""Runs inference of Cohere chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.
        Returns:
            ChatCompletion.
        """
        from cohere.core.api_error import ApiError

        request_config = self._prepare_request(
            messages, response_format, tools
        )

        cohere_messages = self._to_cohere_chatmessage(messages)

        try:
            response = await self._async_client.chat(
                messages=cohere_messages,
                model=self.model_type,
                **request_config,
            )
        except ApiError as e:
            logging.error(f"Cohere API Error: {e.status_code}")
            logging.error(f"Error body: {e.body}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error when calling Cohere API: {e!s}")
            raise

        openai_response = self._to_openai_response(response)

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

    def check_model_config(self):
        r"""Check whether the model configuration contains any unexpected
        arguments to Cohere API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to Cohere API.
        """
        for param in self.model_config_dict:
            if param not in COHERE_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into Cohere model backend."
                )

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode, which sends partial
        results each time. Current it's not supported.

        Returns:
            bool: Whether the model is in stream mode.
        """
        return False
