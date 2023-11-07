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
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam, )
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam, )
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam, )
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam, )
from openai.types.chat.chat_completion_function_message_param import (
    ChatCompletionFunctionMessageParam, )

OpenAISystemMessage = ChatCompletionSystemMessageParam
OpenAIAssistantMessage = ChatCompletionAssistantMessageParam
OpenAIUserMessage = ChatCompletionUserMessageParam
OpenAIFunctionMessage = ChatCompletionFunctionMessageParam
OpenAIMessage = ChatCompletionMessageParam

from .base import BaseMessage  # noqa: E402
from .func_message import FunctionCallingMessage  # noqa: E402

__all__ = [
    'OpenAISystemMessage',
    'OpenAIAssistantMessage',
    'OpenAIUserMessage',
    'OpenAIMessage',
    'BaseMessage',
    'FunctionCallingMessage',
]
0
