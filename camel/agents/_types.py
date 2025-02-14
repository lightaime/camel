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
from typing import Any, Dict, List, Optional, Union

from openai import AsyncStream, Stream
from pydantic import BaseModel, ConfigDict

from camel.messages import BaseMessage
from camel.types import ChatCompletion


class ToolCallRequest(BaseModel):
    r"""The request for tool calling."""

    func_name: str
    args: Dict[str, Any]
    tool_call_id: str


class ModelResponse(BaseModel):
    r"""The response from the model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    response: Union[ChatCompletion, Stream, AsyncStream]
    tool_call_request: Optional[ToolCallRequest]
    output_messages: List[BaseMessage]
    finish_reasons: List[str]
    usage_dict: Dict[str, Any]
    response_id: str
