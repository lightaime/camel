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
from typing import Any, Dict

from pydantic import BaseModel


class ToolCallingRecord(BaseModel):
    r"""Historical records of tools called in the conversation.

    Attributes:
        func_name (str): The name of the tool being called.
        args (Dict[str, Any]): The dictionary of arguments passed to the tool.
        result (Any): The execution result of calling this tool.
        tool_call_id (str): The ID of the tool call, if available.
    """

    func_name: str
    args: Dict[str, Any]
    result: Any
    tool_call_id: str

    def __str__(self) -> str:
        r"""Overridden version of the string function.

        Returns:
            str: Modified string to represent the tool calling.
        """
        return (
            f"Tool Execution: {self.func_name}\n"
            f"\tArgs: {self.args}\n"
            f"\tResult: {self.result}\n"
        )

    def as_dict(self) -> dict[str, Any]:
        r"""Returns the tool calling record as a dictionary.

        Returns:
            dict[str, Any]: The tool calling record as a dictionary.
        """
        return self.model_dump()
