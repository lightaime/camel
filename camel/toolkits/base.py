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

from typing import List, Optional

from camel.toolkits import FunctionTool
from camel.utils import AgentOpsMeta, with_timeout


class BaseToolkit(metaclass=AgentOpsMeta):
    r"""Base class for toolkits.

    Args:
        timeout (Optional[float]): The timeout for the toolkit.
    """

    timeout: Optional[float] = None

    def __init__(self, timeout: Optional[float] = None):
        # check if timeout is a positive number
        if timeout is not None and timeout <= 0:
            raise ValueError("Timeout must be a positive number.")
        self.timeout = timeout

    # Add timeout to all callable methods in the toolkit
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value) and not attr_name.startswith("__"):
                setattr(cls, attr_name, with_timeout(attr_value))

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the
        functions in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects
                representing the functions in the toolkit.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def list_tools(self) -> List[str]:
        r"""Retrieve a list of tool names available in the current class."""
        tools = self.get_tools()
        tool_names = [tool.func.__name__ for tool in tools]
        return tool_names

    def get_a_tool(self, func_name: str) -> List[OpenAIFunction]:
        r"""Retrieve a tool by its function name if it exists within the
        current class.
        """
        tools = self.get_tools()
        for tool in tools:
            if tool.func.__name__ == func_name:
                return [tool]
        raise AttributeError(
            f"{func_name} is not a valid tool name or is not part of "
            f"{self.__class__.__name__}'s module."
        )
