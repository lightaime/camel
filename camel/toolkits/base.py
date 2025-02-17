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

import functools
import threading
from typing import List, Optional

from camel.toolkits import FunctionTool
from camel.utils import AgentOpsMeta


def with_timeout(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        timeout = getattr(self, "timeout", None)
        if timeout is None:
            return func(self, *args, **kwargs)
        result_container = []

        def target():
            result_container.append(func(self, *args, **kwargs))

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            return (
                f"Function `{func.__name__}` execution timed out, exceeded"
                f" {timeout} seconds."
            )
        else:
            return result_container[0]

    return wrapper


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
