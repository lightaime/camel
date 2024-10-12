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
from typing import List, Literal

from camel.interpreters import InternalPythonInterpreter
from camel.toolkits import OpenAIFunction
from camel.utils.commons import export_to_toolkit

from .base import BaseToolkit


class CodeExecutionToolkit(BaseToolkit):
    r"""A tookit for code execution.

    Args:
        sandbox (str): the environment type used to execute code.
    """

    def __init__(
        self,
        sandbox: Literal[
            "internal_python", "jupyter", "docker"
        ] = "internal_python",
        verbose: bool = False,
    ) -> None:
        # TODO: Add support for docker and jupyter.
        self.verbose = verbose
        if sandbox == "internal_python":
            self.interpreter = InternalPythonInterpreter()
        else:
            raise RuntimeError(
                f"The sandbox type `{sandbox}` is not supported."
            )

    @export_to_toolkit
    def execute_code(self, code: str) -> str:
        r"""Execute a given code snippet.

        Args:
            code (str): The input code to the Code Interpreter tool call.

        Returns:
            str: The text output from the Code Interpreter tool call.
        """
        output = self.interpreter.run(code, "python")
        # ruff: noqa: E501
        content = f"Executed the code below:\n```py\n{code}\n```\n> Executed Results:\n{output}"
        if self.verbose:
            print(content)
        return content

    def get_tools(self) -> List[OpenAIFunction]:
        r"""Returns a list of OpenAIFunction objects representing the
        functions in the toolkit.

        Returns:
            List[OpenAIFunction]: A list of OpenAIFunction objects
                representing the functions in the toolkit.
        """
        return [OpenAIFunction(self.execute_code)]
