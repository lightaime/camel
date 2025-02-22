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

import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Dict, List

from colorama import Fore

from camel.interpreters.base import BaseInterpreter
from camel.interpreters.interpreter_error import InterpreterError
from camel.logger import get_logger

logger = get_logger(__name__)


class SubprocessInterpreter(BaseInterpreter):
    r"""SubprocessInterpreter is a class for executing code files or code
    strings in a subprocess.

    This class handles the execution of code in different scripting languages
    (currently Python and Bash) within a subprocess, capturing their
    stdout and stderr streams, and allowing user checking before executing code
    strings.

    Args:
        require_confirm (bool, optional): If True, prompt user before running
            code strings for security. (default: :obj:`True`)
        print_stdout (bool, optional): If True, print the standard output of
            the executed code. (default: :obj:`False`)
        print_stderr (bool, optional): If True, print the standard error of the
            executed code. (default: :obj:`True`)
    """

    _CODE_EXECUTE_CMD_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python {file_name}",
        "bash": "bash {file_name}",
        "r": "Rscript {file_name}",
    }

    _CODE_EXTENSION_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "py",
        "bash": "sh",
        "r": "R",
    }

    _CODE_TYPE_MAPPING: ClassVar[Dict[str, str]] = {
        "python": "python",
        "py3": "python",
        "python3": "python",
        "py": "python",
        "shell": "bash",
        "bash": "bash",
        "sh": "bash",
        "r": "r",
        "R": "r",
    }

    def __init__(
        self,
        require_confirm: bool = True,
        print_stdout: bool = False,
        print_stderr: bool = True,
    ) -> None:
        self.require_confirm = require_confirm
        self.print_stdout = print_stdout
        self.print_stderr = print_stderr

    def run_file(
        self,
        file: Path,
        code_type: str,
    ) -> str:
        r"""Executes a code file in a subprocess and captures its output.

        Args:
            file (Path): The path object of the file to run.
            code_type (str): The type of code to execute (e.g., 'python',
                'bash').

        Returns:
            str: A string containing the captured stdout and stderr of the
                executed code.

        Raises:
            RuntimeError: If the provided file path does not point to a file.
            InterpreterError: If the code type provided is not supported.
        """
        if not file.is_file():
            raise RuntimeError(f"{file} is not a file.")
        code_type = self._check_code_type(code_type)
        if self._CODE_TYPE_MAPPING[code_type] == "python":
            # For Python code, use ast to analyze and modify the code
            import ast

            import astor

            with open(file, 'r') as f:
                source = f.read()

            # Parse the source code
            try:
                tree = ast.parse(source)
                # Get the last node
                if tree.body:
                    last_node = tree.body[-1]
                    # Handle expressions that would normally not produce output
                    # For example: In a REPL, typing '1 + 2' should show '3'

                    if isinstance(last_node, ast.Expr):
                        # Only wrap in print(repr()) if it's not already a
                        # print call
                        if not (
                            isinstance(last_node.value, ast.Call)
                            and isinstance(last_node.value.func, ast.Name)
                            and last_node.value.func.id == 'print'
                        ):
                            # Transform the AST to wrap the expression in print
                            # (repr())
                            # Example transformation:
                            #   Before: x + y
                            #   After:  print(repr(x + y))
                            tree.body[-1] = ast.Expr(
                                value=ast.Call(
                                    # Create print() function call
                                    func=ast.Name(id='print', ctx=ast.Load()),
                                    args=[
                                        ast.Call(
                                            # Create repr() function call
                                            func=ast.Name(
                                                id='repr', ctx=ast.Load()
                                            ),
                                            # Pass the original expression as
                                            # argument to repr()
                                            args=[last_node.value],
                                            keywords=[],
                                        )
                                    ],
                                    keywords=[],
                                )
                            )
                    # Fix missing source locations
                    ast.fix_missing_locations(tree)
                    # Convert back to source
                    modified_source = astor.to_source(tree)
                    # Create a temporary file with the modified source
                    temp_file = self._create_temp_file(modified_source, "py")
                    cmd = shlex.split(f"python {temp_file!s}")
            except SyntaxError:
                # If parsing fails, run the original file
                cmd = shlex.split(
                    self._CODE_EXECUTE_CMD_MAPPING[code_type].format(
                        file_name=str(file)
                    )
                )
        else:
            # For non-Python code, use standard execution
            cmd = shlex.split(
                self._CODE_EXECUTE_CMD_MAPPING[code_type].format(
                    file_name=str(file)
                )
            )

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = proc.communicate()
        return_code = proc.returncode

        # Clean up temporary file if it was created
        if (
            self._CODE_TYPE_MAPPING[code_type] == "python"
            and 'temp_file' in locals()
        ):
            temp_file.unlink()

        if self.print_stdout and stdout:
            print("======stdout======")
            print(Fore.GREEN + stdout + Fore.RESET)
            print("==================")
        if self.print_stderr and stderr:
            print("======stderr======")
            print(Fore.RED + stderr + Fore.RESET)
            print("==================")

        # Build the execution result
        exec_result = ""
        if stdout:
            exec_result += stdout
        if stderr:
            exec_result += f"(stderr: {stderr})"
        if return_code != 0:
            error_msg = f"(Execution failed with return code {return_code})"
            if not stderr:
                exec_result += error_msg
            elif error_msg not in stderr:
                exec_result += error_msg
        return exec_result

    def run(
        self,
        code: str,
        code_type: str,
    ) -> str:
        r"""Generates a temporary file with the given code, executes it, and
            deletes the file afterward.

        Args:
            code (str): The code string to execute.
            code_type (str): The type of code to execute (e.g., 'python',
                'bash').

        Returns:
            str: A string containing the captured stdout and stderr of the
                executed code.

        Raises:
            InterpreterError: If the user declines to run the code or if the
                code type is unsupported.
        """
        code_type = self._check_code_type(code_type)

        # Print code for security checking
        if self.require_confirm:
            logger.info(
                f"The following {code_type} code will run on your "
                "computer: {code}"
            )
            while True:
                choice = input("Running code? [Y/n]:").lower()
                if choice in ["y", "yes", "ye", ""]:
                    break
                elif choice in ["no", "n"]:
                    raise InterpreterError(
                        "Execution halted: User opted not to run the code. "
                        "This choice stops the current operation and any "
                        "further code execution."
                    )
        temp_file_path = self._create_temp_file(
            code=code, extension=self._CODE_EXTENSION_MAPPING[code_type]
        )

        result = self.run_file(temp_file_path, code_type)

        temp_file_path.unlink()
        return result

    def _create_temp_file(self, code: str, extension: str) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=f".{extension}"
        ) as f:
            f.write(code)
            name = f.name
        return Path(name)

    def _check_code_type(self, code_type: str) -> str:
        if code_type not in self._CODE_TYPE_MAPPING:
            raise InterpreterError(
                f"Unsupported code type {code_type}. Currently "
                f"`{self.__class__.__name__}` only supports "
                f"{', '.join(self._CODE_EXTENSION_MAPPING.keys())}."
            )
        return self._CODE_TYPE_MAPPING[code_type]

    def supported_code_types(self) -> List[str]:
        r"""Provides supported code types by the interpreter."""
        return list(self._CODE_EXTENSION_MAPPING.keys())

    def update_action_space(self, action_space: Dict[str, Any]) -> None:
        r"""Updates action space for *python* interpreter"""
        raise RuntimeError(
            "SubprocessInterpreter doesn't support " "`action_space`."
        )