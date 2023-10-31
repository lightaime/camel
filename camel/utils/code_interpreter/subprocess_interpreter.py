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

import shlex
import subprocess
import tempfile
from pathlib import Path

from colorama import Fore

from camel.utils.code_interpreter.interpreter_error import InterpreterError


class SubprocessInterpreter():

    _CODE_EXECUTE_CMD = {
        "python": "python {file_name}",
        "bash": "bash {file_name}",
        "fish": "fish {file_name}",
    }

    _CODE_EXTENTION = {
        "python": "py",
        "bash": "sh",
        "fish": "fish",
    }

    _CODE_TYPE = {
        "python": "python",
        "py3": "python",
        "python3": "python",
        "py": "python",
        "shell": "bash",
        "bash": "bash",
        "sh": "bash",
        "fish": "fish"
    }

    def __init__(
        self,
        user_check: bool = True,
        print_stdout: bool = False,
        print_stderr: bool = True,
    ) -> None:
        self.user_check = user_check
        self.print_stdout = print_stdout
        self.print_stderr = print_stderr

    def run_file(
        self,
        file: Path,
        code_type: str,
    ) -> str:
        if not file.is_file():
            return f"{file} is not a file."
        code_type = self._check_code_type(code_type)
        cmd = shlex.split(
            self._CODE_EXECUTE_CMD[code_type].format(file_name=str(file)))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate()
        if self.print_stdout:
            print("======stdout======\n")
            print(Fore.GREEN + stdout + Fore.RESET + "\n")
            print("==================\n")
        if self.print_stderr:
            print("======stderr======\n")
            print(Fore.RED + stderr + Fore.RESET + "\n")
            print("==================\n")
        return (f"stdout: {stdout}\n"
                f"stderr: {stderr}\n")

    def run_generated_code(
        self,
        code: str,
        code_type: str,
    ) -> str:
        code_type = self._check_code_type(code_type)

        # Print code for security checking
        if self.user_check:
            print(f"The following {code_type} code will run in your computer:")
            print(Fore.CYAN + code + Fore.RESET)
            while True:
                choice = input("Running code? [Y/n]:").lower()
                if choice in ["y", "yes", "ye", ""]:
                    break
                elif choice in ["no", "n"]:
                    raise InterpreterError("User does not run the code")
        temp_file_path = self._create_temp_file(
            code=code, extension=self._CODE_EXTENTION[code_type])

        result = self.run_file(temp_file_path, code_type)

        temp_file_path.unlink()
        return result

    def _create_temp_file(self, code: str, extension: str) -> Path:
        with tempfile.NamedTemporaryFile(mode="w", delete=False,
                                         suffix=f".{extension}") as f:
            f.write(code)
            name = f.name
        return Path(name)

    def _check_code_type(self, code_type: str) -> str:
        if code_type not in self._CODE_TYPE:
            raise InterpreterError(
                f"Unsupported code type {code_type}. Currently CAMEL can "
                f" only execute {', '.join(self._CODE_EXTENTION.keys())}.")
        return self._CODE_TYPE[code_type]
