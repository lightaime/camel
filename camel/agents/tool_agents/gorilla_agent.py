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
from typing import Any, Dict, Optional, Tuple

import openai
from colorama import Fore

from camel.agents.tool_agents import BaseToolAgent
from camel.prompts import ToolAgentsPromptTemplateDict
from camel.typing import ModelType
from camel.utils import print_text_animated


# flake8: noqa :E501
class GorillaAgent(BaseToolAgent):
    r"""Tool agent of <a href="https://github.com/ShishirPatil/gorilla">Gorilla
    </a>, a LLM that comes up with the semantically- and syntactically- correct
    API to invoke. This agent uses the API information generated by Gorrila as
    prompts in other generic LLM for code generation.

    Args:
        name (str): The name of the agent.
        gorilla_model (str, optional): Gorilla model type. Currently it can be
            :obj:`gorilla-7b-hf-v0`, :obj:`gorilla-7b-tf-v0` and
            :obj:`gorilla-7b-th-v0` for Hugging Face, Tensorflow v2 and Torch
            Hub APIs, repectively. (default: :obj:`gorilla-7b-hf-v0`)
        code_model (ModelType, optional): The LLM model to use for generating
            code invoking APIs. (default :obj:`ModelType.GPT_4`)
        remote (bool, optional): Flag indicating whether to run Gorilla
            remotely. Notice that this option does not affect whether the
            generated API running remotely. (default: :obj:`True`)
        logger_color (Any): The color of the logger displayed to the user.
            (default: :obj:`Fore.CYAN`)
    """

    def __init__(
        self,
        name: str,
        gorilla_model: str = "gorilla-7b-hf-v0",
        code_model: ModelType = ModelType.GPT_4,
        remote: bool = True,
        logger_color: Any = Fore.CYAN,
    ) -> None:

        self.name = name
        self.gorilla_model = gorilla_model
        self.code_model = code_model
        self.remote = remote
        self.logger_color = logger_color
        self.description = f"""The `{self.name}` is a tool agent that can perform a variety of tasks.
Given a natural language query, `{self.name}` comes up with the semantically- and syntactically- correct API to invoke.:
Therefore, you should provide a meta task description by a natural language when you use `{self.name}` to do a specified task.
You should not assign the result of `{self.name}.step()` to any variable because this function has no return value.
You should not use any attribute in `{self.name}`. Instead, pass what you want to do by `task` argument in `{self.name}.step()`.

Here are some python code examples of what you can do with this agent:

```
# Text to image
{self.name}.step(task="Draw me a picture of rivers and lakes.", meta_task="I would like to generate an image according to a prompt.")

# Image transformation
{self.name}.step(task="Transform the picture './image.png' to add an island", meta_task="I would like to modify an image given an initial image and a prompt.")

# Unconditional image captioning
{self.name}.step("Can you caption the picture './image.png'?", meta_task="I would like to caption the image.")

# Text summarization
{self.name}.step("Summarize the following text: `./document.txt`", meta_task="I would like to summarize a long text in one or a few sentences.")

# Text generation
{self.name}.step("Generate a science fiction story about a gorilla in English and save it to `./story.txt`.", meta_task="I would like to generate a text based on a prompt.")
```"""

    def reset(self) -> None:
        r"""Resets the chat history of the agent."""
        pass

    def step(
        self,
        task: str,
        meta_task: str,
        remote: Optional[bool] = None,
    ) -> None:
        r"""Runs the agent in single execution mode.

        Args:
            task (str): Concrete task for the agent to perform. Example:
                "Draw me a gorilla."
            meta_task (str): Abstract task used to ask Gorilla to find
                an API for the agent to invoke. Exmaple: "I would like
                to generate an image according to a prompt."
            remote (bool, optional): Flag indicating whether to run the agent
                remotely. Overrides the default setting. (default: :obj:`None`)
        """
        if remote is None:
            remote = self.remote

        if remote is False:
            raise RuntimeError(
                "Running Gorilla locally has not implemented yet.")

        # Change api base to access gorilla remotely
        original_base = openai.api_base
        original_key = openai.api_key
        openai.api_base = "http://34.132.127.197:8000/v1"  # temperal API for research
        openai.api_key = "EMPTY"

        # Get API prompt by gorilla
        completion = openai.ChatCompletion.create(
            model=self.gorilla_model, messages=[{
                "role": "user",
                "content": meta_task
            }])
        api_prompt = completion.choices[0].message.content
        print_text_animated(self.logger_color + f"API info:\n{api_prompt}")

        openai.api_base = original_base
        openai.api_key = original_key

        # Get code by GPT
        code_prompt = ToolAgentsPromptTemplateDict.GORILLA_GENERATE_CODE.format(
            task=task, api_info=api_prompt)
        completion = openai.ChatCompletion.create(
            model=self.code_model.value, messages=[{
                "role": "user",
                "content": code_prompt
            }])
        result = completion.choices[0].message.content
        print_text_animated(self.logger_color + f"Code generation:\n{result}")

        # TODO: Refactor
        _, code = self.get_explanation_and_code(result)
        self.execute_code(code)

    @staticmethod
    def execute_code(code_string: str, global_vars: Dict = None) -> str:
        r"""Executes the given code string. This code is copied from
        embodied_agent.py.

        Args:
            code_string (str): The code string to execute.
            global_vars (Dict, optional): The global variables to use during
                code execution. (default: :obj:`None`)

        Returns:
            str: The execution results.
        """
        # TODO: Refactor this with `CodePrompt.execute`.
        try:
            # Execute the code string
            import io
            import sys
            output_str = io.StringIO()
            sys.stdout = output_str

            global_vars = global_vars or globals()
            local_vars = {}
            exec(
                code_string,
                global_vars,
                local_vars,
            )
            sys.stdout = sys.__stdout__
            output_str.seek(0)

            # If there was no error, return the output
            return (f"- Python standard output:\n{output_str.read()}\n"
                    f"- Local variables:\n{str(local_vars)}")
        except Exception:
            import traceback
            traceback_str = traceback.format_exc()
            sys.stdout = sys.__stdout__
            # If there was an error, return the error message
            return f"Traceback:\n{traceback_str}"

    @staticmethod
    def get_explanation_and_code(
            text_prompt: str) -> Tuple[str, Optional[str]]:
        r"""Extracts the explanation and code from the text prompt.
        This code is copied from embodied_agent.py.

        Args:
            text_prompt (str): The text prompt containing the explanation and
                code.

        Returns:
            Tuple[str, Optional[str]]: The extracted explanation and code.
        """
        # TODO: Refactor this with `BaseMessage.extract_text_and_code_prompts`.
        lines = text_prompt.split("\n")
        idx = 0
        while idx < len(lines) and not lines[idx].lstrip().startswith("```"):
            idx += 1
        explanation = "\n".join(lines[:idx]).strip()
        if idx == len(lines):
            return explanation, None

        idx += 1
        start_idx = idx
        while not lines[idx].lstrip().startswith("```"):
            idx += 1
        code = "\n".join(lines[start_idx:idx]).strip()

        return explanation, code
