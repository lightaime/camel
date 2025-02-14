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
import json
import logging
import re
import textwrap
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from camel.toolkits import FunctionTool
from camel.types import Choice
from camel.types.agents import ToolCallingRecord

logger = logging.getLogger(__name__)


def generate_tool_prompt(tool_schema_list: List[Dict[str, Any]]) -> str:
    r"""Generates a tool prompt based on the provided tool schema list.

    Returns:
        str: A string representing the tool prompt.
    """
    tool_prompts = []

    for tool in tool_schema_list:
        tool_info = tool["function"]
        tool_name = tool_info["name"]
        tool_description = tool_info["description"]
        tool_json = json.dumps(tool_info, indent=4)

        prompt = (
            f"Use the function '{tool_name}' to '{tool_description}':\n"
            f"{tool_json}\n"
        )
        tool_prompts.append(prompt)

    tool_prompt_str = "\n".join(tool_prompts)

    final_prompt = textwrap.dedent(
        f"""\
        You have access to the following functions:

        {tool_prompt_str}

        If you choose to call a function ONLY reply in the following format with no prefix or suffix:

        <function=example_function_name>{{"example_name": "example_value"}}</function>

        Reminder:
        - Function calls MUST follow the specified format, start with <function= and end with </function>
        - Required parameters MUST be specified
        - Only call one function at a time
        - Put the entire function call reply on one line
        - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls.
        """  # noqa: E501
    )
    return final_prompt


def _parse_tool_response(response: str) -> Optional[Dict[str, Any]]:
    r"""Parses the tool response to extract the function name and
    arguments.

    Args:
        response (str): The response from the model containing the
            function call.

    Returns:
        Optional[Dict[str, Any]]: The parsed function name and arguments
            if found, otherwise :obj:`None`.
    """
    function_regex = r"<function=(\w+)>(.*?)</function>"
    match = re.search(function_regex, response)

    if match:
        function_name, args_string = match.groups()
        try:
            args = json.loads(args_string)
            return {"function": function_name, "arguments": args}
        except json.JSONDecodeError as error:
            logger.error(f"Error parsing function arguments: {error}")
            return None
    return None


def extract_tool_call(
    self, response: Any
) -> Optional[ChatCompletionMessageToolCall]:
    r"""Extract the tool call from the model response, if present.

    Args:
        response (Any): The model's response object.

    Returns:
        Optional[ChatCompletionMessageToolCall]: The parsed tool call if
            present, otherwise None.
    """
    # Check if the response contains tool calls
    if (
        self.has_tools
        and not self.model_type.support_native_tool_calling
        and "</function>" in response.choices[0].message.content
    ):
        parsed_content = _parse_tool_response(
            response.choices[0].message.content
        )
        if parsed_content:
            return ChatCompletionMessageToolCall(
                id=str(uuid.uuid4()),
                function=Function(
                    arguments=str(parsed_content["arguments"]).replace(
                        "'", '"'
                    ),
                    name=str(parsed_content["function"]),
                ),
                type="function",
            )
    elif (
        self.has_tools
        and self.model_type.support_native_tool_calling
        and response.choices[0].message.tool_calls
    ):
        return response.choices[0].message.tool_calls[0]

    # No tool call found
    return None


def safe_model_dump(obj) -> Dict[str, Any]:
    r"""Safely dump a Pydantic model to a dictionary.

    This method attempts to use the `model_dump` method if available,
    otherwise it falls back to the `dict` method.
    """
    # Check if the `model_dump` method exists (Pydantic v2)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # Fallback to `dict()` method (Pydantic v1)
    elif hasattr(obj, "dict"):
        return obj.dict()
    else:
        raise TypeError("The object is not a Pydantic model")


def convert_to_function_tool(
    tool: Union[FunctionTool, Callable],
) -> FunctionTool:
    r"""Convert a tool to a FunctionTool from Callable."""
    return tool if isinstance(tool, FunctionTool) else FunctionTool(tool)


def convert_to_schema(
    tool: Union[FunctionTool, Callable, Dict[str, Any]],
) -> Dict[str, Any]:
    r"""Convert a tool to a schema from Callable or FunctionTool."""
    if isinstance(tool, FunctionTool):
        return tool.get_openai_tool_schema()
    elif callable(tool):
        return FunctionTool(tool).get_openai_tool_schema()
    else:
        return tool


def get_info_dict(
    session_id: Optional[str],
    usage: Optional[Dict[str, int]],
    termination_reasons: List[str],
    num_tokens: int,
    tool_calls: List[ToolCallingRecord],
) -> Dict[str, Any]:
    r"""Returns a dictionary containing information about the chat session.

    Args:
        session_id (str, optional): The ID of the chat session.
        usage (Dict[str, int], optional): Information about the usage of
            the LLM.
        termination_reasons (List[str]): The reasons for the termination
            of the chat session.
        num_tokens (int): The number of tokens used in the chat session.
        tool_calls (List[ToolCallingRecord]): The list of function
            calling records, containing the information of called tools.

    Returns:
        Dict[str, Any]: The chat session information.
    """
    return {
        "id": session_id,
        "usage": usage,
        "termination_reasons": termination_reasons,
        "num_tokens": num_tokens,
        "tool_calls": tool_calls,
    }


def handle_logprobs(choice: Choice) -> Optional[List[Dict[str, Any]]]:
    if choice.logprobs is None:
        return None

    tokens_logprobs = choice.logprobs.content

    if tokens_logprobs is None:
        return None

    return [
        {
            "token": token_logprob.token,
            "logprob": token_logprob.logprob,
            "top_logprobs": [
                (top_logprob.token, top_logprob.logprob)
                for top_logprob in token_logprob.top_logprobs
            ],
        }
        for token_logprob in tokens_logprobs
    ]
