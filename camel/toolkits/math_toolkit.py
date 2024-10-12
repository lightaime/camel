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

from typing import List

from camel.toolkits.base import BaseToolkit
from camel.toolkits.openai_function import OpenAIFunction
from camel.utils.commons import export_to_toolkit


@export_to_toolkit
def add(a: int, b: int) -> int:
    r"""Adds two numbers.

    Args:
        a (int): The first number to be added.
        b (int): The second number to be added.

    Returns:
        int: The sum of the two numbers.
    """
    return a + b


@export_to_toolkit
def sub(a: int, b: int) -> int:
    r"""Do subtraction between two numbers.

    Args:
        a (int): The minuend in subtraction.
        b (int): The subtrahend in subtraction.

    Returns:
        int: The result of subtracting :paramref:`b` from :paramref:`a`.
    """
    return a - b


@export_to_toolkit
def mul(a: int, b: int) -> int:
    r"""Multiplies two integers.

    Args:
        a (int): The multiplier in the multiplication.
        b (int): The multiplicand in the multiplication.

    Returns:
        int: The product of the two numbers.
    """
    return a * b


MATH_FUNCS = [
    OpenAIFunction(add),
    OpenAIFunction(sub),
    OpenAIFunction(mul),
]


class MathToolkit(BaseToolkit):
    r"""A class representing a toolkit for mathematical operations. This
    class provides methods for basic mathematical operations such as addition,
    subtraction, and multiplication.
    """

    def get_tools(self) -> List[OpenAIFunction]:
        r"""Returns a list of OpenAIFunction objects representing the
        functions in the toolkit.

        Returns:
            List[OpenAIFunction]: A list of OpenAIFunction objects
                representing the functions in the toolkit.
        """
        return MATH_FUNCS
