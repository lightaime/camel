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
import importlib
import inspect
import pkgutil
from typing import Callable, List, Optional, Union

from camel.toolkits.base import BaseToolkit
from camel.toolkits.openai_function import OpenAIFunction


class ToolManager:
    r"""
    A class representing a manager for dynamically loading and accessing
    toolkits.

    The ToolManager loads all callable toolkits from the `camel.toolkits`
    package and provides methods to list, retrieve, and search them as
    OpenAIFunction objects.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ToolManager, cls).__new__(
                cls, *args, **kwargs
            )
        return cls._instance

    def __init__(self):
        r"""
        Initializes the ToolManager and loads all available toolkits.
        """
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.toolkits = {}
            self.toolkit_classes = {}
            self.toolkit_class_methods = {}
            self._load_toolkits()
            self._load_toolkit_class_and_methods()

    def _load_toolkits(self):
        r"""
        Dynamically loads all toolkit functions from the `camel.toolkits`
        package.

        For each module in the package, it checks for functions decorated with
        `@export_to_toolkit`, which adds the `_is_exported` attribute.
        """
        package = importlib.import_module('camel.toolkits')
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f'camel.toolkits.{module_name}')

            for name, func in inspect.getmembers(module, inspect.isfunction):
                if (
                    hasattr(func, '_is_exported')
                    and func.__module__ == module.__name__
                ):
                    self.toolkits[name] = func

    def _load_toolkit_class_and_methods(self):
        r"""
        Dynamically loads all classes and their exported methods from the
        `camel.toolkits` package.

        For each module in the package, it identifies public classes. For each
        class, it collects only those methods that are decorated with
        `@export_to_toolkit`, which adds the `_is_exported` attribute.
        """
        package = importlib.import_module('camel.toolkits')

        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f'camel.toolkits.{module_name}')

            for name, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ == module.__name__:
                    self.toolkit_classes[name] = cls

                    self.toolkit_class_methods[name] = {
                        method_name: method
                        for method_name, method in inspect.getmembers(
                            cls, inspect.isfunction
                        )
                        if callable(method) and hasattr(method, '_is_exported')
                    }

    def register_tool(
        self,
        toolkit_obj: Union[Callable, object, List[Union[Callable, object]]],
    ) -> List[OpenAIFunction] | str:
        r"""
        Registers a toolkit function or instance and adds it to the toolkits
        list. If the input is a list, it processes each element in the list.

        Parameters:
            toolkit_obj (Union[Callable, object, List[Union[Callable,
                object]]]): The toolkit function(s) or instance(s) to be
                registered.

        Returns:
            Union[List[OpenAIFunction], str]: Returns a list of OpenAIFunction
                instances if the registration is successful. Otherwise,
                returns a message indicating the failure reason.
        """
        res_openai_functions = []
        res_info = ""

        # If the input is a list, process each element
        if isinstance(toolkit_obj, list):
            for obj in toolkit_obj:
                res_openai_functions_part, res_info_part = (
                    self._register_single_tool(obj)
                )
                res_openai_functions.extend(res_openai_functions_part)
                res_info += res_info_part
        else:
            res_openai_functions, res_info = self._register_single_tool(
                toolkit_obj
            )

        return res_openai_functions if res_openai_functions else res_info

    def _register_single_tool(
        self, toolkit_obj: Union[Callable, object]
    ) -> tuple[List[OpenAIFunction], str]:
        r"""
        Helper function to register a single toolkit function or instance.

        Parameters:
            toolkit_obj (Union[Callable, object]): The toolkit function or
                instance to be processed.

        Returns:
            Tuple: A list of OpenAIFunction instances and a result message.
        """
        res_openai_functions = []
        res_info = ""
        if callable(toolkit_obj):
            res = self.add_toolkit_from_function(toolkit_obj)
            if "successfully" in res:
                res_openai_functions.append(OpenAIFunction(toolkit_obj))
            res_info += res
        else:
            res = self.add_toolkit_from_instance(
                **{toolkit_obj.__class__.__name__: toolkit_obj}
            )
            if "Successfully" in res and hasattr(toolkit_obj, 'get_tools'):
                res_openai_functions.extend(toolkit_obj.get_tools())
            res_info += res
        return res_openai_functions, res_info

    def add_toolkit_from_function(self, toolkit_func: Callable):
        r"""
        Adds a toolkit function to the toolkits list.

        Parameters:
            toolkit_func (Callable): The toolkit function to be added.

        Returns:
            Str: A message indicating whether the addition was successful or
                if it  failed.
        """
        if not callable(toolkit_func):
            return "Provided argument is not a callable function."

        func_name = toolkit_func.__name__

        if not func_name:
            return "Function must have a valid name."

        self.toolkits[func_name] = toolkit_func

        return f"Toolkit '{func_name}' added successfully."

    def add_toolkit_from_instance(self, **kwargs):
        r"""
        Add a toolkit class instance to the tool list.
        Custom instance names are supported here.

        Parameters:
            kwargs: The toolkit class instance to be added. Keyword arguments
                where each value is expected to be an instance of BaseToolkit.

        Returns:
            Str: A message indicating whether the addition was successful or
                if it failed.
        """
        messages = []
        for toolkit_instance_name, toolkit_instance in kwargs.items():
            if isinstance(toolkit_instance, BaseToolkit):
                for attr_name in dir(toolkit_instance):
                    attr = getattr(toolkit_instance, attr_name)

                    if callable(attr) and hasattr(attr, '_is_exported'):
                        method_name = f"{toolkit_instance_name}_{attr_name}"

                        self.toolkits[method_name] = attr
                        messages.append(f"Successfully added {method_name}.")
            else:
                messages.append(
                    f"Failed to add {toolkit_instance_name}: "
                    + "Not an instance of BaseToolkit."
                )

        return "\n".join(messages)

    def list_toolkits(self):
        r"""
        Lists the names of all available toolkits.

        Returns:
            List[str]: A list of all toolkit function names available for use.
        """
        return list(self.toolkits.keys())

    def list_toolkit_classes(self):
        r"""
        Lists the names of all available toolkit classes along with their
        methods.

        Returns:
            List[str]: A list of strings in the format 'ClassName: method1,
                method2, ...'.
        """
        result = []

        for class_name, methods in self.toolkit_class_methods.items():
            if methods:
                methods_str = ', '.join(methods)

                formatted_string = f"{class_name}: {methods_str}"

                result.append(formatted_string)

        return result

    def get_toolkit(self, name: str) -> OpenAIFunction | str:
        r"""
        Retrieves the specified toolkit as an OpenAIFunction object.

        Args:
            name (str): The name of the toolkit function to retrieve.

        Returns:
            OpenAIFunction: The toolkit wrapped as an OpenAIFunction.

        Raises:
            ValueError: If the specified toolkit is not found.
        """
        toolkit = self.toolkits.get(name)
        if toolkit:
            return OpenAIFunction(toolkit)
        return f"Toolkit '{name}' not found."

    def get_toolkits(self, names: list[str]) -> list[OpenAIFunction] | str:
        r"""
        Retrieves the specified toolkit as an OpenAIFunction object.

        Args:
            name (str): The name of the toolkit function to retrieve.

        Returns:
            OpenAIFunctions (list): The toolkits wrapped as an OpenAIFunction.

        Raises:
            ValueError: If the specified toolkit is not found.
        """
        toolkits: list[OpenAIFunction] = []
        for name in names:
            current_toolkit = self.toolkits.get(name)
            if current_toolkit:
                toolkits.append(current_toolkit)
        if len(toolkits) > 0:
            return toolkits
        return "Toolkits are not found."

    def _default_search_algorithm(
        self, keyword: str, description: str
    ) -> bool:
        r"""
        Default search algorithm.

        Args:
            keyword (str): The keyword to search for.
            description (str): The description to search within.

        Returns:
            bool: True if a match is found based on similarity, False
            otherwise.
        """
        return keyword.lower() in description.lower()

    def search_toolkits(
        self,
        keyword: str,
        algorithm: Optional[Callable[[str, str], bool]] = None,
    ) -> List[str]:
        r"""
        Searches for toolkits based on a keyword in their descriptions using
        the provided search algorithm.

        Args:
            keyword (str): The keyword to search for in toolkit descriptions.
            algorithm (Callable[[str, str], bool], optional): A custom search
            algorithm function
                that accepts the keyword and description and returns a boolean.
                Defaults to fuzzy matching.

        Returns:
            List[str]: A list of toolkit names whose descriptions match the
            keyword.
        """
        if algorithm is None:
            algorithm = self._default_search_algorithm

        matching_toolkits = []
        for name, func in self.toolkits.items():
            openai_func = OpenAIFunction(func)
            description = openai_func.get_function_description()
            if algorithm(keyword, description) or algorithm(keyword, name):
                matching_toolkits.append(name)

        return matching_toolkits
