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
import ast
import difflib
import importlib
from typing import Any, Callable, Dict, List, Mapping, Optional, Union


class InterpreterError(ValueError):
    r"""An error raised when the interpreter cannot evaluate a Python
    expression, due to syntax error or unsupported operations.
    """

    pass


class PythonInterpreter():
    r"""A customized python interpreter to control the execution of
    LLM-generated codes. The interpreter makes sure the code can only execute
    functions given in action space and import white list. It also supports
    fuzzy variable matching to reveive uncertain input variable name.

    Args:
        action_space (Dict[str, Any]): The dictionary mapping action names to
            the correponding functions or classes.
        import_white_list (List[str], optional): The list storing the
            importable python modules or functions which can be imported in
            code. All submodules and functions of the modules in the list are
            importable. Any other import statements will be rejected. :obj:`.`
            is the seperator between the module and its submodule or its
            function name. (default: :obj:`[]`)
    """

    def __init__(self, action_space: Dict[str, Any],
                 import_white_list: List[str] = []) -> None:
        self.action_space = action_space
        self.state = self.action_space.copy()
        self.fuzz_state = {}
        self.import_white_list = import_white_list

    def execute(self, code: str, state: Dict[str, Any] = {},
                fuzz_state: Dict[str,
                                 Any] = {}, keep_state: bool = True) -> Any:
        r""" Execute the input python codes in a security environment.

        Args:
            code (str): Generated python code to be executed.
            state (Dict[str, Any], optional): External variables that may be
                used in the generated code. (default: :obj:`{}`)
            fuzz_state (Dict[str, Any], optional): External varibles that do
                not have certain varible names. The interpreter will use fuzzy
                matching to these varibales. For example, if :obj:`fuzz_state`
                has a variable :obj:`image`, the generated code can use
                :obj:`input_image` to access it. (default: :obj:`{}`)
            keep_state (bool, optional):  If :obj:`True`, :obj:`state` and
                :obj:`fuzz_state` will be kept for later execution. Otherwise,
                they will be cleared. (default: :obj:`True`)

        Returns:
            Any: An evaluation value of the last line in the code.
        """
        self.state.update(state)
        self.fuzz_state.update(fuzz_state)
        try:
            expression = ast.parse(code)
        except SyntaxError as e:
            raise InterpreterError(f"Syntax error in code: {e}")

        result = None
        for idx, node in enumerate(expression.body):
            try:
                line_result = self._execute_ast(node)
            except InterpreterError as e:
                if not keep_state:
                    self.clear_state()
                msg = (f"Evaluation of the code stopped at line {idx}. "
                       f"See:\n{e}")
                # More information can be provided by `ast.unparse()`,
                # which is new in python 3.9.
                raise InterpreterError(msg)
            if line_result is not None:
                result = line_result

        if not keep_state:
            self.clear_state()

        return result

    def clear_state(self) -> None:
        r"""Initialize :obj:`state` and :obj:`fuzz_state`"""
        self.state = self.action_space.copy()
        self.fuzz_state = {}

    def _execute_ast(self, expression: ast.AST) -> Any:
        if isinstance(expression, ast.Assign):
            # Assignement -> evaluate the assignement which should
            # update the state. We return the variable assigned as it may
            # be used to determine the final result.
            return self._execute_assign(expression)
        elif isinstance(expression, ast.Attribute):
            value = self._execute_ast(expression.value)
            return getattr(value, expression.attr)
        elif isinstance(expression, ast.BinOp):
            # Binary Operator -> return the result value
            return self._execute_binop(expression)
        elif isinstance(expression, ast.Call):
            # Function call -> return the value of the function call
            return self._execute_call(expression)
        elif isinstance(expression, ast.Constant):
            # Constant -> just return the value
            return expression.value
        elif isinstance(expression, ast.Dict):
            # Dict -> evaluate all keys and values
            keys = [self._execute_ast(k) for k in expression.keys]
            values = [self._execute_ast(v) for v in expression.values]
            return dict(zip(keys, values))
        elif isinstance(expression, ast.Expr):
            # Expression -> evaluate the content
            return self._execute_ast(expression.value)
        elif isinstance(expression, ast.For):
            return self._execute_for(expression)
        elif isinstance(expression, ast.FormattedValue):
            # Formatted value (part of f-string) -> evaluate the content
            # and return
            return self._execute_ast(expression.value)
        elif isinstance(expression, ast.If):
            # If -> execute the right branch
            return self._execute_if(expression)
        elif isinstance(expression, ast.Import):
            # Import -> add imported names in self.state and return None.
            for module in expression.names:
                self._execute_import(module)
            return None
        elif isinstance(expression, ast.ImportFrom):
            module = expression.module
            for name in expression.names:
                self._execute_import(module, name)
            return None
        elif hasattr(ast, "Index") and isinstance(expression, ast.Index):
            return self._execute_ast(expression.value)
        elif isinstance(expression, ast.JoinedStr):
            return "".join(
                [str(self._execute_ast(v)) for v in expression.values])
        elif isinstance(expression, ast.List):
            # List -> evaluate all elements
            return [self._execute_ast(elt) for elt in expression.elts]
        elif isinstance(expression, ast.Name):
            # Name -> pick up the value in the state
            return self._execute_name(expression)
        elif isinstance(expression, ast.Subscript):
            # Subscript -> return the value of the indexing
            return self._execute_subscript(expression)
        elif isinstance(expression, ast.Tuple):
            return tuple([self._execute_ast(elt) for elt in expression.elts])
        elif isinstance(expression, ast.UnaryOp):
            # Binary Operator -> return the result value
            return self._execute_unaryop(expression)
        else:
            # For now we refuse anything else. Let's add things as we need
            # them.
            raise InterpreterError(
                f"{expression.__class__.__name__} is not supported.")

    def _execute_assign(self, assign: ast.Assign) -> Any:
        var_names = assign.targets
        result = self._execute_ast(assign.value)

        for var_name in var_names:
            self._assign(var_name, result)
        return result

    def _assign(self, target: Union[ast.Name, ast.Tuple], value: Any):
        if isinstance(target, ast.Name):
            self.state[target.id] = value
        elif isinstance(target, ast.Tuple):
            if len(value) != len(target):
                raise InterpreterError(f"Expected {len(target)} values but got"
                                       f"{len(value)}.")
            for v, r in zip(target.ctx, value):
                self.state[v.id] = r
        else:
            raise InterpreterError(f"Unsupport variable type. Expected"
                                   f"ast.Name or ast.Tuple, got"
                                   f"{type(target)} instead.")

    def _execute_call(self, call: ast.Call) -> Any:
        # callable_func = self._get_func_from_state(call.func)
        callable_func = self._execute_ast(call.func)

        # Todo deal with args
        args = [self._execute_ast(arg) for arg in call.args]
        kwargs = {
            keyword.arg: self._execute_ast(keyword.value)
            for keyword in call.keywords
        }
        return callable_func(*args, **kwargs)

    def _get_func_from_state(self, func: Union[ast.Attribute,
                                               ast.Name]) -> Callable:

        if not isinstance(func, (ast.Attribute, ast.Name)):
            raise InterpreterError(
                f"It is not permitted to invoke functions than the action "
                f"space (tried to execute {func} of type {type(func)}.")

        access_list = []
        while not isinstance(func, ast.Name):
            access_list = [func.attr] + access_list
            func = func.value
        access_list = [func.id] + access_list

        func_name = access_list[0]
        found_func = self.state.get(func_name, None)
        for name in access_list[1:]:
            if found_func:
                try:
                    found_func = getattr(found_func, name)
                except AttributeError as e:
                    raise InterpreterError(
                        f"AttributeError in generated code ({e}).")
            else:
                func_name += f".{name}"
                if func_name in self.state:
                    found_func = self.state[func_name]

        if not found_func:
            raise InterpreterError(
                f"It is not permitted to invoke functions than the action"
                f"space (tried to execute {func}).")

        return found_func

    def _execute_subscript(self, subscript):
        index = self._execute_ast(subscript.slice)
        value = self._execute_ast(subscript.value)
        if isinstance(value, (list, tuple)):
            return value[int(index)]
        if index in value:
            return value[index]
        if isinstance(index, str) and isinstance(value, Mapping):
            close_matches = difflib.get_close_matches(index,
                                                      list(value.keys()))
            if len(close_matches) > 0:
                return value[close_matches[0]]

        raise InterpreterError(f"Could not index {value} with '{index}'.")

    def _execute_name(self, name: ast.Name):
        return self._get_value_from_state(name.id)

    def _execute_condition(self, condition):
        if len(condition.ops) > 1:
            raise InterpreterError(
                "Cannot evaluate conditions with multiple operators")

        left = self._execute_ast(condition.left)
        comparator = condition.ops[0]
        right = self._execute_ast(condition.comparators[0])

        if isinstance(comparator, ast.Eq):
            return left == right
        elif isinstance(comparator, ast.NotEq):
            return left != right
        elif isinstance(comparator, ast.Lt):
            return left < right
        elif isinstance(comparator, ast.LtE):
            return left <= right
        elif isinstance(comparator, ast.Gt):
            return left > right
        elif isinstance(comparator, ast.GtE):
            return left >= right
        elif isinstance(comparator, ast.Is):
            return left is right
        elif isinstance(comparator, ast.IsNot):
            return left is not right
        elif isinstance(comparator, ast.In):
            return left in right
        elif isinstance(comparator, ast.NotIn):
            return left not in right
        else:
            raise InterpreterError(f"Operator not supported: {comparator}")

    def _execute_if(self, if_statement: ast.If):
        result = None
        if self._execute_condition(if_statement.test):
            for line in if_statement.body:
                line_result = self._execute_ast(line)
                if line_result is not None:
                    result = line_result
        else:
            for line in if_statement.orelse:
                line_result = self._execute_ast(line)
                if line_result is not None:
                    result = line_result
        return result

    def _execute_for(self, for_statement: ast.For):
        result = None
        for value in self._execute_ast(for_statement.iter):
            self._assign(for_statement.target, value)
            for line in for_statement.body:
                line_result = self._execute_ast(line)
                if line_result is not None:
                    result = line_result

        return result

    def _execute_import(self, module: Union[str, ast.alias],
                        import_name: Optional[ast.alias] = None) -> None:
        module_name = module if isinstance(module, str) else module.name
        found_name = False
        tmp_name = ""
        full_name = (module_name if import_name is None else module_name +
                     f".{import_name.name}")
        for name in full_name.split("."):
            tmp_name += name if tmp_name == "" else f".{name}"
            if tmp_name in self.import_white_list:
                found_name = True
                break

        if not found_name:
            raise InterpreterError(f"It is not permitted to import modules "
                                   f"than module white list (try to import "
                                   f"{module_name}).")

        if import_name is None:
            alias = (module.asname or module_name)
            self.state[alias] = importlib.import_module(module_name)
        else:
            imported_module = importlib.import_module(module_name)
            alias = import_name.asname or import_name.name
            self.state[alias] = getattr(imported_module, import_name.name)

    def _execute_binop(self, binop: ast.BinOp):
        left = self._execute_ast(binop.left)
        operator = binop.op
        right = self._execute_ast(binop.right)

        if isinstance(operator, ast.Add):
            return left + right
        elif isinstance(operator, ast.Sub):
            return left - right
        elif isinstance(operator, ast.Mult):
            return left * right
        elif isinstance(operator, ast.Div):
            return left / right
        elif isinstance(operator, ast.FloorDiv):
            return left // right
        elif isinstance(operator, ast.Mod):
            return left % right
        elif isinstance(operator, ast.Pow):
            return left**right
        elif isinstance(operator, ast.LShift):
            return left << right
        elif isinstance(operator, ast.RShift):
            return left >> right
        elif isinstance(operator, ast.MatMult):
            return left @ right
        else:
            raise InterpreterError(f"Operator not supported: {operator}")

    def _execute_unaryop(self, unaryop: ast.UnaryOp):
        operand = self._execute_ast(unaryop.operand)
        operator = unaryop.op

        if isinstance(operator, ast.UAdd):
            return +operand
        elif isinstance(operator, ast.USub):
            return -operand
        elif isinstance(operator, ast.Not):
            return not operand
        else:
            raise InterpreterError(f"Operator not supported: {operator}")

    def _get_value_from_state(self, key: str) -> Any:
        if key in self.state:
            return self.state[key]
        else:
            close_matches = (difflib.get_close_matches(
                key, list(self.fuzz_state.keys()), n=1))
            if close_matches:
                return self.fuzz_state[close_matches[0]]
            else:
                raise InterpreterError(f"The variable `{key}` is not defined.")
