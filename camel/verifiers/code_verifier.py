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

import os
from typing import Any, Dict, List, Union

from datasets import Dataset

from camel.interpreters import BaseInterpreter, InterpreterError
from camel.logger import logging

logger = logging.getLogger(__name__)


class CodeVerifier:
    r"""Verifier for code solutions.

    This verifier checks code solutions by:
    1. Validating syntax
    2. Running test cases
    3. Verifying outputs against expected results
    """

    def __init__(
        self,
        interpreter: BaseInterpreter,
        require_confirmation: bool = False,
    ) -> None:
        r"""Initialize the code verifier.

        Args:
            interpreter (BaseInterpreter): The interpreter instance to use for
                code execution
            require_confirmation (bool, optional): Whether to require user
                confirmation before execution. (default: :obj:`False`)
        """
        super().__init__()
        self.interpreter = interpreter
        logger.info(
            "Initialized CodeVerifier with interpreter %s", interpreter
        )

    def verify(self, data: Union[Dataset, Dict[str, Any]]) -> Dataset:
        r"""Verify code solutions.

        Args:
            data (Union[Dataset, Dict[str, Any]]): Data containing code to
                verify

        Returns:
            Dataset: Dataset with verification results added
        """
        if isinstance(data, dict):
            data = Dataset.from_dict(data)

        logger.info("Starting verification of %d examples", len(data))

        def verify_single(example: Dict[str, Any]) -> Dict[str, Any]:
            r"""Verify a single code example.

            Args:
                example (Dict[str, Any]): Example containing code to verify

            Returns:
                Dict[str, Any]: Example with verification results added
            """
            code = example.get("code", "")
            language = example.get("language", "python")

            # Validate language is supported by interpreter
            supported_languages = self.interpreter.supported_code_types()
            if language not in supported_languages:
                logger.warning(
                    "Language %s not supported by interpreter %s. "
                    "Supported languages: %s",
                    language,
                    self.interpreter.__class__.__name__,
                    supported_languages,
                )
                return self._handle_execution_error(
                    example,
                    InterpreterError(f"Language {language} not supported"),
                )

            test_cases = example.get("test_cases", [])

            try:
                self._validate_test_cases(test_cases)
            except ValueError as e:
                logger.warning("Invalid test cases: %s", e)
                return self._handle_execution_error(
                    example, ValueError(f"Invalid test cases: {e!s}")
                )

            logger.debug(
                "Verifying code in %s with %d test cases",
                language,
                len(test_cases),
            )

            # Check syntax first
            try:
                if language == "python":
                    compile(code, '<string>', 'exec')
            except SyntaxError as e:
                logger.warning("Syntax error in code: %s", e)
                return self._handle_syntax_error(example, e)

            try:
                return self._run_test_cases(
                    example, code, language, test_cases
                )
            except Exception as e:
                logger.error("Execution error: %s", e)
                return self._handle_execution_error(example, e)

        # For Parallelization
        default_cpus = max(1, min(8, (os.cpu_count() or 1) // 2))
        num_proc = min(default_cpus, len(data))
        logger.info("Using %d processes for parallel verification", num_proc)

        return data.map(
            verify_single, num_proc=num_proc, desc="Verifying code"
        )

    def _prepare_test_code(
        self,
        code: str,
        test_case: Dict[str, Any],
    ) -> str:
        r"""Prepare code with test case inputs and assertions.

        Args:
            code (str): Original code to test
            test_case (Dict[str, Any]): Test case configuration

        Returns:
            str: Complete test code with assertions
        """
        logger.debug(
            "Preparing test code with inputs: %s", test_case.get("inputs")
        )
        full_code = [code]

        # Add test case setup
        test_setup = [
            f"{k} = {v!r}" for k, v in test_case.get("inputs", {}).items()
        ]
        if test_setup:
            full_code.extend(test_setup)

        # Add test assertions
        test_assertions = []
        for expr, expected in test_case.get("expected", {}).items():
            test_assertions.append(
                f"""
result = {expr}
if result != {expected!r}:
    raise AssertionError(
        f"Test failed:\\n  Expression: {expr}\\n  "
        f"Expected: {expected!r}\\n  Got: {{result}}"
    )
print(f"Test passed: {{result}}")
"""
            )

        if test_assertions:
            full_code.extend(test_assertions)

        return "\n".join(full_code)

    def _validate_test_cases(self, test_cases: List[Dict[str, Any]]) -> None:
        """Validate test cases structure.

        Args:
            test_cases (List[Dict[str, Any]]): List of test cases to validate

        Raises:
            ValueError: If test cases are malformed
        """
        if not isinstance(test_cases, list):
            raise ValueError("Test cases must be provided as a list")

        for i, test_case in enumerate(test_cases):
            if not isinstance(test_case, dict):
                raise ValueError(f"Test case {i} must be a dictionary")
            if not test_case.get("expected"):
                raise ValueError(
                    f"Test case {i} must contain 'expected' results"
                )

    def _handle_syntax_error(
        self,
        example: Dict[str, Any],
        error: SyntaxError,
    ) -> Dict[str, Any]:
        r"""Handle syntax errors in code verification.

        Args:
            example (Dict[str, Any]): The example being verified
            error (SyntaxError): The syntax error that occurred

        Returns:
            Dict[str, Any]: Updated example with error information
        """
        logger.warning(
            "Handling syntax error: %s at line %d", error, error.lineno
        )
        return {
            **example,
            "verification_result": {
                "passed": False,
                "test_results": [],
                "error": f"Syntax error: {error!s}",
                "details": {
                    "type": "syntax_error",
                    "line": error.lineno,
                    "offset": error.offset,
                    "text": error.text,
                },
            },
        }

    def _handle_execution_error(
        self,
        example: Dict[str, Any],
        error: Exception,
    ) -> Dict[str, Any]:
        r"""Handle execution errors in code verification.

        Args:
            example (Dict[str, Any]): The example being verified
            error (Exception): The execution error that occurred

        Returns:
            Dict[str, Any]: Updated example with error information
        """
        logger.error("Handling execution error: %s", error)
        example["verification_result"] = {
            "passed": False,
            "test_results": [],
            "error": str(error),
            "details": {
                "type": "execution_error",
                "message": str(error),
            },
        }
        return example

    def _run_test_cases(
        self,
        example: Dict[str, Any],
        code: str,
        language: str,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        r"""Run test cases for code verification.

        Args:
            example (Dict[str, Any]): The example being verified
            code (str): The code to test
            language (str): Programming language of the code
            test_cases (List[Dict[str, Any]]): List of test cases to run

        Returns:
            Dict[str, Any]: Updated example with test results
        """
        test_results = []
        test_details = []

        if test_cases:
            logger.info("Running %d test cases", len(test_cases))
            for i, test_case in enumerate(test_cases):
                logger.debug("Running test case %d", i + 1)
                test_code = self._prepare_test_code(code, test_case)
                try:
                    output = self.interpreter.run(test_code, language)
                    test_results.append(True)
                    test_details.append(
                        {
                            "test_case": i + 1,
                            "status": "passed",
                            "output": output,
                        }
                    )
                    logger.debug("Test case %d passed", i + 1)
                except Exception as e:
                    test_results.append(False)
                    test_details.append(
                        {
                            "test_case": i + 1,
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    logger.warning("Test case %d failed: %s", i + 1, e)

        passed = all(test_results) if test_results else True
        logger.info("All test cases %s", "passed" if passed else "failed")

        example["verification_result"] = {
            "passed": passed,
            "test_results": test_results,
            "error": None,
            "details": {
                "test_count": len(test_results),
                "tests": test_details,
            },
        }

        return example
