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

from pprint import pprint

from datasets import Dataset

from camel.interpreters import SubprocessInterpreter
from camel.verifiers import CodeVerifier


def main():
    print("\nExample 1: Basic Function Test")
    interpreter = SubprocessInterpreter(require_confirm=False)
    verifier = CodeVerifier(interpreter=interpreter)
    result = verifier.verify(
        {
            "code": ["def add(a, b): return a + b"],
            "language": ["python"],
            "test_cases": [
                [
                    {"inputs": {"a": 1, "b": 2}, "expected": {"add(a, b)": 3}},
                    {
                        "inputs": {"a": -1, "b": 1},
                        "expected": {"add(a, b)": 0},
                    },
                ]
            ],
        }
    )
    pprint(result[0]["verification_result"])

    # Example 2: Multiple Solutions
    print("\nExample 2: Multiple Solutions")
    data = Dataset.from_dict(
        {
            "code": [
                "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                "def factorial(n): return n * factorial(n-1) if n > 1 else 1",
            ],
            "language": ["python", "python"],
            "test_cases": [
                [{"inputs": {"n": 5}, "expected": {"factorial(n)": 120}}],
                [{"inputs": {"n": 5}, "expected": {"factorial(n)": 120}}],
            ],
        }
    )
    results = verifier.verify(data)
    for i, result in enumerate(results):
        print(f"Solution {i+1} result:", result["verification_result"])

    # Example 3: Using subprocess interpreter
    print("\nExample 3: External Imports")
    interpreter = SubprocessInterpreter(require_confirm=False)
    verifier = CodeVerifier(interpreter=interpreter)
    result = verifier.verify(
        {
            "code": [
                """
import numpy as np
def process_array():
    arr = np.array([1, 2, 3])
    return arr.mean()
        """
            ],
            "language": ["python"],
            "test_cases": [
                [{"inputs": {}, "expected": {"process_array()": 2.0}}]
            ],
        }
    )
    print("Result:", result[0]["verification_result"])

    # Example 4: Multi-threaded verification
    print("\nExample 4: Multi-threaded verification")
    interpreter = SubprocessInterpreter(require_confirm=False)
    verifier = CodeVerifier(interpreter=interpreter)
    results = verifier.verify(
        {
            "code": [
                "def square(x):\n    return x * x\nresult = square(4)",
                "def cube(x):\n    return x * x * x\nresult = cube(3)",
                "def double(x):\n    return x + x\nresult = double(5)",
                "def half(x):\n    return x / 2\nresult = half(8)",
            ],
            "language": ["python"] * 4,
            "test_cases": [
                [{"inputs": {}, "expected": {"result": 16}}],
                [{"inputs": {}, "expected": {"result": 27}}],
                [{"inputs": {}, "expected": {"result": 10}}],
                [{"inputs": {}, "expected": {"result": 4.0}}],
            ],
        }
    )
    for i, result in enumerate(results):
        print(f"\nFunction {i+1} result:", result["verification_result"])

    # Example 5: Syntax Error
    print("\nExample 5: Syntax Error")
    result = verifier.verify(
        {
            "code": ["def broken_function(x: return x"],  # Syntax error
            "language": ["python"],
        }
    )
    print("Result:", result[0]["verification_result"])

    # Example 6: Test Case Validation
    print("\nExample 6: Test Case Validation")
    # Invalid test case (not a list)
    result = verifier.verify(
        {
            "code": ["def add(a, b): return a + b"],
            "language": ["python"],
            "test_cases": {"not": "a list"},  # Invalid: not a list
        }
    )
    print("Invalid test case (not a list):", result[0]["verification_result"])


if __name__ == "__main__":
    main()


"""
Example Output:

Example 1: Basic Function Test
Verifying code: 100%|██████████| 1/1 [00:00<00:00, 16.84 examples/s]
{
    'details': {
        'test_count': 2,
        'tests': [
            {
                'output': 'Test passed: 3\n',
                'status': 'passed',
                'test_case': 1
            },
            {
                'output': 'Test passed: 0\n',
                'status': 'passed',
                'test_case': 2
            }
        ]
    },
    'error': None,
    'passed': True,
    'test_results': [True, True]
}

Example 2: Multiple Solutions
Verifying code (num_proc=2):100%|██████████| 2/2 [00:00<00:00,15.64 examples/s]
Solution 1 result: {
    'details': {
        'test_count': 1,
        'tests': [
            {
                'output': 'Test passed: 120\n',
                'status': 'passed',
                'test_case': 1
            }
        ]
    },
    'error': None,
    'passed': True,
    'test_results': [True]
}
Solution 2 result: {
    'details': {
        'test_count': 1,
        'tests': [
            {
                'output': 'Test passed: 120\n',
                'status': 'passed',
                'test_case': 1
            }
        ]
    },
    'error': None,
    'passed': True,
    'test_results': [True]
}

Example 3: External Imports
Verifying code: 100%|██████████| 1/1 [00:00<00:00,  9.29 examples/s]
Result: {
    'details': {
        'test_count': 1,
        'tests': [
            {
                'output': 'Test passed: 2.0\n',
                'status': 'passed',
                'test_case': 1
            }
        ]
    },
    'error': None,
    'passed': True,
    'test_results': [True]
}

Example 4: Multi-threaded verification
Verifying code(num_proc=4):100%|██████████| 4/4 [00:00<00:00, 35.86 examples/s]

Function 1 result: {
    'details': {
        'test_count': 1,
        'tests': [
            {
                'output': 'Test passed: 120\n',
                'status': 'passed',
                'test_case': 1
            }
        ]
    },
    'error': None,
    'passed': True,
    'test_results': [True]
}

Function 2 result: {
    'details': {
        'test_count': 1,
        'tests': [
            {
                'output': 'Test passed: 120\n',
                'status': 'passed',
                'test_case': 1
            }
        ]
    },
    'error': None,
    'passed': True,
    'test_results': [True]
}

Example 5: Syntax Error
Verifying code: 100%|██████████| 1/1 [00:00<00:00, 504.24 examples/s]
Result: {
    'details': {
        'line': 1,
        'offset': 24,
        'text': 'def broken_function(x: return x\n',
        'type': 'syntax_error'
    },
    'error': 'Syntax error: invalid syntax (<string>, line 1)',
    'passed': False,
    'test_results': []
}

Example 6: Test Case Validation
Verifying code:   0%|         | 0/1 [00:00<?, ? examples/s]

2025-02-10 07:41:14,094 - 
camel.verifiers.code_verifier - WARNING - Invalid test cases: Test cases must 
be a list
2025-02-10 07:41:14,094 - 
camel.verifiers.code_verifier - ERROR - Handling execution error: Invalid test 
cases: Test cases must be a list

Verifying code: 100%|██████████| 1/1 [00:00<00:00, 662.71 examples/s]
Result: {
    'details': {
        'message': 'Invalid test cases: Test cases must be provided as a list',
        'type': 'execution_error'
    },
    'error': 'Invalid test cases: Test cases must be provided as a list',
    'passed': False,
    'test_results': []
}
"""
