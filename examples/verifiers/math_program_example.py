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

from camel.interpreters import SubprocessInterpreter
from camel.verifiers import CodeVerifier


def main():
    interpreter = SubprocessInterpreter(require_confirm=False)
    verifier = CodeVerifier(interpreter=interpreter)

    # Example 1: Matrix Operations
    matrix_test = {
        "code": [
            """
import numpy as np

def matrix_multiply(A, B):
    return np.dot(A, B)
"""
        ],
        "language": ["python"],
        "test_cases": [
            [
                {
                    "inputs": {"A": [[1, 2], [3, 4]], "B": [[5, 6], [7, 8]]},
                    "expected": {
                        "np.allclose(matrix_multiply(A, B), "
                        "np.array([[19, 22], [43, 50]]))": True
                    },
                }
            ]
        ],
    }

    print("\nTesting Matrix Operations:")
    result = verifier.verify(matrix_test)
    print(result[0]["verification_result"])

    # Example 2: Linear System Solver
    linear_system_test = {
        "code": [
            """
import numpy as np

def solve_linear_system(A, b):
    return np.linalg.solve(A, b)
"""
        ],
        "language": ["python"],
        "test_cases": [
            [
                {
                    "inputs": {"A": [[2, 1], [1, 3]], "b": [4, 5]},
                    "expected": {
                        "np.allclose(solve_linear_system(A, b), "
                        "np.array([1.4, 1.2]))": True
                    },
                }
            ]
        ],
    }

    print("\nTesting Linear System Solver:")
    result = verifier.verify(linear_system_test)
    print(result[0]["verification_result"])

    # Example 3: Eigenvalue Decomposition
    eigenvalue_test = {
        "code": [
            """
import numpy as np

def compute_eigendecomposition(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # Sort by eigenvalues to ensure consistent order
    idx = eigenvalues.argsort()
    return eigenvalues[idx], eigenvectors[:, idx]
"""
        ],
        "language": ["python"],
        "test_cases": [
            [
                {
                    "inputs": {"A": [[4, -1], [2, 1]]},
                    "expected": {
                        "np.allclose(compute_eigendecomposition(A)[0], "
                        "np.array([2., 3.]))": True
                    },
                }
            ]
        ],
    }

    print("\nTesting Eigenvalue Decomposition:")
    result = verifier.verify(eigenvalue_test)
    print(result[0]["verification_result"])

    # Example 4: Singular Value Decomposition
    svd_test = {
        "code": [
            """
import numpy as np

def compute_svd(A):
    U, s, Vh = np.linalg.svd(A)
    return s  # Return singular values
"""
        ],
        "language": ["python"],
        "test_cases": [
            [
                {
                    "inputs": {"A": [[1, 2], [3, 4], [5, 6]]},
                    "expected": {
                        "np.allclose(compute_svd(A), "
                        "np.array([9.52551809, 0.51430058]))": True
                    },
                }
            ]
        ],
    }

    print("\nTesting SVD:")
    result = verifier.verify(svd_test)
    print(result[0]["verification_result"])

    # Example 5: Optimization (Minimization)
    optimization_test = {
        "code": [
            """
import numpy as np
from scipy.optimize import minimize

def optimize_quadratic(x0):
    # Minimize f(x,y) = x^2 + y^2 + 2x + 4y + 4
    def objective(x):
        return x[0]**2 + x[1]**2 + 2*x[0] + 4*x[1] + 4
    
    result = minimize(objective, x0, method='BFGS')
    return result.x
"""
        ],
        "language": ["python"],
        "test_cases": [
            [
                {
                    "inputs": {"x0": [0, 0]},
                    "expected": {
                        "np.allclose(optimize_quadratic(x0), "
                        "np.array([-1., -2.]))": True
                    },
                }
            ]
        ],
    }

    print("\nTesting Optimization:")
    result = verifier.verify(optimization_test)
    print(result[0]["verification_result"])

    # Example 6: Numerical Integration
    integration_test = {
        "code": [
            """
import numpy as np
from scipy import integrate

def integrate_function(a, b):
    # Integrate sin(x^2) from a to b
    def f(x):
        return np.sin(x**2)
    
    result, _ = integrate.quad(f, a, b)
    return result
"""
        ],
        "language": ["python"],
        "test_cases": [
            [
                {
                    "inputs": {"a": 0, "b": 1},
                    "expected": {
                        "np.allclose(integrate_function(a, b), "
                        "0.3102683017233811)": True
                    },
                }
            ]
        ],
    }

    print("\nTesting Numerical Integration:")
    result = verifier.verify(integration_test)
    print(result[0]["verification_result"])


if __name__ == "__main__":
    main()


"""
Example Output:

Testing Matrix Operations:
Map: 100%|██████████| 1/1 [00:00<00:00,  8.49 examples/s]
{
    'details': {
        'test_count': 1,
        'tests': [{
            'output': 'Test passed: True\n',
            'status': 'passed',
            'test_case': 1
        }]
    },
    'error': None,
    'passed': True,
    'test_results': [True]
}

Testing Linear System Solver:
Map: 100%|██████████| 1/1 [00:00<00:00,  6.27 examples/s]
{
    'details': {
        'test_count': 1,
        'tests': [{
            'output': 'Test passed: True\n',
            'status': 'passed',
            'test_case': 1
        }]
    },
    'error': None,
    'passed': True,
    'test_results': [True]
}

Testing Eigenvalue Decomposition:
Map: 100%|██████████| 1/1 [00:00<00:00, 11.11 examples/s]
{
    'details': {
        'test_count': 1,
        'tests': [{
            'output': 'Test passed: True\n',
            'status': 'passed',
            'test_case': 1
        }]
    },
    'error': None,
    'passed': True,
    'test_results': [True]
}

Testing SVD:
Map: 100%|██████████| 1/1 [00:00<00:00, 10.40 examples/s]
{
    'details': {
        'test_count': 1,
        'tests': [{
            'output': 'Test passed: True\n',
            'status': 'passed',
            'test_case': 1
        }]
    },
    'error': None,
    'passed': True,
    'test_results': [True]
}

Testing Optimization:
Map: 100%|██████████| 1/1 [00:00<00:00,  2.97 examples/s]
{
    'details': {
        'test_count': 1,
        'tests': [{
            'output': 'Test passed: True\n',
            'status': 'passed',
            'test_case': 1
        }]
    },
    'error': None,
    'passed': True,
    'test_results': [True]
}

Testing Numerical Integration:
Map: 100%|██████████| 1/1 [00:00<00:00,  3.61 examples/s]
{
    'details': {
        'test_count': 1,
        'tests': [{
            'output': 'Test passed: True\n',
            'status': 'passed',
            'test_case': 1
        }]
    },
    'error': None,
    'passed': True,
    'test_results': [True]
}
"""
