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
"""OmegaPRM v2 implementation with enhanced search capabilities."""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from model_utils import LM
from search_tree import CandidatePool, SearchTree, State


def separate_steps(steps: List[str], mode: str = 'join') -> Any:
    """Helper function to separate reasoning steps."""
    delimiter = "\n\n"
    if mode == 'join':
        if not isinstance(steps, list):
            raise TypeError(
                "For 'join' mode, 'steps' must be a list of strings."
            )
        return delimiter.join(steps)
    elif mode == 'split':
        if not isinstance(steps, str):
            raise TypeError("For 'split' mode, 'steps' must be a string.")
        return steps.split(delimiter)
    else:
        raise ValueError("Mode should be either 'join' or 'split'.")


def check_correctness(generated_response: str, expected_answer: str) -> bool:
    """Helper function to check correctness of a generated response."""
    sentences = re.split(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s',
        generated_response.strip(),
    )
    last_sentence = sentences[-1] if sentences else ''
    return expected_answer.strip() in last_sentence.strip()


class OmegaPRMV2:
    """Enhanced OmegaPRM algorithm implementation."""

    def __init__(
        self,
        model: LM,
        c_puct: float = 0.2,
        alpha: float = 0.5,
        beta: float = 0.9,
        L: int = 500,
        k: int = 5,
        N: int = 100,
        rollout_budget: int = 1000,
        save_data_tree: bool = True,
    ):
        """
        Initialize the OmegaPRM algorithm.

        Args:
            model (LM): The language model instance
            c_puct (float): Exploration constant
            alpha (float): Weight for MC(s)
            beta (float): Length penalty
            L (int): Maximum solution length
            k (int): Number of rollouts for Monte Carlo estimation
            N (int): Maximum search count
            rollout_budget (int): Maximum number of rollouts
            save_data_tree (bool): Whether to save search tree data
        """
        self.model = model
        self.expected_answer = None
        self.c_puct = c_puct
        self.alpha = alpha
        self.beta = beta
        self.L = L
        self.k = k
        self.N = N
        self.rollout_budget = rollout_budget
        self.save_data_tree = save_data_tree

        self.T = SearchTree()
        self.C = CandidatePool()

        self.n = 0
        self.total_rollouts = 0

    def reset(self):
        """Reset internal state variables."""
        self.T = SearchTree()
        self.C = CandidatePool()
        self.n = 0
        self.total_rollouts = 0

    def monte_carlo_estimation(self, state: State):
        """Perform Monte Carlo estimation for state."""
        if state.MC is not None:  # Skip if already estimated
            return

        # Generate k rollouts
        rollouts = self.model.generate_rollouts(
            state.get_full_solution(), num_copies=self.k
        )
        self.total_rollouts += self.k

        # Check correctness of each rollout
        for rollout in rollouts:
            if check_correctness(rollout, self.expected_answer):
                state.add_rollout(rollout)
            else:
                state.add_incorrect_rollout(rollout)

        # Compute MC score
        state.MC = state.correct_rollouts / self.k if self.k > 0 else 0

    def compute_Q(self, state: State, rollout: str) -> float:
        """Compute Q(s, r) value."""
        if rollout in state.Q:
            return state.Q[rollout]

        # Count words in rollout
        word_count = len(rollout.split())
        length_penalty = math.pow(self.beta, word_count / self.L)
        mc_weight = (
            math.pow(self.alpha, 1 - state.MC) if state.MC is not None else 1
        )

        q_value = mc_weight * length_penalty
        state.Q[rollout] = q_value
        return q_value

    def compute_U(self, state: State) -> float:
        """Compute U(s) value."""
        total_visits = (
            sum(child.N for child in state.children) if state.children else 0
        )
        return self.c_puct * math.sqrt(total_visits) / (1 + state.N)

    def compute_selection_score(self, state: State, rollout: str) -> float:
        """Compute selection score: Score(s, r) = Q(s, r) + U(s)."""
        return self.compute_Q(state, rollout) + self.compute_U(state)

    def selection_phase(self) -> Tuple[Optional[State], Optional[str]]:
        """Select (state, rollout) with highest score from candidate pool."""
        return self.C.pop()

    def add_correct_rollout_to_tree(self, parent_state: State, rollout: str):
        """Add correct rollout to the tree as a child of parent_state."""
        new_state = State(rollout, parent_state)
        parent_state.children.append(new_state)
        self.T.add_state(new_state)
        return new_state

    def binary_search_incorrect_step(
        self, s_ast: State, steps: List[str], left: int, right: int
    ):
        """Recursively perform binary search to find incorrect steps."""
        if left > right:
            return

        mid = (left + right) // 2
        partial_solution = separate_steps(steps[: mid + 1])
        full_solution = s_ast.get_full_solution() + partial_solution

        # Check if this partial solution is correct
        if check_correctness(full_solution, self.expected_answer):
            # The error must be in the latter half
            new_state = self.add_correct_rollout_to_tree(
                s_ast, partial_solution
            )
            self.binary_search_incorrect_step(
                new_state, steps[mid + 1 :], 0, len(steps[mid + 1 :]) - 1
            )
        else:
            # The error must be in this half
            self.binary_search_incorrect_step(s_ast, steps, left, mid - 1)

    def expansion_phase_binary_search(self, parent_state: State, rollout: str):
        """Expansion phase using binary search to find correct parts."""
        steps = separate_steps(rollout, mode='split')
        self.binary_search_incorrect_step(
            parent_state, steps, 0, len(steps) - 1
        )

    def maintenance_phase(self, state: State):
        """Update statistics and candidate pool for incorrect rollouts."""
        state.N += 1

        # Re-compute selection scores for all incorrect rollouts
        for rollout in state.incorrect_rollouts:
            score = self.compute_selection_score(state, rollout)
            self.C.add_or_update(state, rollout, score)

    def run(self, question: str, answer: str) -> List[Dict[str, Any]]:
        """Execute the OmegaPRM algorithm."""
        self.reset()
        self.expected_answer = answer

        # Initialize root state
        root_state = State(question)
        self.T.add_state(root_state)
        self.monte_carlo_estimation(root_state)

        collected_data = []
        while self.n < self.N and self.total_rollouts < self.rollout_budget:
            # Selection phase
            s_ast, r_ast = self.selection_phase()
            if s_ast is None or r_ast is None:
                break

            # Expansion phase
            self.expansion_phase_binary_search(s_ast, r_ast)

            # Maintenance phase
            self.maintenance_phase(s_ast)

            # Collect data if enabled
            if self.save_data_tree:
                collected_data.append(
                    {
                        'iteration': self.n,
                        'total_rollouts': self.total_rollouts,
                        'tree_structure': self.T.root.get_text_with_labels()
                        if self.T.root
                        else None,
                    }
                )

            self.n += 1

        return collected_data
