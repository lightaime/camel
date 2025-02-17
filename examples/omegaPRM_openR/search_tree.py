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
"""Search tree and state management for OmegaPRM."""

import heapq
import itertools
from typing import Any, Dict, List, Optional, Tuple


class State:
    """Represents a state in the search tree."""

    def __init__(self, solution_prefix: str, parent: Optional['State'] = None):
        self.solution_prefix = (
            solution_prefix  # Solution prefix as a single string
        )
        self.parent = parent  # Reference to the parent state
        self.N = 0  # Visit count (number of times selected)
        self.total_rollouts = (
            0  # Total number of rollouts generated from this state
        )
        self.correct_rollouts = 0  # Number of correct rollouts
        self.MC: Optional[float] = None  # Monte Carlo estimation (c/k)
        self.Q: Dict[
            str, float
        ] = {}  # Q(s, r): estimated value for each rollout
        self.R: List[str] = []  # Set of all rollouts from this state
        self.incorrect_rollouts: List[str] = []  # List of incorrect rollouts
        self.children: List['State'] = []  # List of child states

    def add_rollout(self, rollout: str):
        """Add a correct rollout to this state."""
        if rollout not in self.R:
            self.R.append(rollout)
            self.total_rollouts += 1
            self.correct_rollouts += 1

    def add_incorrect_rollout(self, rollout: str):
        """Add an incorrect rollout to this state."""
        if rollout not in self.incorrect_rollouts:
            self.incorrect_rollouts.append(rollout)
            self.total_rollouts += 1

    def get_full_solution(self) -> str:
        """Return the full solution by concatenating
        all parent solution prefixes."""
        if self.parent is None:
            return self.solution_prefix
        return self.parent.get_full_solution() + self.solution_prefix

    def get_new_text(self) -> str:
        """Return the new text added at this node compared to the parent."""
        return self.solution_prefix

    def get_text_with_labels(self) -> Dict[str, Any]:
        """Return a nested dictionary with text and MC values."""
        result = {
            'text': self.get_new_text(),
            'mc_value': self.MC,
            'children': [],
        }
        for child in self.children:
            result['children'].append(child.get_text_with_labels())
        return result


class SearchTree:
    """Represents the search tree for OmegaPRM."""

    def __init__(self):
        self.root: Optional[State] = None
        self.nodes: List[State] = []  # List of all states

    def add_state(self, state: State):
        """Add a new state to the search tree."""
        self.nodes.append(state)
        if self.root is None:
            self.root = state


class CandidatePool:
    """Priority queue with update capability for managing candidate states."""

    def __init__(self):
        self.heap: List[
            Tuple[float, int]
        ] = []  # Heap of (-priority, unique_id)
        self.entry_finder: Dict[
            int, Tuple[float, int]
        ] = {}  # Maps unique_id to (-priority, unique_id)
        self.counter = itertools.count()  # Unique sequence count
        self.id_to_rollout: Dict[
            int, Tuple[State, str]
        ] = {}  # Maps unique_id to (state, rollout)
        self.latest_id_per_rollout: Dict[
            Tuple[int, str], int
        ] = {}  # Maps (state_id, rollout) to unique_id

    def add_or_update(self, state: State, rollout: str, priority: float):
        """Add a new rollout or update the priority of an existing rollout."""
        state_id = id(state)
        rollout_key = (state_id, rollout)

        # Remove previous entry if it exists
        if rollout_key in self.latest_id_per_rollout:
            old_id = self.latest_id_per_rollout[rollout_key]
            if old_id in self.entry_finder:
                del self.entry_finder[old_id]
                del self.id_to_rollout[old_id]

        # Add new entry
        unique_id = next(self.counter)
        entry = (-priority, unique_id)
        self.entry_finder[unique_id] = entry
        self.id_to_rollout[unique_id] = (state, rollout)
        self.latest_id_per_rollout[rollout_key] = unique_id
        heapq.heappush(self.heap, entry)

    def pop(self) -> Tuple[Optional[State], Optional[str]]:
        """Pop the rollout with the highest priority."""
        while self.heap:
            neg_priority, unique_id = heapq.heappop(self.heap)
            if unique_id in self.entry_finder:
                del self.entry_finder[unique_id]
                state, rollout = self.id_to_rollout[unique_id]
                del self.id_to_rollout[unique_id]
                rollout_key = (id(state), rollout)
                if self.latest_id_per_rollout.get(rollout_key) == unique_id:
                    del self.latest_id_per_rollout[rollout_key]
                return state, rollout
        return None, None

    def is_empty(self) -> bool:
        """Check if the candidate pool is empty."""
        return len(self.entry_finder) == 0
