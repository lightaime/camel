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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from camel.agents import ChatAgent
from camel.datasets.base import BaseDataset
from camel.extractors.base import BaseExtractor
from camel.logger import get_logger
from camel.verifiers.base import BaseVerifier, VerificationResult
from camel.verifiers.models import Response, TaskType, VerificationStatus

logger = get_logger(__name__)


class Observation(BaseModel):
    r"""Environment observation.

    Attributes:
        question: The question posed to the LLM.
        context: Additional context for the question.
        metadata: Optional metadata about the observation.
    """

    question: str = Field(..., description="The question posed to the LLM")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context for the question"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata about the observation"
    )


class StepResult(BaseModel):
    r"""Result of an environment step.

    Attributes:
        observation: The next observation.
        reward: Dictionary of reward scores for different aspects.
        done: Whether the episode is complete.
        info: Additional information about the step.
    """

    observation: Observation = Field(..., description="The next observation")
    reward: Dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of reward scores for different aspects",
    )
    done: bool = Field(..., description="Whether the episode is complete")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional information about the step",
    )


class BaseEnvironment(ABC):
    r"""Base class for all RLVR training environments.

    An environment ties everything together. It:
    1. Holds state and manages curriculum progression
    2. Defines reward functions and hint generation
    3. Manages dataset and task selection
    4. Provides reset and step functions
    5. Handles verifier setup and teardown
    6. Enables proactive agent behavior
    7. Supports practice environment creation
    8. Facilitates chain-of-thought verification

    Key Features:
    - Curriculum learning with adaptive difficulty
    - Reward shaping based on solution quality
    - Hint generation from verified solutions
    - Task selection based on agent progress
    - Practice environment generation
    - Chain-of-thought validation
    """

    def __init__(
        self,
        dataset: BaseDataset,
        verifier: BaseVerifier,
        extractor: BaseExtractor,
        max_steps: Optional[int] = None,
        task_type: TaskType = TaskType.SOFTWARE_ENGINEERING,
        teacher_agent: Optional[ChatAgent] = None,
        generator_agent: Optional[ChatAgent] = None,
        curriculum_config: Optional[Dict[str, Any]] = None,
        practice_env_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        r"""Initialize the environment.

        Args:
            dataset: Dataset to sample questions from.
            verifier: Verifier to check responses.
            extractor: Extractor to process LLM responses.
            max_steps: Maximum steps per episode.
            task_type: Type of task being performed.
            teacher_agent: Optional agent for reward shaping and hints
            generator_agent: Optional agent for data generation
            curriculum_config: Configuration for curriculum learning including:
                - difficulty_levels: List of available difficulty levels
                - promotion_threshold: Score needed to advance
                - demotion_threshold: Score triggering level decrease
                - min_questions_per_level: Questions before promotion
            practice_env_config: Configuration for practice environments:
                - max_practice_envs: Maximum concurrent environments
                - difficulty_range: Allowed difficulty variation
                - focus_areas: Specific skills to practice
            **kwargs: Additional environment parameters.
        """
        self.dataset = dataset
        self.verifier = verifier
        self.extractor = extractor
        self.max_steps = max_steps
        self.task_type = task_type
        self.teacher_agent = teacher_agent
        self.generator_agent = generator_agent
        self._metadata = kwargs

        # State tracking
        self._is_setup: bool = False
        self._current_step: int = 0
        self._episode_ended: bool = False
        self._state: Dict[str, Any] = self._get_initial_state()
        self._last_observation: Optional[Observation] = None
        self._episode_history: List[Tuple[Observation, Response]] = []

    @abstractmethod
    async def setup(self) -> None:
        r"""Set up the environment, including verifier initialization."""
        if self._is_setup:
            return

        try:
            # Initialize core components
            if hasattr(self.verifier, 'setup'):
                await self.verifier.setup()
            if hasattr(self.dataset, 'setup'):
                await self.dataset.setup()
            if hasattr(self.extractor, 'setup'):
                await self.extractor.setup()

            # initialize agents if present
            if self.teacher_agent:
                await self.teacher_agent.reset()
            if self.generator_agent:
                await self.generator_agent.reset()

            self._is_setup = True
            logger.info('Environment setup completed successfully')
        except Exception as e:
            logger.error(f'Failed to setup environment: {e}')
            raise

    @abstractmethod
    async def teardown(self) -> None:
        r"""Clean up resources, including verifier teardown."""
        if not self._is_setup:
            return

        try:
            # Cleanup components
            if hasattr(self.verifier, 'cleanup'):
                await self.verifier.cleanup()
            if hasattr(self.dataset, 'cleanup'):
                await self.dataset.cleanup()
            if hasattr(self.extractor, 'cleanup'):
                await self.extractor.cleanup()

            self._is_setup = False
            logger.info('Environment teardown completed successfully')
        except Exception as e:
            logger.error(f'Failed to teardown environment: {e}')
            raise

    @abstractmethod
    async def reset(self) -> Observation:
        r"""Reset the environment to initial state.

        Returns:
            Initial observation for the episode
        """

        if not self._is_setup:
            await self.setup()

        # Reset state
        self._current_step = 0
        self._episode_ended = False
        self._episode_history = []
        self._state = self._get_initial_state()

        # Get initial observation
        observation = self._get_next_observation()
        if observation is None:
            raise RuntimeError("Failed to get initial observation")

        self._last_observation = observation

        return observation

    async def process_response(
        self, response: Response, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], VerificationResult]:
        r"""Process an response through extraction and verification.

        Args:
            response: Response containing the output.
            context: Optional context for extraction.

        Returns:
            Tuple of (extraction_result, verification_result).
        """
        extraction_result = await self.extractor.extract(
            response.llm_response,
            context or {},
        )

        verification_result = await self.verifier.verify(response)
        return extraction_result, verification_result

    @abstractmethod
    async def step(self, response: Response) -> StepResult:
        r"""Take a step in the environment.

        Args:
            response: Response containing output

        Returns:
            StepResult containing next observation, reward, done flag, and info
        """
        if self.max_steps and self._current_step >= self.max_steps:
            return StepResult(
                observation=self._get_terminal_observation(),
                reward={},
                done=True,
                info={"reason": "max_steps_reached"},
            )

        if not self._is_setup:
            raise RuntimeError("Environment not set up. Call setup() first.")
        if self._episode_ended:
            raise RuntimeError("Episode has ended. Call reset() first.")
        if self._last_observation is None:
            raise RuntimeError("No current observation. Call reset() first.")

        self._current_step += 1

        current_obs: Observation = self._last_observation
        self._episode_history.append((current_obs, response))

        # process response
        extraction_result, verification_result = await self.process_response(
            response
        )

        # compute rewards
        rewards = await self.compute_reward(
            response, extraction_result, verification_result
        )

        # check termination
        done = self._is_done()

        next_obs = (
            self._get_terminal_observation()
            if done
            else self._get_next_observation()
        )

        self._last_observation = next_obs
        self._episode_ended = done

        return StepResult(
            observation=next_obs,
            reward=rewards,
            done=done,
            info={
                "extraction_result": extraction_result,
                "verification_result": verification_result,
                "step": self._current_step,
                "state": self._state,
            },
        )

    @abstractmethod
    def _get_initial_state(self) -> Dict[str, Any]:
        r"""Get initial environment state."""

        return {
            "current_datapoint": None,
            "attempts": 0,
            "success_rate": 0.0,
            "rewards": [],
            "termination_reason": None,
        }

    @abstractmethod
    def _get_next_observation(self) -> Observation:
        r"""Get the next observation for the environment.

        Returns:
            Observation for the next step
        """

        if not self.dataset or len(self.dataset) == 0:
            logger.error('Dataset is empty or not initialized')
            return self._get_terminal_observation()

        try:
            datapoint_idx = self._current_step % len(self.dataset)
            datapoint = self.dataset[datapoint_idx]
            if not datapoint:
                logger.error(f'Invalid datapoint at index {datapoint_idx}')
                return self._get_terminal_observation()

            self._state['current_datapoint'] = datapoint
            required_attrs = [
                'question',
                'ground_truth',
                'final_answer',
                'difficulty',
                'chain_of_thought',
            ]

            # Validate required attributes
            missing_attrs = [
                attr for attr in required_attrs if not hasattr(datapoint, attr)
            ]
            if missing_attrs:
                logger.error(
                    f'Datapoint missing required attributes: {missing_attrs}'
                )
                return self._get_terminal_observation()

            observation = Observation(
                question=datapoint.question,
                context={
                    'ground_truth': datapoint.ground_truth,
                    'final_answer': datapoint.final_answer,
                    'difficulty': datapoint.difficulty,
                    'chain_of_thought': datapoint.chain_of_thought,
                },
                metadata={
                    'step': self._current_step,
                    'datapoint_id': str(datapoint_idx),
                    'verified': getattr(datapoint, 'verified', False),
                    **(
                        datapoint.metadata
                        if hasattr(datapoint, 'metadata')
                        and datapoint.metadata
                        else {}
                    ),
                },
            )
            logger.debug(
                f'Generated observation for step {self._current_step}'
            )
            return observation

        except (IndexError, AttributeError) as e:
            logger.error(f'Error getting next observation: {e}')
            return self._get_terminal_observation()
        except Exception as e:
            logger.error(f'Unexpected error getting next observation: {e}')
            return self._get_terminal_observation()

    @abstractmethod
    def _get_terminal_observation(self) -> Observation:
        r"""Get the terminal observation when episode ends.

        Returns:
            Terminal observation
        """
        return Observation(
            question="Episode completed",
            context={},
            metadata={"terminal": True, "final_step": self._current_step},
        )

    @abstractmethod
    async def compute_reward(
        self,
        response: Response,
        extraction_result: Dict[str, Any],
        verification_result: VerificationResult,
    ) -> Dict[str, float]:
        r"""Compute reward scores for different aspects of the response.

        Args:
            response: The response.
            extraction_result: Extracted information from response
            verification_result: Result from the verifier.

        Returns:
            Dictionary of reward scores for different aspects.
        """
        rewards = {}

        # Get success from verification result status
        verification_success = float(
            verification_result.status == VerificationStatus.SUCCESS
        )
        rewards["correctness"] = 1.0 if verification_success > 0.5 else 0.0

        # Update state
        self._state["rewards"].append(rewards)
        total_attempts = self._state["attempts"] + 1
        self._state["success_rate"] = (
            self._state["success_rate"] * (total_attempts - 1)
            + verification_success
        ) / total_attempts

        # Additional reward aspects can be added here
        # For example:
        # - Solution efficiency/structure
        # - Reasoning quality
        # ...

        return rewards

    def _is_done(self) -> bool:
        r"""Check if episode should terminate."""
        if self.max_steps and self._current_step >= self.max_steps:
            return True
        return False

    @property
    def metadata(self) -> Dict[str, Any]:
        r"""Get environment metadata."""
        return self._metadata.copy()

    @property
    def current_step(self) -> int:
        r"""Get current step number."""
        return self._current_step
