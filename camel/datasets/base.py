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
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional

from pydantic import BaseModel, Field

from camel.logger import get_logger

logger = get_logger(__name__)


class DataPoint(BaseModel):
    r"""A single data point in the dataset.

    Attributes:
        question: The question to pose to the LLM
        ground_truth: The ground truth solution
        final_answer: The final answer
        raw_markdown: Raw markdown content for generating rewards/hints
        difficulty: Difficulty level of the question
        chain_of_thought: Long chain-of-thought reasoning
        verified: Whether this chain-of-thought is verified
        metadata: Additional metadata about the data point
    """

    question: str = Field(..., description="The question to pose to the LLM")
    ground_truth: str = Field(..., description="The ground truth solution")
    final_answer: Any = Field(..., description="The final answer")
    raw_markdown: str = Field(
        ..., description="Raw markdown content for generating rewards/hints"
    )
    difficulty: str = Field(
        ..., description="Difficulty level of the question"
    )
    chain_of_thought: Optional[str] = Field(
        None, description="Long chain-of-thought reasoning"
    )
    verified: bool = Field(
        False, description="Whether this chain-of-thought is verified"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata about the data point"
    )


class BaseDataset(ABC):
    r"""A dataset contains questions and ground truth data for training.
    It can be either static (e.g., MATH dataset) or generative
    (using an LLM to generate questions).
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_cache_size: int = int(1e9),  # 1GB default
        preload: bool = False,
        shuffle: bool = True,
        **kwargs,
    ):
        r"""Initialize the dataset.

        Args:
            cache_dir: Directory to cache dataset files.
                (default: :obj:`None`)
            max_cache_size: Maximum cache size in bytes.
                (default: :obj:`1e9` (1GB))
            preload: Whether to preload dataset into memory.
                (default: :obj:`False`)
            shuffle: Whether to shuffle dataset on load.
                (default: :obj:`True`)
            **kwargs: Additional dataset parameters.

        Note:
            The dataset must be initialized by calling setup() before use

        Raises:
            ValueError: If max_cache_size is negative
        """
        self._current_index = 0
        self._is_setup = False
        self._cache: Dict[int, DataPoint] = {}

        # Store configuration parameters
        self._cache_dir = str(cache_dir) if cache_dir is not None else None
        self._max_cache_size = int(max_cache_size)
        self._preload = bool(preload)
        self._shuffle = bool(shuffle)

        # Store all parameters in metadata dict for compatibility
        self._metadata = {
            'cache_dir': self._cache_dir,
            'max_cache_size': self._max_cache_size,
            'preload': self._preload,
            'shuffle': self._shuffle,
            **kwargs,
        }

        if self._max_cache_size < 0:
            raise ValueError("max_cache_size must be positive")

    async def setup(self) -> None:
        r"""Set up the dataset with necessary resources.

        This method:
        1. Creates cache directory if needed
        2. Preloads data if configured
        3. Initializes shuffling if enabled
        4. Validates dataset integrity

        Raises:
            OSError: If cache directory creation fails
            Exception: If dataset initialization fails
        """
        if self._is_setup:
            logger.debug(f"{self.__class__.__name__} already initialized")
            return

        try:
            # Create cache directory if specified
            if self._cache_dir:
                try:
                    os.makedirs(self._cache_dir, exist_ok=True)
                    logger.debug(f"Created cache directory: {self._cache_dir}")
                except OSError as e:
                    logger.error(
                        f"Failed to create cache directory "
                        f"{self._cache_dir}: {e}"
                    )
                    raise

            # Preload dataset if configured
            if self._preload:
                logger.info(f"Preloading {len(self)} items into cache")
                try:
                    for i in range(len(self)):
                        self._cache[i] = await self._get_item_async(i)
                except Exception as e:
                    logger.error(f"Failed to preload dataset: {e}")
                    raise

            # Initialize shuffle indices if needed
            if self._shuffle:
                self._shuffle_indices = list(range(len(self)))
                random.shuffle(self._shuffle_indices)
                logger.debug("Initialized shuffle indices")

            self._is_setup = True
            logger.info(f"{self.__class__.__name__} initialized successfully")

        except Exception as e:
            logger.error(f"Error during {self.__class__.__name__} setup: {e}")
            await self.cleanup()
            raise

    async def cleanup(self) -> None:
        r"""Clean up dataset resources.

        This method handles cleanup of resources and resets the dataset state.
        It ensures:
        1. All resources are properly released
        2. State is reset to initial
        3. Cache is cleared
        4. Cleanup happens even if errors occur
        """
        if not self._is_setup:
            return

        try:
            # Reset dataset state
            self._current_index = 0

            # Clear cache
            self._cache.clear()
            if hasattr(self, '_shuffle_indices'):
                del self._shuffle_indices

            # Clear metadata while preserving init config
            init_config = {
                'cache_dir': self._cache_dir,
                'max_cache_size': self._max_cache_size,
                'preload': self._preload,
                'shuffle': self._shuffle,
            }
            self._metadata = init_config

            logger.info(f"{self.__class__.__name__} cleaned up successfully")

        except Exception as e:
            logger.error(
                f"Error during {self.__class__.__name__} cleanup: {e}"
            )
            raise

        finally:
            # Always mark as uninitialized, even if cleanup fails
            self._is_setup = False

    async def _get_item_async(self, idx: int) -> DataPoint:
        r"""Async wrapper around __getitem__ for preloading.

        Args:
            idx: Index of the item to get

        Returns:
            DataPoint from the dataset
        """
        return self[idx]

    def sample(self) -> DataPoint:
        r"""Sample a random datapoint from the dataset.

        Returns:
            A randomly sampled DataPoint

        Raises:
            RuntimeError: If dataset is not initialized
        """
        if not self._is_setup:
            raise RuntimeError(
                f"{self.__class__.__name__} must be initialized "
                "before sampling"
            )

        if self._shuffle:
            idx = self._shuffle_indices[self._current_index]
        else:
            idx = self._current_index

        self._current_index = (self._current_index + 1) % len(self)
        return self[idx]

    @abstractmethod
    def __len__(self) -> int:
        r"""Return the size of the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> DataPoint:
        r"""Get an item from the dataset.

        Args:
            idx: Index of the item to get

        Returns:
            DataPoint containing question, ground truth, and metadata

        Raises:
            IndexError: If idx is out of bounds
            ValidationError: If data point is invalid
        """
        pass

    def __iter__(self) -> Iterator[DataPoint]:
        r"""Create an iterator over the dataset."""
        return self

    def __next__(self) -> DataPoint:
        r"""Get the next item from the dataset."""
        if self._current_index >= len(self):
            self._current_index = 0
            raise StopIteration
        item = self[self._current_index]
        self._current_index += 1
        return item

    @property
    def metadata(self) -> Dict[str, Any]:
        r"""Get dataset metadata."""
        return self._metadata.copy()

    def reset(self) -> None:
        r"""Reset the dataset iterator and re-shuffle if enabled."""
        self._current_index = 0
        if self._shuffle and self._is_setup:
            random.shuffle(self._shuffle_indices)
