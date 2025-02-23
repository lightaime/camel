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
from types import TracebackType
from typing import Any, Dict, Optional, Type

from typing_extensions import Self

from camel.logger import get_logger
from camel.utils import BatchProcessor

logger = get_logger(__name__)


class BaseExtractor(ABC):
    r"""Base class for all response extractors.

    An extractor takes the response and extracts the relevant parts,
    converting them into a format that the verifier can handle.
    Implements async context manager protocol for proper resource management.
    """

    def __init__(
        self,
        cache_templates: bool = True,
        max_cache_size: int = 1000,
        extraction_timeout: float = 30.0,
        batch_size: int = 10,
        monitoring_interval: float = 5.0,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 85.0,
        **kwargs,
    ):
        r"""Initialize the extractor.

        Args:
            cache_templates: Whether to cache extraction templates.
                (default: :obj:`True`)
            max_cache_size: Maximum number of templates to cache.
                (default: :obj:`1000`)
            extraction_timeout: Maximum time for extraction in seconds.
                (default: :obj:`30.0`)
            batch_size: Size of batches for parallel extraction.
                (default: :obj:`10`)
            monitoring_interval: Interval in seconds between resource checks.
                (default: :obj:`5.0`)
            cpu_threshold: CPU usage percentage threshold for scaling down.
                (default: :obj:`80.0`)
            memory_threshold: Memory usage percentage threshold for scaling
                down. (default: :obj:`85.0`)
            **kwargs: Additional extractor parameters.

        Raises:
            ValueError: If invalid parameter values are provided
        """
        # Store all parameters in metadata dict for compatibility
        self._metadata = {
            'cache_templates': cache_templates,
            'max_cache_size': max_cache_size,
            'extraction_timeout': extraction_timeout,
            'batch_size': batch_size,
            'monitoring_interval': monitoring_interval,
            'cpu_threshold': cpu_threshold,
            'memory_threshold': memory_threshold,
            **kwargs,
        }

        self._is_setup = False
        self._cache: Dict[str, Any] = {}
        self._batch_processor: Optional[BatchProcessor] = None

        # Store configuration parameters
        self._cache_templates = cache_templates
        self._max_cache_size = max_cache_size
        self._extraction_timeout = extraction_timeout
        self._batch_size = batch_size
        self._monitoring_interval = monitoring_interval
        self._cpu_threshold = cpu_threshold
        self._memory_threshold = memory_threshold

    async def setup(self) -> None:
        r"""Set up the extractor with necessary resources.

        This method:
        1. Initializes template cache if enabled
        2. Sets up any parallel processing resources
        3. Validates extraction patterns

        Raises:
            RuntimeError: If initialization fails
        """
        if self._is_setup:
            logger.debug(f"{self.__class__.__name__} already initialized")
            return

        try:
            # Initialize template cache if enabled
            if self._cache_templates:
                self._template_cache: Dict[str, Any] = {}

            # Set up batch processing if needed
            if self._batch_size > 1:
                self._batch_processor = BatchProcessor(
                    initial_batch_size=self._batch_size,
                    monitoring_interval=self._monitoring_interval,
                    cpu_threshold=self._cpu_threshold,
                    memory_threshold=self._memory_threshold,
                )

            self._is_setup = True
            logger.info(f"{self.__class__.__name__} initialized successfully")

        except Exception as e:
            error_msg = f"Error during {self.__class__.__name__} setup: {e}"
            logger.error(error_msg)
            await self.cleanup()
            raise RuntimeError(error_msg) from e

    async def cleanup(self) -> None:
        r"""Clean up extractor resources.

        This method handles cleanup of resources and resets the extractor
        state.
        It ensures:
        1. All resources are properly released
        2. Template cache is cleared
        3. Parallel processing resources are shutdown
        4. State is reset to initial
        5. Cleanup happens even if errors occur

        Raises:
            RuntimeError: If cleanup fails (after resetting initialization
                state)
        """
        if not self._is_setup:
            logger.debug(
                f"{self.__class__.__name__} not initialized, skipping cleanup"
            )
            return

        errors = []
        try:
            # Clear template cache
            if hasattr(self, '_template_cache'):
                try:
                    self._template_cache.clear()
                except Exception as e:
                    errors.append(f"Failed to clear template cache: {e}")

            # Shutdown parallel processing
            if self._batch_processor is not None:
                try:
                    # Get final performance metrics before cleanup
                    metrics = self._batch_processor.get_performance_metrics()
                    logger.info(f"Batch processor final metrics: {metrics}")
                except Exception as e:
                    errors.append(
                        f"Failed to get batch processor metrics: {e}"
                    )

            # Preserve init config in metadata
            self._metadata = {
                'cache_templates': self._cache_templates,
                'max_cache_size': self._max_cache_size,
                'extraction_timeout': self._extraction_timeout,
                'batch_size': self._batch_size,
            }

            if not errors:
                logger.info(
                    f"{self.__class__.__name__} cleaned up successfully"
                )

        except Exception as e:
            errors.append(f"Unexpected error during cleanup: {e}")

        finally:
            # Always mark as uninitialized, even if cleanup fails
            self._is_setup = False
            self._batch_processor = None

        if errors:
            error_msg = (
                f"Errors during {self.__class__.__name__} cleanup: "
                f"{'; '.join(errors)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def __aenter__(self) -> Self:
        r"""Async context manager entry.

        Returns:
            Self reference for context manager usage
        """
        await self.setup()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        r"""Async context manager exit.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        await self.cleanup()

    @abstractmethod
    async def extract(
        self, response: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        r"""Extract relevant parts from a response.

        Extracts:
        1. Question and problem description
        2. Solution code and implementation details
        3. Final answer or output
        4. Chain of thought reasoning steps
        5. Difficulty assessment
        6. Raw markdown for reward/hint generation

        Args:
            response: Raw response from Gurobi examples or agent generation.
            context: Optional context for extraction including:
                - source_type: 'gurobi_example' or 'agent_generated'
                - domain: Problem domain (e.g. 'optimization', 'scheduling')
                - complexity: Expected problem complexity

        Returns:
            Dictionary containing extracted content with keys:
                - question: Extracted problem statement
                - ground_truth: Implementation code
                - final_answer: Expected output
                - chain_of_thought: Reasoning steps
                - difficulty: Assessed difficulty
                - raw_markdown: Content for reward/hint generation

        Raises:
            ValueError: If response is empty or invalid
            KeyError: If required content sections are missing
            RuntimeError: If extractor is not initialized
        """
        if not self._is_setup:
            raise RuntimeError(
                f"{self.__class__.__name__} must be initialized "
                "before extraction"
            )
        if not response or not response.strip():
            raise ValueError("Empty or whitespace-only response")
        raise NotImplementedError("Subclasses must implement extract()")
