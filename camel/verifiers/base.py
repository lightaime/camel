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
import asyncio
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from camel.logger import get_logger
from camel.utils import BatchProcessor

from .models import (
    Response,
    VerificationMetrics,
    VerificationResult,
    VerificationStatus,
)

logger = get_logger(__name__)


class BaseVerifier(ABC):
    r"""Base class for all verifiers.

    Example:
        ```python
        verifier = MyVerifier()
        await verifier.setup()
        result = await verifier.verify(response)
        await verifier.cleanup()
        ```
    Key Features:
    - Async and sync verification support
    - Comprehensive error handling and logging
    - Performance metrics tracking
    - Configurable retry logic
    - Batch verification capabilities
    - Extensible validation framework
    """

    def __init__(
        self,
        max_parallel: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        initial_batch_size: Optional[int] = None,
        monitoring_interval: float = 5.0,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 85.0,
        **kwargs,
    ):
        r"""Initialize the verifier with configuration parameters.

        Args:
            max_parallel: Maximum number of parallel verifications. If None,
                will be determined dynamically based on system resources.
                (default: :obj:`None`)
            timeout: Timeout in seconds for each verification.  (default:
                :obj:`None`)
            max_retries: Maximum number of retry attempts.  (default: :obj:`3`)
            retry_delay: Delay between retries in seconds.  (default:
                :obj:`1.0`)
            initial_batch_size: Initial size for batch processing. If None,
                defaults to 10. (default: :obj:`None`)
            monitoring_interval: Interval in seconds between resource checks.
                (default: :obj:`5.0`)
            cpu_threshold: CPU usage percentage threshold for scaling down.
                (default: :obj:`80.0`)
            memory_threshold: Memory usage percentage threshold for scaling
                down. (default: :obj:`85.0`)
            **kwargs: Additional verifier parameters.
        """
        # Configuration parameters
        self._is_setup: bool = False
        self._max_parallel: Optional[int] = max_parallel
        self._timeout: Optional[float] = timeout
        self._max_retries: int = max_retries
        self._retry_delay: float = retry_delay
        self._initial_batch_size: Optional[int] = initial_batch_size
        self._monitoring_interval: float = monitoring_interval
        self._cpu_threshold: float = cpu_threshold
        self._memory_threshold: float = memory_threshold
        self._batch_processor: BatchProcessor = BatchProcessor()
        self._metrics: VerificationMetrics = VerificationMetrics()
        self._metrics_lock: asyncio.Lock = asyncio.Lock()
        self._start_time: Optional[float] = None

    async def setup(self) -> None:
        r"""Set up the verifier with necessary resources.

        This method initializes:
        1. Metrics tracking
        2. Resource monitoring
        3. Batch processor
        4. Any model-specific resources

        Raises:
            RuntimeError: If setup fails or resources cannot be initialized
        """
        if self._is_setup:
            logger.debug(f"{self.__class__.__name__} already initialized")
            return

        try:
            # Initialize metrics and lock
            self._metrics = VerificationMetrics()
            self._metrics_lock = asyncio.Lock()

            # Set up batch processor with validated parameters
            batch_size = max(1, self._initial_batch_size or 10)
            max_parallel = max(1, self._max_parallel or 1)
            self._batch_processor = BatchProcessor()
            if hasattr(self._batch_processor, 'setup'):
                await self._batch_processor.setup()

            self._is_setup = True
            logger.info(
                f"{self.__class__.__name__} initialized with "
                f"batch_size={batch_size}, max_parallel={max_parallel}"
            )

        except Exception as e:
            error_msg = (
                f"Failed to initialize {self.__class__.__name__}: {e!s}"
            )
            logger.error(error_msg, exc_info=True)
            await self.cleanup()
            raise RuntimeError(error_msg) from e

    async def cleanup(self) -> None:
        r"""Clean up verifier resources.

        This method ensures:
        1. All resources are properly released
        2. Batch processor is shut down
        3. Metrics are saved if needed
        4. State is reset to initial

        Raises:
            RuntimeError: If cleanup fails and resources cannot be properly
                released
        """
        if not self._is_setup:
            return

        error = None
        try:
            # Reset all internal state
            self._batch_processor = BatchProcessor()
            self._metrics = VerificationMetrics()
            self._metrics_lock = asyncio.Lock()
            self._start_time = None

            logger.info(f"{self.__class__.__name__} cleaned up successfully")

        except Exception as e:
            error = e
            logger.error(
                f"Error during {self.__class__.__name__} cleanup: {e!s}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to cleanup {self.__class__.__name__}: {error!s}"
            ) from e

        finally:
            self._is_setup = False

    async def _update_metrics(
        self, duration: float, status: VerificationStatus
    ) -> None:
        r"""Update verification metrics with the results of a verification.

        Args:
            duration: Time taken for the verification in seconds.
            status: Status of the verification.
        """
        async with self._metrics_lock:
            self._metrics.total_verifications += 1
            self._metrics.total_duration += duration
            self._metrics.avg_duration = (
                self._metrics.total_duration
                / self._metrics.total_verifications
            )

            if status == VerificationStatus.SUCCESS:
                self._metrics.successful_verifications += 1
            elif status == VerificationStatus.FAILURE:
                self._metrics.failed_verifications += 1
            elif status == VerificationStatus.TIMEOUT:
                self._metrics.timeout_verifications += 1
            elif status == VerificationStatus.ERROR:
                self._metrics.error_verifications += 1

    async def verify(self, result: Response) -> VerificationResult:
        r"""Perform verification with full error handling and metrics.

        Verifies:
        1. Code solution correctness via execution
        2. Final answer matches expected output
        3. Chain of thought reasoning is valid
        4. Symbolic solver verification matches code execution

        Args:
            result: The response to verify containing code, answer and
                reasoning.

        Returns:
            VerificationResult: Containing status and metadata including:
                - code_execution_status: Success/failure of code execution
                - answer_match: Whether final answer matches ground truth
                - reasoning_valid: Whether chain of thought is valid
                - symbolic_match: Whether symbolic solution matches code

        Raises:
            ValueError: If verification parameters are invalid.
            RuntimeError: If verification fails unexpectedly.
            asyncio.TimeoutError: If verification times out and max retries
                exceeded.
        """
        attempt = 0
        start_time = time.time()

        # Start tracking metrics

        while attempt < self._max_retries:
            try:
                # Handle timeout
                if self._timeout is not None:
                    verification_result = await asyncio.wait_for(
                        self._verify_implementation(result),
                        timeout=self._timeout,
                    )
                else:
                    verification_result = await self._verify_implementation(
                        result
                    )

                duration = time.time() - start_time
                await self._update_metrics(
                    duration, verification_result.status
                )
                verification_result.duration = duration
                verification_result.metadata["attempt"] = attempt + 1
                return verification_result

            except asyncio.TimeoutError:
                attempt += 1
                if attempt == self._max_retries:
                    duration = time.time() - start_time
                    await self._update_metrics(
                        duration, VerificationStatus.TIMEOUT
                    )
                    return VerificationResult(
                        status=VerificationStatus.TIMEOUT,
                        score=None,
                        error_message="Verification timed out after "
                        "all retries.",
                        duration=duration,
                    )
                logger.warning(
                    f"Verification timeout on attempt {attempt}, retrying..."
                )
                await asyncio.sleep(self._retry_delay)

            except Exception as e:
                attempt += 1
                if attempt == self._max_retries:
                    duration = time.time() - start_time
                    await self._update_metrics(
                        duration, VerificationStatus.ERROR
                    )
                    error_msg = (
                        f"Verification failed after {attempt} attempts: {e!s}"
                    )
                    logger.error(error_msg, exc_info=True)
                    return VerificationResult(
                        status=VerificationStatus.ERROR,
                        score=None,
                        error_message=error_msg,
                        duration=duration,
                    )
                logger.warning(
                    f"Verification error on attempt {attempt}: "
                    f"{e!s}, retrying..."
                )
                await asyncio.sleep(self._retry_delay)

        # This is a placeholder return that should never be reached
        # since we always return in one of the above branches
        duration = time.time() - start_time
        await self._update_metrics(duration, VerificationStatus.ERROR)
        return VerificationResult(
            status=VerificationStatus.ERROR,
            score=None,
            error_message="Unexpected code path reached",
            duration=duration,
        )

    @abstractmethod
    async def _verify_implementation(
        self, result: Response
    ) -> VerificationResult:
        r"""Implement the actual verification logic in subclasses.

        Args:
            result: The response to verify.

        Returns:
            VerificationResult: Containing the verification outcome.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement _verify_implementation()"
        )

    async def verify_batch(
        self, results: List[Response], raise_on_error: bool = False
    ) -> List[VerificationResult]:
        r"""Verify multiple results in parallel with controlled concurrency.

        Args:
            results: List of responses to verify.
            raise_on_error: Whether to raise an exception if any verification
                fails. (default: :obj:`False`)

        Returns:
            List[VerificationResult]: One for each input response.

        Raises:
            RuntimeError: If any verification fails and raise_on_error is True.
            asyncio.TimeoutError: If verifications time out and max retries
                exceeded.
        """
        # Get current batch parameters from processor
        max_workers = getattr(self._batch_processor, 'max_workers', 1)
        batch_size = getattr(self._batch_processor, 'batch_size', 1)
        semaphore = asyncio.Semaphore(max_workers)

        async def _verify_with_semaphore(
            response: Response,
        ) -> VerificationResult:
            start_time = time.time()
            try:
                async with semaphore:
                    verification_result = await self.verify(response)
                processing_time = time.time() - start_time
                success = (
                    verification_result.status == VerificationStatus.SUCCESS
                )
                self._batch_processor.adjust_batch_size(
                    success, processing_time
                )
                return verification_result
            except Exception as e:
                processing_time = time.time() - start_time
                self._batch_processor.adjust_batch_size(False, processing_time)
                logger.error(f"Verification failed: {e!s}", exc_info=True)
                return VerificationResult(
                    status=VerificationStatus.ERROR, error_message=str(e)
                )

        # Process in batches
        all_results: List[VerificationResult] = []
        for i in range(0, len(results), batch_size):
            batch = results[i : i + batch_size]
            verification_tasks = [
                _verify_with_semaphore(result) for result in batch
            ]
            try:
                batch_results = await asyncio.gather(*verification_tasks)
                all_results.extend(batch_results)

                # Update metrics after each batch
                metrics = self._batch_processor.get_performance_metrics()
                logger.debug(f"Batch verification metrics: {metrics}")
            except Exception as e:
                logger.error(
                    f"Batch verification failed: {e!s}", exc_info=True
                )
                if raise_on_error:
                    raise RuntimeError(
                        f"Batch verification failed: {e!s}"
                    ) from e

        if raise_on_error and any(
            r.status in {VerificationStatus.ERROR, VerificationStatus.TIMEOUT}
            for r in all_results
        ):
            error_msg = "One or more verifications failed"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        return all_results
