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
        result = await verifier.verify(response)
        ```
    Key Features:
    - Async and sync verification support
    - Comprehensive error handling and logging
    - Performance metrics tracking
    - Configurable retry logic
    - Batch verification capabilities
    - Extensible validation framework
    """

    async def _update_metrics(
        self, duration: float, status: VerificationStatus
    ) -> None:
        """Update verification metrics with the results of a verification.

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
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize batch processor for dynamic resource management
        self.batch_processor = BatchProcessor(
            max_workers=max_parallel,
            initial_batch_size=initial_batch_size,
            monitoring_interval=monitoring_interval,
            cpu_threshold=cpu_threshold,
            memory_threshold=memory_threshold,
        )

        # Internal state
        self._start_time: Optional[float] = None
        self._metrics = VerificationMetrics()
        self._metrics_lock = asyncio.Lock()

    async def verify(self, result: Response) -> VerificationResult:  # type: ignore[return]
        r"""Perform verification with full error handling and metrics.

        Args:
            result: The response to verify.

        Returns:
            VerificationResult: Containing status and metadata.

        Raises:
            ValueError: If verification parameters are invalid.
            RuntimeError: If verification fails unexpectedly.
            asyncio.TimeoutError: If verification times out and max retries
                exceeded.
        """
        attempt = 0
        start_time = time.time()

        # Start tracking metrics

        while attempt < self.max_retries:
            try:
                # Handle timeout
                if self.timeout is not None:
                    verification_result = await asyncio.wait_for(
                        self._verify_implementation(result),
                        timeout=self.timeout,
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
                if attempt == self.max_retries:
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
                await asyncio.sleep(self.retry_delay)

            except Exception as e:
                attempt += 1
                if attempt == self.max_retries:
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
                await asyncio.sleep(self.retry_delay)

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
        max_workers = self.batch_processor.max_workers
        batch_size = self.batch_processor.batch_size
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
                self.batch_processor.adjust_batch_size(
                    success, processing_time
                )
                return verification_result
            except Exception as e:
                processing_time = time.time() - start_time
                self.batch_processor.adjust_batch_size(False, processing_time)
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
                metrics = self.batch_processor.get_performance_metrics()
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
