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
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from camel.agents import ChatAgent
from camel.logger import get_logger
from camel.models.reward import BaseRewardModel, Evaluator
from camel.utils import BatchProcessor, retry_on_error

logger = get_logger(__name__)


class AgentTraceEvaluation(BaseModel):
    correctness: float
    clarity: float
    completeness: float
    feedback: str


class RewardTraceEvaluation(BaseModel):
    feedback: str

    def __init__(self, **data):
        # Allow dynamic score fields while ensuring feedback is present
        super().__init__(**data)

    class Config:
        extra = (
            "allow"  # Allow extra fields for different reward model dimensions
        )


class TraceIteration(BaseModel):
    iteration: int
    trace: str
    evaluation: Union[AgentTraceEvaluation, RewardTraceEvaluation]


class ProblemResult(BaseModel):
    id: Optional[str] = None
    type: Optional[str] = None
    problem: str
    solution: Optional[str] = None
    final_trace: str
    evaluate_success: bool = (
        False  # Tracks if problem meets evaluation standards
    )
    boxed_answer_success: bool = False
    improvement_history: List[TraceIteration]


class STaRPipeline:
    r"""Pipeline for generating self-taught reasoning traces
    using the STaR methodology.

    This implements the STaR paper's approach of:
    1. Initial reasoning trace generation
    2. Self-evaluation
    3. Feedback-based improvement
    4. Iterative refinement

    Args:
        reason_agent (ChatAgent): The chat agent used for generating and
            improving reasoning traces.
        evaluate_agent (ChatAgent): The chat agent used for evaluating
            reasoning traces.
        problems (List[Dict]): List of problem dictionaries to process.
        max_iterations (int, optional): Maximum number of improvement
            iterations. If set to `0`, the pipeline will generate an initial
            trace without any improvement iterations. (default: :obj:`3`)
        score_threshold (Union[float, Dict[str, float]], optional): Quality
            threshold. Can be either a single float value applied to all score,
            or a dictionary mapping score dimensions to their thresholds. For
            example: {"correctness": 0.8, "coherence": 0.7}. If using reward
            model and threshold for a dimension is not specified, will use the
            default value 0.7. (default: :obj:`0.7`)
        reward_model (BaseRewardModel, optional): Model used for evaluating
            reasoning traces. If `None`, uses Agent self-evaluation.
            (default: :obj:`None`)
        output_path (str, optional): Output path for saving traces. If `None`,
            results will only be returned without saving to file.
            (default: :obj:`None`)
        few_shot_examples: Optional[str] = None,
        batch_size (int, optional): Batch size for parallel processing.
            (default: :obj:`10`)
        max_workers (int, optional): Maximum number of worker threads.
            (default: :obj:`4`)
    """

    def __init__(
        self,
        reason_agent: ChatAgent,
        evaluate_agent: ChatAgent,
        problems: List[Dict],
        max_iterations: int = 3,
        score_threshold: Union[float, Dict[str, float]] = 0.7,
        reward_model: Optional[BaseRewardModel] = None,
        output_path: Optional[str] = None,
        few_shot_examples: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
    ):
        r"""Initialize the STaR pipeline.

        Args:
            reason_agent (ChatAgent): The chat agent used for generating and
                improving reasoning traces.
            evaluate_agent (ChatAgent): The chat agent used for evaluating
                reasoning traces.
            problems (List[Dict]): List of problem dictionaries to process.
            max_iterations (int, optional): Maximum number of improvement
                iterations. If set to `0`, the pipeline will generate an
                initial trace without any improvement iterations.
                (default: :obj:`3`)
            score_threshold (Union[float, Dict[str, float]], optional):
                Quality threshold. Can be either a single float value applied
                to average score, or a dictionary mapping score dimensions to
                their thresholds. For example: {"correctness": 0.8,
                "coherence": 0.7}. If using reward model and threshold for a
                dimension is not specified, will use the default value 0.7.
                (default: :obj:`0.7`)
            reward_model (BaseRewardModel, optional): Model used to evaluate
                reasoning traces. If `None`, uses Agent self-evaluation.
                (default: :obj:`None`)
            output_path (str, optional): Output path for saving traces. If
                `None`, results will only be returned without saving to file.
                (default: :obj:`None`)
            few_shot_examples (str, optional): Examples to use for few-shot
                generation. (default: :obj:`None`)
        """
        self.reason_agent = reason_agent
        self.evaluate_agent = evaluate_agent
        self.problems = problems
        self.output_path = output_path
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold
        self.reward_model = reward_model
        self.evaluator = (
            Evaluator(reward_model=reward_model) if reward_model else None
        )
        self.reasoning_traces: List[Dict[str, Any]] = []
        self.few_shot_examples = few_shot_examples
        self.batch_processor = BatchProcessor(max_workers, batch_size)

    async def _batch_process_problems(
        self, problems: List[Dict], rationalization: bool
    ) -> List[ProblemResult]:
        r"""Process multiple problems in parallel batches with dynamic sizing.

        Args:
            problems (List[Dict]): List of problem dictionaries to process.
            rationalization (bool): Whether to use rationalization.

        Returns:
            List[ProblemResult]: List of problem results.
        """
        results = []
        total_problems = len(problems)
        processed = 0

        while processed < total_problems:
            batch_size = self.batch_processor.batch_size
            batch = problems[processed : processed + batch_size]
            batch_start_time = time.time()

            try:
                with ThreadPoolExecutor(
                    max_workers=self.batch_processor.max_workers
                ) as executor:
                    # Create futures with rationalization parameter
                    futures = [
                        executor.submit(
                            self.process_problem,
                            problem=problem,
                            rationalization=rationalization,
                        )
                        for problem in batch
                    ]

                    batch_results = []
                    batch_success = True
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            batch_results.append(result)
                        except Exception as e:
                            logger.error(f"Error processing problem: {e}")
                            batch_success = False
                            continue

                    results.extend(batch_results)
                    processed += len(batch)

                    # Calculate processing time and adjust batch size
                    processing_time = time.time() - batch_start_time
                    self.batch_processor.adjust_batch_size(
                        batch_success, processing_time
                    )

                    # Log progress and performance metrics
                    metrics = self.batch_processor.get_performance_metrics()
                    logger.info(
                        f"Processed {processed}/{total_problems} problems "
                        f"(batch size: {batch_size}, workers: "
                        f"{metrics['current_workers']}, "
                        f"CPU: {metrics['current_cpu']:.1f}%, "
                        f"Memory: {metrics['current_memory']:.1f}%)"
                    )
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                self.batch_processor.adjust_batch_size(False)
                continue

        return results

    async def _batch_evaluate_traces(
        self,
        problems: List[Dict[str, Any]],
        traces: List[str],
        solutions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        r"""Evaluate multiple traces in parallel batches with resource
        monitoring.

        Args:
            problems (List[Dict[str, Any]]): List of problem dictionaries
            traces (List[str]): List of reasoning traces to evaluate
            solutions (Optional[List[str]]): Optional list of solutions

        Returns:
            List[Dict[str, Any]]: List of evaluation results
        """
        if solutions is None:
            solutions = ["null"] * len(problems)

        results = []
        total_traces = len(traces)
        processed = 0

        while processed < total_traces:
            batch_size = self.batch_processor.batch_size
            problem_batch = problems[processed : processed + batch_size]
            trace_batch = traces[processed : processed + batch_size]
            solution_batch = solutions[processed : processed + batch_size]
            batch_start_time = time.time()

            try:
                with ThreadPoolExecutor(
                    max_workers=self.batch_processor.max_workers
                ) as executor:
                    futures = [
                        executor.submit(
                            self.evaluate_trace,
                            problem=problem["problem"],
                            trace=trace,
                            solution=solution,
                        )
                        for problem, trace, solution in zip(
                            problem_batch, trace_batch, solution_batch
                        )
                    ]

                    batch_results = []
                    batch_success = True
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            batch_results.append(result)
                        except Exception as e:
                            logger.error(f"Error evaluating trace: {e}")
                            batch_success = False
                            continue

                    results.extend(batch_results)
                    processed += len(batch_results)

                    # Calculate processing time and adjust batch size
                    processing_time = time.time() - batch_start_time
                    self.batch_processor.adjust_batch_size(
                        batch_success, processing_time
                    )

                    # Log progress and performance metrics
                    metrics = self.batch_processor.get_performance_metrics()
                    logger.info(
                        f"Evaluated {processed}/{total_traces} traces "
                        f"(batch size: {batch_size}, workers: "
                        f"{metrics['current_workers']}, "
                        f"avg time: {metrics['avg_processing_time']:.2f}s, "
                        f"error rate: {metrics['error_rate']:.1f}%)"
                    )
            except Exception as e:
                logger.error(f"Batch evaluation error: {e}")
                self.batch_processor.adjust_batch_size(False)
                continue

        return results

    def _check_score_threshold(self, scores: Dict[str, float]) -> bool:
        r"""Check if scores meet the threshold requirements.

        Args:
            scores (Dict[str, float]): Dictionary of scores for different
                dimensions.

        Returns:
            bool: True if scores meet threshold requirements, False otherwise.
        """
        # If score_threshold is a float, apply it to all dimensions
        if isinstance(self.score_threshold, float):
            return all(
                score >= self.score_threshold for score in scores.values()
            )

        # If score_threshold is a dict, check each dimension with its threshold
        # Use 0 as default threshold for unspecified dimensions
        if isinstance(self.score_threshold, dict):
            for dim, score in scores.items():
                threshold = self.score_threshold.get(dim, 0)
                if score < threshold:
                    return False
            return True

        # If score_threshold is None or invalid type, pass the check
        return True

    def _generate_feedback(self, scores: Dict[str, float]) -> str:
        r"""Generate feedback based on which dimensions need improvement.

        Args:
            scores (Dict[str, float]): Dictionary of scores for different
                dimensions.

        Returns:
            str: Feedback message indicating which dimensions need improvement.
        """
        if isinstance(self.score_threshold, float):
            below_threshold = [
                dim
                for dim, score in scores.items()
                if score < self.score_threshold
            ]
            if not below_threshold:
                return "All dimensions meet the required threshold"
            dims = ", ".join(below_threshold)
            return f"Need improvement in: {dims}"

        if isinstance(self.score_threshold, dict):
            default_threshold = 0
            below_threshold = [
                dim
                for dim, score in scores.items()
                if score < self.score_threshold.get(dim, default_threshold)
            ]
            if not below_threshold:
                return "All dimensions meet their respective thresholds"
            dims = ", ".join(below_threshold)
            return f"Need improvement in: {dims}"

        # If no threshold set, just list all dimensions and their scores
        dims = ", ".join(
            f"{dim}: {score:.2f}" for dim, score in scores.items()
        )
        return f"Current scores - {dims}"

    @retry_on_error()
    def generate_reasoning_trace(self, problem: str) -> str:
        r"""Generate initial reasoning trace for a given problem.

        Args:
            problem (str): The problem text to generate reasoning for.

        Returns:
            str: Generated reasoning trace.
        """
        self.reason_agent.reset()
        few_shot_examples = (
            f"Examples: {self.few_shot_examples}"
            if self.few_shot_examples
            else ""
        )
        prompt = self.REASONING_TEMPLATE.format(
            problem=problem, few_shot_examples=few_shot_examples
        )
        response = self.reason_agent.step(prompt)
        return response.msg.content

    @retry_on_error()
    def evaluate_trace(
        self, problem: str, trace: str, solution: Optional[str] = None
    ) -> Dict[str, Any]:
        r"""Evaluate the quality of a reasoning trace.

        Args:
            problem (str): The original problem text to evaluate against.
            trace (str): The reasoning trace to evaluate.
            solution (Optional[str]): The solution to the problem, if provided.
                (default: :obj:`None`)

        Returns:
            Dict[str, Any]: Evaluation results containing:
                - scores: Dict of evaluation dimensions and their scores
                - feedback: Detailed feedback for improvement

                For Agent self-evaluation, the scores will include:
                - correctness: Score for logical correctness
                - clarity: Score for clarity of explanation
                - completeness: Score for completeness of reasoning

                For reward model evaluation, the scores will depend on
                the model's evaluation dimensions.
        """
        self.evaluate_agent.reset()
        if self.evaluator:
            # Use reward model evaluation
            messages = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": trace},
            ]
            scores = self.evaluator.evaluate(messages)

            # For models that return a single score
            if isinstance(scores, (int, float)) or (
                isinstance(scores, dict) and len(scores) == 1
            ):
                if isinstance(scores, dict):
                    score = next(iter(scores.values()))
                else:
                    score = scores
                scores_dict = {"overall": score}
                return {
                    **scores_dict,
                    "feedback": self._generate_feedback(scores_dict),
                }

            # For models that return multiple dimensions
            return {**scores, "feedback": self._generate_feedback(scores)}
        else:
            # Fallback to original Agent self-evaluation
            solution_text = f"Solution: {solution}" if solution else ""
            prompt = self.EVALUATION_TEMPLATE.format(
                problem=problem, trace=trace, solution=solution_text
            )
            response = self.evaluate_agent.step(
                prompt, response_format=AgentTraceEvaluation
            )
            if response.msg.parsed is None:
                raise AttributeError("Failed to parse evaluation response")
            # Convert dict to AgentTraceEvaluation if needed
            if isinstance(response.msg.parsed, dict):
                evaluation = AgentTraceEvaluation(**response.msg.parsed)
            else:
                evaluation = response.msg.parsed

            return evaluation.model_dump()

    @retry_on_error()
    def improve_trace(
        self,
        problem: str,
        trace: str,
        feedback: str,
        solution: Optional[str] = None,
    ) -> str:
        r"""Generate improved reasoning trace based on feedback.

        Args:
            problem (str): The original problem text.
            trace (str): The current reasoning trace.
            feedback (str): Feedback for improving the trace.
            solution (Optional[str]): The solution to the problem, if provided.
                (default: :obj:`None`)

        Returns:
            str: Improved reasoning trace.
        """
        self.reason_agent.reset()
        solution_text = f"Solution: {solution}" if solution else ""
        prompt = self.IMPROVEMENT_TEMPLATE.format(
            problem=problem,
            trace=trace,
            feedback=feedback,
            solution=solution_text,
        )
        response = self.reason_agent.step(prompt)
        return response.msg.content

    def validate_problem_format(self, problem: Dict) -> None:
        r"""Validate that a problem dictionary has the required format.

        Args:
            problem (Dict): Problem dictionary to validate.

        Raises:
            ValueError: If the problem format is invalid.
        """
        if not isinstance(problem, dict):
            raise ValueError("Problem must be a dictionary.")

        # Check required problem field
        if "problem" not in problem:
            raise ValueError("Problem dictionary must contain 'problem' key.")
        if not isinstance(problem["problem"], str):
            raise ValueError("Problem 'problem' field must be a string.")

        # Optional fields validation
        optional_fields = {"id": str, "type": str, "solution": str}

        for field, expected_type in optional_fields.items():
            if field in problem and not isinstance(
                problem[field], expected_type
            ):
                raise ValueError(
                    f"Problem '{field}' must be of "
                    f"type {expected_type.__name__} if present."
                )

    def _check_boxed_answers(self, solution: str, trace: str) -> bool:
        r"""Check if the boxed answer in the trace matches the solution.

        Args:
            solution (str): The problem solution string.
            trace (str): The reasoning trace string.

        Returns:
            bool: True if boxed answers match, False otherwise
        """
        import re

        # Extract content within \boxed{...}
        solution_match = re.search(r'\\boxed{(.*?)}', solution, re.DOTALL)
        trace_match = re.search(r'\\boxed{(.*?)}', trace, re.DOTALL)

        if solution_match and trace_match:
            # Clean up whitespace and normalize content
            solution_answer = solution_match.group(1).strip()
            trace_answer = trace_match.group(1).strip()
            return solution_answer == trace_answer

        return False

    def process_problem(
        self, problem: Dict, rationalization: bool = False
    ) -> ProblemResult:
        r"""Process a single problem through the STaR pipeline.

        Args:
            problem (Dict): Problem dictionary containing the problem text.
            rationalization (bool, optional): Whether to use rationalization.
                (default: :obj:`False`)

        Returns:
            ProblemResult: Results with final trace and history.

        Raises:
            ValueError: If the problem format is invalid.
        """
        # Validate problem format before processing
        self.validate_problem_format(problem)

        problem_text = problem["problem"]
        solution_text = problem.get("solution", "")
        current_trace = self.generate_reasoning_trace(problem_text)
        improvement_history = []

        # Create batches for parallel evaluation
        batch_problems = []
        batch_traces = []
        batch_solutions = []

        for iteration in range(self.max_iterations):
            # Add to evaluation batch
            batch_problems.append(problem)
            batch_traces.append(current_trace)
            batch_solutions.append(solution_text)

            # Evaluate current trace batch
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                eval_results = loop.run_until_complete(
                    self._batch_evaluate_traces(
                        batch_problems, batch_traces, batch_solutions
                    )
                )
            finally:
                loop.close()

            # Process evaluation results
            eval_dict = eval_results[-1]  # Get latest evaluation
            scores = {k: v for k, v in eval_dict.items() if k != "feedback"}

            # Record iteration history
            if self.evaluator:
                improvement_history.append(
                    TraceIteration(
                        iteration=iteration + 1,
                        trace=current_trace,
                        evaluation=RewardTraceEvaluation(**eval_dict),
                    )
                )
            else:
                improvement_history.append(
                    TraceIteration(
                        iteration=iteration + 1,
                        trace=current_trace,
                        evaluation=AgentTraceEvaluation(
                            **scores, feedback=eval_dict["feedback"]
                        ),
                    )
                )

            # Check if quality threshold met
            if self._check_score_threshold(scores):
                break

            # Generate improved trace
            if rationalization:
                current_trace = self.improve_trace(
                    problem_text,
                    current_trace,
                    eval_dict["feedback"],
                    solution_text,
                )
            else:
                current_trace = self.improve_trace(
                    problem_text, current_trace, eval_dict["feedback"]
                )

        boxed_answer_success = self._check_boxed_answers(
            problem.get("solution", ""), current_trace
        )

        return ProblemResult(
            id=problem.get("id", ""),
            type=problem.get("type", ""),
            problem=problem_text,
            solution=problem.get("solution", ""),
            final_trace=current_trace,
            evaluate_success=self._check_score_threshold(scores),
            boxed_answer_success=boxed_answer_success,
            improvement_history=improvement_history,
        )

    def generate(self, rationalization: bool = False) -> List[Dict[str, Any]]:
        r"""Execute the STaR pipeline on all problems.

        Process problems and return results. If output_path is specified,
        also save results to file.

        Args:
            rationalization (bool, optional): Whether to use rationalization.
                (default: :obj:`False`)

        Returns:
            List[Dict[str, Any]]: List of processed results
        """
        # Pre-allocate results list
        self.reasoning_traces = []

        # Process problems in batches
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            results = loop.run_until_complete(
                self._batch_process_problems(self.problems, rationalization)
            )
        finally:
            loop.close()

        self.reasoning_traces = [result.model_dump() for result in results]

        if self.output_path:
            with open(self.output_path, 'w') as f:
                json.dump({'traces': self.reasoning_traces}, f, indent=2)

        return self.reasoning_traces

    # Templates for generating reasoning, evaluation and improving them.
    REASONING_TEMPLATE = """Let's solve this step by step:
Problem: {problem}
1. First, let's understand what we're asked
2. Let's break this down into parts
3. Let's solve each part systematically
4. Finally, let's verify our solution

{few_shot_examples}

Please show your complete reasoning process."""

    EVALUATION_TEMPLATE = """Please evaluate this reasoning trace and 
provide scores and feedback in valid JSON format.

Problem: {problem}

{solution}

Reasoning Trace:
{trace}

Evaluate for:
1. Correctness (Is each step logically sound?)
2. Clarity (Is the explanation clear and well-structured?)
3. Completeness (Are all necessary steps included?)

Respond ONLY with a JSON object in this exact format:
{{
    "correctness": <score between 0 and 1>,
    "clarity": <score between 0 and 1>,
    "completeness": <score between 0 and 1>,
    "feedback": "<specific feedback for improvement>"
}}"""

    IMPROVEMENT_TEMPLATE = """Based on this feedback, generate an 
improved reasoning trace:
Problem: {problem}

{solution}

Previous Trace:
{trace}

Feedback:
{feedback}

Generate a new, improved reasoning trace that addresses the feedback."""
