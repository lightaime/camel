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

import json
from datetime import datetime
from typing import Annotated, Dict, Optional, Union

from pydantic import BaseModel, Field, confloat

from camel.agents import ChatAgent
from camel.logger import get_logger

# Get a logger for this module
logger = get_logger('CoTDataGenerator')


class AgentResponse(BaseModel):
    r"""Model for structured agent responses.

    A Pydantic model class that represents structured responses from agents,
    including a similarity score that measures the quality of the response.

    Args:
        score (float): A similarity score between 0 and 1 that compares the
            current answer to the correct answer. Must be within the range
            [0, 1].
    """

    score: Annotated[float, confloat(ge=0, le=1)] = Field(
        ...,
        description="""Similarity score between 0 and 1 
        comparing current answer to correct answer""",
    )


class VerificationResponse(BaseModel):
    r"""Model for structured verification responses.

    A Pydantic model class that represents verification results from agents,
    indicating whether an answer is correct or not.

    Args:
        is_correct (bool): Boolean indicating if the answer is correct.
    """

    is_correct: bool = Field(
        ...,
        description="Boolean indicating if the answer is correct",
    )


class CoTDataGenerator:
    r"""Class for generating and managing data through chat agent interactions.

    This module implements a sophisticated Chain of Thought data generation
    system that combines several key algorithms to produce high-quality
    reasoning paths. Methods implemented:

    1. Monte Carlo Tree Search (MCTS)
    2. Binary Search Error Detection
    3. Dual-Agent Verification System
    4. Solution Tree Management

    Args:
        chat_agent (Optional[ChatAgent]): Optional single agent
            for both tasks (legacy mode). (default::obj:`None`)
        generator_agent (Optional[ChatAgent]): Optional specialized agent for
            answer generation. (default::obj:`None`)
        verifier_agent (Optional[ChatAgent]): Optional specialized agent for
            answer verification. (default::obj:`None`)
        golden_answers (Dict[str, str]): Dictionary containing pre-defined
            correct answers for validation and comparison. Required for answer
            verification.
        search_limit (int): Maximum number of search iterations allowed.
            (default::obj:`100`)
    """

    def __init__(
        self,
        chat_agent: Optional[ChatAgent] = None,
        *,
        generator_agent: Optional[ChatAgent] = None,
        verifier_agent: Optional[ChatAgent] = None,
        golden_answers: Dict[str, str],
        search_limit: int = 100,
    ):
        r"""Initialize the CoTDataGenerator.

        This constructor supports both single-agent and dual-agent modes:
        1. Single-agent mode (legacy): Pass a single chat_agent that will be
        used for both generation and verification.
        2. Dual-agent mode: Pass separate generator_agent and verifier_agent
        for specialized tasks.

        Args:
            chat_agent (Optional[ChatAgent]): Optional single agent for both
                tasks (legacy mode). (default::obj:`None`)
            generator_agent (Optional[ChatAgent]): Optional specialized agent
                for answer generation. (default::obj:`None`)
            verifier_agent (Optional[ChatAgent]): Optional specialized agent
                for answer verification. (default::obj:`None`)
            golden_answers (Dict[str, str]): Dictionary containing pre-defined
                correct answers for validation and comparison. Required for
                answer verification.
            search_limit (int): Maximum number of search iterations allowed.
                (default::obj:`100`)
        """
        if chat_agent is not None:
            if generator_agent is not None or verifier_agent is not None:
                raise ValueError(
                    "Cannot specify both chat_agent \
                    and generator/verifier agents"
                )
            self.generator_agent = chat_agent
            self.verifier_agent = chat_agent
        else:
            if generator_agent is None or verifier_agent is None:
                raise ValueError(
                    "Must specify either chat_agent or both generator and "
                    "verifier agents"
                )
            self.generator_agent = generator_agent
            self.verifier_agent = verifier_agent

        self.golden_answers = golden_answers
        self.search_limit = search_limit
        self.solution_tree: Dict[str, Dict[str, Union[str, int]]] = {}
        logger.info(
            "CoTDataGenerator initialized with search_limit=%d", search_limit
        )

    def get_answer(self, question: str, context: str = "") -> str:
        r"""Get an answer from the chat agent for a given question.

        Args:
            question (str): The question to ask.
            context (str): Additional context for the question.
                (default::obj:`""`)

        Returns:
            str: The generated answer.
        """
        prompt = f"""
        Please think step by step and solve this problem: {question}
        Existing content: {context}
        Requirements:
        1. Analyze the problem requirements
        2. List the steps to solve the problem
        3. Execute the solution process
        4. Provide the final answer
        Please explain the thought process of each step in detail.
        """
        self.generator_agent.reset()
        response = self.generator_agent.step(prompt)
        answer = response.msgs[0].content.text
        logger.info("AI thought process:\n%s", answer)
        return answer

    def verify_answer(self, question: str, answer: str) -> bool:
        r"""Verify if a generated answer is semantically equivalent to
        the golden answer for a given question.

        Args:
            question (str): The question being answered.
            answer (str): The answer to verify.

        Returns:
            bool: True if the answer matches the golden answer based on
                semantic equivalence (meaning the core content and meaning are
                the same, even if the exact wording differs).
                False in the following cases:
                - If the provided question doesn't exist in the golden answers
                - If the answer's meaning differs from the golden answer
        """
        golden_answer = self.golden_answers.get(question)
        if not golden_answer:
            raise ValueError(
                f"No golden answer found for question: {question}"
            )

        prompt = (
            f"Question: {question}\n"
            f"Student Answer: {answer}\n"
            f"Correct Answer: {golden_answer}\n"
            "Is the student's answer correct? Please respond with 'true' or "
            "'false' only."
        )
        self.verifier_agent.reset()
        response = self.verifier_agent.step(
            prompt, response_format=VerificationResponse
        )
        is_correct = response.msgs[0].parsed.is_correct  # type:ignore [union-attr]
        logger.info("Answer verification result: %s", is_correct)
        return is_correct

    def monte_carlo_tree_search(
        self, question: str, partial_solution: str = ""
    ) -> float:
        r"""Perform Monte Carlo Tree Search to find the best solution.

        Process:
        a. Selection: Choose promising partial solutions based on previous
        scores
        b. Expansion: Generate new solution steps using the generator agent
        c. Simulation: Evaluate solution quality using similarity scores
        d. Backpropagation: Update solution tree with new findings

        Args:
            question (str): The question to solve.
            partial_solution (str): The current partial solution.
                (default::obj:`""`)

        Returns:
            float: The similarity score between the current
                solution and golden answer.
        """
        if question not in self.golden_answers:
            raise ValueError(
                f"No golden answer found for question: {question}"
            )

        golden_answer = self.golden_answers[question]

        prompt = (
            f"Please evaluate this solution and "
            f"give a score between 0-1:\n"
            f"Question: {question}\n"
            f"Solution: {partial_solution}\n"
            f"Correct answer: {golden_answer}\n"
            f"Return a JSON object with a single field 'score' containing "
            f"a float between 0 and 1, like this: {{'score': 0.85}}\n"
        )
        self.generator_agent.reset()
        response = self.generator_agent.step(
            prompt, response_format=AgentResponse
        )
        agent_response = response.msgs[0].parsed.score  # type: ignore [union-attr]

        return agent_response

    def binary_search_error(self, question: str, solution: str) -> int:
        r"""Use binary search to locate the first error in the solution.
        This method splits the solution into sentences using both English and
        Chinese sentence delimiters and performs binary search to find the
        first error.

        Args:
            question (str): The question being solved.
            solution (str): The complete solution to analyze.

        Returns:
            int: The position of the first error found in the solution.
                Returns -1. If no errors are found (all sentences are correct).
        """
        logger.info("Starting binary search for error location")
        # Split by both English period and Chinese period
        sentences = [
            s.strip()
            for s in solution.replace('。', '.').split('.')
            if s.strip()
        ]

        # First check if the entire solution is correct
        if self.verify_answer(question, solution):
            return -1

        left, right = 0, len(sentences)
        while left < right:
            mid = (left + right) // 2
            partial_solution = '. '.join(sentences[:mid]) + '.'
            logger.info("Checking solution fragment:\n%s", partial_solution)
            # Verify if the current part is correct
            is_correct = self.verify_answer(question, partial_solution)
            if is_correct:
                left = mid + 1
            else:
                right = mid
        logger.info("First error position found: sentence %d", left)
        return left

    def solve(self, question: str) -> str:
        r"""Solve a question using a multi-step approach.

        The solution process follows these steps:
        1. Try to solve directly - if correct, return the solution
        2. If not correct, use Monte Carlo Tree Search to find a good solution
        3. If the solution isn't perfect, use binary search to locate errors
        4. Generate a new solution based on the correct part

        Args:
            question (str): The question to solve.

        Returns:
            str: The best solution found.
        """
        # 1. Try direct solution first
        solution = self.get_answer(question)
        if self.verify_answer(question, solution):
            logger.info("Initial solution is correct")
            return solution

        # 2. If direct solution fails, try Monte Carlo Tree Search
        # to find a solution with high similarity score
        best_solution = ""
        best_score: float = 0.0
        for i in range(self.search_limit):
            # Generate new answer
            current_solution = self.get_answer(question, best_solution)

            # Evaluate solution similarity score
            prompt = (
                f"Please evaluate this solution and "
                f"give a score between 0-1:\n"
                f"Question: {question}\n"
                f"Solution: {current_solution}\n"
                f"Correct answer: {self.golden_answers.get(question, '')}\n"
                f"Return a JSON object with a single field 'score' containing "
                f"a float between 0 and 1, like this: {{'score': 0.85}}\n"
            )
            self.generator_agent.reset()
            response = self.generator_agent.step(prompt)
            try:
                response = self.generator_agent.step(
                    prompt, response_format=AgentResponse
                )
                agent_response = response.msgs[0].parsed.score  # type: ignore [union-attr]
                score = agent_response

                # Exit early if we find a very good solution (score > 0.9)
                if score > 0.9:
                    logger.info(
                        "Found excellent solution with score %.2f. "
                        "Stopping search early.",
                        score,
                    )
                    return current_solution

                if score > best_score:
                    best_score = score
                    best_solution = current_solution

                logger.info(
                    "Current search progress: %d/%d, best score: %.2f",
                    i + 1,
                    self.search_limit,
                    best_score,
                )
            except Exception as e:
                logger.error("Error parsing agent response: %s", str(e))
                continue

        # 3. If the answer is not completely correct,
        # use binary search to locate the error
        error_pos = self.binary_search_error(question, best_solution)

        # If no errors found (error_pos == -1), return the current solution
        if error_pos == -1:
            logger.info("No specific errors found in the solution")
            return best_solution

        # 4. Generate new solution based on correct part
        correct_part = '. '.join(best_solution.split('. ')[:error_pos]) + '.'
        final_solution = self.get_answer(question, correct_part)
        self.solution_tree[question] = {
            "solution": final_solution,
            "error_position": error_pos,
        }
        return final_solution

    def import_qa_from_json(self, data: Union[str, Dict[str, str]]) -> bool:
        r"""Import question and answer data from either a JSON file or a
        dictionary.

        Args:
            data (Union[str, Dict[str, str]]): Either a path to a JSON file
                containing QA pairs or a dictionary of question-answer pairs.
                If a string is provided, it's treated as a file path.
                The expected format is:
                {"question1": "answer1",
                 "question2": "answer2",
                 ...}

        Returns:
            bool: True if import was successful, False otherwise.
        """
        try:
            if isinstance(data, str):
                logger.info("Loading QA pairs from file: %s", data)
                with open(data, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
            else:
                logger.info("Loading QA pairs from provided dictionary")
                qa_data = data

            # Validate the data format
            if not isinstance(qa_data, dict):
                logger.error("Invalid data format: expected dictionary")
                return False

            # Update golden answers
            self.golden_answers.update(qa_data)
            logger.info("Successfully imported %d QA pairs", len(qa_data))
            return True

        except Exception as e:
            logger.error("Error importing QA data: %s", str(e))
            return False

    def export_solutions(self, filepath: str = 'solutions.json') -> None:
        r"""Export the solution process and results to a JSON file.
        Exports the solution tree, golden answers,
         and export timestamp to a JSON file.
        The exported data includes:
        - solutions: The solution tree
            with intermediate steps
        - golden_answers: The reference answers used for verification
        - export_time: ISO format timestamp of the export

        Args:
            filepath (str, optional): Path where the JSON file will be saved.
                (default::obj:`'solutions.json'`)

        Returns:
            None: The method writes to a file and logs the result but does not
                return any value.
        """
        export_data = {
            "solutions": self.solution_tree,
            "golden_answers": self.golden_answers,
            "export_time": datetime.now().isoformat(),
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Solutions exported successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting solutions: {e!s}")
