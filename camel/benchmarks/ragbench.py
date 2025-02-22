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

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

import numpy as np
from datasets import Dataset, load_dataset

from camel.agents import ChatAgent
from camel.benchmarks import BaseBenchmark
from camel.logger import get_logger
from camel.retrievers import AutoRetriever

logger = get_logger(__name__)


class RagasFields:
    r"""Constants for RAGAS evaluation field names."""

    INPUT_CONTEXT = "contexts"
    INPUT_QUESTION = "question"
    INPUT_ANSWER = "answer"


def annotate_dataset(
    dataset: Dataset,
    context_call: Optional[Callable[[Dict[str, Any]], List[str]]],
    answer_call: Optional[Callable[[Dict[str, Any]], str]],
) -> Dataset:
    r"""Annotate the dataset by adding context and answers using the provided
    functions.

    Args:
        dataset (Dataset): The input dataset to annotate.
        context_call (Optional[Callable[[Dict[str, Any]], List[str]]]):
            Function to generate context for each example.
        answer_call (Optional[Callable[[Dict[str, Any]], str]]): Function to
            generate answer for each example.

    Returns:
        Dataset: The annotated dataset with added contexts and/or answers.
    """

    def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
        if context_call:
            example["contexts"] = context_call(example)
        if answer_call:
            example["answer"] = answer_call(example)
        return example

    return dataset.map(process_example)


def rmse(
    input_trues: Sequence[float],
    input_preds: Sequence[float],
) -> Optional[float]:
    r"""Calculate Root Mean Squared Error (RMSE).

    Args:
        input_trues (Sequence[float]): Ground truth values.
        input_preds (Sequence[float]): Predicted values.

    Returns:
        Optional[float]: RMSE value, or None if inputs have different lengths.
    """
    if len(input_trues) != len(input_preds):
        logger.warning("Input lengths mismatch in RMSE calculation")
        return None

    trues = np.array(input_trues)
    preds = np.array(input_preds, dtype=float)

    # Ignore NaN values in predictions
    eval_idx = ~np.isnan(preds)
    if not np.any(eval_idx):
        logger.warning("No valid predictions for RMSE calculation")
        return None

    trues = trues[eval_idx]
    preds = preds[eval_idx]

    return float(np.sqrt(np.mean((preds - trues) ** 2)))


def auroc(trues: Sequence[bool], preds: Sequence[float]) -> float:
    r"""Calculate Area Under Receiver Operating Characteristic Curve (AUROC).

    Args:
        trues (Sequence[bool]): Ground truth binary values.
        preds (Sequence[float]): Predicted probability values.

    Returns:
        float: AUROC score.
    """
    from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

    eval_idx = ~np.isnan(preds)
    if not np.any(eval_idx):
        logger.warning("No valid predictions for AUROC calculation")
        return 0.5  # Return random classifier score

    return float(
        roc_auc_score(np.array(trues)[eval_idx], np.array(preds)[eval_idx])
    )


def ragas_calculate_metrics(
    dataset: Dataset,
    pred_context_relevance_field: Optional[str],
    pred_faithfulness_field: Optional[str],
    metrics_to_evaluate: Optional[List[str]] = None,
    ground_truth_context_relevance_field: str = "relevance_score",
    ground_truth_faithfulness_field: str = "adherence_score",
) -> Dict[str, Optional[float]]:
    r"""Calculate RAGAS evaluation metrics.

    Args:
        dataset (Dataset): The dataset containing predictions and ground truth.
        pred_context_relevance_field (Optional[str]): Field name for predicted
            context relevance.
        pred_faithfulness_field (Optional[str]): Field name for predicted
            faithfulness.
        metrics_to_evaluate (Optional[List[str]]): List of metrics to evaluate.
        ground_truth_context_relevance_field (str): Field name for ground truth
            relevance.
        ground_truth_faithfulness_field (str): Field name for ground truth
            adherence.

    Returns:
        Dict[str, Optional[float]]: Dictionary of calculated metrics.
    """
    metrics_to_evaluate = metrics_to_evaluate or [
        "context_relevancy",
        "faithfulness",
    ]
    calculated_metrics: Dict[str, Optional[float]] = {}

    if (
        "context_relevancy" in metrics_to_evaluate
        and pred_context_relevance_field
    ):
        trues_relevance = dataset[ground_truth_context_relevance_field]
        preds_relevance = dataset[pred_context_relevance_field]
        calculated_metrics["relevance_rmse"] = rmse(
            trues_relevance, preds_relevance
        )

    if "faithfulness" in metrics_to_evaluate and pred_faithfulness_field:
        trues_hallucination = ~np.array(
            dataset[ground_truth_faithfulness_field]
        )
        preds_hallucination = 1 - np.array(
            dataset[pred_faithfulness_field], dtype=float
        )
        calculated_metrics["hallucination_auroc"] = auroc(
            trues_hallucination.tolist(), preds_hallucination.tolist()
        )

    return calculated_metrics


def ragas_evaluate_dataset(
    dataset: Dataset,
    contexts_field_name: Optional[str],
    answer_field_name: Optional[str],
    metrics_to_evaluate: Optional[List[str]] = None,
) -> Dataset:
    r"""Evaluate the dataset using RAGAS metrics.

    Args:
        dataset (Dataset): Input dataset to evaluate.
        contexts_field_name (Optional[str]): Field name containing contexts.
        answer_field_name (Optional[str]): Field name containing answers.
        metrics_to_evaluate (Optional[List[str]]): List of metrics to evaluate.

    Returns:
        Dataset: Dataset with added evaluation metrics.
    """
    from ragas import evaluate  # type: ignore[import]
    from ragas.metrics import (  # type: ignore[import]
        context_relevancy,
        faithfulness,
    )

    metrics_to_evaluate = metrics_to_evaluate or [
        "context_relevancy",
        "faithfulness",
    ]

    # Rename fields if necessary
    if (
        contexts_field_name
        and contexts_field_name != RagasFields.INPUT_CONTEXT
    ):
        dataset = dataset.rename_column(
            contexts_field_name, RagasFields.INPUT_CONTEXT
        )
    if answer_field_name and answer_field_name != RagasFields.INPUT_ANSWER:
        dataset = dataset.rename_column(
            answer_field_name, RagasFields.INPUT_ANSWER
        )

    metrics = []
    if "context_relevancy" in metrics_to_evaluate:
        metrics.append(context_relevancy)
    if "faithfulness" in metrics_to_evaluate:
        metrics.append(faithfulness)

    ragas_result = evaluate(dataset, metrics=metrics)
    return Dataset.from_pandas(ragas_result.to_pandas())


class RAGBenchBenchmark(BaseBenchmark):
    r"""RAGBench Benchmark for evaluating RAG performance.

    This benchmark uses the rungalileo/ragbench dataset to evaluate
    retrieval-augmented generation (RAG) systems. It measures context
    relevancy and faithfulness metrics as described in
    https://arxiv.org/abs/2407.11005.

    Args:
        processes (int, optional): Number of processes for parallel processing.
        subset (str, optional): Dataset subset to use (e.g., "hotpotqa").
        split (str, optional): Dataset split to use (e.g., "test").
    """

    def __init__(
        self,
        processes: int = 1,
        subset: Literal[
            "covidqa",
            "cuad",
            "delucionqa",
            "emanual",
            "expertqa",
            "finqa",
            "hagrid",
            "hotpotqa",
            "msmarco",
            "pubmedqa",
            "tatqa",
            "techqa",
        ] = "hotpotqa",
        split: Literal["train", "test", "validation"] = "test",
    ) -> None:
        super().__init__("ragbench", "rag_bench", "", processes)
        self.subset = subset
        self.split = split
        self.dataset: Optional[Dataset] = None

    def download(self):
        r"""Download the RAGBench dataset."""
        try:
            self.dataset = load_dataset(
                "rungalileo/ragbench", self.subset, split=self.split
            )
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

    def load(self, force_download: bool = False):
        r"""Load the RAGBench dataset.

        Args:
            force_download (bool, optional): Whether to force download the
                data.
        """
        if force_download or self.dataset is None:
            logger.info(
                "%s dataset",
                "Force downloading" if force_download else "Loading",
            )
            self.download()

    def run(  # type: ignore[override, return]
        self,
        agent: ChatAgent,
        auto_retriever: AutoRetriever,
    ) -> Dict[str, Optional[float]]:
        r"""Run the benchmark evaluation.

        Args:
            agent (ChatAgent): Chat agent for generating answers.
            auto_retriever (AutoRetriever): Retriever for finding relevant
                contexts.

        Returns:
            Dict[str, Optional[float]]: Dictionary of evaluation metrics.
        """

        def context_call(example):
            retrieved_info = auto_retriever.run_vector_retriever(
                query=example['question'],
                contents=example['documents'],
                top_k=1,
                return_detailed_info=True,
                similarity_threshold=0.5,
            )
            return [c['text'] for c in retrieved_info['Retrieved Context']]

        def answer_call(example: Dict[str, Any]) -> str:
            user_msg = str(example)
            assistant_response = agent.step(user_msg)
            return assistant_response.msg.content.text

        # Annotate the dataset
        annotated_ds = annotate_dataset(
            self.dataset, context_call, answer_call
        )
        evaluated_ds = ragas_evaluate_dataset(
            annotated_ds,
            contexts_field_name="contexts",
            answer_field_name="answer",
            metrics_to_evaluate=["context_relevancy", "faithfulness"],
        )

        return ragas_calculate_metrics(
            evaluated_ds,
            pred_context_relevance_field="context_relevancy",
            pred_faithfulness_field="faithfulness",
        )
