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

from typing import Any, Dict, List

from camel.agents import ChatAgent
from camel.benchmarks import MathBenchmark, Mode
from camel.logger import get_logger

logger = get_logger(__name__)


class GSM8KBenchmark(MathBenchmark):
    r"""
    Benchmark for evaluating ChatAgents on the GSM8K dataset, a collection of grade-school-level math problems
    sourced from the Hugging Face Hub.

    Attributes:
        DATASET_NAME (str): The name of the dataset.
        DATASET_REPO (str): The repository location of the dataset on Hugging Face.
        QUESTION_COLUMN (str): The column containing math problems.
        ANSWER_COLUMN (str): The column containing solutions.
    """

    import pandas as pd
    from datasets import load_dataset

    DATASET_NAME = "gsm8k"
    DATASET_REPO = "openai/gsm8k"
    QUESTION_COLUMN = "question"
    ANSWER_COLUMN = "answer"

    def __init__(self, data_dir: str, save_to: str, processes: int = 1):
        r"""
        Initializes the GSM8K Benchmark instance.

        Args:
            data_dir (str): Directory for storing the dataset.
            save_to (str): Path for saving benchmark results.
            processes (int, optional): Number of parallel processes. Defaults to 1.
        """
        super().__init__(
            name="GSM8K",
            data_dir=data_dir,
            save_to=save_to,
            processes=processes,
        )
        self._data: Dict[str, List[Dict[str, Any]]] = {}

    def download(self) -> "GSM8KBenchmark":
        r"""
        Ensures the GSM8K dataset is available locally.
        Uses Hugging Face Datasets for automatic caching and management.

        Returns:
            GSM8KBenchmark: The benchmark instance after downloading.
        """
        logger.info("Ensuring GSM8K dataset is downloaded...")
        _ = load_dataset(
            self.DATASET_REPO, 'main', cache_dir=str(self.data_dir)
        )
        logger.info("GSM8K dataset is ready.")
        return self

    def load(self, force_download: bool = False) -> "GSM8KBenchmark":
        r"""
        Loads the GSM8K dataset into memory, optionally forcing a re-download.

        Args:
            force_download (bool, optional): Whether to force re-downloading the dataset. Defaults to False.

        Returns:
            GSM8KBenchmark: The benchmark instance after loading.
        """
        logger.info("Loading GSM8K dataset...")

        dataset = load_dataset(
            self.DATASET_REPO,
            'main',
            cache_dir=str(self.data_dir),
            download_mode="force_redownload"
            if force_download
            else "reuse_dataset_if_exists",
        )

        self._data = {
            "train": dataset["train"].to_pandas().to_dict(orient="records"),
            "test": dataset["test"].to_pandas().to_dict(orient="records"),
        }
        return self

    @property
    def valid(self) -> List[Dict[str, Any]]:
        r"""
        Returns an empty list since GSM8K does not have a validation set.

        Returns:
            List[Dict[str, Any]]: An empty list.
        """
        return []

    def _prepare_dataset(self, dataset: List[Dict[str, Any]]) -> pd.DataFrame:
        r"""
        Prepares the dataset by extracting the numeric solutions from the answer field.

        Args:
            dataset (List[Dict[str, Any]]): The dataset to process.

        Returns:
            pd.DataFrame: The processed dataset with extracted solutions.
        """
        df = pd.DataFrame(dataset)
        df["solution"] = df["answer"].str.extract(r"####\s*(-?\d+)")[0]
        return df

    def _generate_solutions(
        self, agent: ChatAgent, dataset: pd.DataFrame, mode: Mode
    ) -> pd.DataFrame:
        r"""
        Efficiently generates responses for each math problem using the ChatAgent,
        ensuring the agent resets between questions without unnecessary instantiations.

        Args:
            agent (ChatAgent): The agent responsible for generating answers.
            dataset (pd.DataFrame): The dataset containing math problems.
            mode (Mode): The evaluation mode for generating multiple responses.

        Returns:
            pd.DataFrame: The dataset with generated answers.
        """

        def generate_answer(question: str) -> List[str]:
            """Generate `k` responses while resetting the agent after each question."""
            agent.reset()  # Ensuring statelessness
            return [
                agent.step(question).msgs[0].content for _ in range(mode.k)
            ]

        dataset["answers"] = dataset["question"].apply(generate_answer)
        return dataset

    def _preprocess_answers(self, raw_answers: pd.Series) -> pd.Series:
        r"""
        Extracts numeric answers from generated responses using a regular expression.

        Args:
            raw_answers (pd.Series): The series containing raw model-generated responses.

        Returns:
            pd.Series: Extracted numeric answers.
        """
        return raw_answers.str.extract(r"####\s*(-?\d+)")[0]
