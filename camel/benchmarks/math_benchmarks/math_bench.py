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


class MATHBenchmark(MathBenchmark):
    r"""
    Benchmark for evaluating ChatAgents on the MATH dataset, a collection of
    high school-level and competition-style math problems sourced from Hugging Face Hub.

    Attributes:
        DATASET_NAME (str): The name of the dataset.
        DATASET_REPO (str): The repository location of the dataset on Hugging Face.
        DATASET_CONFIGS (List[str]): The different subcategories within the dataset.
    """

    import pandas as pd
    from datasets import load_dataset

    DATASET_NAME = "math"
    DATASET_REPO = "EleutherAI/hendrycks_math"
    DATASET_CONFIGS = [
        'algebra',
        'counting_and_probability',
        'geometry',
        'intermediate_algebra',
        'number_theory',
        'prealgebra',
        'precalculus',
    ]

    def __init__(self, data_dir: str, save_to: str, processes: int = 1):
        r"""
        Initializes the MATH Benchmark instance.

        Args:
            data_dir (str): Directory for storing the dataset.
            save_to (str): Path for saving benchmark results.
            processes (int, optional): Number of parallel processes. Defaults to 1.
        """
        super().__init__(
            name="MATH",
            data_dir=data_dir,
            save_to=save_to,
            processes=processes,
        )
        self._data: Dict[str, List[Dict[str, Any]]] = {}

    def download(self) -> "MATHBenchmark":
        r"""
        Ensures the MATH dataset is available locally.
        Uses Hugging Face Datasets for automatic caching and management.

        Returns:
            MATHBenchmark: The benchmark instance after downloading.
        """
        logger.info("Ensuring MATH dataset is downloaded...")
        for config in self.DATASET_CONFIGS:
            _ = load_dataset(
                self.DATASET_REPO, config, cache_dir=str(self.data_dir)
            )
        logger.info("MATH dataset is ready.")
        return self

    def load(self, force_download: bool = False) -> "MATHBenchmark":
        r"""
        Loads the MATH dataset into memory, optionally forcing a re-download.

        Args:
            force_download (bool, optional): Whether to force re-downloading the dataset. Defaults to False.

        Returns:
            MATHBenchmark: The benchmark instance after loading.
        """
        logger.info("Loading MATH dataset...")

        self._data = {"train": [], "test": []}

        for config in self.DATASET_CONFIGS:
            dataset = load_dataset(
                self.DATASET_REPO,
                config,
                cache_dir=str(self.data_dir),
                download_mode="force_redownload"
                if force_download
                else "reuse_dataset_if_exists",
            )

            # Convert to pandas DataFrame and add a `config` column before converting to dict
            train_df = dataset["train"].to_pandas()
            train_df["config"] = config
            self._data["train"].extend(train_df.to_dict(orient="records"))

            test_df = dataset["test"].to_pandas()
            test_df["config"] = config
            self._data["test"].extend(test_df.to_dict(orient="records"))

        return self

    @property
    def valid(self) -> List[Dict[str, Any]]:
        r"""
        Returns an empty list since the MATH dataset does not have a validation set.

        Returns:
            List[Dict[str, Any]]: An empty list.
        """
        return []

    def _prepare_dataset(self, dataset: List[Dict[str, Any]]) -> pd.DataFrame:
        r"""
        Prepares the dataset by extracting solutions from the provided answers.

        - Renames the "problem" column to "questions" for consistency.
        - Extracts the final answer from solutions wrapped in `\boxed{}`.

        Args:
            dataset (List[Dict[str, Any]]): The dataset to process.

        Returns:
            pd.DataFrame: The processed dataset with extracted solutions.
        """
        df = pd.DataFrame(dataset)
        df.rename(columns={"problem": "questions"}, inplace=True)

        def extract_boxed(text: str) -> str:
            r"""
            Extracts the content inside the first correctly balanced `\boxed{}`.

            Args:
                text (str): The solution text containing `\boxed{}`.

            Returns:
                str: The extracted final answer.

            Raises:
                ValueError: If the answer cannot be extracted properly.
            """
            start_seq = r"\boxed{"
            stack = []  # Stack to track `{}` nesting
            content = []
            inside_boxed = False
            i = 0

            while i < len(text):
                if (
                    text[i : i + len(start_seq)] == start_seq
                    and not inside_boxed
                ):
                    inside_boxed = True
                    stack.append("{")
                    i += len(start_seq)  # Skip `\boxed{`
                    continue

                if inside_boxed:
                    if text[i] == "{":
                        stack.append("{")
                    elif text[i] == "}":
                        stack.pop()
                        if not stack:  # If stack is empty, we've closed `\boxed{}` correctly
                            return "".join(content)

                    content.append(text[i])

                i += 1

            raise ValueError(f"Couldn't extract value from solution: {text}")

        df["solutions"] = df["solution"].apply(extract_boxed)

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
