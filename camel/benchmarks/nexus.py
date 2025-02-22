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

import ast
import json
import logging
import os
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from camel.agents import ChatAgent
from camel.benchmarks.base import BaseBenchmark

logger = logging.getLogger(__name__)


# Define the data class
@dataclass
class NexusSample:
    r"""Nexus benchmark dataset sample."""

    input: str
    output: str


@dataclass
class NexusTool:
    r"""Nexus benchmark tool"""

    function_calls: str
    descriptions: str


dataset_mapping = {
    "NVDLibrary": "Nexusflow/NVDLibraryBenchmark",
    "VirusTotal": "Nexusflow/VirusTotalBenchmark",
    "PlacesAPI": "Nexusflow/PlacesAPIBenchmark",
    "ClimateAPI": "Nexusflow/ClimateAPIBenchmark",
    "OTX": "Nexusflow/OTXAPIBenchmark",
    "VirusTotal-NestedCalls": "Nexusflow/vt_multiapi",
    "VirusTotal-ParallelCalls": "Nexusflow/vt_multiapi",
    "NVDLibrary-NestedCalls": "Nexusflow/CVECPEAPIBenchmark",
}

TOOL_CALLING_PROMPT = """
You are given multiple functions and a user query.

Please proceed with generating a function call for the function \
with the proper arguments that best answers the given prompt.

Respond with nothing but the function call ONLY, such that I can \
directly execute your function call without any post processing \
necessary from my end. Do not use variables. 
If there are more than two function calls, separate them with a semicolon (;).

{tools}

Question: {input}
"""


class NexusBenchmark(BaseBenchmark):
    r"""Nexus Function Calling Benchmark adapted from `NexusRaven V2
    Function Calling Benchmark`
    <https://huggingface.co/collections/Nexusflow/nexusraven-v2-function-calling-benchmark-657a597fb84dbe7a09ebfc3e>.

    Args:
        data_dir (str): The directory to save the data.
        save_to (str): The file to save the results.
        processes (int, optional): The number of processes to use.
            (default: :obj:`1`)
    """

    def __init__(
        self,
        data_dir: str,
        save_to: str,
        processes: int = 1,
    ):
        r"""Initialize the Nexus Function Calling benchmark.

        Args:
            data_dir (str): The directory to save the data.
            save_to (str): The file to save the results.
            processes (int, optional): The number of processes to use for
                parallel processing. (default: :obj:`1`)
        """
        super().__init__("nexus", data_dir, save_to, processes)
        self._data: List[NexusSample] = []  # type: ignore[assignment]

    def download(self):
        r"""Download the Nexus Functional Calling Benchmark dataset."""
        from huggingface_hub import snapshot_download

        for dataset_name, repo_id in dataset_mapping.items():
            local_dir = self.data_dir / dataset_name
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=True,
            )

    def load(self, dataset_name: str, force_download: bool = False):  # type: ignore[override]
        r"""Load the Nexus Benchmark dataset.

        Args:
            dataset_name (str): Name of the specific dataset to be loaded.
            force_download (bool): Whether to force download the data.
        """

        def _load_csv_data(dataset_dir: Path) -> List:
            r"""Load datasets from CSV files."""
            dataset = []
            for file_name in os.listdir(dataset_dir):
                file_path = dataset_dir / file_name
                if file_name.endswith(".csv"):
                    data = pd.read_csv(file_path)
                    for _, sample in data.iterrows():
                        dataset.append(
                            NexusSample(
                                sample["Input"], "".join(sample["Output"])
                            )
                        )
                    continue

                logger.warning(f"Skipping unsupported file: {file_name}")
            return dataset

        def _load_parquet_data(data_dir: Path, dataset_name: str) -> List:
            r"""Load datasets from Parquet files."""
            dataset = []
            if not data_dir.exists():
                raise FileNotFoundError(
                    f"Data directory '{data_dir}' does not exist."
                )

            for file_name in os.listdir(data_dir):
                file_path = data_dir / file_name
                if file_name.endswith(".parquet"):
                    data = pd.read_parquet(file_path)
                    dataset.extend(_process_parquet_data(data, dataset_name))
                    continue

                logger.warning(f"Skipping unsupported file: {file_name}")

            return dataset

        def _process_parquet_data(
            data: pd.DataFrame, dataset_name: str
        ) -> List:
            r"""Process data from Parquet files based on dataset name."""
            dataset: List = []
            dataset_handlers = {
                "NVDLibrary": _process_nvdlibrary,
                "VirusTotal": _process_simple,
                "PlacesAPI": _process_simple,
                "ClimateAPI": _process_simple,
                "OTX": _process_simple,
                "VirusTotal-NestedCalls": _process_nested_calls,
                "VirusTotal-ParallelCalls": _process_parallel_calls,
            }

            if dataset_name not in dataset_handlers:
                logger.warning(
                    f"No specific handler for dataset: {dataset_name}"
                )
                return dataset

            handler = dataset_handlers[dataset_name]
            for _, sample in data.iterrows():
                processed_sample = handler(sample)
                if processed_sample:
                    dataset.append(processed_sample)
            return dataset

        def _process_nvdlibrary(sample) -> NexusSample:
            r"""Process samples for the NVDLibrary dataset."""
            return NexusSample(
                sample["Input"], sample["Output"].replace("r = nvdlib.", "")
            )

        def _process_simple(sample) -> NexusSample:
            r"""Process samples for simple datasets (e.g., VirusTotal)."""
            return NexusSample(sample["Input"], sample["Output"])

        def _process_nested_calls(sample) -> Union[NexusSample, None]:
            r"""Process samples for VirusTotal-NestedCalls dataset."""
            if len(sample["fncall"]) == 1:
                return NexusSample(
                    sample["generated_question"], "".join(sample["fncall"])
                )
            return None

        def _process_parallel_calls(sample) -> Union[NexusSample, None]:
            r"""Process samples for VirusTotal-ParallelCalls dataset."""
            if len(sample["fncall"]) > 1:
                return NexusSample(
                    sample["generated_question"], "; ".join(sample["fncall"])
                )
            return None

        if force_download:
            logger.info("Force downloading data.")
            self.download()

        # Validate dataset name
        if dataset_name not in dataset_mapping:
            available_datasets = list(dataset_mapping.keys())
            raise ValueError(
                f"Dataset '{dataset_name}' is not recognized. "
                f"Available datasets: {available_datasets}"
            )

        # Get the dataset directory
        dataset_dir = self.data_dir / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"The dataset directory for '{dataset_name}' \
                does not exist at {dataset_dir}. "
                "Please download it first."
            )

        # Load the dataset
        if dataset_name == "NVDLibrary-NestedCalls":
            self._data = _load_csv_data(dataset_dir)
        else:
            self._data = _load_parquet_data(dataset_dir / "data", dataset_name)

    @property
    def train(self):
        r"""Get the training set."""
        raise NotImplementedError(
            "Nexus Functional Calling has only a single 'train' set."
        )

    def run(  # type: ignore[override, return]
        self,
        agent: ChatAgent,
        task: Literal[
            "NVDLibrary",
            "VirusTotal",
            "OTX",
            "PlacesAPI",
            "ClimateAPI",
            "VirusTotal-ParallelCalls",
            "VirusTotal-NestedCalls",
            "NVDLibrary-NestedCalls",
        ],
        randomize: bool = False,
        subset: Optional[int] = None,
    ) -> Dict[str, Any]:
        r"""Run the benchmark.

        Args:
            agent (ChatAgent): The agent to run the benchmark.
            task (Literal["NVDLibrary", "VirusTotal", "OTX",
            "PlacesAPI", "ClimateAPI", "VirusTotal-ParallelCalls",
            "VirusTotal-NestedCalls",
            "NVDLibrary-NestedCalls"]): The task to run the benchmark.
            randomize (bool, optional): Whether to randomize the data.
                (default: :obj:`False`)
            subset (Optional[int], optional): The subset of data to run.
                (default: :obj:`None`)

        Returns:
            Dict[str, Any]: The results of the benchmark.
        """

        if task not in dataset_mapping:
            raise ValueError(f"Invalid value for dataset: {task}.")

        logger.info(f"Running Nexus Function Calling benchmark on {task}.")
        self.load(task)
        datas = self._data

        # Shuffle and subset data if necessary
        if randomize:
            random.shuffle(datas)
        if subset:
            datas = datas[:subset]

        logger.info(f"Number of tasks: {len(datas)}")

        # Initialize results storage
        self._results = []

        # Process samples
        tools = construct_tool_descriptions(task)
        with open(self.save_to, "w") as f:
            for sample in tqdm(datas, desc="Running"):
                prompt = construct_prompt(input=sample.input, tools=tools)
                ground_truth_call = sample.output
                try:
                    # Generate response
                    response = agent.step(prompt)
                    agent_call = response.msgs[0].content.text

                    # Evaluate response
                    if agent_call:
                        result = compare_function_calls(
                            agent_call=agent_call,
                            ground_truth_call=ground_truth_call,
                        )
                        self._results.append(
                            {
                                "input": sample.input,
                                "agent_call": agent_call,
                                "ground_truth_call": ground_truth_call,
                                "result": result,
                                "error": None,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Error in processing task: {sample.input}")
                    self._results.append(
                        {
                            "input": sample.input,
                            "agent_call": None,
                            "ground_truth_call": ground_truth_call,
                            "result": 0,
                            "error": str(e),
                        }
                    )

                agent.reset()

                f.write(json.dumps(self._results[-1], indent=2) + "\n")
                f.flush()

        total = len(self._results)
        correct = sum(r["result"] for r in self._results)

        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
        }


# Utility functions
def construct_tool_descriptions(dataset_name: str) -> str:
    r"""Construct tool descriptions from function definitions and
    descriptions."""
    tool_dataset_mapping = {
        "NVDLibrary": "CVECPE",
        "VirusTotal": "VirusTotal",
        "PlacesAPI": "Places",
        "ClimateAPI": "Climate",
        "OTX": "OTX",
        "VirusTotal-NestedCalls": "VT_Multi (Nested)",
        "VirusTotal-ParallelCalls": "VT_Multi (Parallel)",
        "NVDLibrary-NestedCalls": "CVECPE_Multi (Nested)",
    }

    if dataset_name not in tool_dataset_mapping:
        raise ValueError(
            f"Dataset '{dataset_name}' is not recognized. "
            f"Available datasets: {list(dataset_mapping.keys())}"
        )

    # Load the dataset based on the dataset name
    dataset = load_dataset(
        "Nexusflow/Function_Call_Definitions",
        name=tool_dataset_mapping[dataset_name],
    )["train"]

    # Construct tool descriptions
    tools = [
        NexusTool(tool["function_calls"], tool["descriptions"])
        for tool in dataset
    ]

    # Generate the tool prompt
    tool_prompt = "".join(
        f"Function:\ndef {tool.function_calls}:\n"
        + "\"\"\"\n"
        + f"{tool.descriptions}\n"
        + "\"\"\"\n"
        for tool in tools
    )

    return tool_prompt


def construct_prompt(input: str, tools: str) -> str:
    r"Construct prompt from tools and input."
    return TOOL_CALLING_PROMPT.format(tools=tools, input=input)


# Functions for function call evaluation
def parse_function_call(
    call: str,
) -> Tuple[Optional[str], Optional[List[Any]], Optional[Dict[str, Any]]]:
    r"""Parse a function call string to extract the function name,
    positional arguments, and keyword arguments, including
    nested function calls.

    Args:
        call (str): A string in the format `func(arg1, arg2, kwarg=value)`.

    Returns:
        tuple: (function_name (str), positional_args (list),
        keyword_args (dict)) or (None, None, None).
    """

    def preprocess_input(call: str) -> str:
        r"""Remove formatting like code blocks and whitespace."""
        if call.strip().startswith("```python"):
            call = call.strip().removeprefix("```python").removesuffix("```")
        return textwrap.dedent(call).strip()

    def evaluate_arg(arg):
        r"""Recursively evaluate arguments, including nested calls."""
        if isinstance(arg, ast.Call):
            # Recursively parse nested calls
            func_name, args, kwargs = parse_function_call(ast.unparse(arg))
            return func_name, args, kwargs
        elif isinstance(
            arg, ast.Constant
        ):  # Handle literals like numbers, strings, etc.
            return arg.value
        elif isinstance(arg, ast.List):  # Handle list literals
            return [evaluate_arg(el) for el in arg.elts]
        elif isinstance(arg, ast.Dict):  # Handle dictionary literals
            return {
                evaluate_arg(k): evaluate_arg(v)
                for k, v in zip(arg.keys, arg.values)
            }
        elif isinstance(arg, ast.Tuple):  # Handle tuple literals
            return tuple(evaluate_arg(el) for el in arg.elts)
        else:
            return ast.literal_eval(arg)  # Safely evaluate other types

    call = preprocess_input(call)
    parsed_calls = []

    try:
        # Parse the string into an AST
        parsed_calls = call.split(";")
        for single_call in parsed_calls:
            tree = ast.parse(single_call, mode='eval')

            # Ensure it's a function call
            if isinstance(tree.body, ast.Call):
                # Extract function name
                if isinstance(
                    tree.body.func, ast.Name
                ):  # Simple function call
                    func_name = tree.body.func.id
                elif isinstance(
                    tree.body.func, ast.Attribute
                ):  # Attribute function call
                    func_name = (
                        f"{tree.body.func.value.id}.{tree.body.func.attr}"  # type: ignore[attr-defined]
                    )
                else:
                    raise ValueError(f"Unsupported function call: {call}")

                # Extract positional arguments
                args = [evaluate_arg(arg) for arg in tree.body.args]

                # Extract keyword arguments
                kwargs: Dict[str, Any] = {
                    kw.arg: evaluate_arg(kw.value)
                    for kw in tree.body.keywords
                    if kw.arg is not None
                }
                logger.info("Valid call.")
                return func_name, args, kwargs
        else:
            raise ValueError(f"Not a valid function call: {call}")
    except Exception as e:
        logger.info(f"Error parsing call: {call}, {e}")
        return None, None, None


def compare_function_calls(agent_call: str, ground_truth_call: str) -> bool:
    r"""Compare the function name and arguments of
    agent_call and ground_truth_call.
    Args:
        agent_call (str): Function call by agent.
        ground_truth_call (str): Ground truth function call.

    Returns:
        - `True` if the function names and arguments match.
        - `False` otherwise.
    """
    # Parse both calls
    agent_parsed = parse_function_call(agent_call)
    gt_parsed = parse_function_call(ground_truth_call)

    if agent_parsed and gt_parsed:
        return agent_parsed == gt_parsed
    else:
        return False
