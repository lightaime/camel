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
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from camel.agents import ChatAgent
from camel.benchmarks import GSM8KBenchmark, Mode

SAMPLE_DATA = [
    {"question": "What is 5 + 7?", "answer": "#### 12"},
    {"question": "Find the product of 8 and 3.", "answer": "#### 24"},
]


@pytest.fixture
def benchmark():
    r"""Fixture to initialize GSM8KBenchmark with a fully mocked file system."""
    with (
        patch("pathlib.Path.mkdir"),
        patch("pathlib.Path.is_dir", return_value=True),
        patch("pathlib.Path.exists", return_value=True),
    ):
        temp_dir = tempfile.mkdtemp()
        return GSM8KBenchmark(data_dir=Path(temp_dir), save_to=Path(temp_dir))


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.is_dir", return_value=True)
@patch("pathlib.Path.exists", return_value=True)
def test_run(mock_exists, mock_is_dir, mock_file, benchmark):
    r"""Test that GSM8KBenchmark runs correctly and writes expected results."""
    benchmark._data = {"test": SAMPLE_DATA}
    mock_agent = MagicMock(spec=ChatAgent)
    mock_agent.step.return_value.msgs = [MagicMock(content="#### 12")]
    results = benchmark.run(
        agent=mock_agent, on="test", mode=Mode("pass@k", 1)
    )
    assert "correct" in results._results[0]
    mock_file().write.assert_called()


def test_prepare_dataset(benchmark):
    r"""Test that _prepare_dataset extracts solutions correctly."""
    df = benchmark._prepare_dataset(SAMPLE_DATA)
    assert "solution" in df.columns
    assert list(df["solution"]) == ["12", "24"]


def test_preprocess_answers(benchmark):
    r"""Test that _preprocess_answers correctly extracts numeric values from answers."""
    raw_answers = pd.Series(
        ["#### 12", "#### 24", "Mock test with text and numbers 13 #### -7"]
    )
    processed = benchmark._preprocess_answers(raw_answers)
    assert list(processed) == ["12", "24", "-7"]


def test_download():
    r"""Test that GSM8KBenchmark downloads the dataset to the data/ directory."""

    data_dir = Path("data/")
    save_to = Path("data/")

    benchmark = GSM8KBenchmark(data_dir=str(data_dir), save_to=str(save_to))
    benchmark.download()

    assert data_dir.exists(), "Data directory was not created!"
    dataset_files = list(data_dir.glob("**/*"))
    assert len(dataset_files) > 0, "Dataset files were not downloaded!"
