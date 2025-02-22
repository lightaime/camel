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
from unittest.mock import MagicMock, mock_open, patch

from camel.benchmarks import NexusBenchmark

# Sample data for the test
SAMPLE_DATA = [
    {
        "Input": "How do I query the NVD?",
        "Output": "nvdlib.query('CVE-2021-12345')",
    },
]


def test_nexus_benchmark():
    # Patch necessary functions and classes to prevent real file system access
    with (
        patch("os.listdir", return_value=["data.parquet"]),
        patch("pathlib.Path.mkdir"),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_dir", return_value=True),
        patch("pandas.read_parquet") as mock_read_parquet,
        patch("builtins.open", mock_open()) as mock_file,
        patch(
            "camel.benchmarks.nexus.construct_tool_descriptions"
        ) as mock_construct_tools,
    ):
        # Initialize NexusBenchmark with mock paths within the patch context
        data_dir = "/mock_data_dir"
        save_to = "mock_save_to.json"
        benchmark = NexusBenchmark(data_dir=data_dir, save_to=save_to)

        # Setup pandas.read_parquet mock to return SAMPLE_DATA
        mock_df = MagicMock()
        mock_df.iterrows.return_value = [(0, SAMPLE_DATA[0])]
        mock_read_parquet.return_value = mock_df

        # Mock construct_tool_descriptions to return a fixed value
        mock_construct_tools.return_value = r"""Function:\ndef mock_tool():
        \n\"\"\"\nMocked tool description.\n\"\"\"\n"""

        # Mock the ChatAgent's step method to return the expected output
        mock_agent = MagicMock()
        mock_agent.step.return_value.msgs = [
            MagicMock(content="nvdlib.query('CVE-2021-12345')")
        ]

        # Run the benchmark
        result = benchmark.run(
            agent=mock_agent, task="NVDLibrary", randomize=False, subset=1
        )

        # Assert the benchmark loaded the data correctly
        assert len(benchmark._data) == 1
        assert benchmark._data[0].input == "How do I query the NVD?"
        assert benchmark._data[0].output == "nvdlib.query('CVE-2021-12345')"

        # Assert the benchmark results
        assert result["total"] == 1
        assert result["correct"] == 0
        assert result["accuracy"] == 0.0

        # Verify that the results were written correctly to the save_to file
        written_data = json.loads(mock_file().write.call_args[0][0])
        assert written_data["input"] == "How do I query the NVD?"
        assert written_data["agent_call"] == "nvdlib.query('CVE-2021-12345')"
        assert (
            written_data["ground_truth_call"]
            == "nvdlib.query('CVE-2021-12345')"
        )
        assert written_data["result"]
        assert written_data["error"] is None
