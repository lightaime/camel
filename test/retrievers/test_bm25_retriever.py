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

from unittest.mock import MagicMock, patch

import pytest

from camel.retrievers import BM25Retriever


@pytest.fixture
def mock_unstructured_modules():
    with patch('camel.retrievers.bm25_retriever.UnstructuredIO') as mock:
        yield mock


def test_bm25retriever_initialization():
    retriever = BM25Retriever()
    assert retriever.bm25 is None
    assert retriever.content_input_path == ""


def test_process(mock_unstructured_modules):
    mock_instance = mock_unstructured_modules.return_value

    # Create a mock chunk with metadata
    mock_chunk = MagicMock()
    mock_chunk.metadata.to_dict.return_value = {'mock_key': 'mock_value'}

    # Setup mock behavior
    mock_instance.parse_file_or_url.return_value = ["mock_element"]
    mock_instance.chunk_elements.return_value = [mock_chunk]

    bm25_retriever = BM25Retriever()
    bm25_retriever.process(content_input_path="mock_path")

    # Assert that methods are called as expected
    mock_instance.parse_file_or_url.assert_called_once_with("mock_path")
    mock_instance.chunk_elements.assert_called_once()


@patch('camel.retrievers.BM25Retriever')
def test_query(mock_bm25):
    # Setup the mock BM25 instance
    mock_bm25_instance = mock_bm25.return_value
    mock_bm25_instance.get_scores.return_value = [0.8, 0.5]  # Mock scores

    # Create mock chunks as dictionaries with 'text' and 'metadata'
    # It is the standard format for different chunk types
    mock_chunk1 = {'text': 'Chunk 1 text', 'metadata': {'id': 'chunk1'}}

    mock_chunk2 = {'text': 'Chunk 2 text', 'metadata': {'id': 'chunk2'}}

    retriever = BM25Retriever()
    retriever.bm25 = mock_bm25_instance
    retriever.content_input_path = 'dummy_path'
    retriever.chunks = [mock_chunk1, mock_chunk2]

    results = retriever.query('query', top_k=2)
    assert results[0]['similarity score'] == 0.8
    assert results[1]['similarity score'] == 0.5
    assert 'chunk1' in results[0]['metadata']['id']
    assert 'chunk2' in results[1]['metadata']['id']


def test_query_without_initialization():
    retriever = BM25Retriever()
    with pytest.raises(ValueError, match="BM25 model is not initialized"):
        retriever.query('query')
