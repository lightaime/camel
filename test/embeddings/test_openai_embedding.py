# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import os
from unittest.mock import MagicMock, patch
from camel.embeddings import OpenAIEmbedding


def test_openai_embedding():
    embedding_model = OpenAIEmbedding()
    text = "test 1."
    vector = embedding_model.embed(text)
    assert len(vector) == embedding_model.get_output_dim()

    embedding_model = OpenAIEmbedding(dimensions=256)
    text = "test 2"
    vector = embedding_model.embed(text)
    assert len(vector) == embedding_model.get_output_dim() == 256
    

@patch.dict(os.environ, {
    "OPENAI_API_KEY": "fake_openai_api_key",
    "OPENAI_API_BASE_URL": "https://fake-openai-api.com/v1"
})
@patch('openai.OpenAI', autospec=True)
def test_openai_embedding_with_env_url(mock_openai):
    mock_client_instance = mock_openai.return_value
    mock_embeddings = MagicMock()
    mock_embeddings.create.return_value = MagicMock(
        data=[
            MagicMock(embedding=[0.1, 0.2, 0.3]),
        ]
    )
    mock_client_instance.embeddings = mock_embeddings

    embedding = OpenAIEmbedding()
    result = embedding.embed("test text")

    mock_openai.assert_called_once_with(
        timeout=60,
        max_retries=3,
        api_key="fake_openai_api_key",
        base_url="https://fake-openai-api.com/v1"
    )

    assert result == [0.1, 0.2, 0.3]


@patch('openai.OpenAI', autospec=True)
def test_openai_embedding_with_custom_url(mock_openai):
    mock_client_instance = mock_openai.return_value
    mock_embeddings = MagicMock()
    mock_embeddings.create.return_value = MagicMock(
        data=[
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
    )
    mock_client_instance.embeddings = mock_embeddings

    custom_url = "https://custom-openai-api.com/v1"
    embedding = OpenAIEmbedding(url=custom_url)
    result = embedding.embed("test text")

    mock_openai.assert_called_once_with(
        timeout=60,
        max_retries=3,
        api_key=None,
        base_url=custom_url
    )

    assert result == [0.4, 0.5, 0.6]


def test_openai_embedding_output_dim():
    embedding = OpenAIEmbedding()
    assert embedding.get_output_dim() == embedding.model_type.output_dim

    custom_dim = 256
    embedding = OpenAIEmbedding(dimensions=custom_dim)
    assert embedding.get_output_dim() == custom_dim
