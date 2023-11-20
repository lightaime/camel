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
from enum import Enum
from typing import List

from openai import OpenAI

from camel.embeddings import BaseEmbedding


class OpenAIEmbeddingModelType(Enum):
    ADA2 = "text-embedding-ada-002"
    ADA1 = "text-embedding-ada-001"
    BABBAGE1 = "text-embedding-babbage-001"
    CURIE1 = "text-embedding-curie-001"
    DAVINCI1 = "text-embedding-davinci-001"

    @property
    def output_dim(self) -> int:
        if self is OpenAIEmbeddingModelType.ADA2:
            return 1536
        elif self is OpenAIEmbeddingModelType.ADA1:
            return 1024
        elif self is OpenAIEmbeddingModelType.BABBAGE1:
            return 2048
        elif self is OpenAIEmbeddingModelType.CURIE1:
            return 4096
        elif self is OpenAIEmbeddingModelType.DAVINCI1:
            return 12288
        else:
            raise ValueError(f"Unknown model type {self}.")


class OpenAIEmbedding(BaseEmbedding):
    r"""Provides text embedding functionalities using OpenAI's models.

    Args:
        model (OpenAiEmbeddingModel, optional): The model type to be used for
            generating embeddings. (default: :obj:`ModelType.ADA2`)

    Raises:
        RuntimeError: If an unsupported model type is specified.
    """

    def __init__(
        self,
        model_type: OpenAIEmbeddingModelType = OpenAIEmbeddingModelType.ADA2,
    ) -> None:
        self.model_type = model_type
        self.output_dim = model_type.output_dim
        self.client = OpenAI()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        r"""Generates embeddings for the given texts.

        Args:
            texts (List[str]): The texts for which to generate the embeddings.

        Returns:
            List[List[float]]: A list that represents the generated embedding
                as a list of floating-point numbers.
        """
        # TODO: count tokens
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError('OpenAI API key not found.')

        response = self.client.embeddings.create(
            input=texts,
            model=self.model_type.value,
        )
        return [data.embedding for data in response.data]

    def get_output_dim(self) -> int:
        r"""Returns the output dimension of the embeddings.

        Returns:
            int: The dimensionality of the embedding for the current model.
        """
        return self.output_dim
