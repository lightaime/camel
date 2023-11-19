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


class OpenAIEmbeddingModel(Enum):
    ADA2 = "text-embedding-ada-002"
    ADA1 = "text-embedding-ada-001"
    BABBAGE1 = "text-embedding-babbage-001"
    CURIE1 = "text-embedding-curie-001"
    DAVINCI1 = "text-embedding-davinci-001"


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
        model: OpenAIEmbeddingModel = OpenAIEmbeddingModel.ADA2,
    ) -> None:
        self.model = model
        if self.model == OpenAIEmbeddingModel.ADA2:
            self.output_dim = 1536
        elif self.model == OpenAIEmbeddingModel.ADA1:
            self.output_dim = 1024
        elif self.model == OpenAIEmbeddingModel.BABBAGE1:
            self.output_dim = 2048
        elif self.model == OpenAIEmbeddingModel.CURIE1:
            self.output_dim = 4096
        elif self.model == OpenAIEmbeddingModel.DAVINCI1:
            self.output_dim = 12288
        else:
            raise RuntimeError(f"Model type {model} is invalid.")

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
            model=self.model.value,
        )
        return [data.embedding for data in response.data]

    def get_output_dim(self) -> int:
        r"""Returns the output dimension of the embeddings.

        Returns:
            int: The dimensionality of the embedding for the current model.
        """
        return self.output_dim
