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
from typing import List, Optional, Union

from camel.retrievers import AutoRetriever
from camel.toolkits import OpenAIFunction
from camel.toolkits.base import BaseToolkit
from camel.types import StorageType
from camel.utils import Constants
from camel.utils.commons import export_to_toolkit


class RetrievalToolkit(BaseToolkit):
    r"""A class representing a toolkit for information retrieval.

    This class provides methods for retrieving information from a local vector
    storage system based on a specified query.
    """

    def __init__(self, auto_retriever: Optional[AutoRetriever] = None) -> None:
        r"""Initializes a new instance of the RetrievalToolkit class."""
        self.ar = auto_retriever or AutoRetriever(
            vector_storage_local_path="camel/temp_storage",
            storage_type=StorageType.QDRANT,
        )

    @export_to_toolkit
    def information_retrieval(
        self,
        query: str,
        contents: Union[str, List[str]],
        top_k: int = Constants.DEFAULT_TOP_K_RESULTS,
        similarity_threshold: float = Constants.DEFAULT_SIMILARITY_THRESHOLD,
    ) -> str:
        r"""Retrieves information from a local vector storage based on the
        specified query. This function connects to a local vector storage
        system and retrieves relevant information by processing the input
        query. It is essential to use this function when the answer to a
        question requires external knowledge sources.

        Args:
            query (str): The question or query for which an answer is required.
            contents (Union[str, List[str]]): Local file paths, remote URLs or
                string contents.
            top_k (int, optional): The number of top results to return during
                retrieve. Must be a positive integer. Defaults to
                `DEFAULT_TOP_K_RESULTS`.
            similarity_threshold (float, optional): The similarity threshold
                for filtering results. Defaults to
                `DEFAULT_SIMILARITY_THRESHOLD`.

        Returns:
            str: The information retrieved in response to the query, aggregated
                and formatted as a string.

        Example:
            # Retrieve information about CAMEL AI.
            information_retrieval(query = "How to contribute to CAMEL AI?",
                                contents="https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md")
        """
        retrieved_info = self.ar.run_vector_retriever(
            query=query,
            contents=contents,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        return str(retrieved_info)

    def get_tools(self) -> List[OpenAIFunction]:
        r"""Returns a list of OpenAIFunction objects representing the
        functions in the toolkit.

        Returns:
            List[OpenAIFunction]: A list of OpenAIFunction objects
                representing the functions in the toolkit.
        """
        return [
            OpenAIFunction(self.information_retrieval),
        ]
