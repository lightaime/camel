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
from concurrent.futures import ThreadPoolExecutor
from typing import List

from .unstructured_io import UnstructuredIO


class MultiSourceParser:
    def __init__(self):
        r"""A class for parsing and
        processing multiple text sources in parallel.

        Args:
            uio (UnstructuredIO): An instance of UnstructuredIO
            for parsing files and URLs.
            clean_options (List[Tuple]): A list of text cleaning options to be
                applied to the parsed text. Default options include:
                - Replacing unicode quotes
                - Cleaning dashes
                - Removing non-ASCII characters
                - Cleaning extra whitespace
        """
        self.uio = UnstructuredIO()
        self.clean_options = [
            ('replace_unicode_quotes', {}),
            ('clean_dashes', {}),
            ('clean_non_ascii_chars', {}),
            ('clean_extra_whitespace', {}),
        ]

    def parse_multiple_sources(
        self,
        sources: List[str],
        chunk_size: int = 6000,
        overlap: int = 1,
    ) -> List[str]:
        r"""Parse multiple files or URLs and return chunked, cleaned text.

        This method processes multiple sources in parallel, extracts text
        content, combines the extracted elements, chunks them based on size,
        and applies text cleaning operations.

        Args:
            sources (List[str]): List of file paths or URLs to parse. Each
                source can be either a local file path or a valid URL.
            chunk_size (int, optional): Maximum number of characters per chunk.
                Defaults to 6000.
            overlap (int, optional): Number of overlapping chunks to create.
                This helps maintain context between chunks. Defaults to 1.

        Returns:
            List[str]: A list of cleaned text chunks. Each chunk is processed
            according to the clean_options specified in the class.

        Note:
            - If a source fails to parse, a warning message will be printed
            - If no elements are successfully parsed from any source, an empty
              list will be returned
            - Text cleaning is performed in parallel for better performance
        """
        all_elements = []

        # Parse all sources in parallel
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.uio.parse_file_or_url, source): source
                for source in sources
            }
            for future in futures:
                try:
                    elements = future.result()
                    # Only extend if elements were successfully parsed
                    if elements is not None:
                        all_elements.extend(elements)
                    else:
                        source = futures[future]
                        print(f"Warning: No elements extracted from {source}")
                except Exception as e:
                    source = futures[future]
                    print(f"Error processing {source}: {e!s}")

        if not all_elements:
            print("No elements were successfully parsed from any sources")
            return []

        # Chunk the elements
        chunks = self.uio.chunk_elements(
            elements=all_elements,
            chunk_type="chunk_by_title",
            max_characters=chunk_size,
            overlap=overlap,
        )

        # Clean each chunk in parallel
        with ThreadPoolExecutor() as executor:
            cleaned_chunks = list(
                executor.map(
                    lambda chunk: self.uio.clean_text_data(
                        text=str(chunk),
                        clean_options=self.clean_options,
                    ),
                    chunks,
                )
            )

        return cleaned_chunks


def parse_sources(
    sources: List[str],
    chunk_size: int = 6000,
    overlap: int = 1,
) -> List[str]:
    r"""Convenience function to parse and process multiple text sources.

    This is a wrapper function that creates a MultiSourceParser instance and
    uses it to process multiple sources. It provides a simpler interface for
    common use cases.

    Args:
        sources (List[str]): List of file paths or URLs to parse. Each source
            can be either a local file path or a valid URL.
        chunk_size (int, optional): Maximum number of characters per chunk.
            Defaults to 6000.
        overlap (int, optional): Number of overlapping chunks to create.
            This helps maintain context between chunks. Defaults to 1.

    Returns:
        List[str]: A list of cleaned text chunks. Each chunk is processed
        according to the default clean_options in MultiSourceParser.

    Example:
        >>> sources = [
        ...     "https://example.com/doc1.html",
        ...     "path/to/local/file.txt"
        ... ]
        >>> chunks = parse_sources(sources, chunk_size=5000, overlap=2)
        >>> print(len(chunks))  # Number of chunks generated
    """
    parser = MultiSourceParser()
    return parser.parse_multiple_sources(
        sources,
        chunk_size=chunk_size,
        overlap=overlap,
    )


# Example usage with line profiling
if __name__ == "__main__":
    sources = [
        "https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html",
        "https://docs.camel-ai.org/cookbooks/data_generation/self_instruct_data_generation.html",
        "https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html",
        "https://docs.camel-ai.org/cookbooks/data_generation/synthetic_dataevaluation%26filter_with_reward_model.html",
        "https://docs.camel-ai.org/cookbooks/data_generation/data_model_generation_and_structured_output_with_qwen.html#",
    ]

    chunks = parse_sources(sources)
    # Print chunks (optional)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
        print("\n" + "-" * 80)
