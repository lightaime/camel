# test/toolkits/test_semhash.py

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

import pytest
from datasets import load_dataset
from semhash import SemHash
from sentence_transformers import SentenceTransformer


def test_single_dataset_deduplication():
    """Test single dataset deduplication using AG News dataset"""
    # Load a dataset to deduplicate
    texts = load_dataset("ag_news", split="train")["text"]
    
    # Initialize a SemHash instance
    semhash = SemHash.from_records(records=texts)
    
    # Deduplicate the texts
    deduplicated_texts = semhash.self_deduplicate()
    
    # Verify the result
    assert deduplicated_texts is not None
    assert hasattr(deduplicated_texts, 'deduplicated')


def test_cross_dataset_deduplication():
    """Test deduplication across train and test datasets"""
    # Load two datasets to deduplicate
    train_texts = load_dataset("ag_news", split="train")["text"]
    test_texts = load_dataset("ag_news", split="test")["text"]
    
    # Initialize a SemHash instance with training data
    semhash = SemHash.from_records(records=train_texts)
    
    # Deduplicate the test data against the training data
    deduplicated_test_texts = semhash.deduplicate(records=test_texts)
    
    # Verify the result
    assert deduplicated_test_texts is not None
    assert hasattr(deduplicated_test_texts, 'deduplicated')


def test_multi_column_deduplication():
    """Test multi-column dataset deduplication using SQuAD 2.0 dataset"""
    # Load the dataset
    dataset = load_dataset("squad_v2", split="train")
    
    # Convert the dataset to a list of dictionaries
    records = [dict(row) for row in dataset]
    
    # Initialize SemHash with the columns to deduplicate
    semhash = SemHash.from_records(records=records, columns=["question", "context"])
    
    # Deduplicate the records
    deduplicated_records = semhash.self_deduplicate().deduplicated
    
    # Verify the result
    assert deduplicated_records is not None
    assert isinstance(deduplicated_records, list)
    assert all(isinstance(record, dict) for record in deduplicated_records)
    assert all("question" in record and "context" in record 
              for record in deduplicated_records)


def test_deduplication_result_functionality():
    """Test DeduplicationResult functionality"""
    # Load a dataset to deduplicate
    texts = load_dataset("ag_news", split="train")["text"]
    
    # Initialize a SemHash instance
    semhash = SemHash.from_records(records=texts)
    
    # Deduplicate the texts
    deduplication_result = semhash.self_deduplicate()
    
    # Test all functionality of DeduplicationResult
    assert hasattr(deduplication_result, 'deduplicated')
    assert hasattr(deduplication_result, 'duplicates')
    assert hasattr(deduplication_result, 'duplicate_ratio')
    assert hasattr(deduplication_result, 'exact_duplicate_ratio')
    
    # Get the least similar text from the duplicates
    least_similar = deduplication_result.get_least_similar_from_duplicates()
    assert least_similar is not None
    
    # Test rethresholding
    deduplication_result.rethreshold(0.95)
    assert deduplication_result.deduplicated is not None


def test_custom_encoder_with_sentence_transformer():
    """Test using sentence-transformers as custom encoder"""
    # Load a dataset to deduplicate
    texts = load_dataset("ag_news", split="train")["text"]
    
    # Load a sentence-transformers model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize a SemHash with the model and custom encoder
    semhash = SemHash.from_records(records=texts, model=model)
    
    # Deduplicate the texts
    deduplicated_texts = semhash.self_deduplicate()
    
    # Verify the result
    assert deduplicated_texts is not None
    assert hasattr(deduplicated_texts, 'deduplicated')


def test_pandas_dataframe_integration():
    """Test integration with Pandas DataFrames"""
    import pandas as pd
    
    # Load a dataset as a pandas dataframe
    dataframe = load_dataset("ag_news", split="train").to_pandas()
    
    # Convert the dataframe to a list of dictionaries
    dataframe = dataframe.to_dict(orient="records")
    
    # Initialize a SemHash instance with the columns to deduplicate
    semhash = SemHash.from_records(records=dataframe, columns=["text"])
    
    # Deduplicate the texts
    deduplicated_records = semhash.self_deduplicate().deduplicated
    
    # Convert the deduplicated records back to a pandas dataframe
    deduplicated_dataframe = pd.DataFrame(deduplicated_records)
    
    # Verify the result
    assert isinstance(deduplicated_dataframe, pd.DataFrame)
    assert "text" in deduplicated_dataframe.columns