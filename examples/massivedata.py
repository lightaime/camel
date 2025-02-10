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
from camel.loaders import parse_sources

sources = [
    "https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html",
    "https://docs.camel-ai.org/cookbooks/data_generation/self_instruct_data_generation.html",
    "https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html",
    "https://docs.camel-ai.org/cookbooks/data_generation/synthetic_dataevaluation%26filter_with_reward_model.html",
    "https://docs.camel-ai.org/cookbooks/data_generation/data_model_generation_and_structured_output_with_qwen.html#",
]
# Example with custom chunk size and overlap
chunks = parse_sources(sources, chunk_size=10000, overlap=1)
print(f"Successfully parsed {len(chunks)} chunks")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk)
    print("\n" + "-" * 80)
