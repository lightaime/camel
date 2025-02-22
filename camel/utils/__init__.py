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

from .commons import (
    AgentOpsMeta,
    BatchProcessor,
    agentops_decorator,
    api_keys_required,
    check_server_running,
    create_chunks,
    dependencies_required,
    download_github_subdirectory,
    download_tasks,
    func_string_to_callable,
    get_first_int,
    get_local_time,
    get_prompt_template_key_words,
    get_pydantic_major_version,
    get_pydantic_object_schema,
    get_system_information,
    get_task_list,
    handle_http_error,
    is_docker_running,
    json_to_function_code,
    print_text_animated,
    retry_on_error,
    text_extract_from_web,
    to_pascal,
    track_agent,
    with_timeout,
)
from .constants import Constants
from .deduplication import DeduplicationResult, deduplicate_internally
from .response_format import get_pydantic_model
from .token_counting import (
    AnthropicTokenCounter,
    BaseTokenCounter,
    LiteLLMTokenCounter,
    MistralTokenCounter,
    OpenAITokenCounter,
    get_model_encoding,
)

__all__ = [
    "print_text_animated",
    "get_prompt_template_key_words",
    "get_local_time",
    "get_first_int",
    "download_tasks",
    "get_task_list",
    "check_server_running",
    "AnthropicTokenCounter",
    "get_system_information",
    "to_pascal",
    "get_model_encoding",
    "BaseTokenCounter",
    "OpenAITokenCounter",
    "LiteLLMTokenCounter",
    "Constants",
    "text_extract_from_web",
    "create_chunks",
    "dependencies_required",
    "api_keys_required",
    "is_docker_running",
    "MistralTokenCounter",
    "get_pydantic_major_version",
    "get_pydantic_object_schema",
    "func_string_to_callable",
    "json_to_function_code",
    "agentops_decorator",
    "AgentOpsMeta",
    "track_agent",
    "handle_http_error",
    "get_pydantic_model",
    "download_github_subdirectory",
    "generate_prompt_for_structured_output",
    "deduplicate_internally",
    "DeduplicationResult",
    "retry_on_error",
    "BatchProcessor",
    "with_timeout",
]
