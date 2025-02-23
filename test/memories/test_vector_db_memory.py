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
import random
from typing import Dict, List, Tuple

import pytest

from camel.memories import MemoryRecord, VectorDBMemory
from camel.memories.context_creators import ScoreBasedContextCreator
from camel.messages import BaseMessage
from camel.types import ModelType, OpenAIBackendRole, RoleType
from camel.utils.token_counting import OpenAITokenCounter


@pytest.fixture
def memory_and_message(request):
    # Initialize memory with context creator
    memory = VectorDBMemory(
        context_creator=ScoreBasedContextCreator(
            OpenAITokenCounter(ModelType.GPT_4),
            ModelType.GPT_4.token_limit,
        )
    )

    # Add system message
    memory.write_records(
        [
            MemoryRecord(
                message=BaseMessage(
                    "system",
                    role_type=RoleType.DEFAULT,
                    meta_dict=None,
                    content="You are a helpful assistant",
                ),
                role_at_backend=OpenAIBackendRole.SYSTEM,
            )
        ]
    )

    # Add conversations in random order
    messages = request.param["messages"]
    random.shuffle(messages)

    # Process each conversation
    for conv in messages:
        records = [
            MemoryRecord(
                message=BaseMessage(
                    "AI user",
                    role_type=RoleType.USER,
                    meta_dict=None,
                    content=conv["user"],
                ),
                role_at_backend=OpenAIBackendRole.USER,
            ),
            MemoryRecord(
                message=BaseMessage(
                    "AI assistant",
                    role_type=RoleType.ASSISTANT,
                    meta_dict=None,
                    content=conv["assistant"],
                ),
                role_at_backend=OpenAIBackendRole.ASSISTANT,
            ),
        ]
        memory.write_records(records)

    yield memory, messages, request.param["questions"]


@pytest.mark.parametrize(
    "memory_and_message",
    [
        {
            "messages": [
                {
                    "user": "What is renewable energy?",
                    "assistant": "Renewable energy is energy that is collected from renewable resources, which are naturally replenished on a human timescale, such as sunlight, wind, rain, tides, waves, and geothermal heat.",
                    "category": "energy",
                },
                {
                    "user": "How does renewable energy impact the environment?",
                    "assistant": "Renewable energy has a positive impact on the environment as it produces little to no greenhouse gas emissions, reduces air and water pollution, and conserves natural resources.",
                    "category": "energy",
                },
                {
                    "user": "How can I improve my study efficiency",
                    "assistant": "You can improve your study efficiency by setting goals, creating a study schedule, taking breaks, and using active learning techniques.",
                    "category": "education",
                },
                {
                    "user": "What strategies can help with language learning?",
                    "assistant": "Strategies that can help with language learning include practicing regularly, immersing yourself in the language, using flashcards, and speaking with native speakers.",
                    "category": "education",
                },
            ],
            "questions": [
                {
                    "user": "What are the advantages of using solar energy over fossil fuels?",
                    "category": "energy",
                },
                {
                    "user": "How do cultural immersion programs contribute to language learning success?",
                    "category": "education",
                },
            ],
        }
    ],
    indirect=True,
)
def test_vector_db_memory(
    memory_and_message: Tuple[
        VectorDBMemory, List[Dict[str, str]], List[Dict[str, str]]
    ],
):
    memory, messages, questions = memory_and_message

    for question in questions:
        # Add user question to memory
        user_msg = BaseMessage(
            "AI user", RoleType.USER, None, question["user"]
        )
        memory.write_records(
            [
                MemoryRecord(
                    message=user_msg, role_at_backend=OpenAIBackendRole.USER
                )
            ]
        )
        output_messages, _ = memory.get_context()

        # Check that context messages are from correct category
        for msg in output_messages[:-1]:
            for conv in messages:
                if (
                    conv["assistant"] == msg["content"]
                    or conv["user"] == msg["content"]
                ):
                    assert conv["category"] == question["category"]

        # Check final message matches input question
        assert output_messages[-1]["content"] == question["user"]
