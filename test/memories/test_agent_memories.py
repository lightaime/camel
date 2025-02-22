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
from unittest.mock import MagicMock

import pytest

from camel.memories import (
    BaseContextCreator,
    ChatHistoryBlock,
    LongtermAgentMemory,
    MemoryRecord,
    VectorDBBlock,
)
from camel.messages import BaseMessage
from camel.types import OpenAIBackendRole, RoleType


class TestLongtermAgentMemory:
    @pytest.fixture
    def mock_chat_history_block(self):
        return MagicMock(spec=ChatHistoryBlock)

    @pytest.fixture
    def mock_vector_db_block(self):
        return MagicMock(spec=VectorDBBlock)

    @pytest.fixture
    def mock_context_creator(self):
        return MagicMock(spec=BaseContextCreator)

    def test_init_with_default_components(self, mock_context_creator):
        memory = LongtermAgentMemory(mock_context_creator)
        assert memory.chat_history_block is not None
        assert memory.vector_db_block is not None

    def test_init_with_custom_components(
        self,
        mock_chat_history_block,
        mock_vector_db_block,
        mock_context_creator,
    ):
        memory = LongtermAgentMemory(
            mock_context_creator,
            chat_history_block=mock_chat_history_block,
            vector_db_block=mock_vector_db_block,
        )
        assert memory.chat_history_block == mock_chat_history_block
        assert memory.vector_db_block == mock_vector_db_block

    def test_retrieve(
        self,
        mock_chat_history_block,
        mock_vector_db_block,
        mock_context_creator,
    ):
        mock_chat_history_block.retrieve.return_value = ["chat_history_record"]
        mock_vector_db_block.retrieve.return_value = ["vector_db_record"]
        memory = LongtermAgentMemory(
            mock_context_creator,
            chat_history_block=mock_chat_history_block,
            vector_db_block=mock_vector_db_block,
        )

        records = memory.retrieve()
        assert records == ["chat_history_record", "vector_db_record"]

    def test_write_records(
        self,
        mock_chat_history_block,
        mock_vector_db_block,
        mock_context_creator,
    ):
        memory = LongtermAgentMemory(
            mock_context_creator,
            chat_history_block=mock_chat_history_block,
            vector_db_block=mock_vector_db_block,
        )
        records = [
            MemoryRecord(
                message=BaseMessage(
                    role_name="user",
                    role_type=RoleType.USER,
                    content="test message {}".format(i),
                ),
                role_at_backend=OpenAIBackendRole.USER,
            )
            for i in range(5)
        ]

        memory.write_records(records)
        mock_chat_history_block.write_records.assert_called_once_with(records)
        mock_vector_db_block.write_records.assert_called_once_with(records)
        assert memory._current_topic == "test message 4"

    def test_clear(
        self,
        mock_chat_history_block,
        mock_vector_db_block,
        mock_context_creator,
    ):
        memory = LongtermAgentMemory(
            mock_context_creator,
            chat_history_block=mock_chat_history_block,
            vector_db_block=mock_vector_db_block,
        )
        memory.clear()
        mock_chat_history_block.clear.assert_called_once()
        mock_vector_db_block.clear.assert_called_once()
