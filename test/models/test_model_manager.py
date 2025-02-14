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

from typing import TYPE_CHECKING, List

import pytest
from mock import Mock
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)

from camel.models import BaseModelBackend
from camel.models.model_manager import ModelManager


@pytest.mark.model_backend
@pytest.mark.parametrize(
    "models_number, strategy, calls_count, times_each_model_called",
    [
        (1, "round_robin", 9, 9),
        (3, "round_robin", 9, 3),
        (1, "not_existent", 9, 9),
        (3, "not_existent", 9, 3),
        (3, "always_first", 9, 9),
        (3, "random_model", 9, 9),
    ],
)
def test_model_manager(
    models_number: int,
    strategy: str,
    calls_count: int,
    times_each_model_called: int,
):
    models = (
        [Mock(run=Mock()) for _ in range(models_number)]
        if models_number > 1
        else Mock()
    )

    if TYPE_CHECKING:
        assert type(models) is List[BaseModelBackend]
    messages: List = []
    for _ in range(calls_count):
        msg = "message"
        if TYPE_CHECKING:
            assert type(msg) is ChatCompletionSystemMessageParam
        messages.append(msg)
    model_manager = ModelManager(models, scheduling_strategy=strategy)

    assert isinstance(model_manager.models, list)
    assert len(model_manager.models) == models_number
    assert model_manager.current_model_index == 0
    for _ in range(calls_count):
        model_manager.run(messages)

    if strategy in ("not_existent", "round_robin"):
        assert model_manager.scheduling_strategy.__name__ == "round_robin"
        for model in model_manager.models:
            if TYPE_CHECKING:
                assert isinstance(model._run, Mock)
            assert model._run.call_count == times_each_model_called
    if strategy == "always_first":
        assert model_manager.scheduling_strategy.__name__ == "always_first"
        assert models[0].run.call_count == times_each_model_called

    if strategy == "random_model":
        assert model_manager.scheduling_strategy.__name__ == "random_model"
        total_calls = 0
        for model in model_manager.models:
            if TYPE_CHECKING:
                assert isinstance(model._run, Mock)
            total_calls += model._run.call_count
        assert total_calls == times_each_model_called


def test_model_manager_always_first_turns_into_round_robin():
    models = [Mock(run=Mock(side_effect=Exception())) for _ in range(3)]
    model_manager = ModelManager(models, scheduling_strategy="always_first")
    assert model_manager.scheduling_strategy.__name__ == "always_first"
    assert model_manager.current_model_index == 0
    with pytest.raises(Exception):  # noqa: B017
        model_manager.run(["message"])
    assert model_manager.scheduling_strategy.__name__ == "round_robin"
    with pytest.raises(Exception):  # noqa: B017
        model_manager.run(["message"])
    assert model_manager.current_model_index == 1
