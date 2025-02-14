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

import logging
from itertools import cycle
from random import choice
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

from openai import AsyncStream, Stream
from pydantic import BaseModel

from camel.messages import OpenAIMessage
from camel.models.base_model import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    UnifiedModelType,
)
from camel.utils import BaseTokenCounter

logger = logging.getLogger(__name__)


class ModelProcessingError(Exception):
    r"""Raised when an error occurs during model processing."""

    pass


class ModelManager:
    r"""ModelManager choosing a model from provided list.
    Models are picked according to defined strategy.

    Args:
        models(Union[BaseModelBackend, List[BaseModelBackend]]):
            model backend or list of model backends
            (e.g., model instances, APIs)
        scheduling_strategy (str): name of function that defines how
            to select the next model. (default: :str:`round_robin`)
    """

    def __init__(
        self,
        models: Union[BaseModelBackend, List[BaseModelBackend]],
        scheduling_strategy: str = "round_robin",
    ):
        if isinstance(models, list):
            self.models = models
        else:
            self.models = [models]
        self.models_cycle = cycle(self.models)
        self.current_model = self.models[0]

        # Set the scheduling strategy; default is round-robin
        try:
            self.scheduling_strategy = getattr(self, scheduling_strategy)
        except AttributeError:
            logger.warning(
                f"Provided strategy: {scheduling_strategy} is not implemented."
                f"Using default 'round robin'"
            )
            self.scheduling_strategy = self.round_robin

    @property
    def model_type(self) -> UnifiedModelType:
        r"""Return type of the current model.

        Returns:
            Union[ModelType, str]: Current model type.
        """
        return self.current_model.model_type

    @property
    def model_config_dict(self) -> Dict[str, Any]:
        r"""Return model_config_dict of the current model.

        Returns:
            Dict[str, Any]: Config dictionary of the current model.
        """
        return self.current_model.model_config_dict

    @model_config_dict.setter
    def model_config_dict(self, model_config_dict: Dict[str, Any]):
        r"""Set model_config_dict to the current model.

        Args:
            model_config_dict (Dict[str, Any]): Config dictionary to be set at
                current model.
        """
        self.current_model.model_config_dict = model_config_dict

    @property
    def current_model_index(self) -> int:
        r"""Return the index of current model in self.models list.

        Returns:
            int: index of current model in given list of models.
        """
        return self.models.index(self.current_model)

    @property
    def token_limit(self):
        r"""Returns the maximum token limit for current model.

        This method retrieves the maximum token limit either from the
        `model_config_dict` or from the model's default token limit.

        Returns:
            int: The maximum token limit for the given model.
        """
        return self.current_model.token_limit

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Return token_counter of the current model.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        return self.current_model.token_counter

    def add_strategy(self, name: str, strategy_fn: Callable):
        r"""Add a scheduling strategy method provided by user in case when none
            of existent strategies fits.
            When custom strategy is provided, it will be set as
            "self.scheduling_strategy" attribute.

        Args:
            name (str): The name of the strategy.
            strategy_fn (Callable): The scheduling strategy function.
        """
        if not callable(strategy_fn):
            raise ValueError("strategy_fn must be a callable function.")
        setattr(self, name, strategy_fn.__get__(self))
        self.scheduling_strategy = getattr(self, name)
        logger.info(f"Custom strategy '{name}' added.")

    # Strategies
    def round_robin(self) -> BaseModelBackend:
        r"""Return models one by one in simple round-robin fashion.

        Returns:
             BaseModelBackend for processing incoming messages.
        """
        return next(self.models_cycle)

    def always_first(self) -> BaseModelBackend:
        r"""Always return the first model from self.models.

        Returns:
            BaseModelBackend for processing incoming messages.
        """
        return self.models[0]

    def random_model(self) -> BaseModelBackend:
        r"""Return random model from self.models list.

        Returns:
             BaseModelBackend for processing incoming messages.
        """
        return choice(self.models)

    def run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        r"""Process a list of messages by selecting a model based on
            the scheduling strategy.
            Sends the entire list of messages to the selected model,
            and returns a single response.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat
                history in OpenAI API format.

        Returns:
            Union[ChatCompletion, Stream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `Stream[ChatCompletionChunk]` in the stream mode.
        """
        self.current_model = self.scheduling_strategy()

        # Pass all messages to the selected model and get the response
        try:
            response = self.current_model.run(messages, response_format, tools)
        except Exception as exc:
            logger.error(f"Error processing with model: {self.current_model}")
            if self.scheduling_strategy == self.always_first:
                self.scheduling_strategy = self.round_robin
                logger.warning(
                    "The scheduling strategy has been changed to 'round_robin'"
                )
                # Skip already used one
                self.current_model = self.scheduling_strategy()
            raise exc
        return response

    async def arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        r"""Process a list of messages by selecting a model based on
            the scheduling strategy.
            Sends the entire list of messages to the selected model,
            and returns a single response.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat
                history in OpenAI API format.

        Returns:
            Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `AsyncStream[ChatCompletionChunk]` in the stream mode.
        """
        self.current_model = self.scheduling_strategy()

        # Pass all messages to the selected model and get the response
        try:
            response = await self.current_model.arun(
                messages, response_format, tools
            )
        except Exception as exc:
            logger.error(f"Error processing with model: {self.current_model}")
            if self.scheduling_strategy == self.always_first:
                self.scheduling_strategy = self.round_robin
                logger.warning(
                    "The scheduling strategy has been changed to 'round_robin'"
                )
                # Skip already used one
                self.current_model = self.scheduling_strategy()
            raise exc
        return response
