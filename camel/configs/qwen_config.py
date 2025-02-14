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
from __future__ import annotations

from typing import Dict, List, Optional, Union

from camel.configs.base_config import BaseConfig


class QwenConfig(BaseConfig):
    r"""Defines the parameters for generating chat completions using the
    Qwen API. You can refer to the following link for more details:
    https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api

    Args:
        stream (bool, optional): Whether to stream the response.
            (default: :obj:`False`)
        temperature (float, optional): Controls the diversity and
            focus of the generated results. Lower values make the output more
            focused, while higher values make it more diverse.
            (default: :obj:`0.3`)
        top_p (float, optional): Controls the diversity and focus of
            the generated results. Higher values make the output more diverse,
            while lower values make it more focused. (default: :obj:`0.9`)
        presence_penalty (float, optional): Controls the repetition
            content in the generated results. Positive values reduce the
            repetition of content, while negative values increase it.
            (default: :obj:`0.0`)
        response_format (Optional[Dict[str, str]], optional): Specifies the
            format of the returned content. The available values are
            `{"type": "text"}` or `{"type": "json_object"}`. Setting it to
            `{"type": "json_object"}` will output a standard JSON string.
            (default: :obj:`None`)
        max_tokens (Optional[int], optional): Allows the model to
            generate the maximum number of tokens.
            (default: :obj:`None`)
        seed (Optional[int], optional): Sets the seed parameter to make the
            text generation process more deterministic, typically used to
            ensure that the results are consistent across model runs. By
            passing the same seed value (specified by you) in each model call
            while keeping other parameters unchanged, the model is likely to
            return the same result.
            (default: :obj:`None`)
        stop (Optional[Union[str, List]], optional): Using the stop parameter,
            the model will automatically stop generating text when it is about
            to include the specified string or token_id. You can use the stop
            parameter to control the output of the model by passing sensitive
            words. (default: :obj:`None`)
        tools (List, optional): Specifies an array of tools that the model can
            call. It can contain one or more tool objects. During a function
            call process, the model will select one tool from the array.
            (default: :obj:`None`)
        extra_body (Optional[Dict[str, str]], optional): Additional parameters
            to be sent to the Qwen API. If you want to enable internet search,
            you can set this parameter to `{"enable_search": True}`.
            (default: :obj:`None`)
        include_usage (bool, optional): When streaming, specifies whether to
            include usage information in `stream_options`.
            (default: :obj:`True`)
    """

    stream: bool = False
    temperature: float = 0.3
    top_p: float = 0.9
    presence_penalty: float = 0.0
    response_format: Optional[Dict[str, str]] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List]] = None
    extra_body: Optional[Dict[str, str]] = None

    def __init__(self, include_usage: bool = True, **kwargs):
        super().__init__(**kwargs)
        # Only set stream_options when stream is True
        # Otherwise, it will raise error when calling the API
        if self.stream:
            self.stream_options = {"include_usage": include_usage}


QWEN_API_PARAMS = {param for param in QwenConfig.model_fields.keys()}
