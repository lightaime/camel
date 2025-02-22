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
import importlib
import os
from unittest.mock import MagicMock, patch

import pytest
from github import Auth, Github

# Note: `OpenAPIToolkit` does not inherit from `BaseToolkit`, so it cannot
# use the `list_tools` and `get_a_tool` functions.
MODULE_NAMES = [
    'GithubToolkit',
    'MathToolkit',
    'GoogleMapsToolkit',
    'SearchToolkit',
    'SlackToolkit',
    'DalleToolkit',
    'TwitterToolkit',
    'WeatherToolkit',
    'RetrievalToolkit',
    'LinkedInToolkit',
    'RedditToolkit',
    'CodeExecutionToolkit',
]


@patch.object(Github, '__init__', lambda self, *args, **kwargs: None)
@patch.object(Github, 'get_repo', return_value=MagicMock())
@patch.object(Auth.Token, '__init__', lambda self, *args, **kwargs: None)
@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_toolkits(mock_get_repo, module_name):
    with patch.dict(os.environ, {"GITHUB_ACCESS_TOKEN": "fake_token"}):
        toolkit_class = getattr(
            importlib.import_module('camel.toolkits'), module_name
        )

        if module_name == 'GithubToolkit':
            toolkit = toolkit_class('camel-ai/camel')
        else:
            toolkit = toolkit_class()

        tool_names = toolkit.list_tools()

        for tool_name in tool_names:
            tool = toolkit.get_a_tool(tool_name)
            assert (
                tool
            ), f"Tool {tool_name} could not be retrieved in {module_name}"
            assert tool[0].func.__name__ == tool_name, (
                f"Retrieved tool's function name {tool[0].func.__name__}"
                f"does not match {tool_name}"
            )
