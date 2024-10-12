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
from camel.toolkits import ToolManager
from camel.toolkits.github_toolkit import GithubToolkit
from camel.toolkits.openai_function import OpenAIFunction


def pretty_print_list(title, items):
    print(f"\n{'=' * 40}\n{title}:\n{'-' * 40}")
    if not items:
        print("  (No project)")
    else:
        for index, item in enumerate(items, start=1):
            print(f"  {index}. {item}")
    print('=' * 40)


manager = ToolManager()

toolkits = manager.list_toolkits()
toolkit_classes = manager.list_toolkit_classes()

pretty_print_list("Function Toolkits", toolkits)
pretty_print_list("Class Toolkits", toolkit_classes)
"""
===============================================================================
========================================
Function Toolkits:
----------------------------------------
  1. get_dalle_img
  2. add
  3. mul
  4. sub
  5. query_wolfram_alpha
  6. search_duckduckgo
  7. search_google
  8. search_wiki
  9. get_weather_data
========================================

========================================
Class Toolkits:
----------------------------------------
  1. CodeExecutionToolkit: execute_code
  2. GithubToolkit: create_pull_request, retrieve_issue, retrieve_issue_list, 
  retrieve_pull_requests
  3. GoogleMapsToolkit: get_address_description, get_elevation, get_timezone
  4. LinkedInToolkit: create_post, delete_post, get_profile
  5. RedditToolkit: collect_top_posts, perform_sentiment_analysis, 
  track_keyword_discussions
  6. RetrievalToolkit: information_retrieval
  7. SlackToolkit: create_slack_channel, delete_slack_message, 
  get_slack_channel_information, get_slack_channel_message, 
  join_slack_channel, leave_slack_channel, send_slack_message
  8. TwitterToolkit: create_tweet, delete_tweet, get_my_user_profile
========================================
===============================================================================
"""

matching_toolkits_test = manager.search_toolkits('weather')
pretty_print_list("Matching Toolkit", matching_toolkits_test)


def strict_search_algorithm(keyword: str, description: str) -> bool:
    return keyword.lower() in description.lower()


matching_toolkits_custom = manager.search_toolkits(
    'weather', algorithm=strict_search_algorithm
)
pretty_print_list(
    "Custom Algorithm Matching Toolkit", matching_toolkits_custom
)
"""
===============================================================================
========================================
Matching Toolkit:
----------------------------------------
  1. get_weather_data
========================================

========================================
Custom Algorithm Matching Toolkit:
----------------------------------------
  1. get_weather_data
========================================
===============================================================================
"""

tool = manager.get_toolkit('get_weather_data')
if isinstance(tool, OpenAIFunction):
    print("\nFunction Description:")
    print('-' * 40)
    print(tool.get_function_description())
"""
===============================================================================
Function Description:
----------------------------------------
Fetch and return a comprehensive weather report for a given city
as a string. The report includes current weather conditions,
temperature, wind details, visibility, and sunrise/sunset times,
all formatted as a readable string.

The function interacts with the OpenWeatherMap API to
retrieve the data.

===============================================================================
"""


def div(a: int, b: int) -> float:
    r"""Divides two numbers.

    Args:
        a (int): The dividend in the division.
        b (int): The divisor in the division.

    Returns:
        float: The quotient of the division.

    Raises:
        ValueError: If the divisor is zero.
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")

    return a / b


camel_github_toolkit = GithubToolkit(repo_name='camel-ai/camel')

added_tools = manager.register_tool(
    [div, camel_github_toolkit]
)  # manager.register_tool(div) is also supported.

pretty_print_list("Added Tools", added_tools)
pretty_print_list("Available Toolkits for now", manager.list_toolkits())
"""
===============================================================================
========================================
Added Tools:
----------------------------------------
  1. <camel.toolkits.openai_function.OpenAIFunction object at 0x1077b5390>
  2. <camel.toolkits.openai_function.OpenAIFunction object at 0x1077b55a0>
  3. <camel.toolkits.openai_function.OpenAIFunction object at 0x1077b5c30>
  4. <camel.toolkits.openai_function.OpenAIFunction object at 0x1077b6590>
  5. <camel.toolkits.openai_function.OpenAIFunction object at 0x1077b63e0>
========================================
========================================
Available Toolkits for now:
----------------------------------------
  1. get_dalle_img
  2. add
  3. mul
  4. sub
  5. query_wolfram_alpha
  6. search_duckduckgo
  7. search_google
  8. search_wiki
  9. get_weather_data
  10. div
  11. GithubToolkit_create_pull_request
  12. GithubToolkit_retrieve_issue
  13. GithubToolkit_retrieve_issue_list
  14. GithubToolkit_retrieve_pull_requests
========================================
===============================================================================
"""

crab_github_toolkit = GithubToolkit(repo_name='ZackYule/crab')

# Custom instance names are supported here.
manager.add_toolkit_from_instance(
    crab_github_toolkit=crab_github_toolkit,
)

matching_tools_for_github = manager.search_toolkits('github')
pretty_print_list("Matching Tools for GitHub", matching_tools_for_github)
"""
===============================================================================
========================================
Matching Tools for GitHub:
----------------------------------------
  1. GithubToolkit_create_pull_request
  2. GithubToolkit_retrieve_issue
  3. GithubToolkit_retrieve_issue_list
  4. GithubToolkit_retrieve_pull_requests
  5. crab_github_toolkit_create_pull_request
  6. crab_github_toolkit_retrieve_issue
  7. crab_github_toolkit_retrieve_issue_list
  8. crab_github_toolkit_retrieve_pull_requests
========================================
===============================================================================
"""

if isinstance(matching_tools_for_github, list):
    tools_instances = manager.get_toolkits(names=matching_tools_for_github)
    pretty_print_list("Tools Instances", tools_instances)

"""
===============================================================================
========================================
Tools Instances:
----------------------------------------
  1. <bound method GithubToolkit.create_pull_request of <camel.toolkits.
  github_toolkit.GithubToolkit object at 0x104e06aa0>>
  2. <bound method GithubToolkit.retrieve_issue of <camel.toolkits.
  github_toolkit.GithubToolkit object at 0x104e06aa0>>
  3. <bound method GithubToolkit.retrieve_issue_list of <camel.toolkits.
  github_toolkit.GithubToolkit object at 0x104e06aa0>>
  4. <bound method GithubToolkit.retrieve_pull_requests of <camel.toolkits.
  github_toolkit.GithubToolkit object at 0x104e06aa0>>
  5. <bound method GithubToolkit.create_pull_request of <camel.toolkits.
  github_toolkit.GithubToolkit object at 0x105286560>>
  6. <bound method GithubToolkit.retrieve_issue of <camel.toolkits.
  github_toolkit.GithubToolkit object at 0x105286560>>
  7. <bound method GithubToolkit.retrieve_issue_list of <camel.toolkits.
  github_toolkit.GithubToolkit object at 0x105286560>>
  8. <bound method GithubToolkit.retrieve_pull_requests of <camel.toolkits.
  github_toolkit.GithubToolkit object at 0x105286560>>
========================================
===============================================================================
"""
