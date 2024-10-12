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
import os
from typing import Any, Dict, List

from camel.toolkits.base import BaseToolkit
from camel.toolkits.openai_function import OpenAIFunction
from camel.utils.commons import export_to_toolkit


@export_to_toolkit
def search_wiki(entity: str) -> str:
    r"""Search the entity in WikiPedia and return the summary of the
        required page, containing factual information about
        the given entity.

    Args:
        entity (str): The entity to be searched.

    Returns:
        str: The search result. If the page corresponding to the entity
            exists, return the summary of this entity in a string.
    """
    try:
        import wikipedia
    except ImportError:
        raise ImportError(
            "Please install `wikipedia` first. You can install it "
            "by running `pip install wikipedia`."
        )

    result: str

    try:
        result = wikipedia.summary(entity, sentences=5, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as e:
        result = wikipedia.summary(
            e.options[0], sentences=5, auto_suggest=False
        )
    except wikipedia.exceptions.PageError:
        result = (
            "There is no page in Wikipedia corresponding to entity "
            f"{entity}, please specify another word to describe the"
            " entity to be searched."
        )
    except wikipedia.exceptions.WikipediaException as e:
        result = f"An exception occurred during the search: {e}"

    return result


@export_to_toolkit
def search_duckduckgo(
    query: str, source: str = "text", max_results: int = 5
) -> List[Dict[str, Any]]:
    r"""Use DuckDuckGo search engine to search information for
    the given query.

    This function queries the DuckDuckGo API for related topics to
    the given search term. The results are formatted into a list of
    dictionaries, each representing a search result.

    Args:
        query (str): The query to be searched.
        source (str): The type of information to query (e.g., "text",
            "images", "videos"). Defaults to "text".
        max_results (int): Max number of results, defaults to `5`.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary
            represents a search result.
    """
    from duckduckgo_search import DDGS
    from requests.exceptions import RequestException

    ddgs = DDGS()
    responses: List[Dict[str, Any]] = []

    if source == "text":
        try:
            results = ddgs.text(keywords=query, max_results=max_results)
        except RequestException as e:
            # Handle specific exceptions or general request exceptions
            responses.append({"error": f"duckduckgo search failed.{e}"})

        # Iterate over results found
        for i, result in enumerate(results, start=1):
            # Creating a response object with a similar structure
            response = {
                "result_id": i,
                "title": result["title"],
                "description": result["body"],
                "url": result["href"],
            }
            responses.append(response)

    elif source == "images":
        try:
            results = ddgs.images(keywords=query, max_results=max_results)
        except RequestException as e:
            # Handle specific exceptions or general request exceptions
            responses.append({"error": f"duckduckgo search failed.{e}"})

        # Iterate over results found
        for i, result in enumerate(results, start=1):
            # Creating a response object with a similar structure
            response = {
                "result_id": i,
                "title": result["title"],
                "image": result["image"],
                "url": result["url"],
                "source": result["source"],
            }
            responses.append(response)

    elif source == "videos":
        try:
            results = ddgs.videos(keywords=query, max_results=max_results)
        except RequestException as e:
            # Handle specific exceptions or general request exceptions
            responses.append({"error": f"duckduckgo search failed.{e}"})

        # Iterate over results found
        for i, result in enumerate(results, start=1):
            # Creating a response object with a similar structure
            response = {
                "result_id": i,
                "title": result["title"],
                "description": result["description"],
                "embed_url": result["embed_url"],
                "publisher": result["publisher"],
                "duration": result["duration"],
                "published": result["published"],
            }
            responses.append(response)

    # If no answer found, return an empty list
    return responses


@export_to_toolkit
def search_google(
    query: str, num_result_pages: int = 5
) -> List[Dict[str, Any]]:
    r"""Use Google search engine to search information for the given query.

    Args:
        query (str): The query to be searched.
        num_result_pages (int): The number of result pages to retrieve.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary
        represents a website.
            Each dictionary contains the following keys:
            - 'result_id': A number in order.
            - 'title': The title of the website.
            - 'description': A brief description of the website.
            - 'long_description': More detail of the website.
            - 'url': The URL of the website.

            Example:
            {
                'result_id': 1,
                'title': 'OpenAI',
                'description': 'An organization focused on ensuring that
                artificial general intelligence benefits all of humanity.',
                'long_description': 'OpenAI is a non-profit artificial
                intelligence research company. Our goal is to advance
                digital intelligence in the way that is most likely to
                benefit humanity as a whole',
                'url': 'https://www.openai.com'
            }
        title, description, url of a website.
    """
    import requests

    # https://developers.google.com/custom-search/v1/overview
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    # https://cse.google.com/cse/all
    SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

    # Using the first page
    start_page_idx = 1
    # Different language may get different result
    search_language = "en"
    # How many pages to return
    num_result_pages = num_result_pages
    # Constructing the URL
    # Doc: https://developers.google.com/custom-search/v1/using_rest
    url = (
        f"https://www.googleapis.com/customsearch/v1?"
        f"key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&start="
        f"{start_page_idx}&lr={search_language}&num={num_result_pages}"
    )

    responses = []
    # Fetch the results given the URL
    try:
        # Make the get
        result = requests.get(url)
        data = result.json()

        # Get the result items
        if "items" in data:
            search_items = data.get("items")

            # Iterate over 10 results found
            for i, search_item in enumerate(search_items, start=1):
                if "og:description" in search_item["pagemap"]["metatags"][0]:
                    long_description = search_item["pagemap"]["metatags"][0][
                        "og:description"
                    ]
                else:
                    long_description = "N/A"
                # Get the page title
                title = search_item.get("title")
                # Page snippet
                snippet = search_item.get("snippet")

                # Extract the page url
                link = search_item.get("link")
                response = {
                    "result_id": i,
                    "title": title,
                    "description": snippet,
                    "long_description": long_description,
                    "url": link,
                }
                responses.append(response)
        else:
            responses.append({"error": "google search failed."})

    except requests.RequestException:
        # Handle specific exceptions or general request exceptions
        responses.append({"error": "google search failed."})
    # If no answer found, return an empty list
    return responses


@export_to_toolkit
def query_wolfram_alpha(query: str, is_detailed: bool) -> str:
    r"""Queries Wolfram|Alpha and returns the result. Wolfram|Alpha is an
    answer engine developed by Wolfram Research. It is offered as an online
    service that answers factual queries by computing answers from
    externally sourced data.

    Args:
        query (str): The query to send to Wolfram Alpha.
        is_detailed (bool): Whether to include additional details in the
            result.

    Returns:
        str: The result from Wolfram Alpha, formatted as a string.
    """
    try:
        import wolframalpha
    except ImportError:
        raise ImportError(
            "Please install `wolframalpha` first. You can install it by"
            " running `pip install wolframalpha`."
        )

    WOLFRAMALPHA_APP_ID = os.environ.get('WOLFRAMALPHA_APP_ID')
    if not WOLFRAMALPHA_APP_ID:
        raise ValueError(
            "`WOLFRAMALPHA_APP_ID` not found in environment "
            "variables. Get `WOLFRAMALPHA_APP_ID` here: "
            "`https://products.wolframalpha.com/api/`."
        )

    try:
        client = wolframalpha.Client(WOLFRAMALPHA_APP_ID)
        res = client.query(query)
        assumption = next(res.pods).text or "No assumption made."
        answer = next(res.results).text or "No answer found."
    except Exception as e:
        if isinstance(e, StopIteration):
            return "Wolfram Alpha wasn't able to answer it"
        else:
            error_message = f"Wolfram Alpha wasn't able to answer it" f"{e!s}."
            return error_message

    result = f"Assumption:\n{assumption}\n\nAnswer:\n{answer}"

    # Add additional details in the result
    if is_detailed:
        result += '\n'
        for pod in res.pods:
            result += '\n' + pod['@title'] + ':\n'
            for sub in pod.subpods:
                result += (sub.plaintext or "None") + '\n'

    return result.rstrip()  # Remove trailing whitespace


SEARCH_FUNCS = [
    OpenAIFunction(search_wiki),
    OpenAIFunction(search_duckduckgo),
    OpenAIFunction(search_google),
    OpenAIFunction(query_wolfram_alpha),
]


class SearchToolkit(BaseToolkit):
    r"""A class representing a toolkit for web search.

    This class provides methods for searching information on the web using
    search engines like Google, DuckDuckGo, Wikipedia and Wolfram Alpha.
    """

    def get_tools(self) -> List[OpenAIFunction]:
        r"""Returns a list of OpenAIFunction objects representing the
        functions in the toolkit.

        Returns:
            List[OpenAIFunction]: A list of OpenAIFunction objects
                representing the functions in the toolkit.
        """
        return SEARCH_FUNCS
