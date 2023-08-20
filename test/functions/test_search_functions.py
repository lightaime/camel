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
from camel.functions.search_functions import (
    clean_str,
    get_page_abstract,
    search_wiki,
    search_wiki_by_api,
)


def test_clean_str():
    input_str = "Some escaped string with unicode characters: \u2019"
    expected_output = "Some escaped string with unicode characters: ’"
    assert clean_str(input_str) == expected_output


def test_get_page_abstract():
    input_page = "\n".join([
        "This is the first sentence", "This is the second sentence",
        "This is the third sentence", "This is the fourth sentence",
        "This is the fifth sentence", "This is the sixth sentence"
    ])
    expected_output = (
        "This is the first sentence. This is the second sentence. "
        "This is the third sentence. This is the fourth sentence. "
        "This is the fifth sentence.")

    assert get_page_abstract(input_page) == expected_output


def test_search_wiki_normal():
    expected_output = (
        "Erygia sigillata is a species of moth in the family Erebidae found "
        "in Himachal Pradesh, Northern India.[2] The moth was officially "
        "recognized and classified in 1892. This Erebinae-related article "
        "is a stub. You can help Wikipedia by expanding it. Main "
        "pageContentsCurrent eventsRandom articleAbout WikipediaContact "
        "usDonate. HelpLearn to editCommunity portalRecent "
        "changesUpload file.")

    # Test that `search_wiki` returns the expected output
    assert search_wiki("Erygia sigillata") == expected_output


def test_search_wiki_not_found():
    search_output = search_output = search_wiki(
        "South Africa Women Football Team")
    assert search_output.startswith(
        "Could not find South Africa Women Football Team.")


def tests_earch_wiki_by_api():
    expected_output = (
        "New York, often called New York State, is a state in the Northeastern "
        "United States.  A Mid-Atlantic state, New York borders New England, and "
        "has an international border with Canada. With almost 19.7 million residents, "
        "it is the fourth-most populous state in the United States and seventh-most "
        "densely populated as of 2022. New York is the 27th-largest U.S. state by area, "
        "with a total area of 54,556 square miles (141,300 km2).New York has a varied "
        "geography. The southeastern part of the state, known as Downstate, encompasses "
        "New York City (the most populous city in the United States), Long Island (the "
        "most populous island in the United States), and the lower Hudson Valley."
    )
    assert search_wiki_by_api("New York") == expected_output
