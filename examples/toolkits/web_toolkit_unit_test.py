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
# WebToolkit Unit Test

import pytest

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import WebToolkit
from camel.types import ModelPlatformType, ModelType


@pytest.fixture
def web_agent():
    toolkit = WebToolkit()

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )
    assistant_sys_msg = """You are a helpful 
    assistant capable of performing web 
    interactions and answering questions."""

    agent = ChatAgent(
        assistant_sys_msg,
        model=model,
        tools=toolkit.get_tools(),
    )

    return agent


# The first two testcases are relatively more important
def test_1(web_agent):
    task_prompt = r"""According to Girls Who Code, how long did it take in 
    years for the percentage of computer scientists that 
    were women to change by 13% from a starting point of 37%? 
    Please reply with the number. 
    Here are the reference website: `https://girlswhocode.com/about-us`"""
    resp = web_agent.step(task_prompt)
    answer = resp.msgs[0].content

    # The final answer format is for reference only
    assert '22' in answer


def test_2(web_agent):
    task_prompt = r"""In Audre Lorde's poem “Father Son and Holy Ghost”, 
    what is the number of the stanza in which some lines are indented? 
    Please reply with the number. Here are the reference website: 
    `https://www.poetryfoundation.org/poems/46462/father-son-and-holy-ghost`"""
    resp = web_agent.step(task_prompt)
    answer = resp.msgs[0].content

    assert '2' in answer or 'two' in answer.lower()


def test_3(web_agent):
    task_prompt = r"""On the DeepFruits fruit detection graph on 
    Connected Papers from 2016, what feature caused 
    the largest bubble to be the size it is?"""
    resp = web_agent.step(task_prompt)
    answer = resp.msgs[0].content

    assert "citations" in answer.lower()
