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
import asyncio

from camel.agents.chat_agent import ChatAgent
from camel.messages.base import BaseMessage
from camel.tasks.task import Task
from camel.workforce.single_agent_workforce import SingleAgentWorforce
from camel.workforce.task_channel import TaskChannel


async def test_get_dep_tasks_info():
    public_channel = TaskChannel()
    sys_msg = BaseMessage.make_assistant_message(
        role_name="programmer",
        content="You are a python programmer.",
    )
    agent = ChatAgent(sys_msg)
    test_workforce = SingleAgentWorforce('1', 'agent1', agent, public_channel)
    human_task = Task(
        content='develop a python program of investing stock.',
        id='0',
        type='human',
    )
    subtasks = human_task.decompose(agent)
    await test_workforce._process_task(human_task, subtasks)


asyncio.run((test_get_dep_tasks_info()))
