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
import multiprocessing
import os

from camel.agents import ChatAgent
from camel.generators import CodeTaskPromptGenerator, SystemMessageGenerator
from camel.messages import BaseMessage
from camel.types import RoleType, TaskType


def generate_tasks(
    task_generator_prompt: str,
    language: str,
    domain: str,
    start_token: str = "1.",
    num_tasks: int = 10,
    model=None,
) -> None:
    sys_msg_generator = SystemMessageGenerator(task_type=TaskType.DEFAULT)
    assistant_sys_msg = sys_msg_generator.from_dict(
        dict(), role_tuple=("Task Generator", RoleType.DEFAULT)
    )
    assistant_agent = ChatAgent(assistant_sys_msg, model=model)

    user_msg = BaseMessage.make_user_message(
        role_name="Task Generator", content=task_generator_prompt
    )

    assistant_response = assistant_agent.step(user_msg)

    tasks = assistant_response.msg.content.text.split("\n")

    # Filter out the generated response to include the tasks only
    for i, task in enumerate(tasks):
        if start_token in task:
            tasks = tasks[i : i + num_tasks]
            break

    # Ensure exact number of tasks is generated
    assert str(num_tasks) in tasks[-1], print(tasks)

    with open(f"./code/tasks/{language}_{domain}.txt", "w") as file:
        file.write("\n".join(tasks))


def main(model=None) -> None:
    num_tasks = 50
    start_token = "1."

    task_generator_prompt_gen = CodeTaskPromptGenerator(
        num_tasks=num_tasks
    ).from_role_files()

    pool = multiprocessing.Pool()
    for task_generator_prompt, language, domain in task_generator_prompt_gen:
        if not os.path.exists(f"./code/tasks/{language}_{domain}.txt"):
            print(language, domain)

            pool.apply_async(
                generate_tasks,
                (
                    task_generator_prompt,
                    language,
                    domain,
                    start_token,
                    num_tasks,
                    model,
                ),
            )

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
