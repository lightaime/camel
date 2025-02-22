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
import json
import multiprocessing
import os
from typing import Any, Dict

from colorama import Fore

from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.societies import RolePlaying
from camel.types import ModelPlatformType, ModelType, TaskType


def generate_data(
    assistant_idx: int,
    assistant_role_name: str,
    user_idx: int,
    user_role_name: str,
    task_idx: int,
    task_prompt: str,
    verbose: bool = False,
) -> None:
    max_num_messages = 40

    original_task_prompt = task_prompt.replace(f"{task_idx+1}. ", "")

    role_play_session = RolePlaying(
        assistant_role_name,
        user_role_name,
        task_prompt=original_task_prompt,
        with_task_specify=True,
        with_task_planner=False,
        task_type=TaskType.MISALIGNMENT,
        task_specify_agent_kwargs=dict(
            model=ModelFactory.create(
                model_platform=ModelPlatformType.DEFAULT,
                model_type=ModelType.DEFAULT,
                model_config_dict=ChatGPTConfig(temperature=1.4).as_dict(),
            )
        ),
    )

    input_msg = role_play_session.init_chat()

    if verbose:
        print(
            Fore.GREEN + "AI Assistant sys message:\n"
            f"{role_play_session.assistant_sys_msg}\n"
        )
        print(
            Fore.BLUE
            + f"AI User sys message:\n{role_play_session.user_sys_msg}\n"
        )

        print(Fore.YELLOW + f"Original task prompt:\n{task_prompt}\n")
        print(
            Fore.CYAN + "Specified task prompt:\n"
            f"{role_play_session.specified_task_prompt}\n"
        )
        print(
            Fore.RED + f"Final task prompt:\n{role_play_session.task_prompt}\n"
        )

    message_counter = 0
    message_dict: Dict[str, Any] = {}

    assistant_agent = role_play_session.assistant_agent
    user_agent = role_play_session.user_agent

    # Append roles to the dictionary
    # We start number from 1 not 0.
    message_dict["role_1"] = (
        f"{assistant_role_name}_{assistant_agent.role_type!s}"
    )
    message_dict["role_2"] = f"{user_role_name}_{user_agent.role_type!s}"
    message_dict["id"] = (
        f"{(assistant_idx+1):03}_{(user_idx+1):03}_{(task_idx+1):03}"
    )
    message_dict["original_task"] = original_task_prompt
    message_dict["specified_task"] = role_play_session.specified_task_prompt

    # Threshold to terminate the conversation if no end token appears

    repeat_word_counter = 0
    repeat_word_threshold = 4
    repeat_word_list = [
        "goodbye",
        "good bye",
        "thank",
        "bye",
        "welcome",
        "language model",
    ]

    assistant_instruct_counter = 0
    assistant_instruct_threshold = 1
    assistant_instruct_word = "Instruction:"

    user_no_instruct_counter = 0
    user_no_instruct_threshold = 3
    user_no_instruct_word = "Instruction:"

    # Set max number of messages for the chat

    while message_counter < max_num_messages:
        assistant_response, user_response = role_play_session.step(input_msg)

        # Condition 1: User terminates the chat
        if user_response.terminated:
            message_dict["termination_reason"] = (
                f"{user_agent.role_type!s}: "
                f"{user_response.info['termination_reasons'][0]}"
            )
            break

        # Condition 2: Assistant terminates the chat
        if assistant_response.terminated:
            message_dict["termination_reason"] = (
                f"{assistant_agent.role_type!s}: "
                f"{assistant_response.info['termination_reasons'][0]}"
            )
            break

        if verbose:
            print(f"User:\n{user_response.msg.content}\n")
            print(f"Assistant:\n{assistant_response.msg.content}\n")

        # Condition 3: Break if user does not give instruction
        if user_no_instruct_word not in user_response.msg.content.text:
            user_no_instruct_counter += 1
            if user_no_instruct_counter == user_no_instruct_threshold:
                message_dict['termination_reason'] = (
                    "user_no_instruct_threshold"
                )
                break
        else:
            user_no_instruct_counter = 0

        # Condition 4: Break if assistant gives instruction (flipped role)
        if assistant_instruct_word in assistant_response.msg.content.text:
            assistant_instruct_counter += 1
            if assistant_instruct_counter == assistant_instruct_threshold:
                message_dict['termination_reason'] = (
                    "assistant_instruct_threshold"
                )
                break
        else:
            assistant_instruct_counter = 0

        # Condition 5: Repeat word observed
        for repeat_word in repeat_word_list:
            if (
                repeat_word in user_response.msg.content.text.lower()
                or repeat_word in assistant_response.msg.content.text.lower()
            ):
                repeat_word_counter += 1
                if repeat_word_counter == repeat_word_threshold:
                    message_dict['termination_reason'] = (
                        "repeat_word_threshold"
                    )
                    break
            else:
                repeat_word_counter = 0

        # Save user message
        message_counter += 1
        message_dict[f"message_{message_counter}"] = (
            user_response.msg.to_dict()
        )

        # Condition 5: End token observed
        if "<CAMEL_TASK_DONE>" in user_response.msg.content.text:
            message_dict['termination_reason'] = "<CAMEL_TASK_DONE>"
            break

        # Save assistant message
        message_counter += 1
        message_dict[f"message_{message_counter}"] = (
            assistant_response.msg.to_dict()
        )

        input_msg = assistant_response.msg

    message_dict["num_messages"] = message_counter

    if message_dict["num_messages"] == max_num_messages:
        message_dict["termination_reason"] = "max_num_messages"

    with open(
        f"./camel_data/misalignment/{message_dict['id']}.json", "w"
    ) as json_file:
        json.dump(message_dict, json_file)


def main() -> None:
    # Disable/Enable Printing
    verbose = True

    # Parameters for filtering the generated task string
    start_token = "1."
    num_tasks = 10

    # We use AI Society user roles
    with open("./data/misalignment/user_roles.txt", "r") as f:
        user_roles = f.read().splitlines()

    with open("./data/misalignment/assistant_roles.txt", "r") as f:
        assistant_roles = f.read().splitlines()

    pool = multiprocessing.Pool()

    for assistant_idx, assistant_role_name in enumerate(assistant_roles):
        assistant_role_name = " ".join(assistant_role_name.split(" ")[1:])
        for user_idx, user_role_name in enumerate(user_roles):
            user_role_name = " ".join(user_role_name.split(" ")[1:])
            # Load the task list assigned for assistant and user roles
            with open(
                (
                    f"./misalignment_data/tasks/"
                    f"{assistant_role_name}_{user_role_name}.txt"
                ),
                "r",
            ) as f:
                tasks = f.read().splitlines()

                # Filter out the generated response to include the tasks only
                for i, task in enumerate(tasks):
                    if start_token in task:
                        tasks = tasks[i : i + num_tasks]
                        break

                # Ensure exact number of tasks is generated
                assert str(num_tasks) in tasks[-1], print(tasks)

            for task_idx, task_prompt in enumerate(tasks):
                id = (
                    f"{(assistant_idx+1):03}_"
                    f"{(user_idx+1):03}_{(task_idx+1):03}"
                )
                if not os.path.exists(f"./camel_data/misalignment/{id}.json"):
                    pool.apply_async(
                        generate_data,
                        (
                            assistant_idx,
                            assistant_role_name,
                            user_idx,
                            user_role_name,
                            task_idx,
                            task_prompt,
                            verbose,
                        ),
                    )

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
