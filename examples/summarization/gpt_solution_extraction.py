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
import argparse
import concurrent.futures
import itertools
import json
import os
import random
from typing import Dict, Tuple

import numpy as np

from camel.agents import ChatAgent
from camel.prompts import SolutionExtractionPromptTemplateDict
from camel.types import RoleType

parser = argparse.ArgumentParser(
    description='Arguments for conversation summarization.'
)
parser.add_argument(
    '--json_dir',
    type=str,
    help='Directory containing original json files',
    default='../camel/camel_data/ai_society',
)
parser.add_argument(
    '--solution_dir',
    type=str,
    help='Directory for solution json files',
    default='../camel/camel_data/ai_society_solution_extraction',
)
parser.add_argument(
    '--seed', type=int, help='Seed for reproducibility', default=10
)


def flatten_conversation(conversation: Dict) -> str:
    r"""Format a conversation into a string.

    Args:
        conversation (Dict): A dictionary containing
            information about the conversation.

    Returns:
        str: A string containing the specified task and
            all messages in the conversation.

    Raises:
        ValueError: If an unknown role name is encountered
            in the conversation.

    The conversation is formatted in the following format:
    Task: <specified_task>
    User (<role_1>): <message_1>
    Assistant (<role_2>): <message_2>
    ...

    Example:
        >>> conversation = {
        ...     'num_messages': 2,
        ...     'message_1': {'role_name': 'Engineer', 'content': 'Hello'},
        ...     'message_2': {'role_name': 'Programmer',
                              'content': 'Hi there!'},

        ...     'specified_task': 'Answer a greeting'
        ... }
        >>> flatten_conversation(conversation)
        'Task: Answer a greeting
            User (Engineer): Hello
            Assistant (Programmer): Hi there!'

    """

    num_messages = conversation['num_messages']
    assert num_messages >= 2
    role_1 = conversation['message_1']['role_name']
    role_2 = conversation['message_2']['role_name']
    task = conversation['specified_task']

    messages = []
    for i in range(1, num_messages + 1):
        if conversation[f'message_{i}']['role_name'] == role_1:
            message = (
                f"User ({role_1}): " + conversation[f'message_{i}']['content']
            )
        elif conversation[f'message_{i}']['role_name'] == role_2:
            message = (
                f"Assistant ({role_2}): "
                + conversation[f'message_{i}']['content']
            )
        else:
            raise ValueError(
                "Unknown role name: "
                f"{conversation[f'message_{i}']['role_name']}"
            )
        messages.append(message)

    joined_messages = '\n'.join(messages)
    formatted_data = f"Task: {task}\n{joined_messages}"

    return formatted_data


def format_combination(combination: Tuple[int, int, int]):
    assistant_role, user_role, task = combination
    assistant_role_str = str(assistant_role).zfill(3)
    user_role_str = str(user_role).zfill(3)
    task_str = str(task).zfill(3)
    return f"{assistant_role_str}_{user_role_str}_{task_str}"


def solution_extraction(
    conversation: Dict,
    flattened_conversation: str,
    file_name: str,
    args: argparse.Namespace,
) -> None:
    solution_extraction_template = SolutionExtractionPromptTemplateDict()
    assistant_sys_msg_prompt = solution_extraction_template[RoleType.ASSISTANT]

    # We use GPT4 because it has a longer context length
    agent = ChatAgent(assistant_sys_msg_prompt)
    agent.reset()

    prompt = "Here is the conversation:" + flattened_conversation

    assistant_response = agent.step(prompt)
    print(assistant_response.msg.content)

    # Create folder to write solution_extraction to
    if not os.path.exists(args.solution_dir):
        os.makedirs(args.solution_dir)

    # Append to the original JSON conversation file
    conversation['solution_extraction'] = assistant_response.msg.content.text

    # Save new dictionary as JSON file
    save_path = os.path.join(args.solution_dir, f'{file_name}.json')
    with open(save_path, "w") as f:
        json.dump(conversation, f)


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    total_num_assistant_roles = 50
    total_num_user_roles = 50
    total_num_tasks = 1

    subsample_num_assistant_roles = 10
    subsample_num_user_roles = 10
    subsample_num_tasks = 1

    # Randomly subsample `subsample_num_assistant_roles`
    # of the total assistant roles
    subsampled_assistant_roles = random.sample(
        range(1, total_num_assistant_roles + 1), subsample_num_assistant_roles
    )

    # Randomly subsample `subsample_num_user_roles` of the total user roles
    subsampled_user_roles = random.sample(
        range(1, total_num_user_roles + 1), subsample_num_user_roles
    )

    # Randomly subsample `subsample_num_tasks` of the total tasks
    subsampled_tasks = random.sample(
        range(1, total_num_tasks + 1), subsample_num_tasks
    )

    file_names = list(
        itertools.product(
            subsampled_assistant_roles, subsampled_user_roles, subsampled_tasks
        )
    )

    # Formatting is needed to match the names of the original
    # generated JSON files xxx_xxx_xxx.json
    file_names = [
        format_combination(combination) for combination in file_names
    ]

    # Check that all files exist
    for file_name in file_names:
        json_file = os.path.join(args.json_dir, f"{file_name}.json")
        if not os.path.exists(json_file):
            raise ValueError(f"File {json_file} does not exist.")

    # Read in json files and extract solutions
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = []
        for file_name in file_names:
            json_file = os.path.join(args.json_dir, f"{file_name}.json")
            with open(json_file) as f:
                conversation = json.load(f)
            flattened_conversation = flatten_conversation(conversation)
            futures.append(
                executor.submit(
                    solution_extraction,
                    conversation,
                    flattened_conversation,
                    file_name,
                    args,
                )
            )

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Exception: {e}")


if __name__ == "__main__":
    main()
