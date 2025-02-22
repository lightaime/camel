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
from colorama import Fore

from camel.agents import RoleAssignmentAgent
from camel.societies import RolePlaying
from camel.types import TaskType
from camel.utils import print_text_animated

AI_ASSISTANT_ROLE_INDEX = 0
AI_USER_ROLE_INDEX = 1


def main(
    model_for_role_generation=None, model=None, chat_turn_limit=50
) -> None:
    task_prompt = "Develop a trading bot for the stock market."

    role_description_agent = RoleAssignmentAgent(
        model=model_for_role_generation,
    )

    role_description_dict = role_description_agent.run(
        task_prompt=task_prompt, num_roles=2
    )

    ai_assistant_role = list(role_description_dict.keys())[
        AI_ASSISTANT_ROLE_INDEX
    ]
    ai_user_role = list(role_description_dict.keys())[AI_USER_ROLE_INDEX]
    ai_assistant_description = role_description_dict[ai_assistant_role]
    ai_user_description = role_description_dict[ai_user_role]

    sys_msg_meta_dicts = [
        dict(
            assistant_role=ai_assistant_role,
            user_role=ai_user_role,
            assistant_description=ai_assistant_description,
            user_description=ai_user_description,
        )
        for _ in range(2)
    ]

    role_play_session = RolePlaying(
        assistant_role_name=ai_assistant_role,
        user_role_name=ai_user_role,
        task_prompt=task_prompt,
        model=model,
        task_type=TaskType.ROLE_DESCRIPTION,  # Score for role description
        with_task_specify=True,
        task_specify_agent_kwargs=dict(model=model),
        extend_sys_msg_meta_dicts=sys_msg_meta_dicts,
    )

    print(
        Fore.GREEN
        + f"AI Assistant sys message:\n{role_play_session.assistant_sys_msg}\n"
    )
    print(
        Fore.BLUE + f"AI User sys message:\n{role_play_session.user_sys_msg}\n"
    )
    print(
        Fore.GREEN + f"Role description of AI Assistant:\n"
        f"{role_play_session.assistant_sys_msg.role_name}\n"
        f"{role_description_dict[ai_assistant_role]}\n"
    )
    print(
        Fore.BLUE + f"Role description of AI User:\n"
        f"{role_play_session.user_sys_msg.role_name}\n"
        f"{role_description_dict[ai_user_role]}\n"
    )

    print(Fore.YELLOW + f"Original task prompt:\n{task_prompt}\n")
    print(
        Fore.CYAN
        + "Specified task prompt:"
        + f"\n{role_play_session.specified_task_prompt}\n"
    )
    print(Fore.RED + f"Final task prompt:\n{role_play_session.task_prompt}\n")

    n = 0
    input_msg = role_play_session.init_chat()
    while n < chat_turn_limit:
        n += 1
        assistant_response, user_response = role_play_session.step(input_msg)

        if assistant_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI Assistant terminated. "
                    f"Reason: {assistant_response.info['termination_reasons']}"
                )
            )
            break
        if user_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI User terminated. "
                    f"Reason: {user_response.info['termination_reasons']}."
                )
            )
            break

        print_text_animated(
            Fore.BLUE
            + f"AI User: {ai_user_role}\n\n{user_response.msg.content}\n"
        )
        print_text_animated(
            Fore.GREEN
            + f"AI Assistant:{ai_assistant_role}\n\n"
            + f"{assistant_response.msg.content}\n"
        )

        if "CAMEL_TASK_DONE" in user_response.msg.content.text:
            break

        input_msg = assistant_response.msg


if __name__ == "__main__":
    main()
