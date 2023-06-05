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
from camel.agents import EmbodiedAgent, HuggingFaceToolAgent
from camel.generators import SystemMessageGenerator
from camel.messages import UserChatMessage
from camel.typing import ModelType, RoleType


def main():
    # Create an embodied agent
    role_name = "Artist"
    meta_dict = dict(role=role_name, task="Drawing")
    sys_msg = SystemMessageGenerator().from_dict(
        meta_dict=meta_dict,
        role_tuple=(f"{role_name}'s Embodiment", RoleType.EMBODIMENT))
    action_space = [
        HuggingFaceToolAgent(
            'hugging_face_tool_agent',
            model=ModelType.GPT_4.value,
            remote=True,
        )
    ]
    embodied_agent = EmbodiedAgent(
        sys_msg,
        verbose=True,
        action_space=action_space,
    )
    user_msg = UserChatMessage(
        role_name=role_name,
        content=("Draw all the Camelidae species, "
                 "caption the image content, "
                 "save the images by species name."),
    )
    output_message, _, _ = embodied_agent.step(user_msg)
    print(output_message.content)


if __name__ == "__main__":
    main()
