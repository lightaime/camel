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
from PIL import Image

from camel.agents.chat_agent import ChatAgent
from camel.generators import PromptTemplateGenerator
from camel.messages.acl_parameter import Content
from camel.messages.base import BaseMessage
from camel.models import ModelFactory
from camel.toolkits import DalleToolkit
from camel.types import (
    ModelPlatformType,
    ModelType,
    RoleType,
    TaskType,
)


def main(image_paths: list[str]) -> list[str]:
    sys_msg = PromptTemplateGenerator().get_prompt_from_key(
        TaskType.MULTI_CONDITION_IMAGE_CRAFT, RoleType.ASSISTANT
    )
    print("=" * 20 + " SYS MSG " + "=" * 20)
    print(sys_msg)
    print("=" * 49)

    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    )

    dalle_agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        tools=DalleToolkit().get_tools(),
    )

    image_list = [Image.open(image_path) for image_path in image_paths]

    content = Content(
        text="Please generate an image based on the provided images and \
            text, make the backgroup of this image is in the morning'",
        image_list=image_list,
    )

    user_msg = BaseMessage.make_user_message(
        role_name="User",
        content=content,
    )

    response = dalle_agent.step(user_msg)

    print("=" * 20 + " RESULT " + "=" * 20)
    print(response.msg.content)
    print("=" * 48)


if __name__ == "__main__":
    main()

"""
===============================================================================
==================== SYS MSG ====================
You are tasked with creating an image based on
        the provided text and images conditions. Please use your
        imagination and artistic capabilities to visualize and
        draw the images and explain what you are thinking about.
=================================================
==================== RESULT ====================
Here is the generated image of a serene desert scene in the morning:

![Morning Desert Scene](img/3d8310e8-9f14-48be-94db-c66dd0461cd0.png)

The scene features a camel standing on a sand dune, palm trees, and an oasis 
in the background. The sun is rising, casting a soft golden light over the 
landscape with clear skies and a few scattered clouds.
===============================================================================
"""
