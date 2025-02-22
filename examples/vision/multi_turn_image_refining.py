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
import re
from copy import deepcopy

from colorama import Fore
from PIL import Image

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.prompts import PromptTemplateGenerator
from camel.responses import ChatAgentResponse
from camel.toolkits import DalleToolkit
from camel.types import (
    ModelPlatformType,
    ModelType,
    RoleType,
    TaskType,
)
from camel.utils import print_text_animated


class MMChat:
    r"""The class of multimodal chat session.

    NOTE: Currently this example doesn't work properly, since the generated
    image is not included in the response message. Need to add support to
    include Image in response message.
    """

    def __init__(
        self,
    ) -> None:
        self.critic = None
        self.artist = None
        critic_sys = """You need to describe what you see in the figure
and improve the prompt of it.
Reply with the following format:

CRITICS: the image needs to improve...
PROMPT: here is the updated prompt!
        """
        self.critic_sys_msg = BaseMessage.make_assistant_message(
            role_name='Critic', content=critic_sys
        )

        self.artist_sys_msg = BaseMessage.make_assistant_message(
            role_name="Artist",
            content=PromptTemplateGenerator().get_prompt_from_key(
                TaskType.MULTI_CONDITION_IMAGE_CRAFT, RoleType.ASSISTANT
            ),
        )

        self.init_agents()

    def init_agents(self):
        r"""Initialize artist and critic agents with their system messages."""
        model = ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT,
            model_type=ModelType.DEFAULT,
        )

        self.artist = ChatAgent(
            system_message=self.artist_sys_msg,
            model=model,
            tools=DalleToolkit().get_tools(),
        )

        self.artist.reset()

        self.critic = ChatAgent(
            system_message=self.critic_sys_msg, model=model
        )
        self.critic.reset()

    def step(self, initialPrompt: str, iter_num=2) -> ChatAgentResponse:
        r"""Process of the drawing and criticising.

        Returns:
            ChatAgentResponse: it contains the response message of
            the artist agent in the last iteration.

        """

        artist_user_msg = BaseMessage.make_user_message(
            role_name="User", content=initialPrompt
        )
        print(
            Fore.MAGENTA
            + "=" * 10
            + "ARTIST SYS"
            + "=" * 10
            + "\n"
            + self.artist_sys_msg.content
        )
        print(
            Fore.YELLOW
            + "=" * 10
            + "ARTIST USR"
            + "=" * 10
            + "\n"
            + artist_user_msg.content
        )

        pattern = r'''\(.*?/([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-
        [0-9a-fA-F]{4}-[0-9a-fA-F]{12})(\.jpg|\.png)\)'''
        response = self.artist.step(artist_user_msg)
        matches = re.findall(pattern, response.msg.content.text)

        image_paths = [f"./img/{uuid}{ext}" for uuid, ext in matches]
        tmp_paths = deepcopy(image_paths)
        response_msg = re.sub(
            pattern,
            lambda x: "(" + image_paths.pop(0) + ")",
            response.msg.content.text,
        )
        image_paths = deepcopy(tmp_paths)

        print_text_animated(
            Fore.BLUE
            + "=" * 10
            + "ARTIST RES"
            + "=" * 10
            + "\n"
            + response_msg
        )
        print(response_msg)

        i = 0
        while i < iter_num:
            i += 1
            # Resize the image to 128x128
            resized_imgs = [
                Image.open(image_path).resize(
                    (128, 128), Image.Resampling.LANCZOS
                )
                for image_path in image_paths
            ]
            # Save for maintaining the image format
            [
                img.save(f"tmp_{i}.png", "PNG")
                for i, img in enumerate(resized_imgs)
            ]
            saved = [f"tmp_{i}.png" for i in range(len(resized_imgs))]
            image_list = [Image.open(image) for image in saved]

            critic_user_msg = BaseMessage.make_user_message(
                role_name="User",
                content="image:",
                image_list=image_list,
                image_detail="low",
            )
            print(
                Fore.GREEN
                + "=" * 10
                + "CRITIC SYS"
                + "=" * 10
                + "\n"
                + self.critic_sys_msg.content.text
            )
            print(
                Fore.RED
                + "=" * 10
                + "CRITIC USR"
                + "=" * 10
                + "\n"
                + critic_user_msg.content.text
            )
            prompt = self.critic.step(critic_user_msg).msg.content.text
            print_text_animated(
                Fore.CYAN
                + "=" * 10
                + "CRITIC RES"
                + "=" * 10
                + "\n"
                + prompt
                + Fore.RESET
            )

            artist_user_msg = BaseMessage.make_user_message(
                role_name="User",
                content='''Please generate a image based on
                the following prompt: \n'''
                + prompt,
            )
            response = self.artist.step(artist_user_msg)

            matches = re.findall(pattern, response.msg.content.text)
            image_paths = [f"./img/{uuid}{ext}" for uuid, ext in matches]
            tmp_paths = deepcopy(image_paths)
            response_msg = re.sub(
                pattern,
                lambda x, image_paths=image_paths: "("
                + image_paths.pop(0)
                + ")",
                response.msg.content.text,
            )
            image_paths = deepcopy(tmp_paths)
            print_text_animated(
                Fore.BLUE
                + "=" * 10
                + "ARTIST RES"
                + "=" * 10
                + "\n"
                + response_msg
            )
            print(response_msg)

        return response


if __name__ == "__main__":
    session = MMChat()
    res = session.step(
        initialPrompt='''Create an image with pink background,
        a dog is showing a sign with 'I Love Camel'.''',
        iter_num=1,
    )
