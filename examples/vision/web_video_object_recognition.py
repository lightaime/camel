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

from camel.agents import ChatAgent
from camel.generators import PromptTemplateGenerator
from camel.messages import BaseMessage
from camel.messages.acl_parameter import Content
from camel.toolkits.video_toolkit import VideoDownloaderToolkit
from camel.types import (
    RoleType,
    TaskType,
)

parser = argparse.ArgumentParser(description="Arguments for object detection.")
parser.add_argument(
    "--video_url",
    type=str,
    help="URL of the video for screenshot extraction.",
    required=True,
)
parser.add_argument(
    "--timestamps",
    type=int,
    help="Number of screenshots to capture.",
    default=3,
)


def detect_image_obj(image_list) -> None:
    sys_msg = PromptTemplateGenerator().get_prompt_from_key(
        TaskType.OBJECT_RECOGNITION, RoleType.ASSISTANT
    )
    print("=" * 20 + " SYS MSG " + "=" * 20)
    print(sys_msg)
    print("=" * 49)

    agent = ChatAgent(sys_msg)
    content = Content(
        text="Please start the object detection for the \
                      following images!",
        image_list=image_list,
        image_detail="high",
    )
    user_msg = BaseMessage.make_user_message(
        role_name="User",
        content=content,
    )
    assistant_response = agent.step(user_msg)
    print("=" * 20 + " RESULT " + "=" * 20)
    print(assistant_response.msgs[0].content)
    print("=" * 48)


def main() -> None:
    video_url = 'https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4'
    downloader = VideoDownloaderToolkit()

    image_list = downloader.get_video_screenshots(video_url, 3)

    detect_image_obj(image_list)


if __name__ == "__main__":
    main()
"""
===============================================================================
==================== SYS MSG ====================
You have been assigned an object recognition task.
Your mission is to list all detected objects in following image.
Your output should always be a list of strings starting with `1.`, `2.` etc.
Do not explain yourself or output anything else.
=================================================
==================== RESULT ====================
1. Rabbit
2. Grass
3. Rocks
4. Tree roots
5. Background trees
6. Hill
7. Sky
8. Stone structure
================================================
===============================================================================
"""
