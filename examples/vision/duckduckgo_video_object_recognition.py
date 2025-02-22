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
from typing import List

from PIL import Image

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.messages.acl_parameter import Content
from camel.prompts import PromptTemplateGenerator
from camel.toolkits import SearchToolkit, VideoDownloaderToolkit
from camel.types import RoleType, TaskType


def detect_image_obj(image_list: List[Image.Image]) -> None:
    sys_msg = PromptTemplateGenerator().get_prompt_from_key(
        TaskType.OBJECT_RECOGNITION, RoleType.ASSISTANT
    )
    print("=" * 20 + " SYS MSG " + "=" * 20)
    print(sys_msg)
    print("=" * 49)
    agent = ChatAgent(sys_msg)
    content = Content(
        text="Please start the object detection for the following images!",
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


def main():
    # Create an instance of the SearchToolkit
    search_toolkit = SearchToolkit()

    # Example query for DuckDuckGo video search
    query = "The future of AI in education"

    # Perform a DuckDuckGo search with the query, setting source to 'videos'
    results = search_toolkit.search_duckduckgo(
        query=query, source="videos", max_results=5
    )

    # Try to download videos from the search results
    for result in results:
        video_url = result['embed_url']
        if not video_url:
            print(f"No valid video URL provided for result: {result}")
            continue

        print(f"Trying to download video from: {video_url}")
        downloader = VideoDownloaderToolkit()
        image_list = downloader.get_video_screenshots(video_url, 3)
        if image_list and len(image_list) > 0:
            print(
                f'''Successfully downloaded video and captured screenshots 
                from: {video_url}'''
            )
            detect_image_obj(image_list)
            print("Stopping further downloads as we found valid images.")
            break
        else:
            print(f"Failed to capture screenshots from video: {video_url}")

    print("Exited the video download loop.")


if __name__ == "__main__":
    main()

"""
===============================================================================
Successfully downloaded video and captured screenshots 
                from: https://www.youtube.com/embed/RRMVF0PPqZI?autoplay=1
==================== SYS MSG ====================
You have been assigned an object recognition task.
Your mission is to list all detected objects in following image.
Your output should always be a list of strings starting with `1.`, `2.` etc.
Do not explain yourself or output anything else.
=================================================
==================== RESULT ====================
1. Drone
2. Hangar
3. Person (in uniform)
4. Plants
5. Wall (brick)
6. Table
7. Electrical panels
8. Lights
9. Floor
================================================
Stopping further downloads as we found valid images.
Exited the video download loop.
===============================================================================
"""
