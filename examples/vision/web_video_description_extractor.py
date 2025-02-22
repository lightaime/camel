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
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.messages.acl_parameter import Content
from camel.prompts import PromptTemplateGenerator
from camel.toolkits import VideoDownloaderToolkit
from camel.types import RoleType, TaskType

video_url = (
    "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4"
)
downloader = VideoDownloaderToolkit()

# Get the video bytes
video_bytes = downloader.get_video_bytes(video_url)

sys_msg = PromptTemplateGenerator().get_prompt_from_key(
    TaskType.VIDEO_DESCRIPTION, RoleType.ASSISTANT
)

camel_agent = ChatAgent(sys_msg)

content = Content(
    text="These are frames from a video that I want to upload. Generate a \
        compelling description that I can upload along with the video.",
    video_bytes=video_bytes,
)

# Create user message with video bytes
user_msg = BaseMessage.make_user_message(
    role_name="User",
    content=content,
)

# Get response information
response = camel_agent.step(user_msg)
print(response.msgs[0].content)
"""
===============================================================================
Join the delightful adventure of a lovable, chubby bunny as he emerges from
 his cozy burrow to greet the day! Watch as he stretches and yawns, ready to
explore the vibrant, lush world around him. This heartwarming and beautifully 
animated scene is sure to bring a smile to your face and brighten your day. 
Don't miss out on this charming moment of pure joy and wonder! 🌿🐰✨
===============================================================================
"""
