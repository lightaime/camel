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

import asyncio
import os
import re

from camel.agents import ChatAgent
from camel.bots.slack.slack_app import SlackApp
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

slack_bot = SlackApp(
    token=os.getenv("SLACK_TOKEN"),
    app_token=os.getenv("SLACK_APP_TOKEN"),
    socket_mode=True,
)

o1_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
    model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
)
agent = ChatAgent(
    system_message="you are a helpful assistant",
    message_window_size=10,
    model=o1_model,
)


def custom_handler(profile, event) -> str:
    message = profile.text
    print(message)
    message = re.sub(r'<@[\w\d]+>', '', message).strip()
    print(message)
    response = agent.step(message)
    return response.msg.content


slack_bot.set_custom_handler(custom_handler)


async def main():
    try:
        await slack_bot.start()
    except KeyboardInterrupt:
        print("Received Ctrl+C, stopping the bot...")
        await slack_bot.stop()
    finally:
        await slack_bot.stop()


asyncio.run(main())
