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

from camel.agents.chat_agent import ChatAgent
from camel.configs.openai_config import ChatGPTConfig
from camel.data_collector import AlpacaDataCollector
from camel.models.model_factory import ModelFactory
from camel.types.enums import ModelPlatformType, ModelType

model_config_dict = ChatGPTConfig(
    temperature=0.0,
).as_dict()

model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
    model_config_dict=model_config_dict,
)

agent = ChatAgent(
    system_message="You are a helpful assistant",
    model=model,
)

usr_msg = "When is the release date of the video game Portal?"

collector = AlpacaDataCollector().record(agent).start()

# Automatically record the message
resp = agent.step(usr_msg)

print(collector.convert())

print(collector.llm_convert())

collector.reset()

# Manually record the message
collector.step("user", "Tools calling operator", usr_msg)

collector.step(
    "assistant", "Tools calling operator", resp.msgs[0].content.text
)

print(collector.convert())

# ruff: noqa: E501
"""
{'instruction': 'You are a helpful assistantWhen is the release date of the video game Portal?', 'input': '', 'output': 'The video game "Portal" was released on October 10, 2007. It was developed by Valve Corporation and is part of the game bundle known as "The Orange Box," which also included "Half-Life 2" and its episodes.'}
2025-01-19 19:26:09,140 - httpx - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
{'instruction': 'You are a helpful assistant When is the release date of the video game Portal?', 'input': '', 'output': 'The video game "Portal" was released on October 10, 2007. It was developed by Valve Corporation and is part of the game bundle known as "The Orange Box," which also included "Half-Life 2" and its episodes.'}
{'instruction': 'You are a helpful assistantWhen is the release date of the video game Portal?', 'input': '', 'output': 'The video game "Portal" was released on October 10, 2007. It was developed by 
"""
