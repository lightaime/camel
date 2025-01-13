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
from pydantic import BaseModel

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import WeatherToolkit
from camel.types import ModelPlatformType, ModelType

model = ModelFactory.create(
    model_platform=ModelPlatformType.QWEN,
    model_type=ModelType.QWEN_TURBO,
)


class ResponseFormat(BaseModel):
    celsius: str
    fahrenheit: str


agent = ChatAgent(model=model, tools=[WeatherToolkit().get_weather_data])

resp = agent.step(
    "What's the temperature in Beijing?",
    response_format=ResponseFormat,
)
print(resp.msg.content)


# resp = agent.step(
#     "Format your last response.",
#     response_format=ResponseFormat,
# )
# print(resp.msg.content)
