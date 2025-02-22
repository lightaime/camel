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
from camel.agents import ChatAgent
from camel.configs.openai_config import ChatGPTConfig
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.toolkits.search_toolkit import SearchToolkit
from camel.toolkits.weather_toolkit import WeatherToolkit
from camel.types import ModelPlatformType, ModelType

print(WeatherToolkit().list_tools())
print(SearchToolkit().list_tools())
'''
===============================================================================
['get_weather_data']
['search_wiki', 'search_duckduckgo', 'search_google', 'query_wolfram_alpha']
===============================================================================
'''

# Define system message
sys_msg = BaseMessage.make_assistant_message(
    role_name='Tools calling opertor', content='You are a helpful assistant'
)

# Get one tool from one tookit
tools = WeatherToolkit().get_a_tool('get_weather_data')

# Set model config
model_config_dict = ChatGPTConfig(
    temperature=0.0,
).as_dict()

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
    model_config_dict=model_config_dict,
)

# Set agent
camel_agent = ChatAgent(
    system_message=sys_msg,
    model=model,
    tools=tools,
)
camel_agent.reset()

# Define a user message
usr_msg = BaseMessage.make_user_message(
    role_name='CAMEL User', content='help me to know the weather in London.'
)

# Get response information
response = camel_agent.step(usr_msg)
print(response.info['tool_calls'])
"""
===============================================================================
[FunctionCallingRecord(func_name='get_weather_data', args={'city':
'London, GB', 'temp_units': 'celsius', 'wind_units': 'miles_hour',
'visibility_units': 'miles', 'time_units': 'iso'}, result='Weather
in London, GB: 9.25°Celsius, feels like 7.28°Celsius. Max temp:
10.1°Celsius, Min temp: 8.19°Celsius. Wind: 8.052984 miles_hour
at 320 degrees. Visibility: 6.21 miles. Sunrise at 2024-10-10
06:16:21+00:00, Sunset at 2024-10-10 17:18:10+00:00.')]
===============================================================================
"""
