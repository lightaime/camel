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
from camel.models import ModelFactory
from camel.toolkits.function_tool import FunctionTool
from camel.toolkits.web_toolkit import WebToolkit
from camel.types import ModelPlatformType, ModelType

# Initialize the toolkit
toolkit = WebToolkit(headless_mode=False, debug=True)

stagehand_tool = FunctionTool(toolkit.stagehand_tool)
web_text_and_image_tool = FunctionTool(toolkit.stagehand_extract_text_images)
screenshot_tool = FunctionTool(
    toolkit.stagehand_screenshot_and_analyze_with_gpt4o
)

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI, model_type=ModelType.GPT_4O_MINI
)

assistant_sys_msg = """You are a helpful assistant capable of performing 
web interactions and answering questions with 
real-time data. Use the available tools to automate web tasks and 
retrieve necessary information. 
Always include references to the websites you visited and provide the 
extracted text as evidence. 
Verify the authenticity of the information before responding.
"""

# Create a ChatAgent with just this single tool, wrapped in a list
tool_agent = ChatAgent(
    assistant_sys_msg,
    model=model,
    tools=[stagehand_tool, web_text_and_image_tool, screenshot_tool],
)

# Interact with the agent
prompt = "What is the most visited website in the world?"
response = tool_agent.step(prompt)
print(response.msgs[0].content)
