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
from camel.types import ModelPlatformType

model = ModelFactory.create(
    model_platform=ModelPlatformType.AWS_BEDROCK,
    model_type="meta.llama3-70b-instruct-v1:0",
)

camel_agent = ChatAgent(model=model)

user_msg = """Say hi to CAMEL AI, one open-source community dedicated to the 
    study of autonomous and communicative agents."""

response = camel_agent.step(user_msg)
print(response.msgs[0].content)
'''
===============================================================================
Hi CAMEL AI community! It's great to see a dedicated group of individuals 
passionate about the study of autonomous and communicative agents. Your 
open-source community is a fantastic platform for collaboration, knowledge 
sharing, and innovation in this exciting field. I'm happy to interact with you 
and provide assistance on any topics related to autonomous agents, natural 
language processing, or artificial intelligence in general. Feel free to ask 
me any questions, share your projects, or discuss the latest advancements in 
the field. Let's explore the possibilities of autonomous and communicative 
agents together!
===============================================================================
'''
