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
import os
import random
from typing import List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType

load_dotenv()


class LM:
    def __init__(self, model_type, model_name, num_rollouts=5, **kwargs):
        self.model_type = model_type
        self.model_name = model_name
        self.num_rollouts = num_rollouts
        self.max_tokens = kwargs.get('max_tokens', 4096)
        self.temperature_range = kwargs.get('temperature_range', [0.7, 1.0])

        if self.model_type != "camel":
            raise ValueError("Only camel model type is supported")

        # Initialize camel model
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=os.environ.get("OPENAI_COMPATIBILIY_ModelType"),
            api_key=os.environ.get("OPENAI_COMPATIBILIY_API_KEY"),
            url=os.environ.get("OPENAI_COMPATIBILIY_API_BASE_URL"),
            model_config_dict={
                "temperature": random.uniform(*self.temperature_range),
                "max_tokens": self.max_tokens,
            },
        )

        # Initialize chat agent
        self.agent = ChatAgent(
            system_message='''You are a mathematical reasoning expert who
             always solves problems step by step.
For each step:
1. Write down what you're calculating
2. Show the calculation
3. Explain the result
Always show your work, even for simple calculations.
End your solution with the final numerical answer.''',
            model=self.model,
            message_window_size=10,
        )

    def generate(self, question, partial_answer, num_rollouts=None):
        results = []
        if num_rollouts is None:
            num_rollouts = self.num_rollouts

        for _ in tqdm(range(num_rollouts)):
            # Update temperature for each rollout
            self.model.model_config_dict["temperature"] = random.uniform(
                *self.temperature_range
            )

            # Construct the prompt
            if partial_answer:
                prompt = f"""Problem: {question}
Current solution steps:
{partial_answer}
Continue the solution, showing all steps and calculations. 
Make sure to explain each step:"""
            else:
                prompt = f"""Problem: {question}
Please solve this step by step, showing all calculations and
 explaining each step.
Remember to:
1. Break down the problem
2. Show all calculations
3. Explain each step
4. End with the final numerical answer."""

            # Get response from agent
            response = self.agent.step(prompt)
            results.append(response.msgs[0].content)

        return results

    def generate_rollouts(
        self, prompt: str, num_copies: Optional[int] = None
    ) -> List[str]:
        """Generate multiple rollouts for a given prompt.

        Args:
            prompt (str): The input prompt to generate responses for
            num_copies (Optional[int], optional): Number of copies to generate.
                Defaults to None.

        Returns:
            List[str]: List of generated responses
        """
        if num_copies is None:
            num_copies = 1

        rollouts = []
        for _ in range(num_copies):
            response = self.generate(prompt, "")
            if response:
                rollouts.append(response[0])

        return rollouts
