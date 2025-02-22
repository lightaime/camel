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

import json
import os
import random
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from camel.agents import ChatAgent

from .filter import RougeSimilarityFilter
from .filter.instruction_filter import InstructionFilter
from .templates import SelfInstructTemplates


class SelfInstructPipeline:
    r"""A pipeline to generate and manage machine-generated instructions for
    tasks, combining human and machine task samples.

    Args:
        agent (ChatAgent): The agent used to interact and generate
            instructions.
        seed (str): The path to the human-written instructions.
        num_machine_instructions (int): Number of machine-generated
            instructions to generate. (default::obj:`5`)
        data_output_path (Optional[str]): Path to save the generated data.
            (default::obj:`./data_output.json`)
        human_to_machine_ratio (tuple): Ratio of human to machine tasks used
            for instruction generation. (default::obj:`(6, 2)`)
        instruction_filter (InstructionFilter): A filter to validate
            generated instructions. (default::obj:`None`)
        filter_config (Optional[Dict[str, Dict[str, Any]]]): configuration
            for the filter functions registered in FILE_REGISTRY.
            (default::obj:`None`)
        stop_on_first_failure (bool): If True, stops checking filters after
            the first failure.
    """

    def __init__(
        self,
        agent: ChatAgent,
        seed: str,
        num_machine_instructions: int = 5,
        data_output_path: Optional[str] = './data_output.json',
        human_to_machine_ratio: tuple = (6, 2),
        instruction_filter: Optional[InstructionFilter] = None,
        filter_config: Optional[Dict[str, Dict[str, Any]]] = None,
        stop_on_first_failure: bool = False,
    ):
        self.agent = agent
        self.num_machine_instructions = num_machine_instructions
        self.data_output_path = data_output_path
        self.human_to_machine_ratio = human_to_machine_ratio
        self.human_tasks: List[Dict] = []
        self.machine_tasks: List[Dict] = []
        self.load_seed(seed)
        default_config: Dict[str, Dict[str, Any]] = {
            "length": {},
            "keyword": {},
            "punctuation": {},
            "non_english": {},
            "rouge_similarity": {},
        }

        if instruction_filter is not None:
            # custom
            self.instruction_filter = instruction_filter
        else:
            # default
            config_to_use = (
                filter_config if filter_config is not None else default_config
            )
            self.instruction_filter = InstructionFilter(
                config_to_use, stop_on_first_failure
            )

    def load_seed(self, path: str):
        r"""Load seed tasks from a file. Defaults to a predefined seed file if
        no path is provided.

        Args:
            path (str): Path to the seed file.

        Raises:
            FileNotFoundError: If the seed file does not exist.
        """

        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.human_tasks.append(json.loads(line))
        else:
            raise FileNotFoundError(f"Seed file not found at path: {path}")

    def sample_human_tasks(self, count: int) -> List[dict]:
        r"""Sample a specified number of human tasks from the loaded seed.

        Args:
            count (int): Number of human tasks to sample.

        Returns:
            List[dict]: A list of sampled human tasks.
        """
        return random.sample(
            self.human_tasks, min(count, len(self.human_tasks))
        )

    def sample_machine_tasks(self, count: int) -> List[dict]:
        r"""Sample a specified number of machine tasks.

        Args:
            count (int): Number of machine tasks to sample.

        Returns:
            List[dict]: A list of sampled machine tasks, with placeholders if
                insufficient tasks are available.
        """
        available_machine_tasks = len(self.machine_tasks)
        if available_machine_tasks < count:
            sampled_tasks = self.machine_tasks.copy()
            placeholders_needed = count - available_machine_tasks
            sampled_tasks.extend(
                [{'instruction': ""} for _ in range(placeholders_needed)]
            )
            return sampled_tasks

        return random.sample(self.machine_tasks, count)

    def generate_machine_instruction(self) -> List:
        r"""Generate a machine instruction using the agent.

        Combines human and machine tasks based on the configured ratio to
            create a prompt for instruction generation.

        Returns:
            List: The prompt and a machine-generated instruction.
        """

        sampled_human_tasks = self.sample_human_tasks(
            self.human_to_machine_ratio[0]
        )
        sampled_machine_tasks = self.sample_machine_tasks(
            self.human_to_machine_ratio[1]
        )
        prompt = "Below are some tasks:\n\n"

        for idx, task in enumerate(sampled_human_tasks, 1):
            prompt += f"Task {idx}: {task['instruction']}\n"

        current_task_number = len(sampled_human_tasks) + 1
        for idx, task in enumerate(sampled_machine_tasks, current_task_number):
            prompt += f"Task {idx}: {task['instruction']}\n"

        task_num = len(sampled_human_tasks) + len(sampled_machine_tasks) + 1
        prompt += f"Task {task_num}:"
        prompt += (
            "\nNow, please produce exactly one new task that fits the "
            "style of the ones above.\n Do not include any task numbering or "
            "labels like 'Task X:'. Just write the task itself.\n"
            "The task should be a single sentence.\n\n"
        )

        response = self.agent.step(prompt)
        self.agent.reset()
        generated_tasks = [
            line.strip()
            for line in response.msgs[0].content.text.split("\n")
            if line.strip()
        ]
        return [prompt, generated_tasks[0]]

    def identify_instruction(self, instruction: str) -> bool:
        r"""Determine if the given instruction is a classification task.

        Args:
            instruction (str): The instruction to classify.

        Returns:
            bool: True if the instruction is a classification task,
                otherwise False.
        """
        clf_prompt = (
            SelfInstructTemplates.clf_template
            + f"Task: {instruction}\nIs it classification?"
            + "\nRespond in the following structured format:"
            "\n{\n  \"answer\": true\n}\n"
            "or\n"
            "{\n  \"answer\": false\n}\n"
        )
        response = self.agent.step(clf_prompt)
        self.agent.reset()
        try:
            structured_response = AgentResponse.parse_raw(
                response.msgs[0].content.text.strip()
            )
            return structured_response.answer
        except ValueError as e:
            print(f"Error parsing agent response: {e}")
            return False

    def generate_machine_instances(self):
        r"""Generate instances for each machine task based on its
        classification status.
        """
        for instruction in self.machine_tasks:
            instance = self.generate_machine_instance(
                instruction['instruction'], instruction['is_classification']
            )
            instruction['instances'] = instance

    def generate_machine_instance(
        self, instruction: str, classification: bool
    ) -> list[dict]:
        r"""Generate instances for a given instruction.

        Args:
            instruction (str): The instruction to create instances for.
            classification (bool): Whether the instruction is a classification
                task.

        Returns:
            List[dict]: A list of generated instances in input-output format.
        """
        if classification:
            prompt = (
                SelfInstructTemplates.output_first_template_for_clf.format(
                    instruction=instruction
                )
            )
        else:
            prompt = SelfInstructTemplates.input_first_template_for_gen.format(
                instruction=instruction
            )

        response = self.agent.step(prompt)
        generated_text = response.msgs[0].content.text.strip()

        if classification:
            return self.parse_classification_output(generated_text)
        else:
            return self.parse_non_classification_output(generated_text)

    def parse_classification_output(
        self, generated_text: str
    ) -> List[Dict[str, str]]:
        r"""Parse the generated text for classification tasks into input-output
        pairs.

        Args:
            generated_text (str): The raw text generated by the agent for
                classification tasks.

        Returns:
            List[Dict[str, str]]: A list of dictionaries with 'input' and
                'output' keys.
        """
        instances = []
        lines = generated_text.split("\n")
        current_label = None
        current_input = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Class label:"):
                if current_label and current_input:
                    instances.append(
                        {
                            "input": current_input.strip(),
                            "output": current_label.strip(),
                        }
                    )

                current_label = line[len("Class label:") :].strip()
                current_input = None
            else:
                if current_input is None:
                    current_input = line
                else:
                    current_input += f"\n{line}"
        if current_label and current_input:
            instances.append(
                {
                    "input": current_input.strip(),
                    "output": current_label.strip(),
                }
            )

        return instances

    def parse_non_classification_output(
        self, generated_text: str
    ) -> List[Dict[str, str]]:
        r"""Parse the generated text for non-classification tasks into
        input-output pairs.

        Args:
            generated_text (str): The raw text generated by the agent for
                non-classification tasks.

        Returns:
            List[Dict[str, str]]: A list of dictionaries with 'input' and
                'output' keys.
        """
        instances = []
        prev = 0
        lines = generated_text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("Example "):
                prev = i + 1

            elif line.startswith("Output:"):
                instance_input = '\n'.join(lines[prev:i]).strip()
                if instance_input.startswith("Input: "):
                    instance_input = instance_input[len("Input: ") :].strip()
                else:
                    instance_input = instance_input.strip()

                instance_output = line[len("Output:") :].strip()
                i += 1
                while i < len(lines) and not lines[i].strip().startswith(
                    "Example "
                ):
                    instance_output += '\n' + lines[i].strip()
                    i += 1
                i -= 1

                instance_output = instance_output.strip()

                instances.append(
                    {"input": instance_input, "output": instance_output}
                )

                prev = i + 1
            i += 1

        if not instances:
            instances.append({"input": "", "output": "No valid output found."})

        return instances

    def construct_data(self):
        r"""Save the machine-generated tasks to the specified output path
        in JSON format.
        """
        with open(self.data_output_path, 'w') as f:
            json.dump(self.machine_tasks, f, indent=4, ensure_ascii=False)

    def generate(self):
        r"""Execute the entire pipeline to generate machine instructions
        and instances.
        """
        while len(self.machine_tasks) < self.num_machine_instructions:
            prompt, instruction = self.generate_machine_instruction()
            existing_instructions = [
                t["instruction"] for t in self.human_tasks
            ] + [t["instruction"] for t in self.machine_tasks]
            for f in self.instruction_filter.filters:
                if isinstance(f, RougeSimilarityFilter):
                    f.existing_instructions = existing_instructions
            if self.instruction_filter.filter(prompt, instruction):
                instruction_dict = {
                    "id": f"machine_task_{len(self.machine_tasks) + 1}",
                    "instruction": instruction,
                    "is_classification": self.identify_instruction(
                        instruction
                    ),
                }
                self.machine_tasks.append(instruction_dict)
        self.generate_machine_instances()
        self.construct_data()


class AgentResponse(BaseModel):
    answer: bool = Field(
        ...,
        description="Indicates whether the task is "
        "classification (True/False).",
    )
