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
import re
from typing import Any, Dict, List

from camel.agents import ChatAgent
from camel.prompts import PromptTemplateGenerator
from camel.types import TaskType


def parse_question_string(
    question_string: str, category: str
) -> List[Dict[str, Any]]:
    pattern = r'^(\d+)\.\s+(.*?)\s*\n*$'
    questions = []
    for match in re.finditer(pattern, question_string, re.MULTILINE):
        question_id = int(match.group(1))
        text = match.group(2)
        questions.append(
            {'question_id': question_id, 'text': text, 'category': category}
        )
    return questions


def generate_questions(
    examples: str,
    category: str,
    save_file_name: str,
    key: str = 'generate_questions',
    num_questions: int = 20,
    model=None,
) -> None:
    prompt_template = PromptTemplateGenerator().get_prompt_from_key(
        TaskType.EVALUATION, key
    )

    evaluation_dict = {
        'num_questions': num_questions,
        'category': category,
        'examples': examples,
    }

    prompt = prompt_template.format(**evaluation_dict)
    print(prompt)

    agent = ChatAgent("You are a helpful assistant.", model=model)
    agent.reset()

    assistant_response = agent.step(prompt)

    if len(assistant_response.msgs) > 0:
        print(assistant_response.msg.content)

        parsed_assistant_msg = parse_question_string(
            assistant_response.msg.content.text, category
        )

        # save json file
        folder_path = './evaluation_data/questions'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(f"{folder_path}/{save_file_name}.jsonl", "w") as f:
            for item in parsed_assistant_msg:
                json.dump(item, f)
                f.write('\n')


def main(model=None) -> None:
    # generate ai society evaluation questions
    examples = (
        "1. What are the most effective ways to deal with stress?\n"
        "2. Explain the process of natural selection and how it "
        "contributes to the evolution and adaptation of species.\n"
        "3. Can you help me write a formal email to a potential"
        "business partner proposing a joint venture?"
    )

    category = 'generic task Q&A'
    save_file_name = 'questions_ai_society'
    generate_questions(examples, category, save_file_name, model=model)

    # generate coding evaluation questions
    examples = (
        "1. Develop a C++ program that reads a text file line by line and"
        "counts the number of occurrences of a specific word in the file.\n"
        "2. Implement a Jave function to find the longest common"
        " subsequence of two input strings using dynamic programming.\n"
        "3. Implement a machine learning-based chatbot system in Python."
    )

    category = 'coding task'
    save_file_name = 'questions_code'
    generate_questions(examples, category, save_file_name, model=model)

    # generate math evaluation questions
    examples = (
        "1. Given that f(x) = 5x^3 - 2x + 3, find the value of f(2).\n"
        "2. If the endpoints of a line segment are (2, -2) and (10, 4), "
        "what is the length of the segment?\n"
        "3. Solve for x in the equation 3x + 10 = 5(x - 2)."
    )

    category = 'math'
    save_file_name = 'questions_math'
    generate_questions(examples, category, save_file_name, model=model)

    # generate science evaluation questions
    examples = (
        "1. What is the probability of finding a particle with a given energy"
        " in a one-dimensional infinite square well potential when the"
        " potential width is 2 nm and the particle has a mass of 5x10^-26"
        " kg? Use the Schrödinger equation to solve for the allowed energy"
        " states and their wave functions.\n"
        "2. How does habitat loss and fragmentation affect the migratory"
        " patterns and survival of a specific species, such as the monarch"
        " butterfly, over time?\n"
        "3. What is the systematic name of the organic compound with the"
        " molecular formula C6H12O and a ketone functional group located"
        " on the second carbon atom from the left end?"
    )

    category = 'science'

    save_file_name = 'questions_science'
    generate_questions(
        examples, category, save_file_name, num_questions=60, model=model
    )


if __name__ == "__main__":
    main()
