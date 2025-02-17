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
import logging

import yaml
from model_utils import LM
from module import (
    Node,
    calculate_mc_score,
    perform_rollouts,
    process_annotations,
)


def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_json_file(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: A list of dictionaries containing the problem and final answer.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def setup_logging(log_file):
    """
    Set up logging configuration to output to file and console.

    Args:
        log_file (str): Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)

    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    # Direct configuration instead of loading from yaml
    config = {
        'input': {'json_file_path': 'example_problems.json'},
        'output': {
            'file_prefix': 'example',
            'log_file_path': 'example_processing.log',
        },
        'processing': {
            'initial_rollouts': 30,  # 增加初始rollouts数量
            'num_rollouts': 25,  # 增加每次迭代的rollouts数量
            'max_iterations': 150,  # 增加最大迭代次数
        },
        'model': {
            'model_type': 'camel',
            'model_name': 'deepseek-chat',
            'model_args': {
                'max_tokens': 300,  # 增加最大token数
                'temperature_range': [0.6, 0.9],  # 调整温度范围,增加多样性
            },
        },
    }

    # Set up logging
    setup_logging(config['output']['log_file_path'])
    logging.info("Starting data generation process...")

    # Load problems from JSON file
    problems = load_json_file(config['input']['json_file_path'])
    logging.info(f"Loaded {len(problems)} problems")

    # Initialize the language model
    model = LM(
        model_type=config['model']['model_type'],
        model_name=config['model']['model_name'],
        model_args=config['model']['model_args'],
    )

    # Process each problem
    for problem in problems:
        question = problem['problem']
        final_answer = problem['final_answer']

        # Create initial node
        initial_node = Node(question, "", final_answer)
        nodes = [initial_node]

        # Perform initial rollouts
        perform_rollouts(
            initial_node, model, config['processing']['initial_rollouts']
        )
        calculate_mc_score(initial_node)

        # Process annotations
        process_annotations(
            question,
            nodes,
            model,
            filename=f"{config['output']['file_prefix']}_nodes_data.json",
            max_iterations=config['processing']['max_iterations'],
        )

    logging.info("Data generation process completed")


if __name__ == "__main__":
    main()
