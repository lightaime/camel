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
import math
import os
import re

from model_utils import LM
from omegaprm_v2 import OmegaPRMV2


class Node:
    def __init__(self, question, partial_answer, correct_answer):
        self.question = question
        self.partial_answer = partial_answer
        self.correct_answer = correct_answer
        self.mc_score = None
        self.visits = 0
        self.rollouts = []
        self.visited_rollouts = []

    def add_rollout(self, result):
        self.rollouts.append(result)
        self.visited_rollouts.append(False)

    def increment_visits(self):
        self.visits += 1


# Evaluation
def check_correctness(expected_answer, generated_response):
    sentences = re.split(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s',
        generated_response.strip(),
    )
    last_sentence = sentences[-1] if sentences else ''
    return expected_answer.strip() in last_sentence.strip()


def perform_rollouts(node, model: LM, num_rollouts=None):
    correctness_flags = []
    results = model.generate(node.question, node.partial_answer, num_rollouts)
    for result in results:
        node.add_rollout(result)
        is_correct = check_correctness(node.correct_answer, result)
        correctness_flags.append(int(is_correct))
    return node.rollouts, correctness_flags


def calculate_mc_score(node):
    correct_count = sum(
        check_correctness(node.correct_answer, r) for r in node.rollouts
    )
    return correct_count / len(node.rollouts) if node.rollouts else 0


def select_best_node(nodes):
    best_node = None
    best_rollout_idx = -1
    highest_qu_value = -1
    for node in nodes:
        mc_score = (
            node.mc_score
            if node.mc_score is not None
            else calculate_mc_score(node)
        )
        if mc_score in [0, 1]:
            continue
        for idx, rollout in enumerate(node.rollouts):
            if node.visited_rollouts[idx]:
                continue
            q_val = compute_q_value(rollout, mc_score)
            u_val = compute_u_value(node, nodes)
            qu_value = q_val + u_val
            if qu_value > highest_qu_value:
                highest_qu_value = qu_value
                best_node = node
                best_rollout_idx = idx
    if best_rollout_idx != -1 and best_node is not None:
        best_node.visited_rollouts[best_rollout_idx] = True
        return (
            best_node,
            best_node.rollouts[best_rollout_idx],
            highest_qu_value,
        )
    else:
        return None, None, None


def split_text_middle(text):
    text = text.strip()
    mid_idx = len(text) // 2
    if text[mid_idx] != ' ':
        left_space = text.rfind(' ', 0, mid_idx)
        right_space = text.find(' ', mid_idx)
        if left_space == -1:
            split_idx = right_space
        elif right_space == -1:
            split_idx = left_space
        else:
            split_idx = (
                left_space
                if (mid_idx - left_space) <= (right_space - mid_idx)
                else right_space
            )
    else:
        split_idx = mid_idx
    part1 = text[:split_idx].strip()
    part2 = text[split_idx:].strip()
    return part1, part2


def locate_error(node, rollout, model):
    current_span = rollout
    previous_text = ""
    nodes_to_expand = []
    leaf_nodes = []
    while True:
        if len(current_span.split()) < 2:
            break
        left_part, right_part = split_text_middle(current_span)
        print("----")
        print(" Left:", left_part)
        print(" Right:", right_part)
        new_node = Node(
            node.question, previous_text + left_part, node.correct_answer
        )
        perform_rollouts(new_node, model)
        mc_score = calculate_mc_score(new_node)
        new_node.mc_score = mc_score
        if mc_score == 1:
            break
        elif mc_score > 0:
            current_span = right_part
            previous_text += left_part
            nodes_to_expand.append(new_node)
        else:
            current_span = left_part
            leaf_nodes.append(new_node)
    print("----")
    return nodes_to_expand, leaf_nodes


def compute_q_value(
    rollout_text, mc_score, alpha=0.5, beta=0.9, max_length=500
):
    part1 = alpha ** (1 - mc_score)
    part2 = beta ** (len(rollout_text) / max_length)
    return part1 * part2


def compute_u_value(node, all_nodes, exploration_param=0.125):
    total_visits = sum(n.visits for n in all_nodes)
    numerator = math.sqrt(total_visits)
    denominator = 1 + node.visits
    return exploration_param * (numerator / denominator)


def process_annotations(
    question, nodes, model: LM, filename='nodes_data.json', max_iterations=100
):
    """Process annotations using OmegaPRM v2."""
    # Initialize OmegaPRM v2
    omegaprm = OmegaPRMV2(
        model=model,
        c_puct=0.2,
        alpha=0.5,
        beta=0.9,
        L=500,
        k=5,
        N=max_iterations,
        rollout_budget=1000,
        save_data_tree=True,
    )

    # Process each node
    for node in nodes:
        collected_data = omegaprm.run(node.question, node.correct_answer)

        # Save collected data
        for data in collected_data:
            data_entry = {
                'question': node.question,
                'correct_answer': node.correct_answer,
                'iteration': data['iteration'],
                'total_rollouts': data['total_rollouts'],
                'tree_structure': data['tree_structure'],
            }
            append_to_json(filename, data_entry)

    return nodes


# Utils
def append_to_json(filename, data_entry):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
    else:
        data = []
    data.append(data_entry)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data appended to {filename}")
