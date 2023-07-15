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
import copy
import os
import re
from collections import deque
from typing import Any, Dict, List, Optional

import chromadb
import openai
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import pdb
from camel.agents import ChatAgent
from camel.typing import ModelType, RoleType
from camel.messages import BaseMessage

RESULTS_STORE_NAME = "example_table_5"


class BabyAGIAgent(ChatAgent):
    r"""Class for managing conversations of CAMEL BabyAGI Agents.

    Args:
        objective (string): The task objective for the babyagi agent.
        model (ModelType): The LLM model to use for generating
            responses. (default :obj:`ModelType.GPT_3_5_TURBO`)
        model_config (Any, optional): Configuration options for the LLM model.
            (default: :obj:`None`)
        message_window_size (int, optional): The maximum number of previous
            messages to include in the context window. If `None`, no windowing
            is performed. (default: :obj:`None`)
    """

    def __init__(
        self,
        system_message: BaseMessage,
        objective: str,
        model: Optional[ModelType] = None,
        model_config: Optional[Any] = None,
        message_window_size: Optional[int] = None,
        output_language: Optional[str] = None,
    ) -> None:
        super().__init__(system_message, model, model_config,
                         message_window_size, output_language)
        self.objective = objective
        self.results_storage = DefaultResultsStorage()

    def init_messages(self) -> None:
        super().init_messages()
        self.tasks_storage = SingleTaskListStorage()

    def get_ada_embedding(self, text: str):
        text = text.replace("\n", " ")
        return openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002")["data"][0]["embedding"]

    def context_agent(self, query: str, top_results_num: int) -> List[dict]:
        r"""
        Retrieves top n completed tasks as context for a given query.

        Args:
            query (str): The query or objective for retrieving context.
            top_results_num (int): The number of top results to retrieve.

        Returns:
            list: A list of tasks for the query, sorted by relevance.
        """
        results = self.results_storage.query(query=query,
                                             top_results_num=top_results_num)
        return results

    def task_creation_agent(self, objective: str, result: Dict,
                            task_description: str,
                            task_list: List[str]) -> List[dict]:
        r"""
        Create new tasks based on the objective and previous task.

        Args:
            objective (str): The objective for the AI to perform the task.
            task_description (str): The task to be executed by the AI.

        Returns:
            str: The new task list generated by the AI for the given task.
        """
        prompt = f"""
You are to use the result from an execution agent to create new tasks
with the following objective: {objective}.
The last completed task has the result: \n{result["data"]}
This result was based on this task description: {task_description}.\n"""

        if task_list:
            prompt += f"These are incomplete tasks: {', '.join(task_list)}\n"
        prompt += "Based on the result, return a list of tasks to \
be completed in order to meet the objective. "

        if task_list:
            prompt += "These new tasks must not overlap with incomplete tasks."

        prompt += """
Return one task per line in your response.
The result must be a numbered list in the format:

#. First task
#. Second task

The number of each entry must be followed by a period.
If your list is empty, write "There are no tasks to add at this time."
Unless your list is empty, do not include any headers before your numbered list
or follow your numbered list with any other output."""

        # print(f'\n*****TASK CREATION AGENT PROMPT****\n{prompt}\n')
        response = openai.Completion.create(engine="text-davinci-003",
                                            prompt=prompt, max_tokens=1000)
        # print(f'\n****TASK CREATION AGENT RESPONSE****\n{response}\n')
        new_tasks = response.choices[0].text.strip().split('\n')
        new_tasks_list = []
        for task_string in new_tasks:
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                # this may cause error as LLM may generated # instead of number
                # task_id = ''.join(s for s in task_parts[0] if s.isnumeric())
                task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
                if task_name.strip():  # and task_id.isnumeric():
                    new_tasks_list.append(task_name)
        out = [{"task_name": task_name} for task_name in new_tasks_list]
        return out

    def prioritization_agent(self) -> List[dict]:
        r"""
        Reprioritize the task list and returns numbered prioritized list.

        Returns:
            List[dict]: The new prioritized task list generated by the AI.
        """
        task_names = self.tasks_storage.get_task_names()
        bullet_string = '\n'

        prompt = f"""
You are tasked with prioritizing the following tasks:
{bullet_string + bullet_string.join(task_names)}
Consider the ultimate objective of your team: {self.objective}.
Tasks should be sorted from highest to lowest priority,
where higher-priority tasks are those
that act as pre-requisites or are more essential for meeting the objective.
Do not remove any tasks.
Return the ranked tasks as a numbered list in the format:

#. First task
#. Second task

The entries must be consecutively numbered, starting with 1.
The number of each entry must be followed by a period.
Do not include any headers before your ranked list or
follow your list with any other output."""

        # print(f'\n****TASK PRIORITIZATION AGENT PROMPT****\n{prompt}\n')
        response = response = openai.Completion.create(
            engine="text-davinci-003", prompt=prompt, max_tokens=1000)
        response = response.choices[0].text.strip()
        # print(f'\n****TASK PRIORITIZATION AGENT RESPONSE****\n{response}\n')
        if not response:
            # print("Received empty response from priotritization agent.\
            #      Keeping task list unchanged.")
            return
        new_tasks = response.split("\n") if "\n" in response else [response]
        new_tasks_list = []
        for task_string in new_tasks:
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_name = re.sub(r'[^\w\s_]+', '', task_parts[1]).strip()
                if task_name.strip():
                    new_tasks_list.append({
                        "task_name": task_name
                    })

        return new_tasks_list

    def execution_agent(self, input_message: BaseMessage) -> str:
        r"""
        Executes a task based on the given objective and previous context.

        Args:
            objective (str): The goal for the AI to perform the task.
            task (str): The task to be executed by the AI.

        Returns:
            str: The response generated by the AI for the given task.

        """
        context = self.context_agent(query=self.objective, top_results_num=5)
        if context:
            input_message.content += \
            '\n\nConsider the previously completed tasks:' + '\n'.join(
                context)
        response = super(BabyAGIAgent, self).step(input_message)
        response_content = response.msgs[0].content
        return response, response_content
    
    def step(self, input_message: BaseMessage = None) -> dict:
        r"""Performs a single step in solving one task and
        generaing new tasks and priotizing them.

        Returns:
            dict: A info dict containing the current task,
                this task solution, new created tasked and
                information about their priorities.
        """
        log_info = {}
        task_name = input_message.content
        task = {'task_name':task_name}
        log_info['current_task'] = task_name
        # Send to execution function to complete the task
        # based on the context
        response, response_content = self.execution_agent(input_message)
        log_info['task_result'] = response_content
        log_info['new_tasks'] = []
        log_info['prioritized_tasks'] = []
        # Step 2: Enrich result and store in the results storage
        # This is where you should enrich the result if needed
        # avoid system message being added to database
        if (input_message.role_type == RoleType.USER and 
            input_message.content.startswith('Instruction:')):
            enriched_result = {"data": response_content}
            result_id = f"result_{self.tasks_storage.next_task_id()}"
            self.results_storage.add(task, response_content, result_id)

            # Step 3: Create new tasks and re-prioritize task list
            # only the main instance in cooperative mode does that
            new_tasks = self.task_creation_agent(
                self.objective,
                enriched_result,
                task["task_name"],
                self.tasks_storage.get_task_names(),
            )
            # Adding new tasks to task_storage
            for new_task in new_tasks:
                self.tasks_storage.append(new_task)
                log_info['new_tasks'].append(new_task)
                
            prioritized_tasks = self.prioritization_agent()
            self.tasks_storage.replace(prioritized_tasks)
            log_info['prioritized_tasks'] = copy.deepcopy(
                prioritized_tasks)
            
            num_top_tasks = 3
            tmp = '\nThree prioritizd tasks to perform: \n'
            response.msgs[0].content += tmp
            for i in range(num_top_tasks):
                tmp = f"{prioritized_tasks[i]['task_name']}\n"
                response.msgs[0].content += tmp
        response.info['log_info'] = log_info
        return response


# Results storage using local ChromaDB
class DefaultResultsStorage:

    def __init__(self):
        # Create Chroma collection
        chroma_persist_dir = "chroma"
        chroma_client = chromadb.Client(settings=chromadb.config.Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=chroma_persist_dir,
        ))

        metric = "cosine"
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
        self.collection = chroma_client.get_or_create_collection(
            name=RESULTS_STORE_NAME,
            metadata={"hnsw:space": metric},
            embedding_function=embedding_function,
        )

    def add(self, task: Dict, result: str, result_id: str):
        # Continue with the rest of the function
        embeddings = None
        if (len(self.collection.get(ids=[result_id], include=[])["ids"]) >
                0):  # Check if the result already exists
            self.collection.update(
                ids=result_id,
                embeddings=embeddings,
                documents=result,
                metadatas={
                    "task": task["task_name"],
                    "result": result
                },
            )
        else:
            self.collection.add(
                ids=result_id,
                embeddings=embeddings,
                documents=result,
                metadatas={
                    "task": task["task_name"],
                    "result": result
                },
            )

    def query(self, query: str, top_results_num: int) -> List[dict]:
        count: int = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(query_texts=query,
                                        n_results=min(top_results_num, count),
                                        include=["metadatas"])
        return [item["task"] for item in results["metadatas"][0]]


# Task storage supporting only a single instance of BabyAGI
class SingleTaskListStorage:

    def __init__(self):
        self.tasks = deque([])
        self.task_id_counter = 0

    def append(self, task: Dict):
        self.tasks.append(task)

    def replace(self, tasks: List[Dict]):
        self.tasks = deque(tasks)

    def popleft(self):
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def next_task_id(self):
        self.task_id_counter += 1
        return self.task_id_counter

    def get_task_names(self):
        return [t["task_name"] for t in self.tasks]