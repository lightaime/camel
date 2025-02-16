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
from camel.agents import ReActAgent
from camel.messages import BaseMessage
from camel.toolkits import SearchToolkit
from camel.types import RoleType


def main() -> None:
    # Create system message and initialize agent
    system_message = BaseMessage(
        role_name="Assistant",
        role_type=RoleType.ASSISTANT,
        meta_dict={},
        content=(
            "You are a helpful assistant that can search information online. "
            "Use the search tool to find accurate information and "
            "provide detailed answers. "
        ),
    )

    # Initialize toolkit and agent with search_duckduckgo tool
    search_tool = SearchToolkit().search_duckduckgo
    agent = ReActAgent(
        system_message=system_message,
        tools=[search_tool],
        max_steps=5,
    )

    # Example queries that require search
    queries = [
        "What is the current population of Paris?",
        "Who won the Nobel Prize in Physics in 2024?",
    ]

    # Process each query
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        response = agent.step(query)
        print(f"Response:\n{response.info}")
        print("-" * 50)


if __name__ == "__main__":
    main()

"""
Example output:

Query: What is the current population of Paris?
--------------------------------------------------
Response:
{
    "thought": "I found several sources that provide population estimates for 
                Paris. The most relevant ones indicate that the population of 
                Paris in 2023 is approximately 2.1 million for the city proper,
                while the metropolitan area is around 11.2 million. I need to 
                summarize this information clearly.",
    "action": "Finish",
    "observation": "Task completed."
}
--------------------------------------------------

Query: Who won the Nobel Prize in Physics in 2024? 
--------------------------------------------------
Response:
{
    "thought": "I found multiple sources confirming the winners of the 2024 
                Nobel Prize in Physics. The prize was awarded to John J. 
                Hopfield and Geoffrey E. Hinton for their contributions 
                to machine learning and artificial neural networks. 
                I will summarize this information.",
    "action": "Finish", 
    "observation": "Task completed."
}
--------------------------------------------------
"""
