[![Colab][colab-image]][colab-url]
[![Hugging Face][huggingface-image]][huggingface-url]
[![Slack][slack-image]][slack-url]
[![Discord][discord-image]][discord-url]
[![Wechat][wechat-image]][wechat-url]
[![Twitter][twitter-image]][twitter-url]

______________________________________________________________________

# CAMEL: Finding the Scaling Laws of Agents

[![Python Version][python-image]][python-url]
[![PyTest Status][pytest-image]][pytest-url]
[![Documentation][docs-image]][docs-url]
[![Star][star-image]][star-url]
[![Package License][package-license-image]][package-license-url]

<p align="center">
  <a href="https://github.com/camel-ai/camel#community">Community</a> |
  <a href="https://github.com/camel-ai/camel#installation">Installation</a> |
  <a href="https://camel-ai.github.io/camel/">Documentation</a> |
  <a href="https://github.com/camel-ai/camel/tree/HEAD/examples">Examples</a> |
  <a href="https://arxiv.org/abs/2303.17760">Paper</a> |
  <a href="https://github.com/camel-ai/camel#citation">Citation</a> |
  <a href="https://github.com/camel-ai/camel#contributing-to-camel-">Contributing</a> |
  <a href="https://www.camel-ai.org/">CAMEL-AI</a>
</p>

<p align="center">
  <img src='https://raw.githubusercontent.com/camel-ai/camel/master/misc/logo_light.png' width=800>
</p>


## Community
🐫 CAMEL is an open-source community dedicated to finding the scaling laws of agents. We believe that studying these agents on a large scale offers valuable insights into their behaviors, capabilities, and potential risks. To facilitate research in this field, we implement and support various types of agents, tasks, prompts, models, and simulated environments.

Join us ([*Discord*](https://discord.camel-ai.org/), [*WeChat*](https://ghli.org/camel/wechat.png) or [*Slack*](https://join.slack.com/t/camel-ai/shared_invite/zt-2g7xc41gy-_7rcrNNAArIP6sLQqldkqQ)) in pushing the boundaries of finding the scaling laws of agents.

## What Can You Build With CAMEL?

### 🤖 Customize Agents
- Customizable agents are the fundamental entities of the CAMEL architecture. CAMEL empowers you to customize agents using our modular components for specific tasks.

### ⚙️ Build Multi-Agent Systems
- We propose a multi-agent framework to address agents' autonomous cooperation challenges, guiding agents toward task completion while maintaining human intentions.

### 💻 Practical Applications
- The CAMEL framework serves as a generic infrastructure for a wide range of multi-agent applications, including task automation, data generation, and world simulations.


## Why Should You Use CAMEL?

1. Comprehensive Customization and Collaboration:

    - Integrates over 20 advanced model platforms (e.g., commercial models like OpenAI, open-source models such as Llama3, and self-deployment frameworks like Ollama).

    - Supports extensive external tools (e.g., Search, Twitter, Github, Google Maps, Reddit, Slack utilities).
    - Includes memory and prompt components for deep customization.
    - Facilitates complex multi-agent systems with advanced collaboration features.


2. User-Friendly with Transparent Internal Structure:
    - Designed for transparency and consistency in internal structure.

    - Offers comprehensive [tutorials and detailed docstrings](https://docs.camel-ai.org/) for all functions.
    - Ensures an approachable learning curve for newcomers.


## Try It Yourself
We provide a [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing) demo showcasing a conversation between two ChatGPT agents playing roles as a python programmer and a stock trader collaborating on developing a trading bot for stock market.

<p align="center">
  <img src='https://raw.githubusercontent.com/camel-ai/camel/master/misc/framework.png' width=800>
</p>

## Installation

### From PyPI

To install the base CAMEL library:
```bash
pip install camel-ai
```

> **Note**: Some features may not work without their required dependencies. Install `camel-ai[all]` to ensure all dependencies are available, or install specific extras based on the features you need.

```bash
pip install 'camel-ai[all]'  # Replace with options below
```

Available extras:
- `all`: Includes all features below
- `model_platforms`: OpenAI, Google, Mistral, Anthropic Claude, Cohere etc.
- `huggingface`: Transformers, Diffusers, Accelerate, Datasets, PyTorch etc.
- `rag`: Sentence Transformers, Qdrant, Milvus, BM25 etc.
- `storage`: Neo4j, Redis, Azure Blob, Google Cloud Storage, AWS S3  etc.
- `web_tools`: DuckDuckGo, Wikipedia, WolframAlpha, Google Maps, Weather API etc.
- `document_tools`: PDF, Word, OpenAPI, BeautifulSoup, Unstructured etc.
- `media_tools`: Image Processing, Audio Processing, YouTube Download, FFmpeg etc.
- `communication_tools`: Slack, Discord, Telegram, GitHub, Reddit, Notion etc.
- `data_tools`: Pandas, TextBlob, DataCommons, OpenBB, Stripe etc.
- `research_tools`: arXiv, Google Scholar etc.
- `dev_tools`: Docker, Jupyter, Tree-sitter, Code Interpreter etc.

Multiple extras can be combined using commas:
```bash
pip install 'camel-ai[rag,web_tools,document_tools]'  # Example: RAG system with web search and document processing
```

### From Docker

Detailed guidance can be found [here](https://github.com/camel-ai/camel/blob/master/.container/README.md)

## Quick Start

By default, the agent uses the `ModelType.DEFAULT` model from the `ModelPlatformType.DEFAULT`. You can configure the default model platform and model type using environment variables. If these are not set, the agent will fall back to the default settings:

```bash
ModelPlatformType.DEFAULT = "openai"
ModelType.DEFAULT = "gpt-4o-mini"
```

### Setting Default Model Platform and Model Type (Optional)

You can customize the default model platform and model type by setting the following environment variables:
```bash
export DEFAULT_MODEL_PLATFORM_TYPE=<your preferred platform>  # e.g., openai, anthropic, etc.
export DEFAULT_MODEL_TYPE=<your preferred model>  # e.g., gpt-3.5-turbo, gpt-4o-mini, etc.
```

### Setting Your Model API Key (Using OpenAI as an Example)

**For Bash shell (Linux, macOS, Git Bash on Windows):**

```bash
# Export your OpenAI API key
export OPENAI_API_KEY=<insert your OpenAI API key>
OPENAI_API_BASE_URL=<insert your OpenAI API BASE URL>  #(Should you utilize an OpenAI proxy service, kindly specify this)
```

**For Windows Command Prompt:**

```cmd
REM export your OpenAI API key
set OPENAI_API_KEY=<insert your OpenAI API key>
set OPENAI_API_BASE_URL=<insert your OpenAI API BASE URL>  #(Should you utilize an OpenAI proxy service, kindly specify this)
```

**For Windows PowerShell:**

```powershell
# Export your OpenAI API key
$env:OPENAI_API_KEY="<insert your OpenAI API key>"
$env:OPENAI_API_BASE_URL="<insert your OpenAI API BASE URL>"  #(Should you utilize an OpenAI proxy service, kindly specify this)
```

Replace `<insert your OpenAI API key>` with your actual OpenAI API key in each case. Make sure there are no spaces around the `=` sign.


Please note that the environment variable is session-specific. If you open a new terminal window or tab, you will need to set the API key again in that new session.

**For `.env` File:**

To simplify the process of managing API Keys, you can use store information in a `.env` file and load them into your application dynamically.

1. Modify .env file in the root directory of CAMEL and fill the following lines:

```bash
OPENAI_API_KEY=<fill your API KEY here>
```

Replace <fill your API KEY here> with your actual API key.

2. Load the .env file in your Python script: Use the load_dotenv() function from the dotenv module to load the variables from the .env file into the environment. Here's an example:

```python
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()
```
For more details about the key names in project and how to apply key, 
you can refer to [here](https://github.com/camel-ai/camel/.env).

> [!TIP]
> By default, the load_dotenv() function does not overwrite existing environment variables that are already set in your system. It only populates variables that are missing.
If you need to overwrite existing environment variables with the values from your `.env` file, use the `override=True` parameter:
> ```python
> load_dotenv(override=True)
> ```

After setting the OpenAI API key, you can run the `role_playing.py` script. Find tasks for various assistant-user roles [here](https://drive.google.com/file/d/194PPaSTBR07m-PzjS-Ty6KlPLdFIPQDd/view?usp=share_link).

```bash
# You can change the role pair and initial prompt in role_playing.py
python examples/ai_society/role_playing.py
```

Also feel free to run any scripts below that interest you:

```bash
# You can change the role pair and initial prompt in these python files

# Examples of two agents role-playing
python examples/ai_society/role_playing.py

# The agent answers questions by utilizing code execution tools.
python examples/toolkits/code_execution_toolkit.py

# Generating a knowledge graph with an agent
python examples/knowledge_graph/knowledge_graph_agent_example.py  

# Multiple agents collaborating to decompose and solve tasks
python examples/workforce/multiple_single_agents.py 

# Use agent to generate creative image
python examples/vision/image_crafting.py
```
For additional feature examples, see the [`examples`](https://github.com/camel-ai/camel/tree/master/examples) directory.

## Documentation

The [complete documentation](https://camel-ai.github.io/camel/) pages for the CAMEL package. Also, detailed tutorials for each part are provided below:

### Agents
Explore different types of agents, their roles, and their applications.

- **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)**
- **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)**
- **[Embodied Agents](https://docs.camel-ai.org/cookbooks/advanced_features/embodied_agents.html)**
- **[Critic Agents](https://docs.camel-ai.org/cookbooks/advanced_features/critic_agents_and_tree_search.html)**

---

### Key Modules
Core components and utilities to build, operate, and enhance CAMEL-AI agents and societies.

| Module | Description |
|:---|:---|
| **[Models](https://docs.camel-ai.org/key_modules/models.html)** | Model architectures and customization options for agent intelligence. |
| **[Messages](https://docs.camel-ai.org/key_modules/messages.html)** | Messaging protocols for agent communication. |
| **[Memory](https://docs.camel-ai.org/key_modules/memory.html)** | Memory storage and retrieval mechanisms. |
| **[Tools](https://docs.camel-ai.org/key_modules/tools.html)** | Tools integration for specialized agent tasks. |
| **[Prompts](https://docs.camel-ai.org/key_modules/prompts.html)** | Prompt engineering and customization. |
| **[Tasks](https://docs.camel-ai.org/key_modules/tasks.html)** | Task creation and management for agent workflows. |
| **[Loaders](https://docs.camel-ai.org/key_modules/loaders.html)** | Data loading tools for agent operation. |
| **[Storages](https://docs.camel-ai.org/key_modules/storages.html)** | Storage solutions for agent. |
| **[Society](https://docs.camel-ai.org/key_modules/society.html)** | Components for building agent societies and inter-agent collaboration. |
| **[Embeddings](https://docs.camel-ai.org/key_modules/embeddings.html)** | Embedding models for RAG. |
| **[Retrievers](https://docs.camel-ai.org/key_modules/retrievers.html)** | Retrieval methods for knowledge access. |
---

### Cookbooks
Practical guides and tutorials for implementing specific functionalities in CAMEL-AI agents and societies.

### Basic Concepts
| Cookbook | Description |
|:---|:---|
| **[Creating Your First Agent](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agent.html)** | A step-by-step guide to building your first agent. |
| **[Creating Your First Agent Society](https://docs.camel-ai.org/cookbooks/basic_concepts/create_your_first_agents_society.html)** | Learn to build a collaborative society of agents. |
| **[Message Cookbook](https://docs.camel-ai.org/cookbooks/basic_concepts/agents_message.html)** | Best practices for message handling in agents. |

### Advanced Features
| Cookbook | Description |
|:---|:---|
| **[Tools Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_tools.html)** | Integrating tools for enhanced functionality. |
| **[Memory Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_memory.html)** | Implementing memory systems in agents. |
| **[RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_rag.html)** | Recipes for Retrieval-Augmented Generation. |
| **[Graph RAG Cookbook](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_graph_rag.html)** | Leveraging knowledge graphs with RAG. |
| **[Track CAMEL Agents with AgentOps](https://docs.camel-ai.org/cookbooks/advanced_features/agents_tracking.html)** | Tools for tracking and managing agents in operations. |
| **[Agents with Human-in-loop and Tool Approval from HumanLayer](https://docs.camel-ai.org/cookbooks/advanced_features/agents_with_human_in_loop_and_tool_approval.html)** | Implementing human oversight and tool approval within agent workflows using HumanLayer. |

### Data Generation & Model Training
| Cookbook | Description |
|:---|:---|
| **[Data Generation with CAMEL and Finetuning with Unsloth](https://docs.camel-ai.org/cookbooks/data_generation/sft_data_generation_and_unsloth_finetuning_Qwen2_5_7B.html)** | Learn how to generate data with CAMEL and fine-tune models effectively with Unsloth. |
| **[Data Gen with Real Function Calls and Hermes Format](https://docs.camel-ai.org/cookbooks/data_generation/data_gen_with_real_function_calls_and_hermes_format.html)** | Explore how to generate data with real function calls and the Hermes format. |
| **[Self-instruct Data Generation](https://docs.camel-ai.org/cookbooks/data_generation/self_instruct_data_generation.html)** | Explore comprehensive guidelines and best practices for generating self-instructional data, enabling robust model training and improved performance. |
| **[CoT Data Generation and SFT Qwen with Unsolth](https://docs.camel-ai.org/cookbooks/data_generation/cot_data_gen_sft_qwen_unsolth_upload_huggingface.html)** | Discover how to generate CoT data using CAMEL and SFT Qwen with Unsolth, and seamlessly upload your data and model to Huggingface. |
| **[Agentic Data Generation, Evaluation & Filtering with Reward Models](https://docs.camel-ai.org/cookbooks/data_generation/synthetic_dataevaluation&filter_with_reward_model.html)** | Discover methods for generating, evaluating, and filtering agentic data using reward models to enhance the quality and efficiency of your synthetic data pipelines. |
| **[Data Model Generation and Structured Output with Qwen Model](https://docs.camel-ai.org/cookbooks/data_generation/data_model_generation_and_structured_output_with_qwen.html)** |Learn how to generate data models and structured outputs using the Qwen Model for improved data representation.|
| **[Distill Math Reasoning Data from DeepSeek R1](https://docs.camel-ai.org/cookbooks/data_generation/distill_math_reasoning_data_from_deepseek_r1.html)** |Learn  how to set up and leverage CAMEL's data distillation pipline for distilling high-quality maths reasoning data with thought process (Long CoT data)from deepseek R1, and uploading the results to Hugging Face.|
| **[Self-Improving Math Reasoning Data Distillation from DeepSeek R1](https://docs.camel-ai.org/cookbooks/data_generation/self_improving_math_reasoning_data_distillation_from_deepSeek_r1.html)** |Learn how to set up and leverage CAMEL's data distillation pipline for self-improving math reasoning data distillation from deepseek R1, and uploading the results to Hugging Face.|

### Multi-Agent Systems & Applications
| Cookbook | Description |
|:---|:---|
| **[Role-Playing Scraper for Report & Knowledge Graph Generation](https://docs.camel-ai.org/cookbooks/applications/roleplaying_scraper.html)** | Create role-playing agents for data scraping and reporting. |
| **[Create A Hackathon Judge Committee with Workforce](https://docs.camel-ai.org/cookbooks/multi_agent_society/workforce_judge_committee.html)** | Building a team of agents for collaborative judging. |
| **[Dynamic Travel Planner Role-Playing: Multi-Agent System with Real-Time Insights Powered by Dappier](https://docs.camel-ai.org/cookbooks/applications/dynamic_travel_planner.html)** | Explore an innovative approach to travel planning, blending AI-driven role-playing and real-time data for seamless experiences. |
| **[Dynamic Knowledge Graph Role-Playing: Multi-Agent System with dynamic, temporally-aware knowledge graphs](https://docs.camel-ai.org/cookbooks/applications/dyamic_knowledge_graph.html)** |  Builds dynamic, temporally-aware knowledge graphs based on multi-agent systems, which utilize diverse and optional element node deduplication techniques to ensure data integrity and optimize graph structure. |
| **[Customer Service Discord Bot with Agentic RAG](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_SambaNova_with_agentic_RAG.html)** | Learn how to build a robust customer service bot for Discord using Agentic RAG. |
| **[Customer Service Discord Bot with Local Model](https://docs.camel-ai.org/cookbooks/applications/customer_service_Discord_bot_using_local_model_with_agentic_RAG.html)** | Learn how to build a robust customer service bot for Discord using Agentic RAG which supports local deployment. |
| **[Customer Service Discord Bot for Finance with OpenBB](https://docs.camel-ai.org/cookbooks/applications/finance_discord_bot.html)**| Learn how to build a sipmle yet powerful financial data assistant Discord bot using OpenBB tools. |

### Data Processing
| Cookbook | Description |
|:---|:---|
| **[Video Analysis](https://docs.camel-ai.org/cookbooks/data_processing/video_analysis.html)** | Techniques for agents in video data analysis. |
| **[3 Ways to Ingest Data from Websites with Firecrawl](https://docs.camel-ai.org/cookbooks/data_processing/ingest_data_from_websites_with_Firecrawl.html)** | Explore three methods for extracting and processing data from websites using Firecrawl. |
| **[Create AI Agents that work with your PDFs](https://docs.camel-ai.org/cookbooks/data_processing/agent_with_chunkr_for_pdf_parsing.html)** | Learn how to create AI agents that work with your PDFs using Chunkr and Mistral AI. |

## Utilize Various LLMs as Backends

For more details, please see our [`Models Documentation`](https://docs.camel-ai.org/key_modules/models.html#).

## Data (Hosted on Hugging Face)
| Dataset        | Chat format                                                                                         | Instruction format                                                                                               | Chat format (translated)                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| **AI Society** | [Chat format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_chat.tar.gz) | [Instruction format](https://huggingface.co/datasets/camel-ai/ai_society/blob/main/ai_society_instructions.json) | [Chat format (translated)](https://huggingface.co/datasets/camel-ai/ai_society_translated) |
| **Code**       | [Chat format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_chat.tar.gz)             | [Instruction format](https://huggingface.co/datasets/camel-ai/code/blob/main/code_instructions.json)             | x                                                                                          |
| **Math**       | [Chat format](https://huggingface.co/datasets/camel-ai/math)                                        | x                                                                                                                | x                                                                                          |
| **Physics**    | [Chat format](https://huggingface.co/datasets/camel-ai/physics)                                     | x                                                                                                                | x                                                                                          |
| **Chemistry**  | [Chat format](https://huggingface.co/datasets/camel-ai/chemistry)                                   | x                                                                                                                | x                                                                                          |
| **Biology**    | [Chat format](https://huggingface.co/datasets/camel-ai/biology)                                     | x                                                                                                                | x                                                                                          |

## Visualizations of Instructions and Tasks

| Dataset          | Instructions                                                                                                         | Tasks                                                                                                         |
|------------------|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| **AI Society**   | [Instructions](https://atlas.nomic.ai/map/3a559a06-87d0-4476-a879-962656242452/db961915-b254-48e8-8e5c-917f827b74c6) | [Tasks](https://atlas.nomic.ai/map/cb96f41b-a6fd-4fe4-ac40-08e101714483/ae06156c-a572-46e9-8345-ebe18586d02b) |
| **Code**         | [Instructions](https://atlas.nomic.ai/map/902d6ccb-0bbb-4294-83a8-1c7d2dae03c8/ace2e146-e49f-41db-a1f4-25a2c4be2457) | [Tasks](https://atlas.nomic.ai/map/efc38617-9180-490a-8630-43a05b35d22d/2576addf-a133-45d5-89a9-6b067b6652dd) |
| **Misalignment** | [Instructions](https://atlas.nomic.ai/map/5c491035-a26e-4a05-9593-82ffb2c3ab40/2bd98896-894e-4807-9ed8-a203ccb14d5e) | [Tasks](https://atlas.nomic.ai/map/abc357dd-9c04-4913-9541-63e259d7ac1f/825139a4-af66-427c-9d0e-f36b5492ab3f) |

## Implemented Research Ideas from Other Works
We implemented amazing research ideas from other works for you to build, compare and customize your agents. If you use any of these modules, please kindly cite the original works:
- `TaskCreationAgent`, `TaskPrioritizationAgent` and `BabyAGI` from *Nakajima et al.*: [Task-Driven Autonomous Agent](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/). [[Example](https://github.com/camel-ai/camel/blob/master/examples/ai_society/babyagi_playing.py)]

- `PersonaHub` from *Tao Ge et al.*: [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/pdf/2406.20094). [[Example](https://github.com/camel-ai/camel/blob/master/examples/personas/personas_generation.py)]

- `Self-Instruct` from *Yizhong Wang et al.*: [SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/pdf/2212.10560). [[Example](https://github.com/camel-ai/camel/blob/master/examples/datagen/self_instruct/self_instruct.py)]

- `Source2Synth` from *Alisia Lupidi et al.*: [Source2Synth: Synthetic Data Generation and Curation Grounded in Real Data Sources](https://arxiv.org/abs/2409.08239). [[Example](https://github.com/camel-ai/camel/blob/master/examples/datagen/source2synth.py)]

- `STaR` from *Eric Zelikman et al.*: [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465). [[Example](https://github.com/camel-ai/camel/tree/master/examples/datagen/self_improving_cot)]

## Other Research Works Based on Camel
- [Agent Trust](http://agent-trust.camel-ai.org/): Can Large Language Model Agents Simulate Human Trust Behavior?

- [CRAB](https://crab.camel-ai.org/): Cross-environment Agent Benchmark for Multimodal Language Model Agents.

- OASIS: Open Agents Social Interaction Simulations on a Large Scale.

We warmly invite you to use CAMEL for your impactful research.

## News
📢 Added support for Qwen models, Deepseek models to the 🐫 CAMEL framework!. (Nov 28, 2024)
- Integrate SGLang into the 🐫 CAMEL framework. (Dec, 13, 2024)
- Integrated Reward Model into the 🐫 CAMEL framework. (Dec, 13, 2024)
- Added GAIA Benchmark! (Dec, 09, 2024)
- ...
- Released AI Society and Code dataset (April 2, 2023)
- Initial release of `CAMEL` python library (March 21, 2023)

## Citation
```
@inproceedings{li2023camel,
  title={CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society},
  author={Li, Guohao and Hammoud, Hasan Abed Al Kader and Itani, Hani and Khizbullin, Dmitrii and Ghanem, Bernard},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
## Acknowledgment
Special thanks to [Nomic AI](https://home.nomic.ai/) for giving us extended access to their data set exploration tool (Atlas).

We would also like to thank Haya Hammoud for designing the initial logo of our project.

## License

The source code is licensed under Apache 2.0.

## Contributing to CAMEL 🐫
We appreciate your interest in contributing to our open-source initiative. We provide a document of [contributing guidelines](https://github.com/camel-ai/camel/blob/master/CONTRIBUTING.md) which outlines the steps for contributing to CAMEL. Please refer to this guide to ensure smooth collaboration and successful contributions. 🤝🚀

## Contact
For more information please contact camel.ai.team@gmail.com.

[python-image]: https://img.shields.io/badge/Python-3.10%2C%203.11%2C%203.12-brightgreen.svg
[python-url]: https://www.python.org/
[pytest-image]: https://github.com/camel-ai/camel/actions/workflows/pytest_package.yml/badge.svg
[pytest-url]: https://github.com/camel-ai/camel/actions/workflows/pytest_package.yml
[docs-image]: https://img.shields.io/badge/Documentation-grey.svg?logo=github
[docs-url]: https://camel-ai.github.io/camel/index.html
[star-image]: https://img.shields.io/github/stars/camel-ai/camel?label=stars&logo=github&color=brightgreen
[star-url]: https://github.com/camel-ai/camel/stargazers
[package-license-image]: https://img.shields.io/badge/License-Apache_2.0-blue.svg
[package-license-url]: https://github.com/camel-ai/camel/blob/master/licenses/LICENSE

[colab-url]: https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing
[colab-image]: https://colab.research.google.com/assets/colab-badge.svg
[huggingface-url]: https://huggingface.co/camel-ai
[huggingface-image]: https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CAMEL--AI-ffc107?color=ffc107&logoColor=white
[slack-url]: https://join.slack.com/t/camel-ai/shared_invite/zt-2g7xc41gy-_7rcrNNAArIP6sLQqldkqQ
[slack-image]: https://img.shields.io/badge/Slack-CAMEL--AI-blueviolet?logo=slack
[discord-url]: https://discord.camel-ai.org/
[discord-image]: https://img.shields.io/badge/Discord-CAMEL--AI-7289da?logo=discord&logoColor=white&color=7289da
[wechat-url]: https://ghli.org/camel/wechat.png
[wechat-image]: https://img.shields.io/badge/WeChat-CamelAIOrg-brightgreen?logo=wechat&logoColor=white
[twitter-url]: https://twitter.com/CamelAIOrg
[twitter-image]: https://img.shields.io/twitter/follow/CamelAIOrg?style=social&color=brightgreen&logo=twitter
