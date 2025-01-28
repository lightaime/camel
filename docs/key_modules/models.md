# Models

## 1. Concept
The model is the brain of the intelligent agent, responsible for processing all input and output data. By calling different models, the agent can execute operations such as text analysis, image recognition, or complex reasoning according to task requirements. CAMEL offers a range of standard and customizable interfaces, as well as seamless integrations with various components, to facilitate the development of applications with Large Language Models (LLMs). In this part, we will introduce models currently supported by CAMEL and the working principles and interaction methods with models. 

All the codes are also available on colab notebook [here](https://colab.research.google.com/drive/18hQLpte6WW2Ja3Yfj09NRiVY-6S2MFu7?usp=sharing).


## 2. Supported Model Platforms

The following table lists currently supported model platforms by CAMEL.

| Model Platform | Available Models| Multi-modality |
| ----- | ----- | ----- |
| OpenAI | gpt-4o | Y |
| OpenAI | gpt-4o-mini | Y |
| OpenAI | o1 | Y |
| OpenAI | o1-preview | N |
| OpenAI | o1-mini | N |
| OpenAI | gpt-4-turbo | Y |
| OpenAI | gpt-4 | Y |
| OpenAI | gpt-3.5-turbo | N |
| Azure OpenAI | gpt-4o | Y |
| Azure OpenAI | gpt-4-turbo | Y |
| Azure OpenAI | gpt-4 | Y |
| Azure OpenAI | gpt-3.5-turbo | Y |
| OpenAI Compatible | Depends on the provider | ----- |
| Mistral AI | mistral-large-2 | N |
| Mistral AI | pixtral-12b-2409 | Y |
| Mistral AI | ministral-8b-latest | N |
| Mistral AI | ministral-3b-latest | N |
| Mistral AI | open-mistral-nemo | N |
| Mistral AI | codestral | N |
| Mistral AI | open-mistral-7b | N |
| Mistral AI | open-mixtral-8x7b | N |
| Mistral AI | open-mixtral-8x22b | N |
| Mistral AI | open-codestral-mamba | N |
| Anthropic | claude-3-5-sonnet-latest | Y |
| Anthropic | claude-3-5-haiku-latest | N |
| Anthropic | claude-3-haiku-20240307 | Y |
| Anthropic | claude-3-sonnet-20240229 | Y |
| Anthropic | claude-3-opus-latest | Y |
| Anthropic | claude-2.0 | N |
| Gemini | gemini-1.5-pro | Y |
| Gemini | gemini-1.5-flash | Y |
| Gemini | gemini-exp-1114 | Y |
| Lingyiwanwu | yi-lightning | N |
| Lingyiwanwu | yi-large | N |
| Lingyiwanwu | yi-medium | N |
| Lingyiwanwu | yi-large-turbo | N |
| Lingyiwanwu | yi-vision | Y |
| Lingyiwanwu | yi-medium-200k | N |
| Lingyiwanwu | yi-spark | N |
| Lingyiwanwu | yi-large-rag | N |
| Lingyiwanwu | yi-large-fc | N |
| Qwen | qwq-32b-preview | N |
| Qwen | qwen-max | N |
| Qwen | qwen-plus | N |
| Qwen | qwen-turbo | N |
| Qwen | qwen-long | N |
| Qwen | qwen-vl-max | Y |
| Qwen | qwen-vl-plus | Y |
| Qwen | qwen-math-plus | N |
| Qwen | qwen-math-turbo | N |
| Qwen | qwen-coder-turbo | N |
| Qwen | qwen2.5-coder-32b-instruct | N |
| Qwen | qwen2.5-72b-instruct | N |
| Qwen | qwen2.5-32b-instruct | N |
| Qwen | qwen2.5-14b-instruct | N |
| DeepSeek | deepseek-chat | N |
| DeepSeek | deepseek-reasoner | N |
| ZhipuAI | glm-4v | Y |
| ZhipuAI | glm-4 | N |
| ZhipuAI | glm-3-turbo | N |
| InternLM | internlm3-latest | N |
| InternLM | internlm3-8b-instruct | N |
| InternLM | internlm2.5-latest	| N |
| InternLM | internlm2-pro-chat	| N |
| Reka | reka-core | Y |
| Reka | reka-flash | Y |
| Reka | reka-edge | Y |
| Nvidia | https://docs.api.nvidia.com/nim/reference/llm-apis | ----- |
| SambaNova| https://community.sambanova.ai/t/supported-models/193 | ----- |
| Groq | https://console.groq.com/docs/models | ----- |
| Ollama | https://ollama.com/library | ----- |
| vLLM | https://docs.vllm.ai/en/latest/models/supported_models.html | ----- |
| Together AI | https://docs.together.ai/docs/chat-models | ----- |
| LiteLLM | https://docs.litellm.ai/docs/providers | ----- |
| SGLang | https://sgl-project.github.io/references/supported_models.html | ----- |
| AWSBedrock | https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html | ----- |
## 3. Using Models by API calling

Here is an example code to use a specific model (gpt-4o-mini). If you want to use another model, you can simply change these three parameters: `model_platform`, `model_type`, `model_config_dict` .

```python
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from camel.agents import ChatAgent

# Define the model, here in this case we use gpt-4o-mini
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig().as_dict(),
)

# Define an assitant message
system_msg = "You are a helpful assistant."

# Initialize the agent
ChatAgent(system_msg, model=model)
```

And if you want to use an OpenAI-compatible API, you can replace the `model` with the following code:

```python
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="a-string-representing-the-model-type",
    api_key=os.environ.get("OPENAI_COMPATIBILIY_API_KEY"),
    url=os.environ.get("OPENAI_COMPATIBILIY_API_BASE_URL"),
    model_config_dict={"temperature": 0.4, "max_tokens": 4096},
)
```

## 4. Using On-Device Open Source Models
In the current landscape, for those seeking highly stable content generation, OpenAI’s gpt-4o-mini, gpt-4o are often recommended. However, the field is rich with many other outstanding open-source models that also yield commendable results. CAMEL can support developers to delve into integrating these open-source large language models (LLMs) to achieve project outputs based on unique input ideas.

### 4.1 Using Ollama to Set Llama 3 Locally

1. Download [Ollama](https://ollama.com/download).
2. After setting up Ollama, pull the Llama3 model by typing the following command into the terminal:

```bash
ollama pull llama3
```

3. Create a `ModelFile` similar the one below in your project directory. (Optional)

```
FROM llama3

# Set parameters
PARAMETER temperature 0.8
PARAMETER stop Result

# Sets a custom system message to specify the behavior of the chat assistant
# Leaving it blank for now.

SYSTEM """ """
```

4. Create a script to get the base model (llama3) and create a custom model using the `ModelFile` above. Save this as a `.sh` file: (Optional)

```
#!/bin/zsh

# variables
model_name="llama3"
custom_model_name="camel-llama3"

#get the base model
ollama pull $model_name

#create the model file
ollama create $custom_model_name -f ./Llama3ModelFile
```

5. Navigate to the directory where the script and `ModelFile` are located and run the script. Enjoy your Llama3 model, enhanced by CAMEL's excellent agents.

```python
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType

ollama_model = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="llama3",
    url="http://localhost:11434/v1", # Optional
    model_config_dict={"temperature": 0.4},
)

agent_sys_msg = "You are a helpful assistant."

agent = ChatAgent(agent_sys_msg, model=ollama_model, token_limit=4096)

user_msg = "Say hi to CAMEL"

assistant_response = agent.step(user_msg)
print(assistant_response.msg.content)
```

### 4.2 Using vLLM to Set Phi-3 Locally

Install [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html) first.

After setting up vLLM, start an OpenAI compatible server for example by:

```
python -m vllm.entrypoints.openai.api_server --model microsoft/Phi-3-mini-4k-instruct --api-key vllm --dtype bfloat16
```

Create and run following script (more details please refer to this [example](https://github.com/camel-ai/camel/blob/master/examples/models/vllm_model_example.py)):

```python
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType

vllm_model = ModelFactory.create(
    model_platform=ModelPlatformType.VLLM,
    model_type="microsoft/Phi-3-mini-4k-instruct",
    url="http://localhost:8000/v1", # Optional
    model_config_dict={"temperature": 0.0}, # Optional
)

agent_sys_msg = "You are a helpful assistant."

agent = ChatAgent(agent_sys_msg, model=vllm_model, token_limit=4096)

user_msg = "Say hi to CAMEL AI"

assistant_response = agent.step(user_msg)
print(assistant_response.msg.content)
```

### 4.3 Using SGLang to Set meta-llama/Llama Locally

Install [SGLang](https://sgl-project.github.io/start/install.html) first.

Create and run following script (more details please refer to this [example](https://github.com/camel-ai/camel/blob/master/examples/models/sglang_model_example.py)):

```python
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType

sglang_model = ModelFactory.create(
    model_platform=ModelPlatformType.SGLANG,
    model_type="meta-llama/Llama-3.2-1B-Instruct",
    model_config_dict={"temperature": 0.0},
    api_key="sglang",
)

agent_sys_msg = "You are a helpful assistant."

agent = ChatAgent(agent_sys_msg, model=sglang_model, token_limit=4096)

user_msg = "Say hi to CAMEL AI"

assistant_response = agent.step(user_msg)
print(assistant_response.msg.content)
```

## 5. About Model Speed
Model speed is a crucial factor in AI application performance. It affects both user experience and system efficiency, especially in real-time or interactive tasks. In [this notebook](../cookbooks/model_speed_comparison.ipynb), we compared several models, including OpenAI’s GPT-4O Mini, GPT-4O, O1 Preview, and SambaNova's Llama series, by measuring the number of tokens each model processes per second.

Key Insights:
Smaller models like SambaNova’s Llama 8B and OpenAI's GPT-4O Mini typically offer faster responses.
Larger models like SambaNova’s Llama 405B, while more powerful, tend to generate output more slowly due to their complexity.
OpenAI models demonstrate relatively consistent performance, while SambaNova's Llama 8B significantly outperforms others in speed.
The chart below illustrates the tokens per second achieved by each model during our tests:

![Model Speed Comparison](https://i.postimg.cc/4xByytyZ/model-speed.png)

For local inference, we conducted a straightforward comparison locally between vLLM and SGLang. SGLang demonstrated superior performance, with `meta-llama/Llama-3.2-1B-Instruct` reaching a peak speed of 220.98 tokens per second, compared to vLLM, which capped at 107.2 tokens per second.

## 6. Conclusion
In conclusion, CAMEL empowers developers to explore and integrate these diverse models, unlocking new possibilities for innovative AI applications. The world of large language models offers a rich tapestry of options beyond just the well-known proprietary solutions. By guiding users through model selection, environment setup, and integration, CAMEL bridges the gap between cutting-edge AI research and practical implementation. Its hybrid approach, combining in-house implementations with third-party integrations, offers unparalleled flexibility and comprehensive support for LLM-based development. Don't just watch this transformation that is happening from the sidelines.

Dive into the CAMEL documentation, experiment with different models, and be part of shaping the future of AI. The era of truly flexible and powerful AI is here - are you ready to make your mark?
