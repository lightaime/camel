# Tools

For more detailed usage information, please refer to our cookbook: [Tools Cookbook](../cookbooks/agents_with_tools.ipynb)

## 1. Concept
Tools serve as interfaces that allow LLMs and Agents to interact with the world. A tool is essentially a function that has a name, a description, input parameters, and an output type. In this section, we will introduce the tools currently supported by CAMEL and explain how to define your own tools and Toolkits.

**Tools:** Tools are akin to OpenAI Functions and can be easily converted to that format. In CAMEL, we offer a variety of commonly used tools that you can utilize directly. While built-in tools can be quite useful, it's highly likely that you'll need to define your own tools. This guide for detailed instructions on how to create custom tools.

**Toolkits:** Toolkits are collections of tools that are designed to work well together. 

## 2. Get Started

To enhance your agents' capabilities with CAMEL tools, start by installing our additional tools package:
```sh
pip install 'camel-ai[tools]'
```

In CAMEL, a tool is an `FunctionTool` that LLMs can call.


### 2.1 How to Define Your Own Tool?

Developers can create custom tools tailored to their agent’s specific needs with ease. Here's how you can define and use a custom tool:

**Example: Adding Two Numbers**

First, define your function and wrap it using the `FunctionTool`

```python
from camel.toolkits import FunctionTool

def add(a: int, b: int) -> int:
    r"""Adds two numbers.

    Args:
        a (int): The first number to be added.
        b (int): The second number to be added.

    Returns:
        integer: The sum of the two numbers.
    """
    return a + b

# Wrap the function with FunctionTool
add_tool = FunctionTool(add)
```

**Accessing Tool Properties**

Once your tool is defined, you can inspect its properties using built-in methods:

Retrieve the function's name:

```python
print(add_tool.get_function_name())
```

```markdown
>>> add
```

Get the description of what the function does:

```python
print(add_tool.get_function_description())
```

```markdown
>>> Adds two numbers.
```

Fetch the schema representation for OpenAI functions:

```python
print(add_tool.get_openai_function_schema())
```

```markdown
>>> 
{'name': 'add',
 'description': 'Adds two numbers.',
 'parameters': {'properties': {'a': {'type': 'integer',
    'description': 'The first number to be added.'},
   'b': {'type': 'integer', 'description': 'The second number to be added.'}},
  'required': ['a', 'b'],
  'type': 'object'}}
```

Retrieve the tool schema, compatible with OpenAI's structure:

```python
print(add_tool.get_openai_tool_schema())
```

```markdown
>>> 
{'type': 'function',
 'function': {'name': 'add',
  'description': 'Adds two numbers.',
  'parameters': {'properties': {'a': {'type': 'integer',
     'description': 'The first number to be added.'},
    'b': {'type': 'integer', 'description': 'The second number to be added.'}},
   'required': ['a', 'b'],
   'type': 'object'}}}
```

### 2.2 Toolkits

Toolkits are collections of tools that are designed to be used together for specific tasks. All Toolkits expose a `get_tools` method which returns a list of tools. You can therefore do:

```python
from camel.toolkits import SearchToolkit

# Initialize a toolkit
toolkit = SearchToolkit()

# Get list of tools
tools = toolkit.get_tools()
```

To utilize specific tools from the toolkits, you can implement code like the following:

```python
from camel.toolkits import SearchToolkit

google_tool = FunctionTool(SearchToolkit().search_google)
wiki_tool = FunctionTool(SearchToolkit().search_wiki)
```

### 2.3 Passing Tools to `ChatAgent`
Enhance the functionality of a ChatAgent by passing custom tools during initialization. Here's how you can do it:

```python
from camel.agents import ChatAgent

# Initialize a ChatAgent with your custom tools
tool_agent = ChatAgent(
    tools=tools,
)

# Interact with the agent
response = tool_agent.step("A query related to the tool you added")
```

## 3. Built-in Toolkits

CAMEL provides a variety of built-in toolkits that you can use right away. Here's a comprehensive list of available toolkits:

| Toolkit | Description |
|---------|-------------|
| ArxivToolkit | A toolkit for interacting with the arXiv API to search and download academic papers. |
| AskNewsToolkit | A toolkit for fetching news, stories, and other content based on user queries using the AskNews API. |
| CodeExecutionToolkit | A toolkit for code execution which can run code in various sandboxes including internal Python, Jupyter, Docker, subprocess, or e2b. |
| DalleToolkit | A toolkit for image generation using OpenAI's DALL-E model. |
| DappierToolkit | A toolkit for searching real-time data and fetching AI recommendations across key verticals like News, Finance, Stock Market, Sports, Weather and more using the Dappier API. |
| DataCommonsToolkit | A toolkit for querying and retrieving data from the Data Commons knowledge graph, including SPARQL queries, statistical time series data, and property analysis. |
| FunctionTool | A base toolkit for creating function-based tools that OpenAI chat models can call, with support for schema parsing and synthesis. |
| GitHubToolkit | A toolkit for interacting with GitHub repositories, including retrieving issues and creating pull requests. |
| GoogleMapsToolkit | A toolkit for accessing Google Maps services, including address validation, elevation data, and timezone information. |
| GoogleScholarToolkit | A toolkit for retrieving information about authors and their publications from Google Scholar. |
| HumanToolkit | A toolkit for facilitating human-in-the-loop interactions and feedback in AI systems. |
| LinkedInToolkit | A toolkit for LinkedIn operations including creating posts, deleting posts, and retrieving user profile information. |
| MathToolkit | A toolkit for performing basic mathematical operations such as addition, subtraction, and multiplication. |
| MeshyToolkit | A toolkit for working with 3D mesh data and operations. |
| NotionToolkit | A toolkit for retrieving information from Notion pages and workspaces using the Notion API. |
| OpenAPIToolkit | A toolkit for working with OpenAPI specifications and REST APIs. |
| OpenBBToolkit | A toolkit for accessing and analyzing financial market data through the OpenBB Platform, including stocks, ETFs, cryptocurrencies, and economic indicators. |
| RedditToolkit | A toolkit for Reddit operations including collecting top posts, performing sentiment analysis on comments, and tracking keyword discussions. |
| RetrievalToolkit | A toolkit for retrieving information from local vector storage systems based on specified queries. |
| SearchToolkit | A toolkit for performing web searches using various search engines like Google, DuckDuckGo, Wikipedia, and Wolfram Alpha. |
| SlackToolkit | A toolkit for Slack operations including creating channels, joining channels, and managing channel membership. |
| StripeToolkit | A toolkit for processing payments and managing financial transactions via Stripe. |
| TwitterToolkit | A toolkit for Twitter operations including creating tweets, deleting tweets, and retrieving user profile information. |
| VideoToolkit | A toolkit for downloading videos and optionally splitting them into chunks, with support for various video services. |
| WeatherToolkit | A toolkit for fetching weather data for cities using the OpenWeatherMap API. |
| WhatsAppToolkit | A toolkit for interacting with the WhatsApp Business API, including sending messages, managing message templates, and accessing business profile information. |

## 4. Conclusion

Tools are essential for extending the capabilities of CAMEL agents, empowering them to perform a wide range of tasks and collaborate efficiently.
