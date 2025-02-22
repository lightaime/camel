# Agents

## 1. Concept

Agents in CAMEL are autonomous entities capable of performing specific tasks through interaction with language models and other components. Each agent is designed with a particular role and capability, allowing them to work independently or collaboratively to achieve complex goals.

### 1.1. Base Agent Architecture
All CAMEL agents inherit from the `BaseAgent` abstract class, which defines two core methods:
- `reset()`: Resets the agent to its initial state
- `step()`: Performs a single step of the agent's operation

### 1.2. Chat Agent
The `ChatAgent` is the primary implementation that handles conversations with language models. It supports:
- System message configuration for role definition
- Memory management for conversation history
- Tool/function calling capabilities
- Response formatting and structured outputs
- Multiple model backend support with scheduling strategies
- Async operation support

## 2. Types

### 2.1. `ChatAgent`
The main agent implementation for handling conversations with language models. Features include:
- Tool integration and management
- Memory management with customizable window sizes
- Output language control
- Response termination handling
- Structured output support via Pydantic models

### 2.2. `CriticAgent`
Specialized agent for evaluating and critiquing responses or solutions. Used in scenarios requiring quality assessment or validation.

### 2.3. `DeductiveReasonerAgent`
Agent focused on logical reasoning and deduction. Breaks down complex problems into smaller, manageable steps.

### 2.4. `EmbodiedAgent`
Agent designed for embodied AI scenarios, capable of understanding and responding to physical world contexts.

### 2.5. `KnowledgeGraphAgent`
Specialized in building and utilizing knowledge graphs for enhanced reasoning and information management.

### 2.6. `MultiHopGeneratorAgent`
Agent designed for handling multi-hop reasoning tasks, generating intermediate steps to reach conclusions.

### 2.7. `SearchAgent`
Focused on information retrieval and search tasks across various data sources.

### 2.8. `TaskAgent`
Handles task decomposition and management, breaking down complex tasks into manageable subtasks.

## 3. Usage

### 3.1. Basic Chat Agent Usage
```python
from camel.agents import ChatAgent

# Create a chat agent with a system message
agent = ChatAgent(system_message="You are a helpful assistant.")

# Step through a conversation
response = agent.step("Hello, can you help me?")
```

### 3.2. Using Tools with Chat Agent
```python
from camel.agents import ChatAgent
from camel.functions import FunctionTool

# Define a tool
def calculator(a: int, b: int) -> int:
    return a + b

# Create agent with tool
agent = ChatAgent(tools=[calculator])

# The agent can now use the calculator tool in conversations
response = agent.step("What is 5 + 3?")
```

### 3.3. Structured Output
```python
from pydantic import BaseModel
from typing import List

class ResponseFormat(BaseModel):
    points: List[str]
    summary: str

# Create agent with structured output
agent = ChatAgent()
response = agent.step("List benefits of exercise", response_format=ResponseFormat)
```

## 4. Best Practices

### 4.1. Memory Management
- Use appropriate window sizes to manage conversation history
- Consider token limits when dealing with long conversations
- Utilize the memory system for maintaining context

### 4.2. Tool Integration
- Keep tool functions focused and well-documented
- Handle tool errors gracefully
- Use external tools for operations that should be handled by the user

### 4.3. Response Handling
- Implement appropriate response terminators for conversation control
- Use structured outputs when specific response formats are needed
- Handle async operations properly when dealing with long-running tasks

## 5. Advanced Features

### 5.1. Model Scheduling
The agent supports multiple model backends with customizable scheduling strategies:
```python
def custom_strategy(models):
    # Custom model selection logic
    return models[0]

agent.add_model_scheduling_strategy("custom", custom_strategy)
```

### 5.2. Output Language Control
Control the language of agent responses:
```python
agent.set_output_language("Spanish")
```
