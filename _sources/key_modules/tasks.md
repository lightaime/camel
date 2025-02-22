# Task

For more detailed usage information, please refer to our cookbook: [Task Generation Cookbook](../cookbooks/task_generation.ipynb)

## 1. Concept
> In the CAMEL framework, a task is a specific assignment that can be delegated to an agent and resolved by that agent. Tasks represent a higher-level concept than prompts and should be managed by other modules such as the Planner and Workforce. There are two key characteristics of a task: 
> 1. A task can be collaborative, requiring multiple agents to work together. 
> 2. A task can be decomposed and evolved.

### 1.1 Task Attributes

| Attribute | Type | Description |
| ----- | ----- | ----- |
| content | string | A clear and concise description of the task at hand. |
| id | string | An unique string identifier for the task. |
| state | Enum | The task states:  "OPEN", "RUNNING", "DONE", "FAILED", "DELETED". |
| type | string | The type of a task. (TODO) |
| parent | Task | The parent task. |
| subtasks | A list of Task | A list of sub tasks related the original Task. |
| result | string | The Task result. |

### 1.2 Task Methods

| Method | Type | Description |
| ----- | ----- | ----- |
| from_message | classmethod | Load Task from Message. |
| to_message | classmethod | Convert Task to Message. |
| reset | instance | Reset Task to initial state. |
| update_result | instance | Set task result and mark the task as DONE. |
| set_id | instance | Set task id. |
| set_state | instance | Recursively set the state of the task and its subtasks. |
| add_subtask | instance | Add child task. |
| remove_subtask | instance | Delete a subtask by a giving id. |
| get_running_task | instance | Get a RUNNING task. |
| to_string | instance | Convert task to a string. |
| get_result | instance | Get task result to a string. |
| decompose | instance | Decompose a task to a list of sub-tasks. |
| compose | instance | Compose task result by the sub-tasks. |
| get_depth | instance | Get task depth while the root depth is 1. |

## 2. Get Started

Creating a task involves defining its goal (content) and id:

### 2.1 Example of Task definition

```python
from camel.tasks import Task

task = Task(
    content="Weng earns $12 an hour for babysitting. Yesterday, she just did 51 minutes of babysitting. How much did she earn?",
    id="0",
)
```

### 2.2 Example of multiple Tasks with a hierarchical structure.

```python
# Creating the root task
root_task = Task(content="Prepare a meal", id="0")

# Creating subtasks for the root task
sub_task_1 = Task(content="Shop for ingredients", id="1")
sub_task_2 = Task(content="Cook the meal", id="2")
sub_task_3 = Task(content="Set the table", id="3")

# Creating subtasks under "Cook the meal"
sub_task_2_1 = Task(content="Chop vegetables", id="2.1")
sub_task_2_2 = Task(content="Cook rice", id="2.2")

# Adding subtasks to their respective parent tasks
root_task.add_subtask(sub_task_1)
root_task.add_subtask(sub_task_2)
root_task.add_subtask(sub_task_3)

sub_task_2.add_subtask(sub_task_2_1)
sub_task_2.add_subtask(sub_task_2_2)

# Printing the hierarchical task structure
print(root_task.to_string())
```

```markdown
>>>
    Task 0: Prepare a meal
    Task 1: Shop for ingredients
    Task 2: Cook the meal
        Task 2.1: Chop vegetables
        Task 2.2: Cook rice
    Task 3: Set the table
```

## 3. Decomposing and composing a Task

Decomposing or composing a task involves defining its responsible agent, prompt template and agent response parser. Here is an example:

```python
from camel.agents import ChatAgent
from camel.tasks import Task
from camel.tasks.task_prompt import (
    TASK_COMPOSE_PROMPT,
    TASK_DECOMPOSE_PROMPT,
)
from camel.messages import BaseMessage

sys_msg = BaseMessage.make_assistant_message(
    role_name="Assistant", content="You're a helpful assistant"
)
# Set up an agent
agent = ChatAgent(system_message=sys_msg)

task = Task(
    content="Weng earns $12 an hour for babysitting. Yesterday, she just did 51 minutes of babysitting. How much did she earn?",
    id="0",
)


new_tasks = task.decompose(agent=agent, template=TASK_DECOMPOSE_PROMPT)
for t in new_tasks:
    print(t.to_string())
```

```markdown
>>>
Task 0.0: Convert 51 minutes into hours.

Task 0.1: Calculate Weng's earnings for the converted hours at the rate of $12 per hour.

Task 0.2: Provide the final earnings amount based on the calculation.
```

```python
# compose task result by the sub-tasks.
task.compose(agent=agent, template=TASK_COMPOSE_PROMPT)
print(task.result)
```

## 4. TaskManager

TaskManager is used to manage tasks.

| Method | Type | Description |
| ----- | ----- | ----- |
| topological_sort | instance | Sort a list of tasks topologically. |
| set_tasks_dependence | instance | Set relationship between root task and other tasks. Two relationships are currently supported: serial and parallel. |
| evolve | instance | Evolve a task to a new task and here it is only used for data generation. |


### Example

```python
from camel.tasks import (
    Task,
    TaskManager,
)

sys_msg = "You're a helpful assistant"

# Set up an agent
agent = ChatAgent(system_message=sys_msg)


task = Task(
    content="Weng earns $12 an hour for babysitting. Yesterday, she just did 51 minutes of babysitting. How much did she earn?",
    id="0",
)
print(task.to_string())
```

```markdown
>>>Task 0: Weng earns $12 an hour for babysitting. Yesterday, she just did 51 minutes of babysitting. How much did she earn?
```

```python
task_manager = TaskManager(task)
evolved_task = task_manager.evolve(task, agent=agent)
print(evolved_task.to_string())
```

```markdown
>>>Task 0.0: Weng earns $12 an hour for babysitting. Yesterday, she babysat for 1 hour and 45 minutes. If she also received a $5 bonus for exceptional service, how much did she earn in total for that day?
```

## 5. Conclusion
We offers a structured approach to task management, enabling efficient delegation and resolution of tasks by agents. With features such as task decomposition, composition, and hierarchical task structures, CAMEL provides the tools necessary to manage complex workflows. Whether handling simple tasks or intricate, multi-level assignments, CAMEL's task management capabilities ensure that tasks are executed effectively and collaboratively, enhancing overall productivity.