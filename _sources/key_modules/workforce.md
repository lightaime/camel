# Workforce

> Workforce is a system where multiple agents work together to solve tasks.
> By using Workforce, users can quickly set up a multi-agent task solving
> system with customized configurations. In this section, we will give a
> brief view on the architecture of workforce, and how you can configure
> and utilize it to solve tasks.

For more detailed usage information, please refer to our cookbook: [Create A Hackathon Judge Committee with Workforce](https://colab.research.google.com/drive/18ajYUMfwDx3WyrjHow3EvUMpKQDcrLtr?usp=sharing)

## System Design

### Architecture

Workforce follows a hierarchical architecture. A workforce can consist of 
multiple worker nodes, and each of the worker nodes will contain 
one agent or multiple agents as the worker. The worker nodes are managed by 
a coordinator agent inside the workforce, and the coordinator agent will assign 
tasks to the worker nodes according to the description of the worker nodes, 
along with their tool sets.

Alongside the coordinator agent, there is also a task planner agent inside the
workforce. The task planner agent will take the responsibility of decomposing
and composing tasks, so that the workforce can solve the task step by step.

### Communication Mechanism

The communication inside the workforce is based on the task channel. The
initialization of workforce will create a channel that will be shared by
all the nodes. Tasks will later be posted into this channel, and each
worker node will listen to the channel, accept tasks that are assigned
to it from this channel to solve.

When a task is solved, the worker node will post the result back to the
channel, and the result will reside in the channel as a 'dependency' for
other tasks, which will be shared by all the worker nodes.

With this mechanism, the workforce can solve tasks in a collaborative and
efficient way.

### Failure Handling

In the workforce, we have a mechanism to handle failures. When a task fails,
the coordinator agent will take actions to fix it. The actions can be either
decomposing the task into smaller ones and assign them again, or creating a
new worker that is capable of doing the task.

For now, the coordinator will make decisions based on the number of times the
task has been decomposed. If the task has already been decomposed for more
than a certain number of times, the coordinator will take the new worker
creation action; if not, the coordinator will simply take the decomposition
action.

There will also be cases where a task just simply cannot be solved by
agents. Under this circumstance, to prevent the workforce from being stuck
in a infinite agent creation loop, the workforce will be halted if one task
has been failed for a certain number of times (3 by default).

### A Simple Example of the Workflow

Here is a diagram illustrating the workflow with a simple example.

![Workforce Example](https://i.postimg.cc/261dDM9W/work-force.png)

## Get Started

In this section, we will show you how to create a workforce instance, add
worker nodes to the workforce, and finally, how to start the workforce to
solve tasks.

### Create a Workforce Instance

To start using Workforce, you need to first create a workforce instance, and
then add worker nodes to the workforce. Here is an example of how you can do
this:

```python
from camel.societies.workforce import Workforce

# Create a workforce instance
workforce = Workforce("A Simple Workforce")
```

Now we get a workforce instance with description "A Simple Workforce".
However, there is no worker node in the workforce yet.

> **Note: How to further customize the workforce**
>
> You can quickly create a workforce instance with default configurations
> by just providing the description of the workforce. However, you can also
> configure the workforce with more details, such as providing a list of
> worker nodes directly instead of adding them one by one, or configuring
> the coordinator agent and task planner agent by passing the configurations
> to the workforce instance.

### Add Worker Nodes

After creating the workforce instance, you can add worker nodes to the
workforce. To add a worker node, you need to first create worker agents
that actually do the work.

Suppose we have already created an `ChatAgent` that can do web searches 
called `search_agent`. Now we can add this worker agent to the workforce.

```python
# Add the worker agent to the workforce
workforce.add_single_agent_worker(
    "An agent that can do web searches",
    worker=search_agent,
)
```

The adding function follows the fluent interface design pattern, so you can
add multiple worker agents to the workforce in one line, like the following:

```python
workforce.add_single_agent_worker(
    "An agent that can do web searches",
    worker=search_agent,
).add_single_agent_worker(
    "Another agent",
    worker=another_agent,
).add_single_agent_worker(
    "Yet another agent",
    worker=yet_another_agent,
)
```

Now we have some worker agents in the workforce. It's time to create a task and
let the workforce solve it.

> **Note: The description is NOT trivial**
> 
> When adding an agent to the workforce as a worker node, the first argument
> is the description of the worker node. Though it seems trivial, it is 
> actually very important. The description will be used by the coordinator
> agent as the basis to assign tasks to the worker nodes. Therefore, it is
> recommended to provide a clear and concise description of the worker node.

### Start the Workforce

After adding worker nodes to the workforce, you can start the workforce to
solve tasks. First we can define a task:

```python
from camel.tasks import Task

# the id of the task can be any string to mark the task
task = Task(
    content="Make a travel plan for a 3-day trip to Paris.",
    id="0",
)
```

Then we can start the workforce with the task:

```python
task = workforce.process_task(task)
```

The final result of the task will be stored in the `result` attribute of the
task object. You can check the result by:

```python
print(task.result)
```
