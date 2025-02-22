# Prompt

## 1. Concept
The prompt module guides AI models to produce accurate, contextually relevant and personalized outputs. It includes various prompt templates and dictionaries designed for different tasks such as role descriptions, code generation, evaluation, text embedding, misalignment tasks, object recognition and so on. You can also create your own prompt to tailor your own AI agent.

## 2. Get Started

### 2.1 Using Prompt Templates

CAMEL provides various and comprehensive prompt templates for users to easily create AI specialists. For example, here we aggregate them to create a task-specific agent:

```python
from camel.agents import TaskSpecifyAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, TaskType

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
)
task_specify_agent = TaskSpecifyAgent(
    model=model, task_type=TaskType.AI_SOCIETY
)
specified_task_prompt = task_specify_agent.run(
    task_prompt="Improving stage presence and performance skills",
    meta_dict=dict(
        assistant_role="Musician", user_role="Student", word_limit=100
    ),
)

print(f"Specified task prompt:\n{specified_task_prompt}\n")
```

Set `task_type=TaskType.AI_SOCIETY` when creating a task specify agent and it will automatically evoke prompt with template `AISocietyPromptTemplateDict.TASK_SPECIFY_PROMPT` . You can set the role that you want the assistant to play. The output will be like this:

```markdown
>> Response:
Musician will help Student enhance stage presence by practicing engaging eye contact, dynamic movement, and expressive gestures during a mock concert, followed by a review session with video playback to identify strengths and areas for improvement.
```

### 2.2 Using Your Own Prompt

CAMEL also ensures high flexibility for users to create identical prompts. Still using `TaskSpecifyAgent` as an example, here is a demo for you to pass your own prompt template:

```python
from camel.agents import TaskSpecifyAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.prompts import TextPrompt
from camel.types import ModelPlatformType, ModelType

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
)

my_prompt_template = TextPrompt(
    'Here is a task: I\'m a {occupation} and I want to {task}. Help me to make this task more specific.'
)  # Write anything you would like to use as prompt template
task_specify_agent = TaskSpecifyAgent(
    model=model, task_specify_prompt=my_prompt_template
)
response = task_specify_agent.run(
    task_prompt="get promotion",
    meta_dict=dict(occupation="Software Engineer"),
)
print(response)
```

The response will be like this:

```markdown
>>> Response:
Certainly! To make the task of getting a promotion more specific, you can break it down into actionable steps and set clear, measurable goals. Here’s a more detailed plan:

### 1. **Set Clear Objectives**
   - **Identify the Promotion Criteria:** Understand what skills, achievements, and experiences are required for the promotion.
   - **Define Your Desired Position:** Specify the role or title you are aiming for.

### 2. **Skill Development**
   - **Technical Skills:** Identify any technical skills that are necessary for the promotion and create a plan to acquire or improve them (e.g., mastering a new programming language, learning about system architecture, etc.).
   - **Soft Skills:** Focus on improving soft skills such as leadership, communication, and teamwork.

## 3. Introduction to `CodePrompt` Class

In this part, we will explore the `CodePrompt` class, which is a class that represents a code prompt. It extends the `TextPrompt` class, which in turn extends the built-in `str` class. The `CodePrompt` class provides additional functionality related to code execution and handling.

### Importing the `CodePrompt` Class

To use the `CodePrompt` class, you need to import it. Here's an example of how to import the class:

```python
from camel.prompts import CodePrompt
```

### Creating a `CodePrompt` Instance

To create a `CodePrompt` instance, you can simply instantiate the class, providing the code string and the code type as an input argument.

```python
code_prompt = CodePrompt("a = 1 + 1", code_type="python")
```

In this example, we create a `CodePrompt` instance with the code string `"a = 1 + 1"`. We also specify the code type as `"python"`. The code type can be set to `None` if not needed.

### Accessing the Code and Code Type

Once you have a `CodePrompt` instance, you can access the code string and code type as following:

- `code_prompt`: Accesses the code string of the prompt.
- `code_type`: Accesses the type of code associated with the prompt.

```python
print(code_prompt)
# >>> "a = 1 + 1"

print(code_prompt.code_type)
# >>> "python"
```

### Modifying the Code Type

If you need to change the code type associated with a `CodePrompt` instance, you can use the `set_code_type` method. This method takes a code type as a parameter and updates the code type of the instance.

```python
code_prompt = CodePrompt("a = 1 + 1")
print(code_prompt.code_type)
# >>> None

code_prompt.set_code_type("python")
print(code_prompt.code_type) 
# >>> "python"
```

In this example, we change the code type of the `CodePrompt` instance from `None` to `"python"`.

### Executing the Code

The `CodePrompt` class provides a method called `execute` that allows you to execute the code string associated with the prompt. It returns a string containing the stdout and stderr.

```python
code_prompt = CodePrompt("a = 1 + 1\nb = a + 1\nprint(a,b)", code_type="python")
output = code_prompt.execute()
# Running code? [Y/n]: y
print(output)
# >>> 2 3

```


## 4. Write Your Prompts with the `TextPrompt` Class

In this part, we will explore the `TextPrompt` class and understand its functionalities. The `TextPrompt` class is a subclass of the built-in `str` class and provides additional features for working with text prompts. We will cover the following topics:

- Introduction to the `TextPrompt` class
- Using the `TextPrompt` class

### Introduction to the `TextPrompt` class

The `TextPrompt` class represents a text prompt and extends the functionality of the `str` class. It provides a property called `key_words`, which returns a set of strings representing the key words in the prompt.

Here's an example of how to use the `TextPrompt` class:

```python
from camel.prompts import TextPrompt

prompt = TextPrompt('Please enter your name and age: {name}, {age}')
print(prompt)  
>>> 'Please enter your name and age: {name}, {age}'
```

In the above example, we create a `TextPrompt` instance with a format string containing key words for name and age. We can print `TextPrompt` like Python `str`.

### Using the `TextPrompt` class

Once we have created a `TextPrompt` instance, we can use various methods and properties provided by the class to manipulate and work with the text prompt.

### The `key_words` property

The `key_words` property returns a set of strings representing the key words in the prompt.

```python
from camel.prompts import TextPrompt

prompt = TextPrompt('Please enter your name and age: {name}, {age}')
print(prompt.key_words)
>>> {'name', 'age'}
```

In the above example, the `key_words` property returns a set of strings representing the key words in the prompt, which in this case are 'name' and 'age'.

### The `format` method

The `format` method overrides the built-in `str.format` method to allow for partial formatting values in the format string. It replaces the key words in the format string with the provided values.

```python
from camel.prompts import TextPrompt

prompt = TextPrompt('Your name and age are: {name}, {age}')

name, age = 'John', 30
formatted_prompt = prompt.format(name=name, age=age)
print(formatted_prompt)  
>>> "Your name and age are: John, 30"
```

In the above example, we use the `format` method to replace the key words `{name}` and `{age}` with the values 'John' and 30, respectively.

We can also perform partial formatting by providing only some of the values:

```python
from camel.prompts import TextPrompt

prompt = TextPrompt('Your name and age are: {name}, {age}')

name = 'John'
partial_formatted_prompt = prompt.format(name=name)
print(partial_formatted_prompt)  
>>> "Your name and age are: John, {age}"
```

In the above example, we provide only the value for the `name` key word, while the `age` key word remains as it is. This will be helpful when we want to format different key words of `TextPrompt` in different agents.

### Manipulating `TextPrompt` instances

We can perform various string manipulation operations on `TextPrompt` instances, such as concatenation, joining, and applying string methods like Python `str`.

```python
from camel.prompts import TextPrompt

prompt1 = TextPrompt('Hello, {name}!')
prompt2 = TextPrompt('Welcome, {name}!')

# Concatenation
prompt3 = prompt1 + ' ' + prompt2
print(prompt3)  
>>> "Hello, {name}! Welcome, {name}!"

print(isinstance(prompt3, TextPrompt))
>>> True

print(prompt3.key_words)
>>> {'name'}

# Joining
prompt4 = TextPrompt(' ').join([prompt1, prompt2])
print(prompt4)
>>> "Hello, {name}! Welcome, {name}!"

print(isinstance(prompt4, TextPrompt))
>>> True

print(prompt4.key_words)
>>> {'name'}

# Applying string methods
prompt5 = prompt4.upper()
print(prompt5)
>>> "HELLO, {NAME}! WELCOME, {NAME}!"

print(isinstance(prompt5, TextPrompt))
>>> True

print(prompt5.key_words)
>>> {'NAME'}
```

In the above example, we demonstrate concatenation using the `+` operator, joining using the `join` method, and applying the `upper` method to a `TextPrompt` instance. The resulting prompts are also instances of `TextPrompt`.


## 5. Supported Prompt Templates

### 5.1 AISocietyPromptTemplateDict
    
This class defines prompts used in the `AI Society` task with role playing and task handling. For more information about AI Society usage, please refer to corresponding documentation. Template text prompts (defined as attributes) include:

- **GENERATE_ASSISTANTS**: Prompt to list specified number of roles that the AI assistant can play.
- **GENERATE_USERS**: Prompt to list common groups of internet users or occupations.
- **GENERATE_TASKS**: Prompt to list diverse tasks for AI assistants.
- **TASK_SPECIFY_PROMPT**: Specifies a task in more detail with spcified assistant role and user role.
- **ASSISTANT_PROMPT**: Rules of the conversation and
instructions for completing tasks.
- **USER_PROMPT**: Rules of the conversation and instructions for giving instructions to the AI assistant.
- **CRITIC_PROMPT**: Provides selection criteria for critics when choosing proposals from other roles.
### 5.2 CodePromptTemplateDict
    
This class defines prompts for code-related tasks including generating programming languages, domains, tasks, and provide guidelines for AI assistants and users in coding scenarios. Template text prompts include:

- **GENERATE_LANGUAGES**: Prompt to list specified number of computer programming languages.
- **GENERATE_DOMAINS**: Prompt to list common fields of study that programming could help with.
- **GENERATE_TASKS**: Prompt to list diverse tasks that AI assistant can assist for programmers.
- **TASK_SPECIFY_PROMPT**: Prompt to specify a task in more detail.
- **ASSISTANT_PROMPT**: Rules of the conversation and instructions for completing tasks.
- **USER_PROMPT**: Outlines rules for conversation and for users instructing AI coders.
### 5.3 EvaluationPromptTemplateDict
    
This class defines prompts that generate questions to evaluate knowledge. Template text prompt includes:

- **GENERATE_QUESTIONS**: Prompt to generate a set of questions for evaluating knowledge emergence with preliminary knowledge in specific fields. You may provide some examples to enhance its generation quality.
### 5.4 GenerateTextEmbeddingDataPromptTemplateDict
    
This class defines prompts for generating text embedding tasks. These prompts are used to create synthetic data for improving text embeddings. Template text prompts include:

- **GENERATE_TASKS**: Prompt to generate specified number of synthetic text_embedding tasks.
- **ASSISTANT_PROMPT**: Prompt to generate synthetic user queries in JSON format, positive documents, and hard negative documents for specified tasks.
### 5.5 MisalignmentPromptTemplateDict
    
This class defines some misleading prompts to break the AI model alignments. Template text prompts include:

- **DAN_PROMPT**: Prompt to role-playing as DAN (Do Anything Now) for jailbreaking. All the following templates in this dictionary includes DAN_PROMPT.
- **GENERATE_TASKS**: Prompt to list unique malicious tasks that AI assistant can do.
- **TASK_SPECIFY_PROMPT**: Prompt to specify a malicious task in more detail.
- **ASSISTANT_PROMPT**: Rules and instructions for AI assistants when completing tasks.
- **USER_PROMPT**: Outlines rules for conversation and for users instructing AI coders in misalignment tasks.
### 5.6 ObjectRecognitionPromptTemplateDict
    
This class defines a prompt for object recognition tasks:

- **ASSISTANT_PROMPT**: Prompt to detect all objects in images without redundant outputs.
### 5.7 RoleDescriptionPromptTemplateDict
    
This class inherits from `AISocietyPromptTemplateDict` aiming to add descriptions to its prompts.

- **ROLE_DESCRIPTION_PROMPT**: Describes the roles and responsibilities.
### 5.8 SolutionExtractionPromptTemplateDict
    
This class prompts the AI model to focus on finding solution with particular knowledge.

- **ASSISTANT_PROMPT**: Rules of finding and presenting solutions.
### 5.9 TranslationPromptTemplateDict
    
This class prompts a translation AI assistant from English to specified language.

- **ASSISTANT_PROMPT**: Rules of completing translation tasks.
### 5.10 VideoDescriptionPromptTemplateDict
    
This class prompts the AI model to describe a video in words.

- **ASSISTANT_PROMPT**: Rules of completing video description tasks.
