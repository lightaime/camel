# Message

## 1. Concept

The `BaseMessage` class is the base class for message objects used in the CAMEL chat system. It is designed to provide a consistent structure for the messages in the system and allow for easy conversion between different message types.

All the codes are also available on colab notebook [here](https://colab.research.google.com/drive/1qyi4bnAbnYink-FKaAlJG9OipyEWXEsT?usp=sharing).

## 2. Get Started

### 2.1 Creating a `BaseMessage` Instance

To create a `BaseMessage` instance, you need to provide the following arguments:

- `role_name`: The name of the user or assistant role.
- `role_type`: The type of role, either `RoleType.ASSISTANT` or `RoleType.USER`.
- `meta_dict`: An optional metadata dictionary for the message.
- `content`: The content of the message.

Below are optional arguments you can pass:

- `video_bytes`: Optional bytes of a video associated with the message.
- `image_list`: Optional list of PIL Image objects associated with the message.
- `image_detail`: Detail level of the images associated with the message. Default is "auto".
- `video_detail`: Detail level of the videos associated with the message. Default is "low".

Here's an example of creating a `BaseMessage` instance:

```python
from camel.messages import BaseMessage
from camel.types import RoleType

message = BaseMessage(
    role_name="test_user",
    role_type=RoleType.USER,
    content="test content"
)
```

Additionally, the BaseMessage class provides class methods to easily create user and assistant agent messages:

1. Creating a user agent message:

    ```python
    from camel.messages import BaseMessage

    user_message = BaseMessage.make_user_message(
        role_name="user_name", 
        content="test content for user",
    )
    ```

2. Creating an assistant agent message:

    ```python
    from camel.messages import BaseMessage
    
    assistant_message = BaseMessage.make_assistant_message(
        role_name="assistant_name",
        content="test content for assistant",
    )
    ```

### 2.2 Using the Methods of the `BaseMessage` Class

The `BaseMessage` class offers several methods:

1. Creating a new instance with updated content:

    ```python
    new_message = message.create_new_instance("new test content")
    print(isinstance(new_message, BaseMessage))
    >>> True
    ```

2. Converting to an `OpenAIMessage` object:

    ```python
    openai_message = message.to_openai_message(role_at_backend=OpenAIBackendRole.USER)
    print(openai_message == {"role": "user", "content": "test content"})
    >>> True
    ```

3. Converting to an `OpenAISystemMessage` object:

    ```python
    openai_system_message = message.to_openai_system_message()
    print(openai_system_message == {"role": "system", "content": "test content"})
    >>> True
    ```

4. Converting to an `OpenAIUserMessage` object:

    ```python
    openai_user_message = message.to_openai_user_message()
    print(openai_user_message == {"role": "user", "content": "test content"})
    >>> True
    ```

5. Converting to an `OpenAIAssistantMessage` object:

    ```python
    openai_assistant_message = message.to_openai_assistant_message()
    print(openai_assistant_message == {"role": "assistant", "content": "test content"})
    >>> True
    ```

6. Converting to a dictionary:

    ```python
    message_dict = message.to_dict()
    print(message_dict == {
        "role_name": "test_user",
        "role_type": "USER",
        "content": "test content"
    })
    >>> True
    ```


These methods allow you to convert a `BaseMessage` instance into different message types depending on your needs.

## 3. Give `BaseMessage` to `ChatAgent`
```python
from io import BytesIO

import requests
from PIL import Image

from camel.agents import ChatAgent
from camel.messages import BaseMessage
# URL of the image
url = "https://raw.githubusercontent.com/camel-ai/camel/master/misc/logo_light.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Define system message
sys_msg = BaseMessage.make_assistant_message(
    role_name="Assistant",
    content="You are a helpful assistant.",
)

# Set agent
camel_agent = ChatAgent(system_message=sys_msg)

# Set user message
user_msg = BaseMessage.make_user_message(
    role_name="User", content="""what's in the image?""", image_list=[img]
)

# Get response information
response = camel_agent.step(user_msg)
print(response.msgs[0].content)
>>> The image features a logo for "CAMEL-AI." It includes a stylized purple camel graphic alongside the text "CAMEL-AI," which is also in purple. The design appears modern and is likely related to artificial intelligence.
```

## 4. Conclusion
In this session, we introduced the `BaseMessage` class and its conversion to different types of messages. These components play essential roles in the CAMEL chat system, facilitating the creation, management, and interpretation of messages with clarity.