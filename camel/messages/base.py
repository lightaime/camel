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
import base64
import io
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from PIL import Image
from pydantic import BaseModel

from camel.messages import (
    FunctionCallFormatter,
    HermesFunctionFormatter,
    OpenAIAssistantMessage,
    OpenAIMessage,
    OpenAISystemMessage,
    OpenAIUserMessage,
)
from camel.messages.acl_parameter import ACLParameter, Content, Sender
from camel.messages.conversion import ShareGPTMessage
from camel.prompts import CodePrompt, TextPrompt
from camel.types import (
    OpenAIBackendRole,
    OpenAIImageType,
    RoleType,
)
from camel.utils import Constants


class BaseMessage:
    r"""Base class for message objects used in CAMEL chat system.

    Args:
        role_name (str): The name of the user or assistant role.
        role_type (RoleType): The type of role, either `RoleType.ASSISTANT`
            or `RoleType.USER`.
        meta_dict (Optional[Dict[str, Any]]): Additional metadata dictionary
            for the message.
        content (Union[str, Content]): Represents the content of a message,
            which can include text, images, videos, and audio.
        acl_parameter (Optional[ACLParameter]): Access control parameter
            associated with the message, defining sender details.
        parsed (Optional[Union[Type[BaseModel], dict]]): An optional parsed
            object extracted from the content.
        message_id (UUID4): A unique identifier for the message, generated
            automatically and not set by the user.
    """

    def __init__(
        self,
        role_name: str,
        role_type: RoleType,
        content: Union[str, Content],
        acl_parameter: Optional[ACLParameter] = None,
        meta_dict: Optional[Dict[str, Any]] = None,
        parsed: Optional[Union[BaseModel, dict]] = None,
        message_id: Optional[str] = None,
    ):
        self.role_name = role_name
        self.role_type = role_type
        self.acl_parameter = acl_parameter
        self.meta_dict = meta_dict
        self.parsed = parsed
        self.message_id = message_id or str(uuid4())

        """Post-initialization logic for the BaseMessage class.
        - Ensures `content` is properly wrapped as a `Content` object.
        - Initializes `acl_parameter` with a default sender if not specified.
        """
        if isinstance(content, str):
            self.content = Content(text=content)
        elif isinstance(content, Content):
            self.content = content

        if self.acl_parameter is None:
            self.acl_parameter = ACLParameter(
                sender=Sender(name=self.role_name, role_type=self.role_type)
            )
        else:
            self.acl_parameter.sender = Sender(
                name=self.role_name, role_type=self.role_type
            )

    @classmethod
    def make_user_message(
        cls,
        role_name: str,
        content: str,
        meta_dict: Optional[Dict[str, str]] = None,
        acl_parameter: Optional[ACLParameter] = None,
    ) -> "BaseMessage":
        r"""Create a new user message.

        Args:
            role_name (str): The name of the user role.
            content (str): The content of the message.
            meta_dict (Optional[Dict[str, str]]): Additional metadata
                dictionary for the message.
            acl_parameter (Optional[ACLParameter]): Access control parameter
                for the message, defining sender details and permissions.

        Returns:
            BaseMessage: The new user message.
        """
        return cls(
            role_name=role_name,
            role_type=RoleType.USER,
            meta_dict=meta_dict,
            content=content,
            acl_parameter=acl_parameter,
        )

    @classmethod
    def make_assistant_message(
        cls,
        role_name: str,
        content: str,
        meta_dict: Optional[Dict[str, str]] = None,
        acl_parameter: Optional[ACLParameter] = None,
    ) -> "BaseMessage":
        r"""Create a new assistant message.

        Args:
            role_name (str): The name of the assistant role.
            content (str): The content of the message.
            meta_dict (Optional[Dict[str, str]]): Additional metadata
                dictionary for the message.
            acl_parameter (Optional[ACLParameter]): Access control parameter
                for the message, defining sender details and permissions.

        Returns:
            BaseMessage: The new assistant message.
        """
        return cls(
            role_name=role_name,
            role_type=RoleType.ASSISTANT,
            meta_dict=meta_dict,
            content=content,
            acl_parameter=acl_parameter,
        )

    def create_new_instance(self, content: str) -> "BaseMessage":
        r"""Create a new instance of the :obj:`BaseMessage` with updated
        content.

        Args:
            content (str): The new content value.

        Returns:
            BaseMessage: The new instance of :obj:`BaseMessage`.
        """
        return self.__class__(
            role_name=self.role_name,
            role_type=self.role_type,
            meta_dict=self.meta_dict,
            content=content,
        )

    def __add__(self, other: Any) -> Union["BaseMessage", Any]:
        r"""Addition operator override for :obj:`BaseMessage`.

        Args:
            other (Any): The value to be added with.

        Returns:
            Union[BaseMessage, Any]: The result of the addition.
        """
        if isinstance(other, BaseMessage):
            combined_content = self.content.text.__add__(other.content.text)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: '{type(self)}' and "
                f"'{type(other)}'"
            )
        return self.create_new_instance(combined_content)

    def __mul__(self, other: int) -> Union["BaseMessage", Any]:
        r"""Multiplication operator override for :obj:`BaseMessage`.

        Args:
            other (int): The value to be multiplied with.

        Returns:
            Union[BaseMessage, Any]: The result of the multiplication.
        """
        if isinstance(other, int):
            multiplied_content = self.content.text.__mul__(other)
            return self.create_new_instance(multiplied_content)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *: '{type(self)}' and "
                f"'{type(other)}'"
            )

    def __len__(self) -> int:
        r"""Length operator override for :obj:`BaseMessage`.

        Returns:
            int: The length of the content.
        """
        return len(self.content.text)

    def __str__(self) -> str:
        r"""Overridden version of the string function.

        Returns:
            str: Modified string to represent the message content text.
        """
        return self.content.text

    def __contains__(self, item: str) -> bool:
        r"""Contains operator override for :obj:`BaseMessage`.

        Args:
            item (str): The item to check for containment.

        Returns:
            bool: :obj:`True` if the item is contained in the content,
                :obj:`False` otherwise.
        """
        return item in self.content.text

    def extract_text_and_code_prompts(
        self,
    ) -> Tuple[List[TextPrompt], List[CodePrompt]]:
        r"""Extract text and code prompts from the message content.

        Returns:
            Tuple[List[TextPrompt], List[CodePrompt]]: A tuple containing a
                list of text prompts and a list of code prompts extracted
                from the content.
        """
        text_prompts: List[TextPrompt] = []
        code_prompts: List[CodePrompt] = []

        lines = self.content.text.split("\n")
        idx = 0
        start_idx = 0
        while idx < len(lines):
            while idx < len(lines) and (
                not lines[idx].lstrip().startswith("```")
            ):
                idx += 1
            text = "\n".join(lines[start_idx:idx]).strip()
            text_prompts.append(TextPrompt(text))

            if idx >= len(lines):
                break

            code_type = lines[idx].strip()[3:].strip()
            idx += 1
            start_idx = idx
            while not lines[idx].lstrip().startswith("```"):
                idx += 1
            code = "\n".join(lines[start_idx:idx]).strip()
            code_prompts.append(CodePrompt(code, code_type=code_type))

            idx += 1
            start_idx = idx

        return text_prompts, code_prompts

    @classmethod
    def from_sharegpt(
        cls,
        message: ShareGPTMessage,
        function_format: Optional[FunctionCallFormatter[Any, Any]] = None,
        role_mapping=None,
    ) -> "BaseMessage":
        r"""Convert ShareGPT message to BaseMessage or FunctionCallingMessage.
        Note tool calls and responses have an 'assistant' role in CAMEL

        Args:
            message (ShareGPTMessage): ShareGPT message to convert.
            function_format (FunctionCallFormatter, optional): Function call
                formatter to use. (default: :obj:`HermesFunctionFormatter()`.
            role_mapping (Dict[str, List[str, RoleType]], optional): Role
                mapping to use. Defaults to a CAMEL specific mapping.

        Returns:
            BaseMessage: Converted message.
        """
        from camel.messages import FunctionCallingMessage

        if role_mapping is None:
            role_mapping = {
                "system": ["system", RoleType.USER],
                "human": ["user", RoleType.USER],
                "gpt": ["assistant", RoleType.ASSISTANT],
                "tool": ["assistant", RoleType.ASSISTANT],
            }
        role_name, role_type = role_mapping[message.from_]

        if function_format is None:
            function_format = HermesFunctionFormatter()

        # Check if this is a function-related message
        if message.from_ == "gpt":
            func_info = function_format.extract_tool_calls(message.value)
            if (
                func_info and len(func_info) == 1
            ):  # TODO: Handle multiple tool calls
                # Including cleaned content is useful to
                # remind consumers of non-considered content
                clean_content = re.sub(
                    r"<tool_call>.*?</tool_call>",
                    "",
                    message.value,
                    flags=re.DOTALL,
                ).strip()

                return FunctionCallingMessage(
                    role_name=role_name,
                    role_type=role_type,
                    meta_dict=None,
                    content=clean_content,
                    func_name=func_info[0].__dict__["name"],
                    args=func_info[0].__dict__["arguments"],
                )
        elif message.from_ == "tool":
            func_r_info = function_format.extract_tool_response(message.value)
            if func_r_info:
                return FunctionCallingMessage(
                    role_name=role_name,
                    role_type=role_type,
                    meta_dict=None,
                    content="",
                    func_name=func_r_info.__dict__["name"],
                    result=func_r_info.__dict__["content"],
                )

        # Regular message
        return cls(
            role_name=role_name,
            role_type=role_type,
            meta_dict=None,
            content=message.value,
        )

    def to_sharegpt(
        self,
        function_format: Optional[FunctionCallFormatter] = None,
    ) -> ShareGPTMessage:
        r"""Convert BaseMessage to ShareGPT message

        Args:
            function_format (FunctionCallFormatter): Function call formatter
                to use. Defaults to Hermes.
        """

        if function_format is None:
            function_format = HermesFunctionFormatter()

        # Convert role type to ShareGPT 'from' field
        if self.role_type == RoleType.USER:
            from_ = "system" if self.role_name == "system" else "human"
        else:  # RoleType.ASSISTANT
            from_ = "gpt"

        # Function conversion code in FunctionCallingMessage
        return ShareGPTMessage(from_=from_, value=self.content.text)  # type: ignore[call-arg]

    def to_openai_message(
        self,
        role_at_backend: OpenAIBackendRole,
    ) -> OpenAIMessage:
        r"""Converts the message to an :obj:`OpenAIMessage` object.

        Args:
            role_at_backend (OpenAIBackendRole): The role of the message in
                OpenAI chat system.

        Returns:
            OpenAIMessage: The converted :obj:`OpenAIMessage` object.
        """
        if role_at_backend == OpenAIBackendRole.SYSTEM:
            return self.to_openai_system_message()
        elif role_at_backend == OpenAIBackendRole.USER:
            return self.to_openai_user_message()
        elif role_at_backend == OpenAIBackendRole.ASSISTANT:
            return self.to_openai_assistant_message()
        else:
            raise ValueError(f"Unsupported role: {role_at_backend}.")

    def to_openai_system_message(self) -> OpenAISystemMessage:
        r"""Converts the message to an :obj:`OpenAISystemMessage` object.

        Returns:
            OpenAISystemMessage: The converted :obj:`OpenAISystemMessage`
                object.
        """
        return {"role": "system", "content": self.content.text}

    def to_openai_user_message(self) -> OpenAIUserMessage:
        r"""Converts the message to an :obj:`OpenAIUserMessage` object.

        Returns:
            OpenAIUserMessage: The converted :obj:`OpenAIUserMessage` object.
        """
        hybrid_content: List[Any] = []
        hybrid_content.append(
            {
                "type": "text",
                "text": self.content.text,
            }
        )
        if self.content.image_list and len(self.content.image_list) > 0:
            for image in self.content.image_list:
                if image.format is None:
                    raise ValueError(
                        f"Image's `format` is `None`, please "
                        f"transform the `PIL.Image.Image` to  one of "
                        f"following supported formats, such as "
                        f"{list(OpenAIImageType)}"
                    )

                image_type: str = image.format.lower()
                if image_type not in OpenAIImageType:
                    raise ValueError(
                        f"Image type {image.format} "
                        f"is not supported by OpenAI vision model"
                    )
                with io.BytesIO() as buffer:
                    image.save(fp=buffer, format=image.format)
                    encoded_image = base64.b64encode(buffer.getvalue()).decode(
                        "utf-8"
                    )
                image_prefix = f"data:image/{image_type};base64,"
                hybrid_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{image_prefix}{encoded_image}",
                            "detail": self.content.image_detail,
                        },
                    }
                )

        if self.content.video_bytes:
            import imageio.v3 as iio

            base64Frames: List[str] = []
            frame_count = 0
            # read video bytes
            video = iio.imiter(
                self.content.video_bytes,
                plugin=Constants.VIDEO_DEFAULT_PLUG_PYAV,
            )

            for frame in video:
                frame_count += 1
                if (
                    frame_count % Constants.VIDEO_IMAGE_EXTRACTION_INTERVAL
                    == 0
                ):
                    # convert frame to numpy array
                    frame_array = np.asarray(frame)
                    frame_image = Image.fromarray(frame_array)

                    # Get the dimensions of the frame
                    width, height = frame_image.size

                    # resize the frame to the default image size
                    new_width = Constants.VIDEO_DEFAULT_IMAGE_SIZE
                    aspect_ratio = width / height
                    new_height = int(new_width / aspect_ratio)
                    resized_img = frame_image.resize((new_width, new_height))

                    # encode the image to base64
                    with io.BytesIO() as buffer:
                        image_format = OpenAIImageType.JPEG.value
                        image_format = image_format.upper()
                        resized_img.save(fp=buffer, format=image_format)
                        encoded_image = base64.b64encode(
                            buffer.getvalue()
                        ).decode("utf-8")

                    base64Frames.append(encoded_image)

            for encoded_image in base64Frames:
                item = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                        "detail": self.content.video_detail,
                    },
                }

                hybrid_content.append(item)

        if len(hybrid_content) > 1:
            return {
                "role": "user",
                "content": hybrid_content,
            }
        # This return just for str message
        else:
            return {
                "role": "user",
                "content": self.content.text,
            }

    def to_openai_assistant_message(self) -> OpenAIAssistantMessage:
        r"""Converts the message to an :obj:`OpenAIAssistantMessage` object.

        Returns:
            OpenAIAssistantMessage: The converted :obj:`OpenAIAssistantMessage`
                object.
        """
        return {"role": "assistant", "content": self.content.text}

    def to_dict(self) -> Dict:
        r"""Converts the message to a dictionary.

        Returns:
            dict: The converted dictionary.
        """
        return {
            "role_name": self.role_name,
            "role_type": self.role_type,
            **(self.meta_dict or {}),
            "content": self.content.to_dict()
            if self.content.image_list or self.content.video_bytes
            else self.content.text,
            "acl_parameter": self.acl_parameter.model_dump_json(),  # type: ignore[union-attr]
        }
