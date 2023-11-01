# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict
from uuid import UUID, uuid4

from camel.messages.base import BaseMessage
from camel.messages.func_message import FunctionCallingMessage
from camel.typing import OpenAIBackendRole


@dataclass(frozen=True)
class MemoryRecord():
    r"""
    The basic message storing unit in the CAMEL memory system.
    """
    message: BaseMessage
    role_at_backend: OpenAIBackendRole
    uuid: UUID = field(default_factory=uuid4)
    extra_info: Dict[str, str] = field(default_factory=dict)

    _MESSAGE_TYPES: ClassVar[dict] = {
        "BaseMessage": BaseMessage,
        "FunctionCallingMessage": FunctionCallingMessage
    }

    @classmethod
    def from_dict(cls, record_dict: Dict[str, Any]):
        r"""Reconstruct a :obj:`MemoryRecord` from the input dict.

        Args:
            record_dict(Dict[str, Any]): A dict generated by :meth:`to_dict`.
        """
        message_cls = cls._MESSAGE_TYPES[record_dict["message"]["__class__"]]
        kwargs: Dict = record_dict["message"].copy()
        kwargs.pop("__class__")
        reconstructed_message = message_cls(**kwargs)
        return cls(
            uuid=UUID(record_dict["uuid"]),
            message= reconstructed_message,
            role_at_backend=record_dict["role_at_backend"],
            extra_info=record_dict["extra_info"],
        )

    def to_dict(self):
        r"""Convert the :obj:`MemoryRecord` to a dict for serialization purposes.
        """
        return {
            "uuid": str(self.uuid),
            "message": {
                "__class__": self.message.__class__.__name__,
                **asdict(self.message)
            },
            "role_at_backend": self.role_at_backend,
            "extra_info": self.extra_info
        }

    def to_openai_message(self):
        r"""Converts the record to an :obj:`OpenAIMessage` object.
        """
        return self.message.to_openai_message(self.role_at_backend.value)


@dataclass(frozen=True)
class ContextRecord():
    r"""The result of memory retrieving.
    """
    memory_record: MemoryRecord
    importance: float
