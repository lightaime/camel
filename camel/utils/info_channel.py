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
import asyncio
from typing import Any, Dict


class Channel:
    def __init__(self, name: str):
        r"""Channel class for communication between different tasks and agents.

        Args:
            name (str): name of the channel
        """
        self.name = name
        self.input_queues: Dict[str, asyncio.Queue]= {}
        self.output_queues: Dict[str, asyncio.Queue] = {}

    def connect(self, channel: "Channel"):
        r"""Connect to another channel.

        Args:
            channel(Channel): another channel to connect
        """
        self.input_queues[channel.name] = asyncio.Queue()
        channel.input_queues[self.name] = asyncio.Queue()
        self.output_queues[channel.name] = channel.input_queues[self.name]
        channel.output_queues[self.name] = self.input_queues[channel.name]

    async def receive_from(self, name: str):
        r"""Receive message from another channel.

        Args:
            name (str): name of the channel to receive from
        """
        if name not in self.input_queues:
            raise ValueError(f"Channel {name} not connected")
        return await self.input_queues[name].get()

    async def send_to(self, name: str, message: Any):
        r"""Send message to another channel.

        Args:
            name (str): name of the channel to send to
            message (Any): message to send
        """
        if name not in self.output_queues:
            raise ValueError(f"Channel {name} not connected")
        await self.output_queues[name].put(message)

    def empty(self, name: str):
        r"""Check if the queue is empty.

        Args:
            name (str): name of the channel to check
        """
        return self.input_queues[name].empty()


class ChannelManagement:
    def __init__(self):
        r"""Channel management class for managing channels and tasks.
        """
        self.tasks = []
        self.channels = {}

    def register_channel(self, name: str):
        r"""Register a channel.

        Args:
            name (str): name of the channel
        """
        if name not in self.channels:
            channel = Channel(name)
            self.channels[name] = channel
            for _name in self.channels:
                if _name != name:
                    channel.connect(self.channels[_name])
        else:
            channel = self.channels[name]

        return channel

    def register_task(self, func, **params):
        r"""Register a task.

        Args:
            func: function to run
            **params: parameters for the function
        """
        self.tasks.append(func(**params))

    async def run(self):
        r"""Run all the tasks.
        """
        await asyncio.gather(*self.tasks)