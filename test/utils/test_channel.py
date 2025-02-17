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

from camel.utils.info_channel import Channel, ChannelManagement

check = False

class A:
    def __init__(self, name):
        self.count = 0
        self.name = name

    async def task(self, channel):
         while True:
            self.count += 1
            await channel.send_to("B", f"{self.name} says hello {self.count}")
            await channel.receive_from("B")
            if self.count == 1:
                await channel.send_to("B", "stop")
                break


class B:
    def __init__(self, name):
        self.count = 0
        self.name = name

    async def task(self, channel):
        while True:
            message = await channel.receive_from("A")
            self.count += 1
            await channel.send_to("A", f"{self.name} says hello {self.count}")
            if message == "stop":
                global check
                check = True
                break


async def main():
    manager = ChannelManagement()
    a = A("A")
    b = B("B")
    channel_a = manager.register_channel(a.name)
    channel_b = manager.register_channel(b.name)
    manager.register_task(a.task, channel=channel_a)
    manager.register_task(b.task, channel=channel_b)

    await manager.run()

    assert isinstance(channel_a, Channel)
    assert isinstance(channel_b, Channel)
    assert manager.channels["A"] == channel_a
    assert manager.channels["B"] == channel_b
    assert check


def test_channel():
    asyncio.run(main())