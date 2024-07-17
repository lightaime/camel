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
from __future__ import annotations

import asyncio
import random
from collections import deque
from typing import Dict, List, Optional

from camel.agents.manager_agent import ChatAgent, ManagerAgent
from camel.messages.base import BaseMessage
from camel.tasks.task import Task
from camel.workforce import BaseWorkforce, LeafWorkforce
from camel.workforce.task_channel import Packet, PacketStatus, TaskChannel
from camel.workforce.utils import get_workforces_info


class InternalWorkforce(BaseWorkforce):
    r"""A workforce that manages multiple workforces and agents. It will
    split the task it receives into subtasks and assign them to the
    workforces/agents under it, and also handles the situation when the task
    fails.

    Args:
        workforce_id (str): ID for the workforce.
        description (str): Description of the workforce.
        workforces (List[BaseWorkforce]): List of workforces under this
            workforce.
        manager_agent_config (dict): Configuration parameters for the
            manager agent.
        channel (TaskChannel): Communication channel for the workforce.
    """

    def __init__(
        self,
        workforce_id: str,
        description: str,
        workforces: List[BaseWorkforce],
        manager_agent_config: dict,
        task_agent_config: dict,
        initial_task: Task,
        channel: TaskChannel,
    ) -> None:
        super().__init__(workforce_id, description, channel)
        self.workforces = workforces
        self.manager_agent = ManagerAgent()
        sys_msg = BaseMessage.make_assistant_message(
            role_name="task_planner",
            content="You are going to decompose tasks.",
        )
        self.task_agent = ChatAgent(sys_msg)
        self.workforce_info = get_workforces_info(workforces)
        self.initial_task = initial_task
        # self.task_collection = TaskManager(self.initial_task)

    def assign_task(
        self,
        task: Task,
        workforce_info: dict,
        failed_log: Optional[str] = None,
    ) -> str:
        r"""Assigns a task to an internal workforce if capable, otherwise
        create a new workforce.

        Parameters:
            task (Task): The task to be assigned.
            failed_log (Optional[str]): Optional log of a previous failed
                attempt.
            workforce_info (str): Information about the internal workforce.

        Returns:
            str: ID of the assigned workforce.
        """
        # Note: The following are mock outputs for workforce example
        return random.choice(['1', '2', '3'])

    async def create_workforce_for_task(self, task: Task) -> LeafWorkforce:
        r"""Creates a new workforce for a given task. One of the actions that
        the manager agent can take when a task has failed.

        Args:
            task (Task): The task for which the workforce is created.

        Returns:
            LeafWorkforce: The created workforce.
        """
        # Note: The following are mock outputs for workforce example
        sys_msg = BaseMessage.make_assistant_message(
            role_name="product owner",
            content="You are familiar with internet.",
        )
        new_agent = ChatAgent(sys_msg)
        new_workforce = LeafWorkforce(
            str(len(self.workforces) + 1), 'new_agent', new_agent, self.channel
        )
        self.workforces.append(new_workforce)
        task = [asyncio.create_task(new_workforce.start())]
        print('start listening...')
        await asyncio.gather(*task)
        return new_workforce

    async def decompose_task_to_packets(
        self, task: Task, failed: bool
    ) -> List[Packet]:
        r"""Decompose a task into a packet and set dependencies."""
        packet_lst: List[Packet] = []
        dependencies: List[str] = []
        subtasks = task.decompose(self.task_agent)
        task.subtasks = subtasks

        for subtask in subtasks:
            # get assign_id by calling assign_task function.
            # If failed, create a workforce.
            if failed:
                create_result = await self.create_workforce_for_task(subtask)
                assign_id = create_result.workforce_id
                print('assign_id:', assign_id)
            else:
                assign_id = self.assign_task(
                    task=subtask, workforce_info=self.workforce_info
                )
            # set status as Taskstatus.ASSIGNED
            packet = Packet(
                subtask, self.workforce_id, assign_id, list(dependencies)
            )
            print(subtask, 'dependencies:', dependencies)

            packet_lst.append(packet)
            # get dependencies via the index of the subtask list
            dependencies.append(subtask.id)

        return packet_lst

    async def get_finished_task(self) -> Packet:
        r"""Get the task that's published by the workforce and just get
        finished from the channel."""
        return await self.channel.get_returned_task_by_publisher(
            self.workforce_id
        )

    async def delete_composed_subtask(self, parent_packet: Packet) -> None:
        for subtask in parent_packet.task.subtasks:
            await self.channel.remove_task(subtask.id)

    async def send_packet(self) -> None:
        next_packet = self.pending_packets[0]
        if next_packet.status == PacketStatus.FAILED:
            next_packet.task.compose(self.task_agent)
            next_packet.status = PacketStatus.COMPLETED
            await self.delete_composed_subtask(next_packet)

        await self.channel.send_task(next_packet)

    async def listening(self) -> Dict[str, Packet]:
        r"""Continuously listen to the channel, post task to the channel and
        track the status of posted tasks.
        """
        print(f'listening {self.workforce_id}')
        if self.initial_task is not None:
            # TODO: split the initial task into subtasks and assign the
            #  first one to the workforces
            packets = await self.decompose_task_to_packets(
                task=self.initial_task, failed=False
            )
            # Insert packets at the tail of the queue and send to channel
            self.pending_packets.extend(packets)
            await self.send_packet()

            # initial_packet is set to failed to trigger the compose
            initial_packet = Packet(
                self.initial_task,
                self.workforce_id,
                self.workforce_id,
                None,
                PacketStatus.FAILED,
            )
            self.pending_packets.append(initial_packet)

        while self.running and self.pending_packets:
            finished_task = await self.get_finished_task()
            if finished_task.status == PacketStatus.COMPLETED:
                # close the task, indicating that the task is completed and
                # known by the manager
                self.pending_packets.popleft()
                await self.channel.return_task(
                    finished_task.task.id, PacketStatus.CLOSED
                )
                if not self.pending_packets:
                    break
                # TODO: mark the task as completed, assign the next task
                await self.send_packet()

            elif finished_task.status == PacketStatus.FAILED:
                # remove the failed task from the channel
                await self.channel.remove_task(finished_task.task.id)
                # TODO: apply action when the task fails

                packets = await self.decompose_task_to_packets(
                    task=finished_task.task, failed=True
                )
                # Insert packets at the head of the queue
                self.pending_packets.extendleft(reversed(packets))

                await self.send_packet()
        await self.stop()
        return self.channel._task_dict

    async def start(self) -> None:
        r"""Start the internal workforce and all the workforces under it."""
        self.running = True
        self.pending_packets: deque = deque()
        tasks = [
            asyncio.create_task(workforce.start())
            for workforce in self.workforces
        ]
        print('start listening...')
        tasks.append(asyncio.create_task(self.listening()))
        results = await asyncio.gather(*tasks)
        listening_result = results[-1]
        print(listening_result['0'].result)

    async def stop(self) -> None:
        r"""Stop the internal workforce and all the workforces under it."""
        self.running = False
        for workforce in self.workforces:
            await workforce.stop()
