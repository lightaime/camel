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
import asyncio
from typing import TYPE_CHECKING, List, Optional, Union

from camel.agents import ChatAgent
from camel.bots import DiscordApp
from camel.retrievers import AutoRetriever
from camel.types import StorageType

try:
    from unstructured.documents.elements import Element
except ImportError:
    Element = None

if TYPE_CHECKING:
    from discord import Message


class BotAgent:
    def __init__(
        self,
        contents: Union[str, List[str], "Element", List["Element"]] = None,
        auto_retriever: Optional[AutoRetriever] = None,
        similarity_threshold: float = 0.5,
        vector_storage_local_path: str = "local_data/",
        top_k: int = 1,
        return_detailed_info: bool = True,
    ):
        r"""Initialize the BotAgent instance.

        Args:
            contents (Union[str, List[str], Element, List[Element]], optional)
                : The content to be retrieved.
            auto_retriever (Optional[AutoRetriever], optional): An instance of
                AutoRetriever for vector search.
            similarity_threshold (float): Threshold for vector similarity when
                retrieving content.
            vector_storage_local_path (str): Path to local vector storage for
                the retriever.
            top_k (int): Number of top results to retrieve.
            return_detailed_info (bool): Whether to return detailed
                information from the retriever.
        """

        assistant_sys_msg = '''
            Objective: 
                You are a customer service bot designed to assist users
                with inquiries related to our open-source project. 
                Your responses should be informative, concise, and helpful.

            Instructions:
                Understand User Queries: Carefully read and understand the
                        user's question. Focus on keywords and context to
                        determine the user's intent.
                Search for Relevant Information: Use the provided dataset
                        and refer to the RAG (file to find answers that 
                        closely match the user's query. The RAG file 
                        contains detailed interactions and should be your 
                        primary resource for crafting responses.
                Provide Clear and Concise Responses: Your answers should 
                        be clear and to the point. Avoid overly technical
                        language unless the user's query indicates 
                        familiarity with technical terms.
                Encourage Engagement: Where applicable, encourage users
                        to contribute to the project or seek further
                        assistance.

            Response Structure:
                Greeting: Begin with a polite greeting or acknowledgment.
                Main Response: Provide the main answer to the user's query.
                Additional Information: Offer any extra tips or direct the
                        user to additional resources if necessary.
                Closing: Close the response politely, encouraging
                        further engagement if appropriate.
            bd
            Tone:
                Professional: Maintain a professional tone that 
                        instills confidence in the user.
                Friendly: Be approachable and friendly to make users 
                        feel comfortable.
                Helpful: Always aim to be as helpful as possible,
                        guiding users to solutions.        
        '''

        self._agent = ChatAgent(
            assistant_sys_msg,
        )

        self._auto_retriever = None
        self._contents = contents
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold
        self._return_detailed_info = return_detailed_info

        self._auto_retriever = auto_retriever or AutoRetriever(
            vector_storage_local_path=vector_storage_local_path,
            storage_type=StorageType.QDRANT,
        )

    async def process(self, message: str) -> str:
        r"""Process the user message, retrieve relevant content, and generate
        a response.

        Args:
            message (str): The user's query message.

        Returns:
            str: The assistant's response message.
        """
        user_raw_msg = message
        print("User message:", user_raw_msg)
        if self._auto_retriever:
            retrieved_content = self._auto_retriever.run_vector_retriever(
                query=user_raw_msg,
                contents=self._contents,
                top_k=self._top_k,
                similarity_threshold=self._similarity_threshold,
                return_detailed_info=self._return_detailed_info,
            )
            user_raw_msg = (
                f"Here is the query to you: {user_raw_msg}\n"
                f"Based on the retrieved content: {retrieved_content}, \n"
                f"answer the query"
            )

        assistant_response = self._agent.step(user_raw_msg)
        return assistant_response.msg.content.text


class DiscordBot(DiscordApp):
    def __init__(
        self,
        agent: BotAgent,
        token: Optional[str] = None,
        channel_ids: Optional[list[int]] = None,
    ):
        r"""Initializes the DiscordBot instance to handle Discord messages and
        communicate with BotAgent.

        Args:
            agent (BotAgent): The BotAgent responsible for processing messages
                and generating responses.
            token (Optional[str]): The token used to authenticate the bot with
                Discord.
            channel_ids (Optional[list[int]]): A list of Discord channel IDs
                where the bot is allowed to interact.
        """
        super().__init__(token=token, channel_ids=channel_ids)
        self.agent: BotAgent = agent

    async def on_message(self, message: 'Message') -> None:
        r"""Event handler for received messages. This method processes incoming
        messages, checks whether the message is from the bot itself, and
        determines whether the bot should respond based on channel ID and
        mentions. Then processes the message using the BotAgent, and responds.

        Args:
            message (discord.Message): The received message object.
        """
        # If the message author is the bot itself,
        # do not respond to this message
        if message.author == self._client.user:
            return

        # If allowed channel IDs are provided,
        # only respond to messages in those channels
        if self.channel_ids and message.channel.id not in self.channel_ids:
            return

        # Only respond to messages that mention the bot
        if not self._client.user or not self._client.user.mentioned_in(
            message
        ):
            return

        user_raw_msg = message.content
        response = await self.agent.process(user_raw_msg)
        await message.channel.send(response)


async def process_message(agent: BotAgent, msg_queue: asyncio.Queue):
    r"""Continuously processes messages from the queue and sends responses.

    This function waits for new messages in the queue, processes each message
    using the `BotAgent` instance, and sends the response back to Discord.

    Args:
        agent (BotAgent): An instance of `BotAgent` that processes the received
            messages.
        msg_queue (asyncio.Queue): The queue from which messages are retrieved
            for processing.
    """
    while True:
        message: "Message" = await msg_queue.get()
        user_raw_msg = message.content

        # Process the message using the agent and get the response
        response = await agent.process(user_raw_msg)
        # message.reply(response)
        await message.channel.send(response)
        msg_queue.task_done()


async def main():
    agent = BotAgent()

    # Initialize the DiscordBot with the message queue
    discord_bot = DiscordBot(agent=agent)

    await discord_bot.start()


if __name__ == "__main__":
    asyncio.run(main())
