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
import logging
from typing import TYPE_CHECKING, List, Optional, Union

from slack_bolt.context.async_context import AsyncBoltContext
from slack_bolt.context.say.async_say import AsyncSay
from slack_sdk.web.async_client import AsyncWebClient

from camel.agents import ChatAgent
from camel.bots import SlackApp, SlackEventBody
from camel.retrievers import AutoRetriever
from camel.types import StorageType

if TYPE_CHECKING:
    from unstructured.documents.elements import Element

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        content = '''
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
            content,
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
        user_raw_msg = message
        logger.info("User message:", user_raw_msg)
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


class SlackBot(SlackApp):
    def __init__(
        self,
        agent: BotAgent,
        token: Optional[str] = None,
        scopes: Optional[str] = None,
        signing_secret: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        """Initializes the SlackBot instance to handle Slack messages and
        communicate with BotAgent.

        Args:
            agent (BotAgent): The BotAgent responsible for processing messages
                and generating responses.
            token (Optional[str]): Slack API token for authentication.
            scopes (Optional[str]): Slack app scopes for permissions.
            signing_secret (Optional[str]): Signing secret for verifying Slack
                requests.
            client_id (Optional[str]): Slack app client ID.
            client_secret (Optional[str]): Slack app client secret.
        """
        super().__init__(
            token=token,
            scopes=scopes,
            signing_secret=signing_secret,
            client_id=client_id,
            client_secret=client_secret,
        )
        self.agent: BotAgent = agent

    async def on_message(
        self,
        context: "AsyncBoltContext",
        client: "AsyncWebClient",
        event: dict,
        body: dict,
        say: "AsyncSay",
    ):
        """Handles incoming Slack messages, processes them with BotAgent, and
        responds.

        Args:
            context (AsyncBoltContext): Context object that contains
                information about the Slack event.
            client (AsyncWebClient): Slack Web API client for making API
                requests.
            event (dict): Event data containing details of the incoming Slack
                message.
            body (dict): Full request body from Slack.
            say (AsyncSay): A function to send a response back to the Slack
                channel.
        """
        await context.ack()
        event_body = SlackEventBody(**body)
        user_raw_msg = event_body.event.text
        response = await agent.process(user_raw_msg)
        await say(response)


if __name__ == "__main__":
    agent = BotAgent()

    slack_bot = SlackBot(agent=agent)
    slack_bot.run(3000)
