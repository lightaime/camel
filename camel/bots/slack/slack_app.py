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
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

from slack_sdk.oauth.installation_store.async_installation_store import (
    AsyncInstallationStore,
)

from camel.bots.slack.models import (
    SlackAppMentionEventBody,
    SlackAppMentionEventProfile,
    SlackEventBody,
    SlackEventProfile,
)
from camel.utils import dependencies_required

if TYPE_CHECKING:
    from slack_bolt.context.async_context import AsyncBoltContext
    from slack_bolt.context.say.async_say import AsyncSay
    from slack_sdk.web.async_client import AsyncWebClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlackApp:
    r"""Represents a Slack app that is powered by a Slack Bolt `AsyncApp`.

    This class is responsible for initializing and managing the Slack
    application by setting up event handlers, running the app server, and
    handling events such as messages and mentions from Slack.

    Args:
        token (Optional[str]): Slack API token for authentication.
        scopes (Optional[str]): Slack app scopes for permissions.
        signing_secret (Optional[str]): Signing secret for verifying Slack
            requests.
        client_id (Optional[str]): Slack app client ID.
        client_secret (Optional[str]): Slack app client secret.
        redirect_uri_path (str): The URI path for OAuth redirect, defaults to
            "/slack/oauth_redirect".
        installation_store (Optional[AsyncInstallationStore]): The installation
            store for handling OAuth installations.
        socket_mode (bool): A flag to enable socket mode for the Slack app,
    """

    @dependencies_required('slack_bolt')
    def __init__(
        self,
        token: Optional[str] = None,
        app_token: Optional[str] = None,
        scopes: Optional[str] = None,
        signing_secret: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri_path: str = "/slack/oauth_redirect",
        installation_store: Optional[AsyncInstallationStore] = None,
        socket_mode: bool = False,
        oauth_mode: bool = False,
    ) -> None:
        r"""Initializes the SlackApp instance by setting up the Slack Bolt app
        and configuring event handlers and OAuth settings.

        Args:
            token: The Slack bot token. (xoxb-)
            app_token (Optional[str]): The Slack app token. (xapp-)
            scopes (Optional[str]): The scopes for Slack app permissions.
            signing_secret (Optional[str]): The signing secret for verifying
                requests.
            client_id (Optional[str]): The Slack app client ID.
            client_secret (Optional[str]): The Slack app client secret.
            redirect_uri_path (str): The URI path for handling OAuth redirects
                (default is "/slack/oauth_redirect").
            installation_store (Optional[AsyncInstallationStore]): An optional
                installation store for OAuth installations.
            socket_mode (bool): A flag to enable socket mode for the Slack app.
                (default is False). If want to use socket mode, set it to True.
                And Open the socket mode in the Slack API Applications.
                url: https://api.slack.com/apps/
            oauth_mode (bool): A flag to enable OAuth mode for the Slack app.
        """
        from slack_bolt.adapter.starlette.async_handler import (
            AsyncSlackRequestHandler,
        )

        self.token = token or os.getenv("SLACK_TOKEN")
        self.scopes: Optional[str] = scopes or os.getenv("SLACK_SCOPES")
        self.signing_secret: Optional[str] = signing_secret or os.getenv(
            "SLACK_SIGNING_SECRET"
        )
        self.client_id: Optional[str] = client_id or os.getenv(
            "SLACK_CLIENT_ID"
        )
        self.client_secret: Optional[str] = client_secret or os.getenv(
            "SLACK_CLIENT_SECRET"
        )
        self.app_token: Optional[str] = app_token or os.getenv(
            "SLACK_APP_TOKEN"
        )
        self.custom_handler: Optional[
            Callable[[SlackEventProfile, SlackEventBody], Any]
        ] = None
        self.socket_mode: bool = socket_mode
        self.oauth_mode: bool = oauth_mode
        self.handlers: Dict[str, Callable[[SlackEventProfile, SlackEventBody], Any]] = {}
        if not self.socket_mode:
            if not all([self.token, self.signing_secret]):
                raise ValueError(
                    "`SLACK_TOKEN` and `SLACK_SIGNING_SECRET` environment "
                    "variables must be set. Get it here: "
                    "`https://api.slack.com/apps`."
                )
        else:
            if not all([self.app_token, self.token]):
                raise ValueError(
                    "`SLACK_APP_TOKEN` and `SLACK_TOKEN` environment "
                    "variables must be set. Get it here: "
                    "`https://api.slack.com/apps`."
                )
        self._installation_store = installation_store
        self._app = self._initialize_app(
            installation_store=installation_store,
            redirect_uri_path=redirect_uri_path,
        )
        if not self.socket_mode:
            self._handler = AsyncSlackRequestHandler(self._app)
        else:
            self._handler = None
        self.setup_handlers()

    def _initialize_app(
        self,
        installation_store: Optional[AsyncInstallationStore] = None,
        redirect_uri_path: str = "/slack/oauth_redirect",
    ) -> Any:
        r"""Initializes the Slack Bolt app with the given installation store
        and OAuth settings.

        Args:
            installation_store (Optional[AsyncInstallationStore]): The
                installation store for handling OAuth installations.
            redirect_uri_path (str): The URI path for handling OAuth redirects
                (default is "/slack/oauth_redirect").

        Returns:
            Any: The initialized Slack Bolt app.
        """
        from slack_bolt.app.async_app import AsyncApp
        from slack_bolt.oauth.async_oauth_settings import AsyncOAuthSettings

        if self.oauth_mode:
            return AsyncApp(
                oauth_settings=AsyncOAuthSettings(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    scopes=self.scopes,
                    redirect_uri_path=redirect_uri_path,
                    installation_store=installation_store,
                ),
                logger=logger,
                signing_secret=self.signing_secret,
                token=self.token,
            )
        else:
            return AsyncApp(
                logger=logger,
                signing_secret=self.signing_secret,
                token=self.token,
            )

    def setup_handlers(self) -> None:
        r"""Sets up the event handlers for Slack events, such as `app_mention`
        and `message`.

        This method registers the `app_mention` and `on_message` event handlers
        with the Slack Bolt app to respond to Slack events.
        """
        self._app.event("app_mention")(self.app_mention)
        self._app.event("message")(self.on_message)

    async def start(self) -> None:
        r"""Starts the Slack Bolt app asynchronously."""
        from slack_bolt.adapter.socket_mode.async_handler import (
            AsyncSocketModeHandler,
        )

        if self.socket_mode:
            self._handler = AsyncSocketModeHandler(self._app, self.app_token)
            await self._handler.start_async()
        else:
            logger.info(
                "please use `start()` instead of `run()` for HTTP server."
            )
            return

    def run(
        self,
        port: int = 3000,
        path: str = "/slack/events",
        host: Optional[str] = None,
    ) -> None:
        r"""Starts the Slack Bolt app server to listen for incoming Slack
        events.

        Args:
            port (int): The port on which the server should run (default is
                3000).
            path (str): The endpoint path for receiving Slack events (default
                is "/slack/events").
            host (Optional[str]): The hostname to bind the server (default is
                None).
        """
        self._app.start(port=port, path=path, host=host)

    async def app_mention(
        self,
        context: "AsyncBoltContext",
        client: "AsyncWebClient",
        event: Dict[str, Any],
        body: Dict[str, Any],
        say: "AsyncSay",
    ) -> None:
        r"""Event handler for `app_mention` events.

        This method is triggered when someone mentions the app in Slack.

        Args:
            context (AsyncBoltContext): The Slack Bolt context for the event.
            client (AsyncWebClient): The Slack Web API client.
            event (Dict[str, Any]): The event data for the app mention.
            body (Dict[str, Any]): The full request body from Slack.
            say (AsyncSay): A function to send a response back to the channel.
        """
        await context.ack()
        token = await self._get_bot_token(context)
        event_profile = SlackAppMentionEventProfile(**event)
        event_body = SlackAppMentionEventBody(**body)
        logger.info(f"app_mention, context: {context}")
        logger.info(f"app_mention, client: {client}")
        logger.info(f"app_mention, event_profile: {event_profile}")
        logger.info(f"app_mention, event_body: {event_body}")
        logger.info(f"app_mention, say: {say}")
        handler = self.handlers.get("mention", self.handlers.get('default'))
        if handler:
            response = handler(event_profile, event_body)
            await say(text=response, token=token)

    async def on_message(
        self,
        context: "AsyncBoltContext",
        client: "AsyncWebClient",
        event: Dict[str, Any],
        body: Dict[str, Any],
        say: "AsyncSay",
    ) -> None:
        r"""Event handler for `message` events.

        This method is triggered when the app receives a message in Slack.

        Args:
            context (AsyncBoltContext): The Slack Bolt context for the event.
            client (AsyncWebClient): The Slack Web API client.
            event (Dict[str, Any]): The event data for the message.
            body (Dict[str, Any]): The full request body from Slack.
            say (AsyncSay): A function to send a response back to the channel.
        """
        await context.ack()
        if self.mention_me(context, SlackEventBody(**body)):
            return
        event_profile = SlackEventProfile(**event)
        event_body = SlackEventBody(**body)
        token = await self._get_bot_token(context)
        logger.info(f"on_message, context: {context}")
        logger.info(f"on_message, client: {client}")
        logger.info(f"on_message, event_profile: {event_profile}")
        logger.info(f"on_message, event_body: {event_body}")
        logger.info(f"on_message, say: {say}")
        logger.info(f"Received message: {event_profile.text}")
        handler = self.handlers.get("message", self.handlers.get('default'))
        if handler:
            response = handler(event_profile, event_body)
            await say(text=response, token=token)
        
    def mention_me(
        self, context: "AsyncBoltContext", body: SlackEventBody
    ) -> bool:
        r"""Check if the bot is mentioned in the message.

        Args:
            context (AsyncBoltContext): The Slack Bolt context for the event.
            body (SlackEventBody): The body of the Slack event.

        Returns:
            bool: True if the bot is mentioned in the message, False otherwise.
        """
        message = body.event.text
        bot_user_id = context.bot_user_id
        mention = f"<@{bot_user_id}>"
        return mention in message

    async def stop(self) -> None:
        r"""Stops the Slack Bolt app asynchronously."""
        from slack_bolt.adapter.socket_mode.async_handler import (
            AsyncSocketModeHandler,
        )

        if self.socket_mode:
            if isinstance(self._handler, AsyncSocketModeHandler):
                await self._handler.close_async()

    async def _get_bot_token(self, context: "AsyncBoltContext") -> str:
        r"""Helper function to get the bot token.
        Args:
            context (AsyncBoltContext): The Slack Bolt context for the event.
        Returns:
            str: The bot token.
        Raises:
            ValueError: If the bot token is unavailable and self.token is
        """
        from slack_sdk.oauth.installation_store import Bot

        if self._installation_store:
            bot: Optional[Bot] = await self._installation_store.async_find_bot(
                team_id=context.team_id, enterprise_id=context.enterprise_id
            )
            if bot is not None and bot.bot_token is not None:
                return bot.bot_token
        if self.token is None:
            raise ValueError(
                "Bot token is unavailable, and self.token is None."
            )
        return self.token
    
    def set_custom_handler(
            self, 
            handler: Callable[[SlackEventProfile, SlackEventBody], Any], 
            message_type: Optional[str] = 'default') -> None:
        r"""Sets a custom message handler for the Slack app for specific message types.

        Args:
            handler (Callable[[SlackEventProfile, SlackEventBody], Any]): 
                The handler function.
            message_type (Optional[str]): The specific message type to handle 
                (e.g., 'mention', 'message', etc.). 
                Defaults to 'default' if not specified.

        Returns: None
        """
        self.handlers[message_type] = handler
