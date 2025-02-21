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
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from camel.bots import SlackApp


class TestSlackApp(unittest.TestCase):
    def setUp(self):
        # Temporarily set environment variables for testing
        os.environ["SLACK_APP_TOKEN"] = "fake_app_token"
        os.environ["SLACK_TOKEN"] = "fake_token"
        os.environ["SLACK_SCOPES"] = "channels:read,chat:write"
        os.environ["SLACK_SIGNING_SECRET"] = "fake_signing_secret"
        os.environ["SLACK_CLIENT_ID"] = "fake_client_id"
        os.environ["SLACK_CLIENT_SECRET"] = "fake_client_secret"

    def tearDown(self):
        # Clean up environment variables after the tests, if they exist
        for var in [
            "SLACK_TOKEN",
            "SLACK_APP_TOKEN",
            "SLACK_SCOPES",
            "SLACK_SIGNING_SECRET",
            "SLACK_CLIENT_ID",
            "SLACK_CLIENT_SECRET",
        ]:
            if var in os.environ:
                del os.environ[var]

    @patch('slack_bolt.app.async_app.AsyncApp')
    def test_init_without_token_raises_error(self, mock_async_app):
        # Temporarily clear SLACK_TOKEN, SLACK_SCOPES and SLACK_SIGNING_SECRET
        # to test the ValueError
        for var in [
            "SLACK_TOKEN",
            "SLACK_APP_TOKEN",
            "SLACK_SCOPES",
            "SLACK_SIGNING_SECRET",
        ]:
            if var in os.environ:
                del os.environ[var]

        with self.assertRaises(ValueError):
            SlackApp()

    @patch('slack_bolt.app.async_app.AsyncApp')
    def test_init_with_token(self, mock_async_app):
        app = SlackApp(
            token="fake_token1",
            app_token="fake_app_token1",
            scopes="channels:read,chat:write,commands",
            signing_secret="fake_signing_secret1",
            client_id="fake_client_id1",
            client_secret="fake_client_secret1",
        )
        # Assert correct initialization of SlackApp attributes
        self.assertEqual(app.token, "fake_token1")
        self.assertEqual(app.app_token, "fake_app_token1")
        self.assertEqual(app.scopes, "channels:read,chat:write,commands")
        self.assertEqual(app.signing_secret, "fake_signing_secret1")
        self.assertEqual(app.client_id, "fake_client_id1")
        self.assertEqual(app.client_secret, "fake_client_secret1")

    @patch('slack_bolt.app.async_app.AsyncApp')
    def test_setup_handlers(self, mock_async_app):
        mock_app = mock_async_app.return_value
        app = SlackApp()
        app.setup_handlers()

        # Ensure event handlers are properly registered
        mock_app.event.assert_any_call("app_mention")
        mock_app.event.assert_any_call("message")

    @patch('slack_bolt.app.async_app.AsyncApp')
    async def test_handle_request(self, mock_async_app):
        app = SlackApp()
        mock_handler = AsyncMock()
        app._handler = mock_handler

        mock_request = MagicMock()  # Use MagicMock for the request
        await app.handle_request(mock_request)

        # Verify the request was handled once
        mock_handler.handle.assert_called_once_with(mock_request)

    @patch('slack_bolt.app.async_app.AsyncApp')
    async def test_app_mention(self, mock_async_app):
        app = SlackApp()

        context = MagicMock()
        client = MagicMock()
        event = {"text": "Hello, SlackApp!"}
        body = {"body": "some body content"}
        say = AsyncMock()

        await app.app_mention(context, client, event, body, say)

        # Test that the event was logged and say() was called
        self.assertIn("app_mention", context.method_calls[0][0])
        self.assertIn("app_mention", client.method_calls[0][0])
        say.assert_called_once()

    @patch('slack_bolt.app.async_app.AsyncApp')
    async def test_on_message(self, mock_async_app):
        app = SlackApp()

        context = MagicMock()
        client = MagicMock()
        event = {"text": "This is a test message"}
        body = {"body": "some body content"}
        say = AsyncMock()

        await app.on_message(context, client, event, body, say)

        # Check that context.ack() and say() were called
        context.ack.assert_called_once()
        say.assert_called_once()

    @patch('slack_bolt.app.async_app.AsyncApp')
    async def test_mention_me(self, mock_async_app):
        app = SlackApp()

        context = MagicMock()
        context.bot_user_id = "U12345"
        body = MagicMock()
        body.event.text = "Hello <@U12345>"

        # Test when the bot is mentioned
        is_mentioned = app.mention_me(context, body)
        self.assertTrue(is_mentioned)

        # Test when the bot is not mentioned
        body.event.text = "Hello there!"
        is_not_mentioned = app.mention_me(context, body)
        self.assertFalse(is_not_mentioned)


if __name__ == '__main__':
    unittest.main()
