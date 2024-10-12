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
import datetime
import os
from http import HTTPStatus
from http.client import responses
from typing import List, Optional, Tuple, Union

import requests

from camel.toolkits import OpenAIFunction
from camel.toolkits.base import BaseToolkit
from camel.utils.commons import export_to_toolkit

TWEET_TEXT_LIMIT = 280


class TwitterToolkit(BaseToolkit):
    r"""A class representing a toolkit for Twitter operations.

    This class provides methods for creating a tweet, deleting a tweet, and
    getting the authenticated user's profile information.
    """

    @export_to_toolkit
    def create_tweet(
        self,
        *,
        text: str,
        poll_options: Optional[List[str]] = None,
        poll_duration_minutes: Optional[int] = None,
        quote_tweet_id: Optional[Union[int, str]] = None,
    ) -> str:
        r"""Creates a new tweet, optionally including a poll or a quote tweet,
        or simply a text-only tweet.

        This function sends a POST request to the Twitter API to create a new
        tweet. The tweet can be a text-only tweet, or optionally include a poll
        or be a quote tweet. A confirmation prompt is presented to the user
        before the tweet is created.

        Args:
            text (str): The text of the tweet. The Twitter character limit for
                a single tweet is 280 characters.
            poll_options (Optional[List[str]]): A list of poll options for a
                tweet with a poll.
            poll_duration_minutes (Optional[int]): Duration of the poll in
                minutes for a tweet with a poll. This is only required
                if the request includes poll_options.
            quote_tweet_id (Optional[Union[int, str]]): Link to the tweet being
                quoted.

        Note:
            You can only provide either the `quote_tweet_id` parameter or
            the pair of `poll_duration_minutes` and `poll_options` parameters,
            not both.

        Returns:
            str: A message indicating the success of the tweet creation,
                including the tweet ID and text. If the request to the
                Twitter API is not successful, the return is an error message.

        Reference:
            https://developer.twitter.com/en/docs/twitter-api/tweets/
            manage-tweets/api-reference/post-tweets
            https://github.com/xdevplatform/Twitter-API-v2-sample-code/blob/
            main/Manage-Tweets/create_tweet.py
        """
        # validate text
        if text is None:
            return "Text cannot be None"
        elif len(text) > TWEET_TEXT_LIMIT:
            return "Text must not exceed 280 characters."

        # Validate poll options and duration
        if (poll_options is None) != (poll_duration_minutes is None):
            return (
                "Error: Both `poll_options` and `poll_duration_minutes` must "
                "be provided together or not at all."
            )

        # Validate exclusive parameters
        if quote_tweet_id is not None and (
            poll_options or poll_duration_minutes
        ):
            return (
                "Error: Cannot provide both `quote_tweet_id` and "
                "(`poll_options` or `poll_duration_minutes`)."
            )

        # Print the parameters that are not None
        params = {
            "text": text,
            "poll_options": poll_options,
            "poll_duration_minutes": poll_duration_minutes,
            "quote_tweet_id": quote_tweet_id,
        }
        print("You are going to create a tweet with following parameters:")
        for key, value in params.items():
            if value is not None:
                print(f"{key}: {value}")

        # Add a confirmation prompt at the beginning of the function
        confirm = input(
            "Are you sure you want to create this tweet? (yes/no): "
        )
        if confirm.lower() != "yes":
            return "Execution cancelled by the user."

        oauth = self._get_oauth_session()
        json_data = {}

        if poll_options is not None and poll_duration_minutes is not None:
            json_data["poll"] = {
                "options": poll_options,
                "duration_minutes": poll_duration_minutes,
            }

        if quote_tweet_id is not None:
            json_data["quote_tweet_id"] = str(quote_tweet_id)  # type: ignore[assignment]

        json_data["text"] = text  # type: ignore[assignment]

        # Making the request
        response = oauth.post(
            "https://api.twitter.com/2/tweets",
            json=json_data,
        )

        if response.status_code != HTTPStatus.CREATED:
            error_type = self._handle_http_error(response)
            # use string concatenation to satisfy flake8
            return (
                "Request returned a(n) "
                + str(error_type)
                + ": "
                + str(response.status_code)
                + " "
                + response.text
            )

        # Saving the response as JSON
        json_response = response.json()

        tweet_id = json_response["data"]["id"]
        tweet_text = json_response["data"]["text"]

        response_str = (
            f"Create tweet successful. "
            f"The tweet ID is: {tweet_id}. "
            f"The tweet text is: '{tweet_text}'."
        )

        return response_str

    @export_to_toolkit
    def delete_tweet(self, tweet_id: str) -> str:
        r"""Deletes a tweet with the specified ID for an authorized user.

        This function sends a DELETE request to the Twitter API to delete
        a tweet with the specified ID. Before sending the request, it
        prompts the user to confirm the deletion.

        Args:
            tweet_id (str): The ID of the tweet to delete.

        Returns:
            str: A message indicating the result of the deletion. If the
                deletion was successful, the message includes the ID of the
                deleted tweet. If the deletion was not successful, the message
                includes an error message.

        Reference:
            https://developer.twitter.com/en/docs/twitter-api/tweets/
            manage-tweets/api-reference/delete-tweets-id
        """
        # Print the parameters that are not None
        if tweet_id is not None:
            print(
                f"You are going to delete a tweet with the following "
                f"ID: {tweet_id}"
            )

        # Add a confirmation prompt at the beginning of the function
        confirm = input(
            "Are you sure you want to delete this tweet? (yes/no): "
        )
        if confirm.lower() != "yes":
            return "Execution cancelled by the user."

        oauth = self._get_oauth_session()

        # Making the request
        response = oauth.delete(
            f"https://api.twitter.com/2/tweets/{tweet_id}",
        )

        if response.status_code != HTTPStatus.OK:
            error_type = self._handle_http_error(response)
            # use string concatenation to satisfy flake8
            return (
                "Request returned a(n) "
                + str(error_type)
                + ": "
                + str(response.status_code)
                + " "
                + response.text
            )

        # Saving the response as JSON
        json_response = response.json()
        # `deleted_status` may be True or False.
        # Defaults to False if not found.
        deleted_status = json_response.get("data", {}).get("deleted", False)
        response_str = (
            f"Delete tweet successful: {deleted_status}. "
            f"The tweet ID is: {tweet_id}. "
        )
        return response_str

    @export_to_toolkit
    def get_my_user_profile(self) -> str:
        r"""Retrieves and formats the authenticated user's Twitter
        profile info.

        This function sends a GET request to the Twitter API to retrieve the
        authenticated user's profile information, including their pinned tweet.
        It then formats this information into a readable report.

        Returns:
            str: A formatted report of the authenticated user's Twitter profile
                information. This includes their ID, name, username,
                description, location, most recent tweet ID, profile image URL,
                account creation date, protection status, verification type,
                public metrics, and pinned tweet information. If the request to
                the Twitter API is not successful,
                the return is an error message.

        Reference:
            https://developer.twitter.com/en/docs/twitter-api/users/lookup/
            api-reference/get-users-me
        """
        oauth = self._get_oauth_session()

        tweet_fields = ["created_at", "text"]
        user_fields = [
            "created_at",
            "description",
            "id",
            "location",
            "most_recent_tweet_id",
            "name",
            "pinned_tweet_id",
            "profile_image_url",
            "protected",
            "public_metrics",
            "url",
            "username",
            "verified_type",
        ]
        params = {
            "expansions": "pinned_tweet_id",
            "tweet.fields": ",".join(tweet_fields),
            "user.fields": ",".join(user_fields),
        }

        response = oauth.get(
            "https://api.twitter.com/2/users/me", params=params
        )

        if response.status_code != HTTPStatus.OK:
            error_type = self._handle_http_error(response)
            error_message = "Request returned a(n) {}: {} {}".format(
                error_type, response.status_code, response.text
            )
            return error_message

        json_response = response.json()

        user_info = json_response.get('data', {})
        tweets = json_response.get('includes', {}).get('tweets', [{}])[0]

        user_report = ""
        user_report += f"ID: {user_info['id']}. "
        user_report += f"Name: {user_info['name']}. "
        user_report += f"Username: {user_info['username']}. "

        # Define the part of keys that need to be repeatedly processed
        user_info_keys = [
            'description',
            'location',
            'most_recent_tweet_id',
            'profile_image_url',
        ]
        for key in user_info_keys:
            value = user_info.get(key)
            if user_info.get(key):
                user_report += (
                    f"{key.replace('_', ' ').capitalize()}: {value}. "
                )

        if 'created_at' in user_info:
            created_at = datetime.datetime.strptime(
                user_info['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            date_str = created_at.strftime('%B %d, %Y at %H:%M:%S')
            user_report += f"Account created at: {date_str}. "

        protection_status = "private" if user_info['protected'] else "public"
        user_report += (
            f"Protected: This user's Tweets are {protection_status}. "
        )

        verification_messages = {
            'blue': (
                "The user has a blue verification, typically reserved for "
                "public figures, celebrities, or global brands. "
            ),
            'business': (
                "The user has a business verification, typically "
                "reserved for businesses and corporations. "
            ),
            'government': (
                "The user has a government verification, typically "
                "reserved for government officials or entities. "
            ),
            'none': "The user is not verified. ",
        }
        verification_type = user_info.get('verified_type', 'none')
        user_report += (
            f"Verified type: {verification_messages.get(verification_type)}"
        )

        if 'public_metrics' in user_info:
            user_report += "Public metrics: "
            metrics = user_info['public_metrics']
            user_report += (
                f"The user has {metrics.get('followers_count', 0)} followers, "
                f"is following {metrics.get('following_count', 0)} users, "
                f"has made {metrics.get('tweet_count', 0)} tweets, "
                f"is listed in {metrics.get('listed_count', 0)} lists, "
                f"and has received {metrics.get('like_count', 0)} likes. "
            )

        if 'pinned_tweet_id' in user_info:
            user_report += f"Pinned tweet ID: {user_info['pinned_tweet_id']}. "

        if 'created_at' in tweets and 'text' in tweets:
            user_report += "\nPinned tweet information: "
            tweet_created_at = datetime.datetime.strptime(
                tweets['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            user_report += (
                f"Pinned tweet created at "
                f"{tweet_created_at.strftime('%B %d, %Y at %H:%M:%S')} "
                f"with text: '{tweets['text']}'."
            )

        return user_report

    def get_tools(self) -> List[OpenAIFunction]:
        r"""Returns a list of OpenAIFunction objects representing the
        functions in the toolkit.

        Returns:
            List[OpenAIFunction]: A list of OpenAIFunction objects
                representing the functions in the toolkit.
        """
        return [
            OpenAIFunction(self.create_tweet),
            OpenAIFunction(self.delete_tweet),
            OpenAIFunction(self.get_my_user_profile),
        ]

    def _get_twitter_api_key(self) -> Tuple[str, str]:
        r"""Retrieve the Twitter API key and secret from environment variables.

        Returns:
            Tuple[str, str]: A tuple containing the Twitter API key and secret.

        Raises:
            ValueError: If the API key or secret is not found in the
                environment variables.
        """
        # Get `TWITTER_CONSUMER_KEY` and `TWITTER_CONSUMER_SECRET` here:
        # https://developer.twitter.com/en/portal/products/free
        TWITTER_CONSUMER_KEY = os.environ.get("TWITTER_CONSUMER_KEY")
        TWITTER_CONSUMER_SECRET = os.environ.get("TWITTER_CONSUMER_SECRET")

        if not TWITTER_CONSUMER_KEY or not TWITTER_CONSUMER_SECRET:
            missing_keys = ", ".join(
                [
                    "TWITTER_CONSUMER_KEY" if not TWITTER_CONSUMER_KEY else "",
                    "TWITTER_CONSUMER_SECRET"
                    if not TWITTER_CONSUMER_SECRET
                    else "",
                ]
            ).strip(", ")
            raise ValueError(
                f"{missing_keys} not found in environment variables. Get them "
                "here: `https://developer.twitter.com/en/portal/products/free`."
            )
        return TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET

    def _get_oauth_session(self) -> requests.Session:
        r"""Initiates an OAuth1Session with Twitter's API and returns it.

        The function first fetches a request token, then prompts the user to
        authorize the application. After the user has authorized the
        application and provided a verifier (PIN), the function fetches an
        access token. Finally, a new OAuth1Session is created with the access
        token and returned.

        Raises:
            RuntimeError: If an error occurs while fetching the OAuth access
                token or the OAuth request token.

        Returns:
            requests_oauthlib.OAuth1Session: An OAuth1Session object
                authenticated with the user's access token.

        Reference:
            https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/main/
            Manage-Tweets/create_tweet.py
            https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/main/
            User-Lookup/get_users_me_user_context.py
        """
        try:
            from requests_oauthlib import OAuth1Session
        except ImportError:
            raise ImportError(
                "Please install `requests_oauthlib` first. You can "
                "install it by running `pip install "
                "requests_oauthlib`."
            )

        consumer_key, consumer_secret = self._get_twitter_api_key()

        # Get request token
        request_token_url = (
            "https://api.twitter.com/oauth/request_token"
            "?oauth_callback=oob&x_auth_access_type=write"
        )
        oauth = OAuth1Session(consumer_key, client_secret=consumer_secret)

        try:
            fetch_response = oauth.fetch_request_token(request_token_url)
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while fetching the OAuth access token: {e}"
            )

        resource_owner_key = fetch_response.get("oauth_token")
        resource_owner_secret = fetch_response.get("oauth_token_secret")

        # Get authorization
        base_authorization_url = "https://api.twitter.com/oauth/authorize"
        authorization_url = oauth.authorization_url(base_authorization_url)
        print("Please go here and authorize: %s" % authorization_url)
        verifier = input("Paste the PIN here: ")

        # Get the access token
        access_token_url = "https://api.twitter.com/oauth/access_token"
        oauth = OAuth1Session(
            consumer_key,
            client_secret=consumer_secret,
            resource_owner_key=resource_owner_key,
            resource_owner_secret=resource_owner_secret,
            verifier=verifier,
        )

        try:
            oauth_tokens = oauth.fetch_access_token(access_token_url)
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while fetching the OAuth request token: {e}"
            )

        # Create a new OAuth1Session with the access token
        oauth = OAuth1Session(
            consumer_key,
            client_secret=consumer_secret,
            resource_owner_key=oauth_tokens["oauth_token"],
            resource_owner_secret=oauth_tokens["oauth_token_secret"],
        )
        return oauth

    def _handle_http_error(self, response: requests.Response) -> str:
        r"""Handles the HTTP response by checking the status code and
        returning an appropriate message if there is an error.

        Args:
            response (requests.Response): The HTTP response to handle.

        Returns:
            str: A string describing the error, if any. If there is no error,
                the function returns an "Unexpected Exception" message.

        Reference:
            https://github.com/tweepy/tweepy/blob/master/tweepy/client.py#L64
        """
        if response.status_code in responses:
            # For 5xx server errors, return "Twitter Server Error"
            if 500 <= response.status_code < 600:
                return "Twitter Server Error"
            else:
                error_message = responses[response.status_code] + " Error"
                return error_message
        elif not 200 <= response.status_code < 300:
            return "HTTP Exception"
        else:
            return "Unexpected Exception"
