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

import os
import time
from typing import Any, Dict, List

from requests.exceptions import RequestException

from camel.toolkits import OpenAIFunction
from camel.toolkits.base import BaseToolkit


class RedditToolkit(BaseToolkit):
    r"""A class representing a toolkit for Reddit operations."""

    def __init__(self, retries: int = 3, delay: int = 0):
        from praw import Reddit  # type: ignore[import-untyped]

        self.retries = retries
        self.delay = delay

        client_id = os.environ.get("REDDIT_CLIENT_ID", "")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
        user_agent = os.environ.get("REDDIT_USER_AGENT", "")

        if not all([client_id, client_secret, user_agent]):
            print(
                "Reddit API credentials are not set. "
                "Please set the environment variables."
            )

        self.reddit = Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            request_timeout=30,  # Set a timeout to handle delays
        )

    def _retry_request(self, func, *args, **kwargs):
        r"""Retries a function in case of network-related errors."""
        for attempt in range(self.retries):
            try:
                return func(*args, **kwargs)
            except RequestException as e:
                print(f"Attempt {attempt + 1}/{self.retries} failed: {e}")
                if attempt < self.retries - 1:
                    time.sleep(self.delay)
                else:
                    raise

    def collect_top_posts(
        self,
        subreddit_name: str,
        post_limit: int = 5,
        comment_limit: int = 5,
    ) -> List[Dict[str, Any]]:
        r"""Collects the top posts and comments from a specified subreddit."""
        subreddit = self._retry_request(self.reddit.subreddit, subreddit_name)
        top_posts = self._retry_request(subreddit.top, limit=post_limit)
        data = []

        for post in top_posts:
            post_data = {
                "Post Title": post.title,
                "Comments": [
                    {"Comment Body": comment.body, "Upvotes": comment.score}
                    for comment in self._retry_request(
                        lambda post=post: list(post.comments)
                    )[:comment_limit]
                ],
            }
            data.append(post_data)
            time.sleep(self.delay)  # Add a delay to avoid hitting rate limits

        return data

    def perform_sentiment_analysis(
        self, data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        from textblob import TextBlob

        for item in data:
            # Sentiment analysis should be done on 'Comment Body'
            item["Sentiment Score"] = TextBlob(
                item["Comment Body"]
            ).sentiment.polarity

        return data

    def track_keyword_discussions(
        self,
        subreddits: List[str],
        keywords: List[str],
        post_limit: int = 10,
        comment_limit: int = 10,
        textblob_available: bool = False,
    ) -> List[Dict[str, Any]]:
        r"""Tracks discussions about specific keywords in specified
        subreddits"""
        data = []

        for subreddit_name in subreddits:
            subreddit = self._retry_request(
                self.reddit.subreddit, subreddit_name
            )
            top_posts = self._retry_request(subreddit.top, limit=post_limit)

            for post in top_posts:
                for comment in self._retry_request(
                    lambda post=post: list(post.comments)
                )[:comment_limit]:
                    # Print comment body for debugging
                    if any(
                        keyword.lower() in comment.body.lower()
                        for keyword in keywords
                    ):
                        comment_data = {
                            "Subreddit": subreddit_name,
                            "Post Title": post.title,
                            "Comment Body": comment.body,
                            "Upvotes": comment.score,
                        }
                        data.append(comment_data)
                # Add a delay to avoid hitting rate limits
                time.sleep(self.delay)
        if textblob_available:
            data = self.perform_sentiment_analysis(data)
        return data

    def get_tools(self) -> List[OpenAIFunction]:
        r"""Returns a list of OpenAIFunction objects representing the
        functions in the toolkit.
        """
        return [
            OpenAIFunction(self.collect_top_posts),
            OpenAIFunction(self.perform_sentiment_analysis),
            OpenAIFunction(self.track_keyword_discussions),
        ]


REDDIT_FUNCS: List[OpenAIFunction] = RedditToolkit().get_tools()
