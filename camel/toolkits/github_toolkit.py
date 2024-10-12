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
from datetime import datetime, timedelta
from typing import List, Optional

from pydantic import BaseModel

from camel.toolkits.base import BaseToolkit
from camel.toolkits.openai_function import OpenAIFunction
from camel.utils import dependencies_required
from camel.utils.commons import export_to_toolkit


def get_github_access_token() -> str:
    r"""Retrieve the GitHub access token from environment variables.

    Returns:
        str: A string containing the GitHub access token.

    Raises:
        ValueError: If the API key or secret is not found in the environment
            variables.
    """
    # Get `GITHUB_ACCESS_TOKEN` here: https://github.com/settings/tokens
    github_token = os.environ.get("GITHUB_ACCESS_TOKEN")

    if not github_token:
        raise ValueError(
            "`GITHUB_ACCESS_TOKEN` not found in environment variables. Get it "
            "here: `https://github.com/settings/tokens`."
        )
    return github_token


class GithubIssue(BaseModel):
    r"""Represents a GitHub issue.

    Attributes:
        title (str): The title of the issue.
        body (str): The body/content of the issue.
        number (int): The issue number.
        file_path (str): The path of the file associated with the issue.
        file_content (str): The content of the file associated with the issue.
    """

    title: str
    body: str
    number: int
    file_path: str
    file_content: str

    def __str__(self) -> str:
        r"""Returns a string representation of the issue.

        Returns:
            str: A string containing the title, body, number, file path, and
                file content of the issue.
        """
        return (
            f"Title: {self.title}\n"
            f"Body: {self.body}\n"
            f"Number: {self.number}\n"
            f"File Path: {self.file_path}\n"
            f"File Content: {self.file_content}"
        )


class GithubPullRequestDiff(BaseModel):
    r"""Represents a single diff of a pull request on Github.

    Attributes:
        filename (str): The name of the file that was changed.
        patch (str): The diff patch for the file.
    """

    filename: str
    patch: str

    def __str__(self) -> str:
        return f"Filename: {self.filename}\nPatch: {self.patch}"


class GithubPullRequest(BaseModel):
    r"""Represents a pull request on GitHub.

    Attributes:
        title (str): The title of the GitHub pull request.
        body (str): The body/content of the GitHub pull request.
        diffs (List[GithubPullRequestDiff]): A list of diffs for the PR.
    """

    title: str
    body: str
    diffs: List[GithubPullRequestDiff]

    def __str__(self) -> str:
        diff_summaries = '\n'.join(str(diff) for diff in self.diffs)
        return (
            f"Title: {self.title}\n"
            f"Body: {self.body}\n"
            f"Diffs: {diff_summaries}\n"
        )


class GithubToolkit(BaseToolkit):
    r"""A class representing a toolkit for interacting with GitHub
    repositories. This class provides methods for retrieving open issues,
    retrieving specific issues, and creating pull requests in a GitHub
    repository.

    Args:
        repo_name (str): The name of the GitHub repository.
        access_token (Optional[str], optional): The access token to
            authenticate with GitHub. If not provided, it will be obtained
            using the `get_github_access_token` method. (default: :obj:`None`)
    """

    @dependencies_required('github')
    def __init__(
        self, repo_name: str, access_token: Optional[str] = None
    ) -> None:
        r"""Initializes a new instance of the GitHubToolkit class.

        Args:
            repo_name (str): The name of the GitHub repository.
            access_token (Optional[str], optional): The access token to
                authenticate with GitHub. If not provided, it will be obtained
                using the `get_github_access_token` method.
                (default: :obj:`None`)
        """
        if access_token is None:
            access_token = get_github_access_token()

        from github import Auth, Github

        self.github = Github(auth=Auth.Token(access_token))
        self.repo = self.github.get_repo(repo_name)

    def get_tools(self) -> List[OpenAIFunction]:
        r"""Returns a list of OpenAIFunction objects representing the
        functions in the toolkit.

        Returns:
            List[OpenAIFunction]: A list of OpenAIFunction objects
                representing the functions in the toolkit.
        """
        return [
            OpenAIFunction(self.retrieve_issue_list),
            OpenAIFunction(self.retrieve_issue),
            OpenAIFunction(self.create_pull_request),
            OpenAIFunction(self.retrieve_pull_requests),
        ]

    @export_to_toolkit
    def retrieve_issue_list(self) -> List[GithubIssue]:
        r"""Retrieve a list of open issues from the repository.

        Returns:
            List[GithubIssue]: A list of GithubIssue objects representing
                the open issues.
        """
        issues = self.repo.get_issues(state='open')
        return [
            GithubIssue(
                title=issue.title,
                body=issue.body,
                number=issue.number,
                file_path=issue.labels[
                    0
                ].name,  # we require file path to be the first label in the PR
                file_content=self.retrieve_file_content(issue.labels[0].name),
            )
            for issue in issues
            if not issue.pull_request
        ]

    @export_to_toolkit
    def retrieve_issue(self, issue_number: int) -> Optional[str]:
        r"""Retrieves an issue from a GitHub repository. This function
        retrieves an issue from a specified repository using the issue number.

        Args:
            issue_number (int): The number of the issue to retrieve.

        Returns:
            Optional[str]: A formatted report of the retrieved issue.
        """
        issues = self.retrieve_issue_list()
        for issue in issues:
            if issue.number == issue_number:
                return str(issue)
        return None

    @export_to_toolkit
    def retrieve_pull_requests(
        self, days: int, state: str, max_prs: int
    ) -> List[str]:
        r"""Retrieves a summary of merged pull requests from the repository.
        The summary will be provided for the last specified number of days.

        Args:
            days (int): Number of days to retrieve merged PRs for.
            state (str): The state of PRs to retrieve. Can be open or closed.
            max_prs (int): The maximum number of PRs to retrieve.

        Returns:
             List[str]: A list of merged pull request summaries.
        """
        pull_requests = self.repo.get_pulls(state=state)
        merged_prs = []
        earliest_date: datetime = datetime.utcnow() - timedelta(days=days)

        for pr in pull_requests[:max_prs]:
            if (
                pr.merged
                and pr.merged_at is not None
                and pr.merged_at.timestamp() > earliest_date.timestamp()
            ):
                pr_details = GithubPullRequest(
                    title=pr.title, body=pr.body, diffs=[]
                )

                # Get files changed in the PR
                files = pr.get_files()

                for file in files:
                    diff = GithubPullRequestDiff(
                        filename=file.filename, patch=file.patch
                    )
                    pr_details.diffs.append(diff)

                merged_prs.append(str(pr_details))
        return merged_prs

    @export_to_toolkit
    def create_pull_request(
        self,
        file_path: str,
        new_content: str,
        pr_title: str,
        body: str,
        branch_name: str,
    ) -> str:
        r"""Creates a pull request. This function creates a pull request in
        specified repository, which updates a file in the specific path with
        new content. The pull request description contains information about
        the issue title and number.

        Args:
            file_path (str): The path of the file to be updated in the
                repository.
            new_content (str): The specified new content of the specified file.
            pr_title (str): The title of the issue that is solved by this pull
                request.
            body (str): The commit message for the pull request.
            branch_name (str): The name of the branch to create and submit the
                pull request from.

        Returns:
            str: A formatted report of whether the pull request was created
                successfully or not.
        """
        sb = self.repo.get_branch(self.repo.default_branch)
        self.repo.create_git_ref(
            ref=f"refs/heads/{branch_name}", sha=sb.commit.sha
        )

        file = self.repo.get_contents(file_path)
        from github.ContentFile import ContentFile

        if isinstance(file, ContentFile):
            self.repo.update_file(
                file.path, body, new_content, file.sha, branch=branch_name
            )
            pr = self.repo.create_pull(
                title=pr_title,
                body=body,
                head=branch_name,
                base=self.repo.default_branch,
            )

            if pr is not None:
                return f"Title: {pr.title}\n" f"Body: {pr.body}\n"
            else:
                return "Failed to create pull request."
        else:
            raise ValueError("PRs with multiple files aren't supported yet.")

    def retrieve_file_content(self, file_path: str) -> str:
        r"""Retrieves the content of a file from the GitHub repository.

        Args:
            file_path (str): The path of the file to retrieve.

        Returns:
            str: The decoded content of the file.
        """
        file_content = self.repo.get_contents(file_path)

        from github.ContentFile import ContentFile

        if isinstance(file_content, ContentFile):
            return file_content.decoded_content.decode()
        else:
            raise ValueError("PRs with multiple files aren't supported yet.")
