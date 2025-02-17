🐫 **Welcome to CAMEL!** 🐫

Thank you for your interest in contributing to the CAMEL project! 🎉 We're excited to have your support. As an open-source initiative in a rapidly evolving and open-ended field, we wholeheartedly welcome contributions of all kinds. Whether you want to introduce new features, enhance the infrastructure, improve documentation, asking issues, add more examples, implement state-of-the-art research ideas, or fix bugs, we appreciate your enthusiasm and efforts. 🙌  You are welcome to join our [discord](https://discord.camel-ai.org/) for more efficient communication. 💬

## Join Our Community 🌍

### Schedule an Introduction Call 📞 
- English speakers: [here](https://calendly.com/roman-georgio/camel-ai-introduction-call?month=2024-05)
- Chinese speakers: [here](https://calendly.com/sisi-qu/welcome-to-camel-onboarding-meeting?month=2024-05)

### Developer Meeting Time & Link 💻
- English speakers: Mondays at 5 PM GMT+1. Join via Discord: [Meeting Link](https://discord.gg/FFe4nB8MJj?event=1313319275708289034)
- Chinese Speakers: Mondays at 9 PM UTC+8. Join via TecentMeeting: [Meeting Link](https://meeting.tencent.com/dm/057wap1eeCSY)

### Our Communication Channels 💬
- **Discord:** [Join here](https://discord.camel-ai.org/)
- **WeChat:** Scan the QR code [here](https://ghli.org/camel/wechat.png)
- **Slack:** [Join here](https://join.slack.com/t/camel-ai/shared_invite/zt-2g7xc41gy-_7rcrNNAArIP6sLQqldkqQ)

## Guidelines 📝

### Contributing to the Code 👨‍💻👩‍💻

If you're eager to contribute to this project, that's fantastic! We're thrilled to have your support. 

- If you are a contributor from the community:
  - Follow the [Fork-and-Pull-Request](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow when opening your pull requests.
- If you are a member of [CAMEL-AI.org](https://github.com/camel-ai):
  - Follow the [Checkout-and-Pull-Request](https://dev.to/ceceliacreates/how-to-create-a-pull-request-on-github-16h1) workflow when opening your pull request; this will allow the PR to pass all tests that require [GitHub Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets).

Make sure to mention any related issues and tag the relevant maintainers too. 💪

Before your pull request can be merged, it must pass the formatting, linting, and testing checks. You can find instructions on running these checks locally under the **Common Actions** section below. 🔍

Ensuring excellent documentation and thorough testing is absolutely crucial. Here are some guidelines to follow based on the type of contribution you're making:

- If you fix a bug:
  - Add a relevant unit test when possible. These can be found in the `test` directory.
- If you make an improvement:
  - Update any affected example console scripts in the `examples` directory, Gradio demos in the `apps` directory, and documentation in the `docs` directory.
  - Update unit tests when relevant.
- If you add a feature:
  - Include unit tests in the `test/unit` directory, which can be run locally.
  - Include integration tests in the `test/integration` directory, which needs web requests like OpenAI.
  - Add a demo script in the `examples` directory.
 
We're a small team focused on building great things. If you have something in mind that you'd like to add or modify, opening a pull request is the ideal way to catch our attention. 🚀

### Contributing to the Cookbook Writing 📚

We are excited that you want to contribute to our cookbook! The CAMEL cookbook is an essential resource for users, and we want to make sure that it remains high-quality, accurate, and helpful.

We use Google Colab as the primary platform for writing and reviewing our cookbooks, and we have a specific [template](https://colab.research.google.com/drive/17SiWWjoK7l8Sy9FBsGKUHC6zuEsLt2yX?usp=sharing) that contributors should follow to maintain consistency. Please feel free to share any suggestions for improvements or additions to the template.

Here’s how you can contribute to writing cookbooks:

#### 1. Writing the Cookbook

- Template Usage: We have a [template](https://colab.research.google.com/drive/17SiWWjoK7l8Sy9FBsGKUHC6zuEsLt2yX?usp=sharing) for cookbook entries. Always use this template to ensure uniformity in style and structure.
- Start in Colab: Begin writing your cookbook in Google Colab. Colab allows us to run interactive code and include explanations side-by-side, which is ideal for a practical, hands-on cookbook.
- Correctness: Ensure that the **LLM-generated responses are correct**. Each recipe in the cookbook should be verified with real code execution, and the results must be accurate.
- Clear and Concise Explanations: Provide clear explanations of the code, its purpose, and how it works. Make sure any complex logic or steps are well documented and explained in simple terms. Avoid installing all optional extras unless they are needed.
- Interactive Elements: Whenever applicable, add interactive code cells in Colab that users can directly run and modify.

##### 1.2. Developing cookbooks for in-progress features
You can install the latest version of CAMEL from the main branch or a topic branch. This allows you to use the latest codebase, or in-progress features in your cookbook. 

`!pip install "git+https://github.com/camel-ai/camel.git@master#egg=camel-ai[all]"` 

Changing the branch and extras section (e.g. remove `#egg=camel-ai[all]`) will behave as expected.

Remember to change to a release version before sharing the cookbook with the community for long-term stability. Also note that a topic branch will stop working if the branch is deleted.

#### 2. Reviewing the Cookbook
Once the initial draft of the cookbook is ready:

- Review in Colab: The review process for cookbooks is done in Colab. We will leave comments and suggestions directly in the Colab document. Reviewers will focus on:
 - Accuracy: Ensure that the code is correct, performs the expected tasks, and generates the intended results.
 - Clarity: Check that the explanations are clear, detailed, and easy to understand.
 - Structure: Ensure that the flow of the cookbook is logical and easy to follow, with a step-by-step approach.
 - Formatting: Verify that the formatting follows the template guidelines (e.g., headers, code blocks, comments).
 - Feedback Process: If there are issues or suggestions for improvement, the reviewer will leave comments directly in the Colab notebook. The contributor should address these comments and update the Colab file accordingly.

#### 3. Submitting the Cookbook
When the Colab cookbook is ready for integration:

- Download the Cookbook: Once the Colab notebook is finalized and reviewed, download the notebook as a .ipynb file.
- Create a Pull Request: Open a pull request to add the cookbook to the docs folder of the repository. This pull request will include the downloadable .ipynb file and also include any necessary documentation or references to integrate the cookbook into the main docs.

#### 4. Principles to Follow
To ensure that the cookbook meets the highest standards, please keep the following principles in mind:

- High Quality: Every cookbook entry must be of high quality. This includes accurate code, well-written explanations, and comprehensive tests (if applicable).
- LLM-Generated Responses: If you use a large language model (LLM) to generate part of the cookbook content, ensure that the responses are correct and validated with real-world examples. Do not include any code that is incorrect or doesn't work as expected.
- Reproducibility: Ensure that all code in the cookbook is reproducible. Any user following the steps should be able to get the same results by running the code on their own machine.
- Accessibility: Make sure the content is accessible to a wide range of users, from beginners to advanced users. Provide context where necessary and define technical terms.

By following these guidelines, you'll help us maintain a high-quality, helpful, and engaging cookbook for the CAMEL community. Thank you for contributing to this valuable resource! 🌟

### Contributing to Code Reviews 🔍
This part outlines the guidelines and best practices for conducting code reviews in CAMEL. The aim is to ensure that all contributions are of high quality, align with the project's goals, and are consistent with our coding standards.

#### Purpose of Code Reviews
- Maintain Code Quality: Ensure that the codebase remains clean, readable, and maintainable.
- Knowledge Sharing: Facilitate knowledge sharing among contributors and help new contributors learn best practices.
- Bug Prevention: Catch potential bugs and issues before they are merged into the main branch.
- Consistency: Ensure consistency in style, design patterns, and architecture across the project.

#### Review Process Overview
- Reviewers should check the code for functionality, readability, consistency, and compliance with the project’s coding standards.
- If changes are necessary, the reviewer should leave constructive feedback.
- The contributor addresses feedback and updates the PR.
- The reviewer re-reviews the updated code.
- Once the code is approved by at least two reviewer, it can be merged into the main branch.
- Merging should be done by a maintainer or an authorized contributor.

#### Code Review Checklist
- Functionality
  - Correctness: Does the code perform the intended task? Are edge cases handled?
  - Testing: Is there sufficient test coverage? Do all tests pass?
  - Security: Are there any security vulnerabilities introduced by the change?
  - Performance: Does the code introduce any performance regressions?

- Code Quality
  - Readability: Is the code easy to read and understand? Is it well-commented where necessary?
  - Maintainability: Is the code structured in a way that makes future changes easy?
  - Style: Does the code follow the project’s style guidelines?
  Currently we use Ruff for format check and take [Google Python Style Guide]("https://google.github.io/styleguide/pyguide.html") as reference.
  - Documentation: Are public methods, classes, and any complex logic well-documented?
- Design
  - Consistency: Does the code follow established design patterns and project architecture?
  - Modularity: Are the changes modular and self-contained? Does the code avoid unnecessary duplication?
  - Dependencies: Are dependencies minimized and used appropriately?

#### Reviewer Responsibilities
- Timely Reviews: Reviewers should strive to review PRs promptly to keep the project moving.
- Constructive Feedback: Provide feedback that is clear, constructive, and aimed at helping the contributor improve.
- Collaboration: Work with the contributor to address any issues and ensure the final code meets the project’s standards.
- Approvals: Only approve code that you are confident meets all the necessary criteria.

#### Common Pitfalls
- Large PRs: Avoid submitting PRs that are too large. Break down your changes into smaller, manageable PRs if possible.
- Ignoring Feedback: Address all feedback provided by reviewers, even if you don’t agree with it—discuss it instead of ignoring it.
- Rushed Reviews: Avoid rushing through reviews. Taking the time to thoroughly review code is critical to maintaining quality.

Code reviews are an essential part of maintaining the quality and integrity of our open source project. By following these guidelines, we can ensure that CAMEL remains robust, secure, and easy to maintain, while also fostering a collaborative and welcoming community.

### Guideline for Writing Docstrings

This guideline will help you write clear, concise, and structured docstrings for contributing to `CAMEL`.

#### 1. Use the Triple-Quoted String with `r"""` (Raw String)
Begin the docstring with `r"""` to indicate a raw docstring. This prevents any issues with special characters and ensures consistent formatting, especially in documentation tools like Sphinx.

#### 2. Provide a Brief Class or Method Description
- Start with a concise summary of the purpose and functionality.
- Keep each line under `79` characters.
- The summary should start on the first line without a linebreak.

Example:
```python
r"""Class for managing conversations of CAMEL Chat Agents.
"""
```

#### 3. Document Parameters in the Args Section
- Use an `Args`: section for documenting constructor or function parameters.
- Maintain the `79`-character limit for each line, and indent continuation lines by 4 spaces.
- Follow this structure:
  - Parameter Name: Match the function signature.
  - Type: Include the type (e.g., `int`, `str`, custom types like `BaseModelBackend`).
  - Description: Provide a brief explanation of the parameter's role.
  - Default Value: Use (`default: :obj:<default_value>`) to indicate default values.

Example:
```markdown
Args:
    system_message (BaseMessage): The system message for initializing 
        the agent's conversation context.
    model (BaseModelBackend, optional): The model backend to use for 
        response generation. Defaults to :obj:`OpenAIModel` with 
        `GPT_4O_MINI`. (default: :obj:`OpenAIModel` with `GPT_4O_MINI`)
```

### Principles 🛡️

#### Naming Principle: Avoid Abbreviations in Naming

- Abbreviations can lead to ambiguity, especially since variable names and code in CAMEL are directly used by agents.
- Use clear, descriptive names that convey meaning without requiring additional explanation. This improves both human readability and the agent's ability to interpret the code.

Examples:

- Bad: msg_win_sz
- Good: message_window_size

By adhering to this principle, we ensure that CAMEL remains accessible and unambiguous for both developers and AI agents.

#### Logging Principle: Use `logger` Instead of `print`

Avoid using `print` for output. Use Python's `logging` module (`logger`) to ensure consistent, configurable, and professional logging.

Examples:

- Bad: 
  ```python
  print("Process started")
  print(f"User input: {user_input}")
  ```
- Good: 
  ```python
  Args:
  logger.info("Process started")
  logger.debug(f"User input: {user_input}")
  ```


### Board Item Create Workflow 🛠️
At CAMEL, we manage our project through a structured workflow that ensures efficiency and clarity in our development process. Our workflow includes stages for issue creation and pull requests (PRs), sprint planning, and reviews.

#### Issue Item Stage:
Our [issues](https://github.com/camel-ai/camel/issues) page on GitHub is regularly updated with bugs, improvements, and feature requests. We have a handy set of labels to help you sort through and find issues that interest you. Feel free to use these labels to keep things organized.

When you start working on an issue, please assign it to yourself so that others know it's being taken care of.

When creating a new issue, it's best to keep it focused on a specific bug, improvement, or feature. If two issues are related or blocking each other, it's better to link them instead of merging them into one.

We do our best to keep these issues up to date, but considering the fast-paced nature of this field, some may become outdated. If you come across any such issues, please give us a heads-up so we can address them promptly. 👀

Here’s how to engage with our issues effectively:
- Go to [GitHub Issues](https://github.com/camel-ai/camel/issues), create a new issue, choose the category, and fill in the required information.
- Ensure the issue has a proper title and update the Assignees, Labels, Projects (select Backlog status), Development, and Milestones.
- Discuss the issue during team meetings, then move it to the Analysis Done column.
- At the beginning of each sprint, share the analyzed issue and move it to the Sprint Planned column if you are going to work on this issue in the sprint.

#### Pull Request Item Stage:

- Go to [GitHub Pulls](https://github.com/camel-ai/camel/pulls), create a new PR, choose the branch, and fill in the information, linking the related issue.
- Ensure the PR has a proper title and update the Reviewers (convert to draft), Assignees, Labels, Projects (select Developing status), Development, and Milestones.
- If the PR is related to a roadmap, link the roadmap to the PR.
- Move the PR item through the stages: Developing, Stuck, Reviewing (click ready for review), Merged. The linked issue will close automatically when the PR is merged.

**Labeling PRs:**
- **feat**: For new features (e.g., `feat: Add new AI model`)
- **fix**: For bug fixes (e.g., `fix: Resolve memory leak issue`)
- **docs**: For documentation updates (e.g., `docs: Update contribution guidelines`)
- **style**: For code style changes (e.g., `style: Refactor code formatting`)
- **refactor**: For code refactoring (e.g., `refactor: Optimize data processing`)
- **test**: For adding or updating tests (e.g., `test: Add unit tests for new feature`)
- **chore**: For maintenance tasks (e.g., `chore: Update dependencies`)

### Sprint Planning & Review 🎯

#### Definition

Sprint planning defines what can be delivered in the sprint and how it will be achieved. Sprint review allows stakeholders to review and provide feedback on recent work.

#### Practice

- **Sprint Duration**: Two weeks for development, one week for review.
- **Sprint Planning & Review**: Conducted biweekly during the dev meeting (around 30 minutes).
- **Planning**: Founder highlights the sprint goal and key points; developers pick items for the sprint.
- **Review**: Feedback on delivered features and identification of improvement areas.

### Getting Help 🆘

Our aim is to make the developer setup as straightforward as possible. If you encounter any challenges during the setup process, don't hesitate to reach out to a maintainer. We're here to assist you and ensure that the experience is smooth not just for you but also for future contributors. 😊

In line with this, we do have specific guidelines for code linting, formatting, and documentation in the codebase. If you find these requirements difficult or even just bothersome to work with, please feel free to get in touch with a maintainer. We don't want these guidelines to hinder the integration of good code into the codebase, so we're more than happy to provide support and find a solution that works for you. 🤝

## Quick Start 🚀

To get started with CAMEL, follow these steps:

```sh
# Clone github repo
git clone https://github.com/camel-ai/camel.git

# Change directory into project directory
cd camel

# Activate camel virtual environment
poetry shell

# Install camel from source
poetry install --with dev,docs -E all  # (Suggested for developers, needed to pass all tests)

# The following command installs a pre-commit hook into the local git repo,
# so every commit gets auto-formatted and linted.
pre-commit install

# Run camel's pre-commit before push
pre-commit run --all-files

# Run all tests, expect most of them would fail
pytest test

# Exit the virtual environment
exit
```

These commands will install all the necessary dependencies for running the package, examples, linting, formatting, tests, and coverage.

To verify that everything is set up correctly, run `pytest .` This will ensure that all tests pass successfully. ✅

> [!TIP]
> You need to config different API Keys as environment variables to pass all tests.

## Common Actions 🔄

### Update dependencies

Whenever you add, update, or delete any dependencies in `pyproject.toml`, please run `poetry lock` to synchronize the dependencies with the lock file.

### Linting & Formatting ✨

```bash
poetry run ruff check .
poetry run ruff format .
```

For extra validation of type hints:

```bash
mypy --namespace-packages -p camel
mypy --namespace-packages -p test
mypy --namespace-packages -p examples
mypy --namespace-packages -p apps
```

### Coverage 📊

Code coverage measures the extent to which unit tests cover the code, helping identify both robust and less robust areas of the codebase.

To generate a report showing the current code coverage, execute one of the following commands.

To include all source files into coverage:

```bash
coverage erase
coverage run --source=. -m pytest .
coverage html
# Open htmlcov/index.html
```

To include only tested files:
```bash
pytest --cov --cov-report=html
```

The coverage report will be generated at `htmlcov/index.html`.

## Tests 🧪

Unit tests cover modular logic that doesn't require calls to outside APIs. Currently, the test setup requires an OpenAI API key to test the framework, making them resemble integration tests.

To run all tests including those that use OpenAI API, use the following command:

```bash
pytest .
```

To quickly run only local isolated unit and integration tests:
```bash
pytest --fast-test-mode .
```
this is defined in `conftest.py`.
which is equivalent to
```bash
pytest -m "not optional and not model_backend"
```

If you're developing with VSCode, make sure to create a `.env` file in the repository root and include your OpenAI API key:

```
OPENAI_API_KEY=sk-XXXXXXXX
OPENAI_API_BASE_URL=https://XXXXXXXX (Should you utilize an OpenAI proxy service, kindly specify this)
```

### Unit & Integration
As integration tests require token and third-party setup, it is costly to run locally. Ideally, developers should run unit test locally and run integration tests in github/CI/CD.

The recommended way to run the integration tests is to create a draft PR and all tests will be run on github. It is not recommended to run integration tests locally.

To run a single unit test locally, use
```bash
pytest ./test/agents/test_agent_base.py
``**

**From now on, a developer is expected to add unit tests under `test/unit` and integration tests under `test/integration`.** We would probably fix the existing tests, but we do want to make sure the tests in the future are categorized correctly.

Note: We have some tech debt that unit tests and integration tests are not decoupled. 

### Use Decorator
Ideally you should use pytest decorators for all the tests, check pytest.ini for marker definitions. For example, you should use unit for unit tests and integration for integration tests.

## Documentation 📚

### Contribute to Documentation 📝

The documentation is primarily generated automatically by [Sphinx](https://www.sphinx-doc.org/en/master/) using the code.

We kindly request that you provide comprehensive documentation for all classes and methods to ensure high-quality documentation coverage.

### Build Documentation Locally 🛠️

To build the documentation locally, follow these steps:

```bash
cd docs
make html
```

More guidelines about building and hosting documentations locally can be found [here](https://github.com/camel-ai/camel/blob/master/docs/README.md).

## Versioning and Release 🚀

As of now, CAMEL is actively in development and just published preview version to PyPI.

CAMEL follows the [semver](https://semver.org/) versioning standard. As pre-1.0 software, even patch releases may contain [non-backwards-compatible changes](https://semver.org/#spec-item-4). Currently, the major version is 0, and the minor version is incremented. Releases are made once the maintainers feel that a significant body of changes has accumulated.


## License 📜

The source code of the CAMEL project is licensed under Apache 2.0. Your contributed code will be also licensed under Apache 2.0 by default. To add license to you code, you can manually copy-paste it from `license_template.txt` to the head of your files or run the `update_license.py` script to automate the process:

```bash
python licenses/update_license.py . licenses/license_template.txt
```

This script will add licenses to all the `*.py` files or update the licenses if the existing licenses are not the same as `license_template.txt`.

## Giving Credit 🎉

If your contribution has been included in a release, we'd love to give you credit on Twitter, but only if you're comfortable with it!

If you have a Twitter account that you would like us to mention, please let us know either in the pull request or through another communication method. We want to make sure you receive proper recognition for your valuable contributions. 😄
