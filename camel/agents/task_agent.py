import re
from typing import Any, Dict, Optional, Tuple, Union

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.messages import SystemMessage, UserChatMessage
from camel.prompts import PromptTemplateGenerator, TextPrompt
from camel.typing import ModelType, RoleType, TaskType


class TaskSpecifyAgent(ChatAgent):
    r"""An agent that Specifies a given task prompt by prompting the user to
    provide more details.

    Attributes:
        DEFAULT_WORD_LIMIT (int): The default word limit for the task prompt.
        task_specify_prompt (TextPrompt): The prompt for specifying the task.

    Args:
        model (ModelType): The type of model to use for the agent.
            (default: :obj:`ModelType.GPT_3_5_TURBO`)
        task_type (TaskType): The type of task for which to generate a prompt.
            (default: :obj:`TaskType.AI_SOCIETY`)
        model_config (Any): The configuration for the model.
            (default: :obj:`None`)
        task_specify_prompt (Optional[TextPrompt]): The prompt for specifying
            the task. (default: :obj:`None`)
        word_limit (int): The word limit for the task prompt.
            (default: :obj:`50`)
    """
    DEFAULT_WORD_LIMIT = 50

    def __init__(
        self,
        model: ModelType = ModelType.GPT_3_5_TURBO,
        task_type: TaskType = TaskType.AI_SOCIETY,
        model_config: Any = None,
        task_specify_prompt: Optional[Union[str, TextPrompt]] = None,
        word_limit: int = DEFAULT_WORD_LIMIT,
    ) -> None:
        if task_specify_prompt is None:
            task_specify_prompt_template = (
                PromptTemplateGenerator().get_task_specify_prompt(task_type))

            self.task_specify_prompt = task_specify_prompt_template.format(
                word_limit=word_limit)
        else:
            self.task_specify_prompt = task_specify_prompt

        model_config = model_config or ChatGPTConfig(temperature=1.0)

        system_message = SystemMessage(
            role_name="Task Specifier",
            role_type=RoleType.ASSISTANT,
            content="You can make a task more specific.",
        )
        super().__init__(system_message, model, model_config)

    def step(
        self,
        original_task_prompt: Union[str, TextPrompt],
        meta_dict: Optional[Dict[str, Any]] = None,
    ) -> TextPrompt:
        r"""Specify the given task prompt by providing more details.

        Args:
            original_task_prompt (Union[str, TextPrompt]): The original task
                prompt.
            meta_dict (Optional[Dict[str, Any]]): A dictionary containing
                additional information to include in the prompt.
                (default: :obj:`None`)

        Returns:
            TextPrompt: The specified task prompt.
        """
        self.reset()
        self.task_specify_prompt = self.task_specify_prompt.format(
            task=original_task_prompt)

        if meta_dict is not None:
            self.task_specify_prompt = self.task_specify_prompt.format(
                **meta_dict)

        task_msg = UserChatMessage(role_name="Task Specifier",
                                   content=self.task_specify_prompt)
        specified_task_msgs, terminated, _ = super().step(task_msg)
        specified_task_msg = specified_task_msgs[0]

        if terminated:
            raise RuntimeError("Task specification failed.")
        else:
            return TextPrompt(specified_task_msg.content)


class TaskPlannerAgent(ChatAgent):
    r"""An agent that helps divide a task into subtasks based on the input
    task prompt.

    Attributes:
        task_planner_prompt (TextPrompt): A prompt for the agent to divide
            the task into subtasks.

    Args:
        model (ModelType): The type of model to use for the agent.
            (default: :obj:`ModelType.GPT_3_5_TURBO`)
        model_config (Any): The configuration for the model.
            (default: :obj:`None`)
    """

    def __init__(
        self,
        model: ModelType = ModelType.GPT_3_5_TURBO,
        model_config: Any = None,
    ) -> None:
        self.task_planner_prompt = TextPrompt(
            "Divide this task into subtasks: {task}. Be concise.")

        system_message = SystemMessage(
            role_name="Task Planner",
            role_type=RoleType.ASSISTANT,
            content="You are a helpful task planner.",
        )
        super().__init__(system_message, model, model_config)

    def step(
        self,
        task_prompt: Union[str, TextPrompt],
    ) -> TextPrompt:
        r"""Generate subtasks based on the input task prompt.

        Args:
            task_prompt (Union[str, TextPrompt]): The prompt for the task to
                be divided into subtasks.

        Returns:
            TextPrompt: A prompt for the subtasks generated by the agent.
        """
        # TODO: Maybe include roles information.
        self.reset()
        self.task_planner_prompt = self.task_planner_prompt.format(
            task=task_prompt)

        task_msg = UserChatMessage(role_name="Task Planner",
                                   content=self.task_planner_prompt)
        sub_tasks_msgs, terminated, _ = super().step(task_msg)

        if sub_tasks_msgs is None:
            raise RuntimeError("Got None Subtasks messages.")
        if terminated:
            raise RuntimeError("Task planning failed.")

        sub_tasks_msg = sub_tasks_msgs[0]
        return TextPrompt(sub_tasks_msg.content)

class RoleAssignmentAgent(ChatAgent):
    r"""
    An agent that generates role names based on the task prompt.
    Attributes:
        role_assignment_prompt (TextPrompt): A prompt for the agent to generate
        role names.
    args:
        model (ModelType): The tupe of model to use for the agent.
            (default: :obj: 'ModelType.GPT_3_5_TURBO')
        model_config (Any): The configuration for the model.
            (default: :obj: 'None')
    """

    def __init__(
        self,
        model: ModelType = ModelType.GPT_3_5_TURBO,
        model_config: Any = None,
    ) -> None:
        self.role_assignment_prompt = TextPrompt(
            'Given this task, "{task}", generate two role names, ' +
            'one for the AI user and one for the AI assistant.')

        system_message = SystemMessage(
            role_name="Role Assigner",
            role_type=RoleType.ASSISTANT,
            content="You assign roles based on tasks.",
        )

        super().__init__(system_message, model, model_config)

    def step(
        self,
        task_prompt: Union[str, TextPrompt],
    ) -> Tuple[TextPrompt, TextPrompt]:
        r"""
        Generate role names based on the input task prompt.
        Args:
            task_prompt (Union[str, TextPrompt]): The prompt
            for the task based
            on which the roles are to be generated.
            Returns:
                Tuple[TextPrompt, Textprompt]: The role names generated by the
                agent for
                the AI user and AI assistant, respectively.
        """

        self.reset()

        task_msg = AssistantChatMessage(
            role_name="Role Assigner",
            content=self.role_assignment_prompt.format(task=task_prompt),
        )

        role_msgs, terminated, _ = super().step(task_msg)

        ai_user_role_name = None
        ai_assistant_role_name = None

        for message in role_msgs:
            content = message.content
            names = content.split("\n")

            ai_user_role_name = names[1].split(":")[1].strip()
            ai_assistant_role_name = names[0].split(":")[1].strip()

        if ai_user_role_name is None or ai_assistant_role_name is None:
            raise RuntimeError("Got None or insufficient Role messages. ")
        if terminated:
            raise RuntimeError("Role assignment failed.")

        return ai_user_role_name, ai_assistant_role_name

    def step_completion(
        self,
        num_roles: Optional[int] = 2,
        task_prompt: Union[str, TextPrompt] = None,
    ) -> Tuple[TextPrompt, TextPrompt]:
        r""" "
        Generate role names based on the input task prompt.

        Args:
            num_roles (int): The number of roles to generate.
                (default: :obj:`2`)
            task_prompt (Union[str, TextPrompt]): The prompt
                for the task based on which the roles are to be generated.

        Returns:
            Tuple[TextPrompt, Textprompt]: The role names generated by the
                agent for the AI user and AI assistant, respectively.
        """
        expert_prompts = "\n".join(
            f"Domain expert {i + 1}: <|blank|>\n"
            f"Associated competencies, professional characteristics, duties "
            f"and workflows: <|blank|>. End.\n" for i in range(num_roles))
        role_assignment_generation_prompt = TextPrompt(
            "You are the boss, you need to recruit experts in {num_roles} " +
            "different fields to solve the task.\n" +
            "Please tell me which domain experts should be recruited, " +
            "and what competencies, professional characteristics, duties " +
            "and workflows to complete the task.\n" +
            "ONLY return the content in BLANK.\n\n" + "===== TASK =====\n" +
            "{task}\n\n" + "===== PROMPT =====\n" + expert_prompts)
        role_assignment_generation = role_assignment_generation_prompt.format(
            num_roles=num_roles, task=task_prompt)

        role_assignment_generation_msg = AssistantChatMessage(
            role_name="Role Assigner", content=role_assignment_generation)

        output_completions, terminated, info = super().step_completion(
            input_prompt_for_completion=role_assignment_generation_msg,
            max_tokens=100 * num_roles,  # 100 maximum tokens per role
        )

        # Distribute the output completions into role names and descriptions
        role_names = re.findall(
            r"Domain expert \d: (.+?)\nAssociated competencies,",
            output_completions[0]["text"],
            re.DOTALL,
        )
        role_descriptions = re.findall(
            r"Associated competencies, professional characteristics, duties"
            " and workflows: (.+?) End.",
            output_completions[0]["text"],
            re.DOTALL,
        )
        role_description_dict = {
            role_name: description
            for role_name, description in zip(role_names, role_descriptions)
        }

        if len(role_names) != num_roles or len(role_descriptions) != num_roles:
            raise RuntimeError("Got None or insufficient Role messages. ")
        if terminated:
            raise RuntimeError("Role assignment failed.")

        return role_names, role_description_dict, terminated, info
