"""
ExpelConstructor Implementation

This module implements the concrete prompt constructor for the ExpeL framework,
handling the complete prompt building process including system messages, few-shots,
rules/insights, and task descriptions.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from copy import deepcopy

from langchain_core.messages import ChatMessage
from langchain_core.prompts import PromptTemplate

from .base import BaseConstructor

# Import prompt templates
try:
    from prompts.templates.human import (
        human_instruction_fewshot_message_prompt,
        human_task_message_prompt,
        RULE_TEMPLATE,
    )
except ImportError:
    # Fallback if imports not available
    def human_instruction_fewshot_message_prompt(message_type):
        raise NotImplementedError("human_instruction_fewshot_message_prompt not available")

    class human_task_message_prompt:
        @staticmethod
        def format_messages(task):
            raise NotImplementedError("human_task_message_prompt not available")

    RULE_TEMPLATE = {}


class ExpelConstructor(BaseConstructor):
    """
    Concrete prompt constructor implementation for ExpeL framework.

    This class handles the complete prompt construction process,
    including system messages, few-shots, rules, and task descriptions.
    """

    def __init__(self, *args, **kwargs):
        """Initialize ExpelConstructor with automatic rule template selection."""
        super().__init__(*args, **kwargs)

        # If no rule_template provided, try to get benchmark-specific template
        if self.rule_template is None and self.benchmark_name in RULE_TEMPLATE:
            self.rule_template = RULE_TEMPLATE[self.benchmark_name]

    def build_system_prompt(self) -> List[ChatMessage]:
        """
        Build system prompt messages.

        This method creates the system message that introduces the agent
        and provides environment-specific instructions.
        """
        system_prompt = self.system_prompt.format_messages(
            instruction=self.system_instruction,
            ai_name=self.ai_name
        )
        return system_prompt

    def build_fewshot_prompt(self,
                            fewshots: List[str],
                            prompt_type: str = 'react_type') -> List[ChatMessage]:
        """
        Build few-shot example prompt messages.

        This method integrates few-shot examples with human instructions
        following the original _build_fewshot_prompt logic.
        """
        messages = []

        if human_instruction_fewshot_message_prompt is not None and self.human_instruction is not None:
            messages.append(
                human_instruction_fewshot_message_prompt('message_style_kwargs').format_messages(
                    instruction=self.human_instruction.format_messages(
                        **self.human_instruction_kwargs)[0].content,
                    fewshots='\n\n'.join(fewshots)
                )[0]
            )

        return messages

    def build_rules_prompt(self, rules: str) -> List[ChatMessage]:
        """
        Build rules/insights prompt messages (evaluation phase).

        This method formats and inserts rules/insights extracted during
        the training phase 2 for use in evaluation.
        """
        if self.rule_template is None:
            return []

        return [self.rule_template.format_messages(rules=rules)[0]]

    def build_task_prompt(self, task: str) -> List[ChatMessage]:
        """
        Build task description prompt messages.

        This method creates the task description message that presents
        the current task to the agent.
        """
        return [human_task_message_prompt.format_messages(task=task)[0]]

    def collapse_prompts(self, prompt_history: List[ChatMessage]) -> List[ChatMessage]:
        """
        Collapse consecutive messages of the same type.

        This method merges consecutive messages of the same type to optimize
        prompt structure and reduce redundancy. Based on the original collapse_prompts logic.
        """
        if not prompt_history:
            return []

        new_prompt_history = []
        scratch_pad = prompt_history[0].content
        last_message_type = type(prompt_history[0])

        for message in prompt_history[1:]:
            current_message_type = type(message)
            if current_message_type == last_message_type:
                scratch_pad += '\n' + message.content
            else:
                new_prompt_history.append(last_message_type(content=scratch_pad))
                scratch_pad = message.content
                last_message_type = current_message_type

        # Handle the last accumulated message
        new_prompt_history.append(last_message_type(content=scratch_pad))

        return new_prompt_history

    def build_complete_prompt(self,
                             fewshots: List[str],
                             task: str,
                             rules: Optional[str] = None,
                             is_training: bool = True,
                             no_rules: bool = False,
                             remove_task_suffix: Optional[Callable] = None) -> List[ChatMessage]:
        """
        Build complete prompt combining all components.

        This method replicates the original _build_agent_prompt logic,
        combining system messages, few-shots, rules, and task descriptions.
        """
        prompt_history = []

        # 1. Add system prompt
        system_messages = self.build_system_prompt()
        prompt_history.extend(system_messages)

        # 2. Add few-shot examples
        fewshot_messages = self.build_fewshot_prompt(fewshots)
        prompt_history.extend(fewshot_messages)

        # 3. Collapse messages
        prompt_history = self.collapse_prompts(prompt_history)

        # 4. Insert rules before task (evaluation phase only)
        if not is_training and not no_rules and rules:
            rules_messages = self.build_rules_prompt(rules)
            prompt_history.extend(rules_messages)

        # 5. Add task description
        task_content = task
        if remove_task_suffix:
            task_content = remove_task_suffix(task)

        task_messages = self.build_task_prompt(task_content)
        prompt_history.extend(task_messages)

        # 6. Final collapse
        prompt_history = self.collapse_prompts(prompt_history)

        return prompt_history

    def update_prompt_with_fewshots(self,
                                   current_prompt: List[ChatMessage],
                                   old_fewshots: Union[List[str], str],
                                   new_fewshots: Union[List[str], str]) -> List[ChatMessage]:
        """
        Update existing prompt with new few-shot examples.

        This method replaces old few-shot examples with new ones in the prompt history,
        following the original update_dynamic_prompt_components logic.
        """
        if not old_fewshots or not new_fewshots:
            return current_prompt

        # Handle both list and string inputs
        if isinstance(old_fewshots, list):
            old_fewshots_str = '\n\n'.join(old_fewshots)
        else:
            old_fewshots_str = old_fewshots

        if isinstance(new_fewshots, list):
            new_fewshots_str = '\n\n'.join(new_fewshots)
        else:
            new_fewshots_str = new_fewshots

        updated_prompt = []
        replaced = False

        for message in current_prompt:
            if old_fewshots_str in message.content:
                message_type = type(message)
                updated_content = message.content.replace(old_fewshots_str, new_fewshots_str)
                updated_prompt.append(message_type(content=updated_content))
                replaced = True
            else:
                updated_prompt.append(message)

        return updated_prompt

    def extract_fewshots_from_prompt(self, prompt_history: List[ChatMessage]) -> Optional[str]:
        """
        Extract current few-shot examples from prompt history.

        This method searches for and extracts the current few-shot examples
        from the prompt history for replacement purposes.
        """
        for message in prompt_history:
            if "(END OF EXAMPLES)" in message.content:
                # Extract content between instruction and (END OF EXAMPLES)
                content = message.content
                if '\n\n' in content:
                    parts = content.split('\n\n')
                    for i, part in enumerate(parts):
                        if "(END OF EXAMPLES)" in part:
                            if i > 0:
                                # Return the part before (END OF EXAMPLES)
                                fewshots_part = '\n\n'.join(parts[1:i])
                                return fewshots_part
        return None

    def remove_task_suffix(self, task: str) -> str:
        """
        Remove benchmark-specific task suffixes.

        This method handles benchmark-specific task cleaning logic.
        """
        if self.benchmark_name == 'alfworld':
            return task.split('___')[0]
        return task

    def build_complete_prompt_with_insertions(self,
                                            fewshots: List[str],
                                            task: str,
                                            rules: Optional[str] = None,
                                            before_task_content: Optional[List[ChatMessage]] = None,
                                            after_task_content: Optional[List[ChatMessage]] = None,
                                            is_training: bool = True,
                                            no_rules: bool = False) -> List[ChatMessage]:
        """
        Build complete prompt with flexible insertion points.

        This method extends build_complete_prompt to support arbitrary content insertion
        before and after the task description.
        """
        prompt_history = []

        # 1. Add system prompt
        system_messages = self.build_system_prompt()
        prompt_history.extend(system_messages)

        # 2. Add few-shot examples
        fewshot_messages = self.build_fewshot_prompt(fewshots)
        prompt_history.extend(fewshot_messages)

        # 3. Collapse messages
        prompt_history = self.collapse_prompts(prompt_history)

        # 4. Insert rules before task (evaluation phase only)
        if not is_training and not no_rules and rules:
            rules_messages = self.build_rules_prompt(rules)
            prompt_history.extend(rules_messages)

        # 5. Insert arbitrary content before task
        if before_task_content:
            prompt_history.extend(before_task_content)

        # 6. Add task description
        task_content = self.remove_task_suffix(task)
        task_messages = self.build_task_prompt(task_content)
        prompt_history.extend(task_messages)

        # 7. Insert arbitrary content after task
        if after_task_content:
            prompt_history.extend(after_task_content)

        # 8. Final collapse
        prompt_history = self.collapse_prompts(prompt_history)

        return prompt_history

    def build_incremental_prompt(self,
                                 base_prompt: List[ChatMessage],
                                 conversation_history: List[ChatMessage]) -> List[ChatMessage]:
        """
        Build incremental prompt by combining base prompt with conversation history.

        This method creates the complete prompt for LLM interaction by combining
        the base prompt structure with ongoing conversation history.
        """
        complete_prompt = deepcopy(base_prompt)
        complete_prompt.extend(conversation_history)
        return self.collapse_prompts(complete_prompt)