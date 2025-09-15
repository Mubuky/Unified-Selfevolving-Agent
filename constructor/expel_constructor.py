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

        # Initialize conversation management
        self.conversation_history = []
        self.base_prompt_length = 0

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

    # ==================== Conversation Management Methods ====================

    def initialize_conversation(self,
                              fewshots: List[str],
                              task: str,
                              rules: Optional[str] = None,
                              is_training: bool = True,
                              no_rules: bool = False) -> None:
        """
        Initialize conversation with base prompt structure.

        This method sets up the initial conversation state with system messages,
        few-shots, rules, and task descriptions.
        """
        base_prompt = self.build_complete_prompt(
            fewshots=fewshots,
            task=task,
            rules=rules,
            is_training=is_training,
            no_rules=no_rules
        )

        self.conversation_history = base_prompt
        self.base_prompt_length = len(base_prompt)

    def add_conversation_turn(self, message: ChatMessage) -> None:
        """
        Add a conversation turn (user message or AI response).

        This method appends a new message to the conversation history
        and maintains the conversation state.
        """
        self.conversation_history.append(message)

    def update_dynamic_components(self,
                                update_callback: Optional[Callable] = None,
                                **update_kwargs) -> None:
        """
        Update dynamic components in the conversation.

        This method allows for dynamic updates of conversation components
        such as few-shots, system instructions, etc. through a callback.

        Args:
            update_callback: Function to call for dynamic updates
            **update_kwargs: Additional arguments for the update callback
        """
        if update_callback is not None:
            # Store conversation turns after base prompt
            conversation_turns = self.conversation_history[self.base_prompt_length:]

            # Call update callback (typically agent's update_dynamic_prompt_components)
            update_callback(**update_kwargs)

            # Rebuild base prompt with updated components
            # Note: fewshots and other components should be updated by callback
            if hasattr(self, '_last_build_params'):
                base_prompt = self.build_complete_prompt(**self._last_build_params)
                self.conversation_history = base_prompt + conversation_turns
                self.base_prompt_length = len(base_prompt)

    def prepare_llm_input(self, collapse_messages: bool = True) -> List[ChatMessage]:
        """
        Prepare the final prompt for LLM input.

        This method processes the conversation history and returns the
        final prompt structure ready for LLM invocation.
        """
        if collapse_messages:
            return self.collapse_prompts(self.conversation_history)
        return self.conversation_history

    def reset_conversation(self) -> None:
        """
        Reset the conversation history to empty state.

        This method clears the conversation history and resets
        the conversation management state.
        """
        self.conversation_history = []
        self.base_prompt_length = 0

    def get_conversation_length(self) -> int:
        """
        Get the total length of the conversation history.

        Returns the number of messages in the current conversation.
        """
        return len(self.conversation_history)

    def get_conversation_turns_only(self) -> List[ChatMessage]:
        """
        Get only the conversation turns (excluding base prompt).

        Returns only the dynamic conversation parts without the
        initial system/few-shot/task setup.
        """
        return self.conversation_history[self.base_prompt_length:]

    def rebuild_with_updated_fewshots(self,
                                    old_fewshots: Union[List[str], str],
                                    new_fewshots: Union[List[str], str]) -> None:
        """
        Rebuild conversation with updated few-shot examples.

        This method updates the few-shot examples in the base prompt
        while preserving the conversation turns.
        """
        # Store conversation turns
        conversation_turns = self.get_conversation_turns_only()

        # Update base prompt with new fewshots
        base_updated = self.update_prompt_with_fewshots(
            self.conversation_history[:self.base_prompt_length],
            old_fewshots,
            new_fewshots
        )

        # Rebuild conversation history
        self.conversation_history = base_updated + conversation_turns
        self.base_prompt_length = len(base_updated)

    def store_build_parameters(self, **params) -> None:
        """
        Store the parameters used for building the base prompt.

        This allows for rebuilding the base prompt when dynamic updates occur.
        """
        self._last_build_params = params

    # ==================== Advanced Conversation Management ====================

    def prompt_agent_for_llm(self,
                           llm_callable: Callable,
                           long_context_llm_callable: Callable,
                           update_callback: Optional[Callable] = None,
                           testing: bool = False,
                           print_callback: Optional[Callable] = None,
                           token_counter: Optional[Callable] = None,
                           long_pass: Optional[bool] = None) -> str:
        """
        Complete agent prompting workflow for LLM interaction.

        This method handles the full workflow: dynamic updates → prompt preparation → LLM call.
        """
        # Update dynamic components if callback provided
        if update_callback is not None:
            self.update_dynamic_components(update_callback=update_callback)

        # Prepare final prompt
        prompt_history = self.prepare_llm_input(collapse_messages=True)

        # Handle testing mode
        if testing:
            if print_callback:
                print('###################################')
                for prompt in prompt_history:
                    print_callback(prompt, token_counter)
            return input()

        # Call LLM with error handling
        try:
            return llm_callable(prompt_history, stop=['\n', '\n\n'])
        except Exception as e:  # Catch BadRequestError and others
            if 'BadRequest' in str(type(e)):
                while long_pass is None:
                    res = input('Changing to long context LLM. Press Enter to continue.\n')
                    if res == 'pass':
                        long_pass = True
                    elif res != '':
                        continue
                    break
                return long_context_llm_callable(prompt_history, stop=['\n', '\n\n'])
            else:
                raise e

    def handle_agent_step(self,
                        llm_parser: Callable,
                        llm_callable: Callable,
                        long_context_llm_callable: Callable,
                        current_step: int,
                        update_callback: Optional[Callable] = None,
                        testing: bool = False,
                        print_callback: Optional[Callable] = None,
                        token_counter: Optional[Callable] = None,
                        long_pass: Optional[bool] = None) -> tuple:
        """
        Handle a complete agent step including LLM interaction and response parsing.

        Returns: (message, message_type, others) tuple from llm_parser
        """
        # Get LLM response
        llm_response = self.prompt_agent_for_llm(
            llm_callable=llm_callable,
            long_context_llm_callable=long_context_llm_callable,
            update_callback=update_callback,
            testing=testing,
            print_callback=print_callback,
            token_counter=token_counter,
            long_pass=long_pass
        )

        # Parse response
        message, message_type, others = llm_parser(llm_response, current_step, False)

        # Add parsed message to conversation
        self.add_conversation_turn(message)

        return message, message_type, others

    def handle_observation(self,
                         observation_message: Any,
                         operation: str = 'append',
                         last_observation_content: Optional[str] = None) -> None:
        """
        Handle environment observation in the conversation.

        Args:
            observation_message: The observation message to add
            operation: 'append' or 'replace' operation
            last_observation_content: Content to replace (for 'replace' operation)
        """
        if operation == 'append':
            self.add_conversation_turn(observation_message)
        elif operation == 'replace' and last_observation_content:
            # Handle replacement in conversation history
            for message in self.conversation_history:
                if last_observation_content in message.content:
                    message.content = message.content.replace(
                        last_observation_content,
                        observation_message.content
                    )
                    break

    def get_prompt_history_for_compatibility(self) -> List[ChatMessage]:
        """
        Get conversation history for backward compatibility.

        Returns the current conversation history for agents that still
        reference prompt_history directly.
        """
        return self.conversation_history

    def update_fewshots_dynamically(self,
                                  old_fewshots: Union[List[str], str],
                                  new_fewshots: Union[List[str], str]) -> None:
        """
        Update few-shot examples dynamically while preserving conversation.

        This method is specifically for ExpelAgent's dynamic few-shot replacement.
        """
        self.rebuild_with_updated_fewshots(old_fewshots, new_fewshots)

    def insert_rules_or_insights(self, rules: str) -> None:
        """
        Insert rules or insights into the conversation base prompt.

        This method rebuilds the base prompt with rules included.
        """
        if hasattr(self, '_last_build_params'):
            # Store conversation turns
            conversation_turns = self.get_conversation_turns_only()

            # Update build parameters with rules
            build_params = self._last_build_params.copy()
            build_params['rules'] = rules
            build_params['is_training'] = False
            build_params['no_rules'] = False

            # Rebuild base prompt with rules
            base_prompt = self.build_complete_prompt(**build_params)

            # Update conversation
            self.conversation_history = base_prompt + conversation_turns
            self.base_prompt_length = len(base_prompt)

    def get_conversation_statistics(self) -> dict:
        """
        Get statistics about the current conversation.

        Returns information useful for debugging and monitoring.
        """
        return {
            'total_messages': len(self.conversation_history),
            'base_prompt_length': self.base_prompt_length,
            'conversation_turns': len(self.get_conversation_turns_only()),
            'last_build_params': getattr(self, '_last_build_params', None)
        }