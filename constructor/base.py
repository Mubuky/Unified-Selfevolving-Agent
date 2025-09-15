"""
Abstract Base Class for ExpeL Prompt Constructors

This module defines the interface for all prompt construction systems used in the ExpeL framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from langchain_core.messages import ChatMessage
from langchain_core.prompts import PromptTemplate


class BaseConstructor(ABC):
    """
    Abstract base class for ExpeL prompt construction systems.

    This class defines the interface for managing prompt construction across
    training and evaluation phases, including system messages, few-shots, rules, and tasks.
    """

    def __init__(self,
                 benchmark_name: str,
                 system_instruction: Union[str, Dict[str, str]],
                 human_instruction: Callable,
                 system_prompt: Callable,
                 rule_template: Optional[PromptTemplate] = None,
                 ai_name: str = "ExpelAgent",
                 human_instruction_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize the constructor with configuration.

        Args:
            benchmark_name: Name of the benchmark (alfworld, webshop, etc.)
            system_instruction: System-level instructions for the agent
            human_instruction: Human instruction template
            system_prompt: System prompt template
            rule_template: Template for inserting rules/insights (evaluation phase)
            ai_name: Name of the AI agent
            human_instruction_kwargs: Additional kwargs for human instruction
        """
        self.benchmark_name = benchmark_name
        self.system_instruction = system_instruction
        self.human_instruction = human_instruction
        self.system_prompt = system_prompt
        self.rule_template = rule_template
        self.ai_name = ai_name
        self.human_instruction_kwargs = human_instruction_kwargs or {}

    @abstractmethod
    def build_system_prompt(self) -> List[ChatMessage]:
        """
        Build system prompt messages.

        Returns:
            List of system messages
        """
        pass

    @abstractmethod
    def build_fewshot_prompt(self,
                            fewshots: List[str],
                            prompt_type: str = 'react_type') -> List[ChatMessage]:
        """
        Build few-shot example prompt messages.

        Args:
            fewshots: List of few-shot example strings
            prompt_type: Type of prompt being built

        Returns:
            List of few-shot messages
        """
        pass

    @abstractmethod
    def build_rules_prompt(self, rules: str) -> List[ChatMessage]:
        """
        Build rules/insights prompt messages (evaluation phase).

        Args:
            rules: Formatted rules/insights string

        Returns:
            List of rules messages
        """
        pass

    @abstractmethod
    def build_task_prompt(self, task: str) -> List[ChatMessage]:
        """
        Build task description prompt messages.

        Args:
            task: Current task description

        Returns:
            List of task messages
        """
        pass

    @abstractmethod
    def collapse_prompts(self, prompt_history: List[ChatMessage]) -> List[ChatMessage]:
        """
        Collapse consecutive messages of the same type.

        Args:
            prompt_history: List of chat messages

        Returns:
            Collapsed list of messages
        """
        pass

    @abstractmethod
    def build_complete_prompt(self,
                             fewshots: List[str],
                             task: str,
                             rules: Optional[str] = None,
                             is_training: bool = True,
                             no_rules: bool = False) -> List[ChatMessage]:
        """
        Build complete prompt combining all components.

        Args:
            fewshots: List of few-shot examples
            task: Current task description
            rules: Rules/insights string (evaluation phase)
            is_training: Whether in training mode
            no_rules: Whether to skip rules insertion

        Returns:
            Complete prompt history
        """
        pass

    def get_constructor_info(self) -> Dict[str, Any]:
        """
        Get information about the constructor.

        Returns:
            Dictionary with constructor information
        """
        return {
            'constructor_type': self.__class__.__name__,
            'benchmark': self.benchmark_name,
            'ai_name': self.ai_name,
            'has_rule_template': self.rule_template is not None,
            'human_instruction_kwargs': self.human_instruction_kwargs
        }

    def __repr__(self) -> str:
        """String representation of the constructor."""
        return f"{self.__class__.__name__}(benchmark={self.benchmark_name}, ai_name={self.ai_name})"