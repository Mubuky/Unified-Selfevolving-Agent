from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Tuple, Callable
from langchain_core.messages import HumanMessage

class BaseManager(ABC):
    """
    Abstract base class for insight extraction and rule management.

    This class defines the interface for managing the insights extraction process
    in the Training Phase 2 of the ExpeL framework, including critique generation,
    rule parsing, and rule updating operations.
    """

    @abstractmethod
    def create_rules(
        self,
        training_ids: List[int],
        cache_fold: int = None,
        load_cache_fold: int = None,
        logging_dir: str = None,
        run_name: str = 'run',
        loaded_dict: Dict[str, Any] = None,
        loaded_log: str = None,
        eval_idx_list: List[int] = None,
        saving_dict: bool = False,
    ) -> str:
        """
        Main method for extracting rules from training trajectories.

        Args:
            training_ids: List of training task IDs to process
            cache_fold: Current fold number for caching
            load_cache_fold: Fold number to load from cache
            logging_dir: Directory for saving logs
            run_name: Name of the current run
            loaded_dict: Previously saved agent state dictionary
            loaded_log: Previously saved log content
            eval_idx_list: List of evaluation indices
            saving_dict: Whether to save intermediate dictionaries

        Returns:
            Complete log of the rule extraction process
        """
        pass

    @abstractmethod
    def prompt_critique(
        self,
        success_history: str,
        fail_history: str,
        existing_rules: List[str] = None,
        task: str = None,
        reflections: List[str] = None
    ) -> str:
        """
        Generate critiques by prompting the LLM with success/failure trajectories.

        Args:
            success_history: Successful task trajectory
            fail_history: Failed task trajectory
            existing_rules: Current rule set for updating
            task: Task description
            reflections: Previous reflections

        Returns:
            Generated critique text from LLM
        """
        pass

    @abstractmethod
    def build_critique_prompt(
        self,
        success_history: str,
        fail_history: str = None,
        existing_rules: List[str] = None,
        task: str = None,
        reflections: List[str] = None
    ) -> List[HumanMessage]:
        """
        Build the prompt for critique generation.

        Args:
            success_history: Successful task trajectory
            fail_history: Failed task trajectory
            existing_rules: Current rule set
            task: Task description
            reflections: Previous reflections

        Returns:
            List of formatted messages for LLM prompting
        """
        pass

    @abstractmethod
    def task_critique(
        self,
        task: str,
        return_log: bool = False
    ) -> Union[None, str]:
        """
        Generate critiques for a specific task by comparing success/failure trajectories.

        Args:
            task: Task to generate critiques for
            return_log: Whether to return detailed log

        Returns:
            Log of critique generation process if return_log=True
        """
        pass

    @abstractmethod
    def success_critique(
        self,
        training_ids: List[int]
    ) -> None:
        """
        Generate critiques from successful trajectories only.

        Args:
            training_ids: List of training task IDs to process
        """
        pass

    @abstractmethod
    def failure_critique(self) -> None:
        """
        Generate critiques from multiple failed trajectories for the same task.
        """
        pass

    @abstractmethod
    def parse_rules(self, llm_text: str) -> List[Tuple[str, str]]:
        """
        Parse LLM output to extract rule operations (ADD/EDIT/REMOVE/AGREE).

        Args:
            llm_text: Raw text output from LLM

        Returns:
            List of (operation, rule_text) tuples
        """
        pass

    @abstractmethod
    def update_rules(
        self,
        rules: List[Tuple[str, int]],
        operations: List[Tuple[str, str]],
        list_full: bool = False
    ) -> List[Tuple[str, int]]:
        """
        Update rule list based on parsed operations.

        Args:
            rules: Current list of (rule_text, count) tuples
            operations: List of (operation, rule_text) tuples
            list_full: Whether the rule list is at capacity

        Returns:
            Updated list of (rule_text, count) tuples
        """
        pass