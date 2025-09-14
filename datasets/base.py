"""
Base Dataset Class for ExpeL Framework

This module defines the abstract base class for all ExpeL datasets.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple


class BaseDataset(ABC):
    """
    Abstract base class for ExpeL datasets.

    This class defines the interface that all ExpeL datasets must implement.
    It handles configuration management, data range processing, and provides
    the abstract method for loading tasks.
    """

    def __init__(self, cfg):
        """
        Initialize the dataset with configuration.

        Args:
            cfg: Hydra configuration object containing benchmark settings
        """
        self.cfg = cfg
        self.benchmark_name = cfg.benchmark.name
        self.task_file = cfg.benchmark.task_file
        self.task_prefix = cfg.benchmark.task_prefix
        # Always enable debug mode for now since it's helpful for data range validation
        self._debug_mode = True

    def get_data_range(self, mode: str = 'train') -> Optional[Tuple[int, int]]:
        """
        Get data range based on mode and configuration.

        Args:
            mode: Data mode ('train' or 'eval')

        Returns:
            Tuple of (start, end) indices, or None for full dataset
        """
        if hasattr(self.cfg.benchmark, 'data_split') and self.cfg.benchmark.data_split:
            if mode == 'eval':
                data_range = tuple(self.cfg.benchmark.data_split.eval_range)
            else:  # mode == 'train' or default
                data_range = tuple(self.cfg.benchmark.data_split.train_range)

            if self._debug_mode:
                print(f"[DEBUG] Using data_split config for {mode} mode: {data_range}")
            return data_range
        else:
            # Fallback to original logic for backwards compatibility
            if self._debug_mode:
                print(f"[DEBUG] No data_split config found for {mode} mode, using fallback")
            return None

    @abstractmethod
    def load_tasks(self, mode: str = 'train') -> List[Dict[str, Any]]:
        """
        Load tasks for the specified mode.

        Args:
            mode: Data mode ('train' or 'eval')

        Returns:
            List of task dictionaries, each containing:
                - 'task': Task description string
                - 'env_kwargs': Environment-specific kwargs
                - 'env_name': Environment name
        """
        pass

    def _apply_data_range(self, all_data: List[Any], mode: str) -> List[Any]:
        """
        Apply data range filtering to the full dataset.

        Args:
            all_data: Complete list of data items
            mode: Data mode ('train' or 'eval')

        Returns:
            Filtered list of data items
        """
        data_range = self.get_data_range(mode)

        if data_range is not None:
            selected_data = all_data[data_range[0]:data_range[1]]
        else:
            # Fallback to original logic
            if hasattr(self.cfg.benchmark, 'dataset') and hasattr(self.cfg.benchmark.dataset, 'num_train_games'):
                if self.cfg.benchmark.dataset.num_train_games > 0:
                    selected_data = all_data[:self.cfg.benchmark.dataset.num_train_games]
                else:
                    selected_data = all_data
            else:
                selected_data = all_data

        if self._debug_mode:
            print(f"[DEBUG] {self.benchmark_name} {mode} mode: Total data loaded: {len(all_data)}, "
                  f"Selected range: {data_range}, Final data count: {len(selected_data)}")

        return selected_data

    def __len__(self, mode: str = 'train') -> int:
        """
        Get the number of tasks for the specified mode.

        Args:
            mode: Data mode ('train' or 'eval')

        Returns:
            Number of tasks
        """
        return len(self.load_tasks(mode))

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return f"{self.__class__.__name__}(benchmark={self.benchmark_name}, file={self.task_file})"