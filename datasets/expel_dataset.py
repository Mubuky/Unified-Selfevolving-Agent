"""
ExpelDataset Implementation

This module implements the concrete ExpelDataset class that handles
data loading for different benchmarks in the ExpeL framework.
"""

import json
import random
from typing import List, Dict, Any

from .base import BaseDataset
from utils import get_env_name_from_gamefile

class ExpelDataset(BaseDataset):
    """
    Concrete dataset implementation for ExpeL framework.

    This class handles data loading for different benchmarks including
    ALFWorld, HotpotQA, WebShop, and FEVER.
    """

    def __init__(self, cfg):
        """
        Initialize ExpelDataset with configuration.

        Args:
            cfg: Hydra configuration object
        """
        super().__init__(cfg)

        # Initialize benchmark-specific settings
        self._init_benchmark_settings()

    def _init_benchmark_settings(self):
        """Initialize benchmark-specific settings and data loaders."""
        self.benchmark_loaders = {
            'hotpotqa': self._load_hotpotqa_tasks,
            'alfworld': self._load_alfworld_tasks,
            'webshop': self._load_webshop_tasks,
            'fever': self._load_fever_tasks,
        }

        # Taken from ReAct Github - for FEVER benchmark
        self.idxs = list(range(7405))
        random.Random(233).shuffle(self.idxs)

    def load_tasks(self, mode: str = 'train') -> List[Dict[str, Any]]:
        """
        Load tasks for the specified mode.

        Args:
            mode: Data mode ('train' or 'eval')

        Returns:
            List of task dictionaries
        """
        if self.benchmark_name not in self.benchmark_loaders:
            raise ValueError(f"Unsupported benchmark: {self.benchmark_name}")

        loader_func = self.benchmark_loaders[self.benchmark_name]
        return loader_func(mode)

    def _load_hotpotqa_tasks(self, mode: str) -> List[Dict[str, Any]]:
        """Load HotpotQA tasks."""
        with open(self.task_file, "r") as f:
            all_data = json.load(f)

        selected_data = self._apply_data_range(all_data, mode)

        tasks = [
            {
                'task': f'{self.task_prefix}{row["question"]}',
                'env_kwargs': {
                    'question': row['question'],
                    'key': row['answer'],
                },
                'env_name': 'hotpotqa',
            }
            for row in selected_data
        ]

        return tasks

    def _load_alfworld_tasks(self, mode: str) -> List[Dict[str, Any]]:
        """Load ALFWorld tasks."""
        with open(self.task_file, "r") as f:
            all_data = json.load(f)

        selected_data = self._apply_data_range(all_data, mode)

        tasks = [
            {
                'task': f'{self.task_prefix}{row["goal"]}',
                'env_kwargs': {
                    'config': self.cfg.benchmark,
                    "gamefile": row["gamefile"],
                },
                'env_name': get_env_name_from_gamefile(row['gamefile'])
            }
            for row in selected_data
        ]

        return tasks

    def _load_webshop_tasks(self, mode: str) -> List[Dict[str, Any]]:
        """Load WebShop tasks."""
        with open(self.task_file, "r") as f:
            all_data = json.load(f)

        selected_data = self._apply_data_range(all_data, mode)

        tasks = [
            {
                'task': f'{self.task_prefix}{row["task"]}',
                'env_kwargs': {
                    'session_idx': row["session_idx"],
                },
                'env_name': 'webshop'
            }
            for row in selected_data
        ]

        return tasks

    def _load_fever_tasks(self, mode: str) -> List[Dict[str, Any]]:
        """Load FEVER tasks."""
        # NOTE: FEVER implementation is commented out in the original code
        # This is a placeholder implementation
        # from .fever.fever import FeverEnv  # Would need to be imported if used

        # Use first 100 tasks for fever (from original implementation)
        selected_idxs = self.idxs[:100]

        # Apply data range if specified
        if hasattr(self.cfg.benchmark, 'data_split') and self.cfg.benchmark.data_split:
            data_range = self.get_data_range(mode)
            if data_range is not None:
                selected_idxs = selected_idxs[data_range[0]:data_range[1]]

        # This would require FeverEnv to be imported and working
        tasks = [
            {
                'task': f'{self.task_prefix}FEVER_TASK_{idx}',  # Placeholder
                'env_kwargs': {
                    'idx': idx,
                },
                'env_name': 'fever',
            }
            for idx in selected_idxs
        ]

        return tasks

    def get_tasks_by_benchmark(self, benchmark_name: str, mode: str = 'train') -> List[Dict[str, Any]]:
        """
        Get tasks for a specific benchmark (utility method).

        Args:
            benchmark_name: Name of the benchmark
            mode: Data mode ('train' or 'eval')

        Returns:
            List of task dictionaries
        """
        original_benchmark = self.benchmark_name
        self.benchmark_name = benchmark_name
        try:
            tasks = self.load_tasks(mode)
        finally:
            self.benchmark_name = original_benchmark
        return tasks

    def get_benchmark_info(self) -> Dict[str, Any]:
        """
        Get information about the current benchmark.

        Returns:
            Dictionary with benchmark information
        """
        return {
            'name': self.benchmark_name,
            'task_file': self.task_file,
            'task_prefix': self.task_prefix,
            'supported_benchmarks': list(self.benchmark_loaders.keys())
        }