"""
Base Storage Class for ExpeL Framework

This module defines the abstract base class for all ExpeL storage systems.
It manages the data persistence and transfer chain across training and evaluation phases.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class BaseStorage(ABC):
    """
    Abstract base class for ExpeL storage systems.

    This class defines the interface for managing the data transfer chain:
    Training Phase 1 (Experience) → Training Phase 2 (Insights) → Evaluation
    """

    def __init__(self, cfg):
        """
        Initialize the storage system with configuration.

        Args:
            cfg: Hydra configuration object containing storage settings
        """
        self.cfg = cfg
        self.benchmark_name = cfg.benchmark.name
        self.log_dir = cfg.log_dir
        self._setup_paths()

    def _setup_paths(self):
        """Setup directory paths for different storage locations."""
        self.base_path = Path(self.log_dir) / self.benchmark_name / self.cfg.agent_type
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Define paths for different phases
        self.insights_path = self.base_path / 'extracted_insights'
        self.insights_path.mkdir(exist_ok=True)

        self.eval_path = self.base_path / 'eval'
        self.eval_path.mkdir(exist_ok=True)

    # ==================== Training Phase 1: Experience Collection ====================

    @abstractmethod
    def save_experience(
        self,
        run_name: str,
        agent_dict: Dict[str, Any],
        log: str,
        true_log: str
    ) -> None:
        """
        Save experience collection data.

        Args:
            run_name: Name of the training run
            agent_dict: Agent state dictionary containing trajectories
            log: Execution log text
            true_log: Complete execution log with all details
        """
        pass

    @abstractmethod
    def load_experience(
        self,
        run_name: str
    ) -> Dict[str, Any]:
        """
        Load experience data for insights extraction.

        Args:
            run_name: Name of the training run to load

        Returns:
            Dictionary containing agent state and logs
        """
        pass

    # ==================== Training Phase 2: Insights Extraction ====================

    @abstractmethod
    def save_insights(
        self,
        run_name: str,
        agent_dict: Dict[str, Any],
        log: str,
        original_run_name: Optional[str] = None
    ) -> None:
        """
        Save training phase 2 data (insights extraction).

        Args:
            run_name: Name of the insights extraction run
            agent_dict: Agent state dictionary containing insights and rules
            log: Execution log text
            original_run_name: Original training phase 1 run name for reference
        """
        pass

    @abstractmethod
    def load_insights(
        self,
        run_name: str
    ) -> Dict[str, Any]:
        """
        Load insights extraction data for evaluation.

        Args:
            run_name: Name of the insights extraction run to load

        Returns:
            Dictionary containing agent state with insights and logs
        """
        pass

    # ==================== Evaluation Phase ====================

    @abstractmethod
    def save_evaluation_results(
        self,
        run_name: str,
        agent_dict: Dict[str, Any],
        log: str,
        true_log: str
    ) -> None:
        """
        Save evaluation phase results.

        Args:
            run_name: Name of the evaluation run
            agent_dict: Agent state dictionary containing evaluation results
            log: Execution log text
            true_log: Complete execution log with all details
        """
        pass

    @abstractmethod
    def load_evaluation_checkpoint(
        self,
        run_name: str
    ) -> Dict[str, Any]:
        """
        Load evaluation checkpoint for resuming evaluation.

        Args:
            run_name: Name of the evaluation run to resume

        Returns:
            Dictionary containing evaluation state and logs
        """
        pass

    # ==================== Generic Methods ====================

    @abstractmethod
    def exists(self, run_name: str, phase: str) -> bool:
        """
        Check if a run exists for a specific phase.

        Args:
            run_name: Name of the run
            phase: Phase name ('train', 'insights', 'eval')

        Returns:
            True if the run exists, False otherwise
        """
        pass

    @abstractmethod
    def get_run_path(self, run_name: str, phase: str) -> Path:
        """
        Get the storage path for a specific run and phase.

        Args:
            run_name: Name of the run
            phase: Phase name ('train', 'insights', 'eval')

        Returns:
            Path object pointing to the run's storage location
        """
        pass

    # ==================== Utility Methods ====================

    def get_available_runs(self, phase: str) -> List[str]:
        """
        Get list of available run names for a specific phase.

        Args:
            phase: Phase name ('train', 'insights', 'eval')

        Returns:
            List of available run names
        """
        if phase == 'train':
            pattern = '*.pkl'
            search_path = self.base_path
        elif phase == 'insights':
            pattern = '*.pkl'
            search_path = self.insights_path
        elif phase == 'eval':
            pattern = '*.pkl'
            search_path = self.eval_path
        else:
            return []

        pkl_files = list(search_path.glob(pattern))
        return [f.stem for f in pkl_files]

    def cleanup_run(self, run_name: str, phase: str) -> None:
        """
        Clean up files for a specific run and phase.

        Args:
            run_name: Name of the run
            phase: Phase name ('train', 'insights', 'eval')
        """
        run_path = self.get_run_path(run_name, phase)
        if run_path.exists():
            # Remove .pkl, .txt, and _true.txt files
            for suffix in ['.pkl', '.txt', '_true.txt']:
                file_path = run_path.parent / f"{run_name}{suffix}"
                if file_path.exists():
                    file_path.unlink()

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about the storage system.

        Returns:
            Dictionary with storage information
        """
        return {
            'benchmark': self.benchmark_name,
            'base_path': str(self.base_path),
            'insights_path': str(self.insights_path),
            'eval_path': str(self.eval_path),
            'available_train_runs': self.get_available_runs('train'),
            'available_insights_runs': self.get_available_runs('insights'),
            'available_eval_runs': self.get_available_runs('eval')
        }

    def __repr__(self) -> str:
        """String representation of the storage system."""
        return f"{self.__class__.__name__}(benchmark={self.benchmark_name}, base_path={self.base_path})"