"""
ExpelStorage Implementation

This module implements the concrete ExpelStorage class that handles
the data persistence and transfer chain for the ExpeL framework.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from copy import deepcopy

from .base import BaseStorage

# Import utility functions - handle import error gracefully
try:
    from utils import save_trajectories_log, load_trajectories_log
except ImportError:
    # Fallback implementations if utils not available
    def save_trajectories_log(path, log=None, dicts=None, true_log=None, **kwargs):
        raise NotImplementedError("save_trajectories_log not available")

    def load_trajectories_log(path, **kwargs):
        raise NotImplementedError("load_trajectories_log not available")


class ExpelStorage(BaseStorage):
    """
    Concrete storage implementation for ExpeL framework.

    This class handles the complete data transfer chain across the three phases:
    1. Training Phase 1: Experience Collection
    2. Training Phase 2: Insights Extraction
    3. Evaluation: Performance Testing
    """

    def __init__(self, cfg):
        """
        Initialize ExpelStorage with configuration.

        Args:
            cfg: Hydra configuration object
        """
        super().__init__(cfg)
        self._debug_mode = getattr(cfg, 'debug', True)

    def _log_operation(self, operation: str, details: str):
        """Log storage operations if debug mode is enabled."""
        if self._debug_mode:
            print(f"[STORAGE] {operation}: {details}")

    # ==================== Training Phase 1: Experience Collection ====================

    def save_experience(
        self,
        run_name: str,
        agent_dict: Dict[str, Any],
        log: str,
        true_log: str
    ) -> None:
        """
        Save experience collection data.

        This saves the agent's state after collecting experiences from training tasks,
        including succeeded_trial_history and failed_trial_history.
        """
        self._log_operation(
            "SAVE_EXPERIENCE",
            f"run_name={run_name}, path={self.base_path}"
        )

        # Package data in the format expected by save_trajectories_log
        dicts = [deepcopy(agent_dict)]

        save_trajectories_log(
            path=str(self.base_path),
            log=log,
            dicts=dicts,
            true_log=true_log,
            run_name=run_name
        )

    def load_experience(
        self,
        run_name: str
    ) -> Dict[str, Any]:
        """
        Load experience data for insights extraction.

        Returns the agent state and logs from the experience collection phase.
        """
        self._log_operation(
            "LOAD_EXPERIENCE",
            f"run_name={run_name}, path={self.base_path}"
        )

        return load_trajectories_log(
            path=str(self.base_path),
            run_name=run_name,
            load_log=True,
            load_dict=True,
            load_true_log=True
        )

    # ==================== Training Phase 2: Insights Extraction ====================

    def save_insights(
        self,
        run_name: str,
        agent_dict: Dict[str, Any],
        log: str,
        original_run_name: Optional[str] = None
    ) -> None:
        """
        Save training phase 2 data (insights extraction).

        This saves the agent's state after extracting insights and rules
        from the collected experiences.
        """
        self._log_operation(
            "SAVE_INSIGHTS",
            f"run_name={run_name}, path={self.insights_path}, original={original_run_name}"
        )

        # Package data in the format expected by save_trajectories_log
        dicts = [deepcopy(agent_dict)]

        # Add reference to original training run if provided
        if original_run_name:
            dicts[0]['_original_training_run'] = original_run_name

        save_trajectories_log(
            path=str(self.insights_path),
            log=log,
            dicts=dicts,
            true_log=None,  # Don't save true_log for insights phase
            run_name=run_name,
            save_true_log=False
        )

    def load_insights(
        self,
        run_name: str
    ) -> Dict[str, Any]:
        """
        Load insights extraction data for evaluation.

        Returns the agent state with extracted insights and rules.
        """
        # Handle case where run_name already contains extracted_insights/ prefix
        if run_name.startswith('extracted_insights/'):
            actual_run_name = run_name[len('extracted_insights/'):]
            load_path = str(self.insights_path)
        else:
            actual_run_name = run_name
            load_path = str(self.insights_path)

        self._log_operation(
            "LOAD_INSIGHTS",
            f"run_name={run_name} -> actual_run_name={actual_run_name}, path={load_path}"
        )

        return load_trajectories_log(
            path=load_path,
            run_name=actual_run_name,
            load_log=True,
            load_dict=True,
            load_true_log=False
        )

    # ==================== Evaluation Phase ====================

    def save_evaluation_results(
        self,
        run_name: str,
        agent_dict: Dict[str, Any],
        log: str,
        true_log: str
    ) -> None:
        """
        Save evaluation phase results.

        This saves the evaluation results and performance metrics.
        """
        self._log_operation(
            "SAVE_EVALUATION",
            f"run_name={run_name}, path={self.eval_path}"
        )

        # Package data in the format expected by save_trajectories_log
        dicts = [deepcopy(agent_dict)]

        save_trajectories_log(
            path=str(self.eval_path),
            log=log,
            dicts=dicts,
            true_log=true_log,
            run_name=run_name
        )

    def load_evaluation_checkpoint(
        self,
        run_name: str
    ) -> Dict[str, Any]:
        """
        Load evaluation checkpoint for resuming evaluation.

        Returns the evaluation state for resuming interrupted evaluation.
        """
        # Handle case where run_name already contains eval/ prefix
        if run_name.startswith('eval/'):
            actual_run_name = run_name[len('eval/'):]
            load_path = str(self.eval_path)
        else:
            actual_run_name = run_name
            load_path = str(self.eval_path)

        self._log_operation(
            "LOAD_EVALUATION_CHECKPOINT",
            f"run_name={run_name} -> actual_run_name={actual_run_name}, path={load_path}"
        )

        return load_trajectories_log(
            path=load_path,
            run_name=actual_run_name,
            load_log=True,
            load_dict=True,
            load_true_log=True
        )

    # ==================== Generic Methods ====================

    def exists(self, run_name: str, phase: str) -> bool:
        """
        Check if a run exists for a specific phase.

        Args:
            run_name: Name of the run
            phase: Phase name ('train', 'insights', 'eval')

        Returns:
            True if the run exists, False otherwise
        """
        run_path = self.get_run_path(run_name, phase)
        pkl_file = run_path.parent / f"{run_name}.pkl"
        return pkl_file.exists()

    def get_run_path(self, run_name: str, phase: str) -> Path:
        """
        Get the storage path for a specific run and phase.

        Args:
            run_name: Name of the run
            phase: Phase name ('train', 'insights', 'eval')

        Returns:
            Path object pointing to the run's storage location
        """
        if phase == 'train':
            return self.base_path / f"{run_name}.pkl"
        elif phase == 'insights':
            return self.insights_path / f"{run_name}.pkl"
        elif phase == 'eval':
            return self.eval_path / f"{run_name}.pkl"
        else:
            raise ValueError(f"Unknown phase: {phase}")

    # ==================== High-level Interface Methods ====================

    def save_checkpoint(
        self,
        phase: str,
        run_name: str,
        agent_dict: Dict[str, Any],
        log: str,
        true_log: Optional[str] = None
    ) -> None:
        """
        Generic checkpoint saving method.

        Args:
            phase: Phase name ('train', 'insights', 'eval')
            run_name: Name of the run
            agent_dict: Agent state dictionary
            log: Execution log
            true_log: Complete execution log (optional)
        """
        if phase == 'train':
            self.save_experience(run_name, agent_dict, log, true_log or log)
        elif phase == 'insights':
            self.save_insights(run_name, agent_dict, log)
        elif phase == 'eval':
            self.save_evaluation_results(run_name, agent_dict, log, true_log or log)
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def load_checkpoint(
        self,
        phase: str,
        run_name: str
    ) -> Dict[str, Any]:
        """
        Generic checkpoint loading method.

        Args:
            phase: Phase name ('train', 'insights', 'eval')
            run_name: Name of the run

        Returns:
            Dictionary containing loaded data
        """
        if phase == 'train':
            return self.load_experience(run_name)
        elif phase == 'insights':
            return self.load_insights(run_name)
        elif phase == 'eval':
            return self.load_evaluation_checkpoint(run_name)
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def get_data_transfer_chain_info(self) -> Dict[str, Any]:
        """
        Get information about the complete data transfer chain.

        Returns:
            Dictionary with transfer chain information
        """
        return {
            'storage_type': self.__class__.__name__,
            'benchmark': self.benchmark_name,
            'transfer_chain': {
                'phase1_train': {
                    'purpose': 'Experience Collection',
                    'input': 'Task environment interactions',
                    'output': 'succeeded_trial_history, failed_trial_history',
                    'path': str(self.base_path),
                    'available_runs': self.get_available_runs('train')
                },
                'phase2_insights': {
                    'purpose': 'Insights Extraction',
                    'input': 'Experience trajectories from Phase 1',
                    'output': 'rule_items, extracted insights',
                    'path': str(self.insights_path),
                    'available_runs': self.get_available_runs('insights')
                },
                'phase3_eval': {
                    'purpose': 'Performance Evaluation',
                    'input': 'Insights and rules from Phase 2',
                    'output': 'Performance metrics, evaluation results',
                    'path': str(self.eval_path),
                    'available_runs': self.get_available_runs('eval')
                }
            }
        }