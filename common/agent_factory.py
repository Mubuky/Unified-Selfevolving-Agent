"""
Shared agent factory to eliminate code duplication between train.py and insight_extraction.py
"""
import os
from pathlib import Path
from functools import partial
from omegaconf import DictConfig
from typing import Dict, Any, Optional, Tuple

from agent import AGENT
from prompts.templates.system import system_message_prompt
from prompts.templates.human import HUMAN_CRITIQUES
from prompts import (
    SYSTEM_INSTRUCTION,
    HUMAN_INSTRUCTION,
    FEWSHOTS,
    REFLECTION_FEWSHOTS,
    HUMAN_REFLECTION_INSTRUCTION,
    SYSTEM_REFLECTION_INSTRUCTION,
    SYSTEM_CRITIQUE_INSTRUCTION,
    RULE_TEMPLATE,
    LLM_PARSER,
    OBSERVATION_FORMATTER,
    STEP_IDENTIFIER,
    CYCLER,
    STEP_CYCLER,
    REFLECTION_PREFIX,
    PREVIOUS_TRIALS_FORMATTER,
    STEP_STRIPPER,
    CRITIQUE_SUMMARY_SUFFIX,
)
from envs import ENVS, INIT_TASKS_FN
from memory import EMBEDDERS, RETRIEVERS
from models import LLM_CLS
from utils import get_fewshot_max_tokens
from storage import ExpelStorage

class AgentFactory:
    """Factory class for creating agents with shared configuration logic."""

    @staticmethod
    def get_api_credentials(cfg: DictConfig) -> Tuple[str, Optional[str]]:
        """Get OpenAI API credentials based on testing mode."""
        if cfg.testing:
            return 'NO_KEY_FOR_TESTING', None
        else:
            return os.environ.get('OPENAI_API_KEY'), os.environ.get('BASE_URL')

    @staticmethod
    def get_log_path(cfg: DictConfig) -> Path:
        """Generate standard log path."""
        return Path('/'.join([cfg.log_dir, cfg.benchmark.name, cfg.agent_type]))

    @staticmethod
    def initialize_storage(cfg: DictConfig) -> ExpelStorage:
        """Initialize storage system."""
        return ExpelStorage(cfg)

    @staticmethod
    def handle_auto_resume_confirmation(
        cfg: DictConfig,
        checkpoint_exists: bool,
        checkpoint_path: str = None
    ) -> None:
        """Handle auto-resume and overwrite confirmation logic - matches train.py exactly."""
        auto_resume = getattr(cfg, 'auto_resume', False)

        # This logic only applies when not resuming (matches train.py else branch)
        if checkpoint_exists and cfg.run_name != 'test' and not auto_resume:
            while True:
                res = input(f"Are you sure to overwrite '{cfg.run_name}'? (Y/N)\n").lower()
                if res == 'n':
                    exit(0)
                elif res == 'y':
                    break
        elif checkpoint_exists and auto_resume:
            if checkpoint_path:
                print(f"Auto-resume mode: Existing checkpoint will be overwritten")
            else:
                print(f"Auto-resume mode: Existing checkpoint will be overwritten")

    @staticmethod
    def create_agent(
        cfg: DictConfig,
        openai_api_key: str,
        base_url: Optional[str],
        mode: str = 'train',
        task_idx: Optional[int] = None,
        max_fewshot_tokens_override: Optional[int] = None,
        use_direct_max_fewshot_tokens: bool = False
    ):
        """
        Create agent with standardized configuration.

        Args:
            cfg: Hydra configuration
            openai_api_key: OpenAI API key
            base_url: Base URL for API
            mode: Mode for task loading ('train' or 'eval')
            task_idx: Starting task index (for resuming)
            max_fewshot_tokens_override: Override for max_fewshot_tokens
        """
        # Handle max_fewshot_tokens logic
        if max_fewshot_tokens_override is not None:
            max_fewshot_tokens = max_fewshot_tokens_override
        elif use_direct_max_fewshot_tokens:
            # For insight_extraction.py - use direct value without auto conversion
            max_fewshot_tokens = cfg.agent.retrieval_kwargs.max_fewshot_tokens
        elif cfg.agent.retrieval_kwargs.max_fewshot_tokens == 'auto':
            max_fewshot_tokens = get_fewshot_max_tokens(cfg.benchmark.name)
        else:
            max_fewshot_tokens = cfg.agent.retrieval_kwargs.max_fewshot_tokens

        agent_kwargs = {
            'name': cfg.ai_name,
            'system_instruction': SYSTEM_INSTRUCTION[cfg.benchmark.name],
            'human_instruction': HUMAN_INSTRUCTION[cfg.benchmark.name],
            'tasks': INIT_TASKS_FN[cfg.benchmark.name](cfg, mode=mode),
            'fewshots': FEWSHOTS[cfg.benchmark.name],
            'system_prompt': system_message_prompt,
            'env': ENVS[cfg.benchmark.name],
            'max_steps': cfg.benchmark.max_steps,
            'openai_api_key': openai_api_key,
            'base_url': base_url,
            'llm': cfg.agent.llm,
            'llm_builder': LLM_CLS,
            'reflection_fewshots': REFLECTION_FEWSHOTS[cfg.benchmark.name],
            'reflection_task_prompt': HUMAN_REFLECTION_INSTRUCTION[cfg.benchmark.name],
            'reflection_system_instruction': SYSTEM_REFLECTION_INSTRUCTION[cfg.benchmark.name],
            'reflection_system_prompt': SYSTEM_INSTRUCTION[cfg.benchmark.name],
            'max_relfection_depth': cfg.agent.max_reflection_depth if 'max_reflection_depth' in cfg.agent.keys() else 0,
            'system_critique_instructions': SYSTEM_CRITIQUE_INSTRUCTION[cfg.benchmark.name],
            'human_critiques': HUMAN_CRITIQUES,
            'max_num_rules': cfg.agent.max_num_rules if 'max_num_rules' in cfg.agent.keys() else 0,
            'rule_template': RULE_TEMPLATE[cfg.benchmark.name],
            'truncate_strategy': cfg.agent.truncate_strategy if 'truncate_strategy' in cfg.agent.keys() else None,
            'llm_parser': LLM_PARSER[cfg.benchmark.name],
            'observation_formatter': OBSERVATION_FORMATTER[cfg.benchmark.name],
            'embedder': EMBEDDERS(cfg.agent.retrieval_kwargs.embedder_type),
            'embedder_path': cfg.agent.retrieval_kwargs.embedder_path,
            'step_stripper': STEP_STRIPPER[cfg.benchmark.name],
            'retriever_cls': RETRIEVERS(cfg.agent.retrieval_kwargs.retriever_type),
            'message_splitter': CYCLER[cfg.benchmark.name],
            'identifier': STEP_IDENTIFIER[cfg.benchmark.name],
            'message_step_splitter': partial(STEP_CYCLER, benchmark=cfg.benchmark.name),
            'reflection_prefix': REFLECTION_PREFIX[cfg.benchmark.name],
            'previous_trials_formatter': PREVIOUS_TRIALS_FORMATTER[cfg.benchmark.name],
            'success_critique_num': cfg.agent.success_critique_num,
            'fewshot_strategy': cfg.agent.fewshot_strategy,
            'critique_truncate_strategy': cfg.agent.critique_truncate_strategy,
            'critique_summary_suffix': CRITIQUE_SUMMARY_SUFFIX,
            'testing': cfg.testing,
            'benchmark_name': cfg.benchmark.name,
            'reranker': cfg.agent.retrieval_kwargs.reranker,
            'buffer_retrieve_ratio': cfg.agent.retrieval_kwargs.buffer_retrieve_ratio,
            'max_fewshot_tokens': max_fewshot_tokens,
        }

        # Add task_idx only if provided (for train.py resume logic)
        if task_idx is not None:
            agent_kwargs['task_idx'] = task_idx

        return AGENT[cfg.agent_type](**agent_kwargs)

    @staticmethod
    def filter_agent_dict(agent_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Filter agent dictionary for saving, removing complex objects."""
        from agent.reflect import Count
        return {
            k: v for k, v in agent_dict.items()
            if type(v) in [list, set, str, bool, int, dict, Count]
            and k not in ['openai_api_key', 'llm']
        }

    @staticmethod
    def get_training_phase_no_load_list() -> list:
        """Get the standard no_load_list for training phase checkpoint loading."""
        return [
            'testing', 'max_relfection_depth', 'fewshot_strategy',
            'max_fewshot_tokens', 'task_idx', 'idx2task', 'task2idx', 'tasks'
        ]

    @staticmethod
    def get_insights_extraction_no_load_list() -> list:
        """Get the standard no_load_list for insights extraction checkpoint loading."""
        return [
            'ai_message', 'message_type_format', 'max_num_rules', 'testing',
            'human_critiques', 'system_critique_instructions', 'fewshot_strategy',
            'success', 'halted', 'fail', 'task_idx', 'prompt_history',
            'critique_truncate_strategy', 'success_critique_num', 'reflection_fewshots',
            'reflection_system_prompt', 'reflection_prefix', 'reflection_prompt_history',
            'reflections', 'previous_trial', 'perform_reflection', 'increment_task',
            'reflection_system_kwargs', 'prepend_human_instruction', 'name', 'tasks',
            'human_instruction_kwargs', 'all_system_instruction', 'all_fewshots',
            'max_steps', 'ordered_summary', 'fewshots', 'system_instruction',
            'num_fewshots', 'curr_step', 'log_idx', 'pretask_idx', 'reflect_interaction_idx',
            'truncated', 'reward', 'terminated', 'autoqregressive_model_instruction',
            'failed_training_task_idx', '_train', 'task', 'eval_idx_list', 'starting_fold',
            'starting_idx', 'critique_summary_suffix'
        ]