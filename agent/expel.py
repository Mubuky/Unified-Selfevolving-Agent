import random
from typing import List, Dict, Callable, Union, Any, Tuple
import re
from functools import partial

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import numpy as np
import openai
from scipy.spatial.distance import cosine

from agent import ReflectAgent, ReactAgent
from agent.reflect import Count
from utils import random_divide_list, save_trajectories_log, get_env_name_from_task
from memory import Trajectory
from retrieval import ExpelRetrieval
from manager import ExpelManager

from copy import deepcopy

class ExpelAgent(ReflectAgent):
    def __init__(self,
                 system_critique_instructions: Dict[str, str],
                 human_critiques: Dict[str, PromptTemplate],
                 rule_template: PromptTemplate,
                 max_num_rules: Union[int, str],
                 truncate_strategy: str,
                 embedder: Callable,
                 embedder_path: str,
                 step_stripper: Callable,
                 retriever_cls: Callable,
                 success_critique_num: int,
                 fewshot_strategy: str,
                 critique_truncate_strategy: str,
                 benchmark_name: str,
                 critique_summary_suffix: str,
                 max_fewshot_tokens: int,
                 reranker: str,
                 buffer_retrieve_ratio: int,
                 *args,
                 **kwargs,
                 ) -> None:
        self.benchmark_name = benchmark_name
        self.system_critique_instructions = system_critique_instructions
        self.human_critiques = human_critiques
        self.max_num_rules = max_num_rules
        self.rule_template = rule_template
        self.truncate_strategy = truncate_strategy
        self.critique_truncate_strategy = critique_truncate_strategy
        self.embedder = embedder(model_name=embedder_path)
        self.fewshot_strategy = fewshot_strategy
        self.retriever_cls = retriever_cls
        self.step_stripper = step_stripper
        self.success_critique_num = success_critique_num
        self.reranker = reranker
        self.buffer_retrieve_ratio = buffer_retrieve_ratio
        self.failed_training_task_idx = []
        self.critique_summary_suffix = critique_summary_suffix
        self.max_fewshot_tokens = max_fewshot_tokens
        self.eval_successes = []
        self.succeeded_trial_history: Dict[str, Trajectory] = {}
        self.failed_trial_history: Dict[str, Trajectory] = {}
        self.critiques = {}
        self.all_success_critiques = {}
        self.past_reflections = {}
        self.rule_items = []
        self.rule_items_with_count = []
        self.cache_rules = {}
        self._train = True
        super().__init__(benchmark_name=benchmark_name, *args, **kwargs)
        self.idx2task = {idx: task['task'] for idx, task in enumerate(self.tasks)}
        self.task2idx = {task['task']: idx for idx, task in enumerate(self.tasks)}

        # Update constructor with rule template for ExpelAgent
        if hasattr(self, 'constructor') and self.rule_template is not None:
            self.constructor.rule_template = self.rule_template

        # Initialize retrieval system
        self.retrieval = ExpelRetrieval(
            embedder=self.embedder,
            benchmark_name=self.benchmark_name,
            fewshot_strategy=self.fewshot_strategy,
            reranker=self.reranker,
            buffer_retrieve_ratio=self.buffer_retrieve_ratio,
            max_fewshot_tokens=self.max_fewshot_tokens,
            num_fewshots=self.num_fewshots,
            message_splitter=self.message_splitter,
            identifier=self.identifier,
            step_stripper=self.step_stripper,
            message_step_splitter=self.message_step_splitter,
            remove_task_suffix=self.remove_task_suffix,
            token_counter=self.token_counter
        )

        # Initialize insights extraction manager
        self.manager = ExpelManager(
            agent=self,
            system_critique_instructions=self.system_critique_instructions,
            human_critiques=self.human_critiques,
            rule_template=self.rule_template,
            max_num_rules=self.max_num_rules,
            critique_truncate_strategy=self.critique_truncate_strategy,
            success_critique_num=self.success_critique_num,
            critique_summary_suffix=self.critique_summary_suffix,
            benchmark_name=self.benchmark_name,
            testing=kwargs.get('testing', False),
        )

    @property
    def training(self) -> bool:
        return self._train

    def train(self) -> None:
        self._train = True

    def eval(self) -> None:
        self._train = False

    def next_task(self) -> bool:
        # storing reflections
        if self.task not in self.past_reflections:
            self.past_reflections[self.task] = []
        if self.reflections != []:
            self.past_reflections[self.task].append(self.reflections[-1])

        # only reflect on the task if the task is training task
        if self.training:
            # record the tasks
            history = self.log_history(include_task=False)
            trajectory = Trajectory(
                task=self.remove_task_suffix(self.task),
                trajectory=history,
                reflections=self.reflections,
                splitter=self.message_splitter,
                identifier=self.identifier,
                step_splitter=self.message_step_splitter,
            )
            self.succeeded_trial_history = deepcopy(self.succeeded_trial_history)
            self.failed_trial_history = deepcopy(self.failed_trial_history)

            # first time doing the task
            if self.task not in self.failed_trial_history:
                self.succeeded_trial_history[self.task] = []
                self.failed_trial_history[self.task] = []
            # if changing task, reflect accordingly
            if self.increment_task:
                if self.is_success():
                    self.succeeded_trial_history[self.task].append(trajectory)
                else:
                    self.failed_trial_history[self.task].append(trajectory)
                    # record the task index that failed
                    self.failed_training_task_idx.append(self.task_idx)
            else:
                self.failed_trial_history[self.task].append(trajectory)

        return ReflectAgent.next_task(self)

    ################# CRITIQUES #################

    def task_critique(self, task: str, return_log: bool = False) -> Union[None, str]:
        """Delegate task critique generation to the manager."""
        return self.manager.task_critique(task=task, return_log=return_log)

    def success_critique(self, training_ids: List[int]) -> None:
        """Delegate success critique generation to the manager."""
        return self.manager.success_critique(training_ids=training_ids)

    def failure_critique(self) -> None:
        """Delegate failure critique generation to the manager."""
        return self.manager.failure_critique()

    def _build_critique_prompt(self, success_history: str, fail_history: str = None, existing_rules: List[str] = None, task: str = None, reflections: List[str] = None) -> List[HumanMessage]:
        """Delegate critique prompt building to the manager."""
        return self.manager.build_critique_prompt(
            success_history=success_history,
            fail_history=fail_history,
            existing_rules=existing_rules,
            task=task,
            reflections=reflections
        )

    def prepare_new_eval(self) -> None:
        self.succeeded_trial_history = {}
        self.failed_trial_history = {}

    def prompt_critique(
        self, success_history: str, fail_history: str,
        existing_rules: List[str] = None, task: str = None, reflections: List[str] = None) -> str:
        """Delegate critique generation to the manager."""
        return self.manager.prompt_critique(
            success_history=success_history,
            fail_history=fail_history,
            existing_rules=existing_rules,
            task=task,
            reflections=reflections
        )

    ################# EVALUATION #################

    def run(self, mode: str, eval_idx: int = None, reset: bool = True):
        # normal training step
        if mode == 'train':
            return ReflectAgent.run(self, reset)
        # testing step
        if mode == 'eval':
            self.task = self.tasks[eval_idx]['task']
            self.set_env(self.tasks[eval_idx]['env_kwargs'], max_steps=self.max_steps)
            ret = ReactAgent.run(self, reset)
            if self.is_success():
                self.eval_successes.append(eval_idx)
            return ret
        raise NotImplementedError

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
        """Delegate insights extraction to the manager."""
        return self.manager.create_rules(
            training_ids=training_ids,
            cache_fold=cache_fold,
            load_cache_fold=load_cache_fold,
            logging_dir=logging_dir,
            run_name=run_name,
            loaded_dict=loaded_dict,
            loaded_log=loaded_log,
            eval_idx_list=eval_idx_list,
            saving_dict=saving_dict,
        )

    def insert_before_task_prompt(self):
        # if training then reflect
        if self.training:
            return ReflectAgent.insert_before_task_prompt(self)
        # if eval, add the manual through constructor
        if not self.no_rules and hasattr(self, 'rules'):
            self.constructor.insert_rules_or_insights(self.rules)

    def after_step(self) -> None:
        pass

    def setup_vectorstore(self) -> None:
        """
        Setup vector store using the modularized retrieval system.
        """
        # Use retrieval system to setup documents
        self.retrieval.setup_documents(
            succeeded_trial_history=self.succeeded_trial_history,
            all_fewshots=self.all_fewshots,
            env=self.env
        )

    def update_dynamic_prompt_components(self, reset: bool = False):
        """
        Update dynamic prompt components using the modularized retrieval system.
        """
        if reset:
            ReactAgent.update_dynamic_prompt_components(self)
            return

        # Do not dynamically update during training
        if self.training or self.fewshot_strategy == 'none':
            return

        old_fewshots = '\n\n'.join(self.fewshots)

        # Setup vector store using retrieval system
        self.setup_vectorstore()

        # Build trajectory and queries for current context using retrieval system
        if self.prompt_history == []:
            trajectory = None
            queries = self.retrieval.build_query_vectors(self.task, None, self.prompt_history)
        else:
            history = self.log_history(include_task=False)
            trajectory = Trajectory(
                task=self.remove_task_suffix(self.task),
                trajectory=history,
                reflections=list(self.reflections),
                splitter=self.message_splitter,
                identifier=self.identifier,
                step_splitter=self.message_step_splitter,
            )
            queries = self.retrieval.build_query_vectors(self.task, trajectory, self.prompt_history)

        # Select retrieval strategy and get fewshots using retrieval system
        if self.fewshot_strategy == 'random':
            vectorstore = self.retrieval.create_filtered_vectorstore('random', self.env.env_name)
            self.retrieval.vectorstore = vectorstore
            self.vectorstore = vectorstore
            self.fewshots = self.retrieval.retrieve_topk_documents(queries, 'task', self.task)
        elif self.fewshot_strategy == 'rotation':
            # Use task to retrieve if no trajectory available
            if trajectory is None or self.prompt_history == [] or len(trajectory.thoughts) < 1 or trajectory.thoughts[0] == '':
                vectorstore = self.retrieval.create_filtered_vectorstore('task_similarity', self.env.env_name)
                self.retrieval.vectorstore = vectorstore
                self.vectorstore = vectorstore
                self.fewshots = self.retrieval.retrieve_topk_documents(queries, 'task', self.task)
            else:
                last_step_type = self.identifier(self.message_splitter(trajectory.trajectory)[-1])
                if last_step_type == 'thought':
                    vectorstore = self.retrieval.create_filtered_vectorstore('thought_similarity', self.env.env_name)
                    self.retrieval.vectorstore = vectorstore
                    self.vectorstore = vectorstore
                    self.fewshots = self.retrieval.retrieve_topk_documents(queries, 'thought', self.task)
                elif last_step_type == 'observation':
                    vectorstore = self.retrieval.create_filtered_vectorstore('step_similarity', self.env.env_name)
                    self.retrieval.vectorstore = vectorstore
                    self.vectorstore = vectorstore
                    self.fewshots = self.retrieval.retrieve_topk_documents(queries, 'step', self.task)
        elif self.fewshot_strategy == 'task_thought_similarity':
            # Use task to retrieve
            if trajectory is None or self.prompt_history == [] or len(trajectory.thoughts) < 1 or trajectory.thoughts[0] == '':
                vectorstore = self.retrieval.create_filtered_vectorstore('task_similarity', self.env.env_name)
                self.retrieval.vectorstore = vectorstore
                self.vectorstore = vectorstore
                self.fewshots = self.retrieval.retrieve_topk_documents(queries, 'task', self.task)
            else:
                vectorstore = self.retrieval.create_filtered_vectorstore('thought_similarity', self.env.env_name)
                self.retrieval.vectorstore = vectorstore
                self.vectorstore = vectorstore
                self.fewshots = self.retrieval.retrieve_topk_documents(queries, 'thought', self.task)
        elif self.fewshot_strategy == 'task_similarity':
            # Retrieve task as the query, and task as the keys for successful trials
            vectorstore = self.retrieval.create_filtered_vectorstore('task_similarity', self.env.env_name)
            self.retrieval.vectorstore = vectorstore
            self.vectorstore = vectorstore
            self.fewshots = self.retrieval.retrieve_topk_documents(queries, 'task', self.task)
        elif self.fewshot_strategy == 'thought_similarity':
            if trajectory is None or self.prompt_history == [] or len(trajectory.thoughts) < 1 or trajectory.thoughts[0] == '':
                ReactAgent.update_dynamic_prompt_components(self)
            else:
                # Use the latest thoughts to retrieve fewshots
                vectorstore = self.retrieval.create_filtered_vectorstore('thought_similarity', self.env.env_name)
                self.retrieval.vectorstore = vectorstore
                self.vectorstore = vectorstore
                self.fewshots = self.retrieval.retrieve_topk_documents(queries, 'thought', self.task)
        elif self.fewshot_strategy == 'action_similarity':
            if trajectory is None or self.prompt_history == [] or len(trajectory.actions) < 1:
                ReactAgent.update_dynamic_prompt_components(self)
            else:
                # Use the latest actions to retrieve fewshots
                vectorstore = self.retrieval.create_filtered_vectorstore('action_similarity', self.env.env_name)
                self.retrieval.vectorstore = vectorstore
                self.vectorstore = vectorstore
                self.fewshots = self.retrieval.retrieve_topk_documents(queries, 'action', self.task)
        elif self.fewshot_strategy == 'step_similarity':
            if trajectory is None or self.prompt_history == [] or len(trajectory.observations) < 1:
                ReactAgent.update_dynamic_prompt_components(self)
            else:
                vectorstore = self.retrieval.create_filtered_vectorstore('step_similarity', self.env.env_name)
                self.retrieval.vectorstore = vectorstore
                self.vectorstore = vectorstore
                self.fewshots = self.retrieval.retrieve_topk_documents(queries, 'step', self.task)
        else:
            raise NotImplementedError

        # Update fewshots dynamically through constructor
        self.constructor.update_fewshots_dynamically(old_fewshots, self.fewshots)

        # Update prompt_history reference for compatibility
        self.prompt_history = self.constructor.get_prompt_history_for_compatibility()

