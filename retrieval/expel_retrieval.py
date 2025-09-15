"""
ExpelRetrieval Implementation

This module implements the concrete retrieval system for the ExpeL framework,
handling FAISS-based vector retrieval for experience-based learning.
"""

import random
from typing import Dict, List, Any, Optional, Callable
from functools import partial
from copy import deepcopy

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from scipy.spatial.distance import cosine

from .base import BaseRetrieval
from memory import Trajectory
from utils import get_env_name_from_task


class ExpelRetrieval(BaseRetrieval):
    """
    Concrete retrieval implementation for ExpeL framework.

    This class handles the complete FAISS-based vector retrieval system,
    including document setup, query construction, and similarity search.
    """

    def setup_documents(self,
                       succeeded_trial_history: Dict[str, List[Any]],
                       all_fewshots: Any,
                       env: Any) -> None:
        """
        Setup documents from succeeded trial history and few-shot examples.

        This method is equivalent to the original setup_vectorstore() method.
        """
        self.keys2task = {'thought': {}, 'task': {}, 'step': {}, 'reflection': {}, 'action': {}}
        self.docs = []
        combined_history = dict(succeeded_trial_history)

        # Process few-shot examples
        if isinstance(all_fewshots, list):
            for fewshot in all_fewshots:
                if self.benchmark_name in ['hotpotqa', 'fever']:
                    task = fewshot.split('\n')[0]
                    trajectory = '\n'.join(fewshot.split('\n')[1:])
                elif self.benchmark_name == 'webshop':
                    task = '\n'.join(fewshot.split('\n')[:2])
                    trajectory = '\n'.join(fewshot.split('\n')[2:])

                cleaned_traj = Trajectory(
                    task=self.remove_task_suffix(task),
                    trajectory=trajectory,
                    reflections=[],
                    splitter=self.message_splitter,
                    identifier=self.identifier,
                    step_splitter=partial(
                        self.message_step_splitter,
                        stripper=self.step_stripper
                    ),
                )
                combined_history.update({task: [cleaned_traj]})

        elif isinstance(all_fewshots, dict):
            fewshot_offset = 100000
            for env_task, fewshots in all_fewshots.items():
                for fewshot in fewshots:
                    if self.benchmark_name in ['alfworld']:
                        task = '\n'.join(fewshot.split('\n')[:3]) + '___' + str(fewshot_offset)
                        trajectory = '\n'.join(fewshot.split('\n')[3:])

                    cleaned_traj = Trajectory(
                        task=self.remove_task_suffix(task),
                        trajectory=trajectory,
                        reflections=[],
                        splitter=self.message_splitter,
                        identifier=self.identifier,
                        step_splitter=partial(
                            self.message_step_splitter,
                            stripper=self.step_stripper
                        ),
                    )
                    combined_history.update({task: [cleaned_traj]})
                    fewshot_offset += 1

        # Create documents from combined history
        for task in combined_history:
            if combined_history[task] != []:
                self.docs.append(Document(
                    page_content=self.remove_task_suffix(task),
                    metadata={
                        'type': 'task',
                        'task': task,
                        'env_name': get_env_name_from_task(task, self.benchmark_name)
                    }
                ))

            for i, traj in enumerate(combined_history[task]):
                cleaned_traj = Trajectory(
                    task=self.remove_task_suffix(task),
                    trajectory=traj.trajectory,
                    reflections=list(traj.reflections),
                    splitter=self.message_splitter,
                    identifier=self.identifier,
                    step_splitter=partial(
                        self.message_step_splitter,
                        stripper=self.step_stripper
                    ),
                )

                cleaned_thoughts: List[str] = cleaned_traj.thoughts
                cleaned_steps: List[str] = cleaned_traj.steps
                cleaned_reflections: List[str] = cleaned_traj.reflections
                cleaned_actions: List[str] = cleaned_traj.actions

                # Add documents for different content types
                self.docs.extend([
                    Document(page_content=action, metadata={
                        'type': 'action', 'task': task,
                        'env_name': get_env_name_from_task(task, self.benchmark_name)
                    }) for action in cleaned_actions
                ])

                self.docs.extend([
                    Document(page_content=thought, metadata={
                        'type': 'thought', 'task': task,
                        'env_name': get_env_name_from_task(task, self.benchmark_name)
                    }) for thought in cleaned_thoughts
                ])

                self.docs.extend([
                    Document(page_content=step, metadata={
                        'type': 'step', 'task': task,
                        'env_name': get_env_name_from_task(task, self.benchmark_name)
                    }) for step in cleaned_steps
                ])

                if cleaned_reflections != []:
                    self.docs.extend([
                        Document(page_content=reflection, metadata={
                            'type': 'reflection', 'task': task,
                            'env_name': get_env_name_from_task(task, self.benchmark_name)
                        }) for reflection in cleaned_reflections
                    ])

                # Build keys2task mapping
                for thought in cleaned_thoughts:
                    self.keys2task['thought'][thought] = (task, i)
                for step in cleaned_steps:
                    self.keys2task['step'][step] = (task, i)
                for reflection in cleaned_reflections:
                    self.keys2task['reflection'][reflection] = (task, i)
                for action in cleaned_actions:
                    self.keys2task['action'][action] = (task, i)

        self.combined_history = combined_history

    def build_query_vectors(self,
                           task: str,
                           trajectory: Any,
                           prompt_history: List) -> Dict[str, str]:
        """
        Build query vectors for retrieval based on current context.
        """
        if prompt_history == []:
            queries = {'task': self.step_stripper(self.remove_task_suffix(task), step_type='task')}
        else:
            # Build queries from current trajectory
            steps = self.message_splitter(trajectory.steps[-1])
            step_types = [self.identifier(step) for step in steps]

            # If the step is not complete, use the previous step
            if 'observation' not in step_types and self.fewshot_strategy == 'step':
                steps = self.message_splitter(trajectory.steps[-2])
                step_types = [self.identifier(step) for step in steps]

            cleaned_step = '\n'.join([self.step_stripper(step, step_type) for step, step_type in zip(steps, step_types)])

            queries = {
                'task': self.step_stripper(self.remove_task_suffix(task), step_type='task'),
                'thought': '' if len(trajectory.thoughts) < 1 or trajectory.thoughts[0] == '' else self.step_stripper(trajectory.thoughts[-1], step_type='thought'),
                'step': cleaned_step,
                'action': self.step_stripper(trajectory.actions[-1], step_type='action') if len(trajectory.actions) > 1 else '',
            }

        return queries

    def create_filtered_vectorstore(self,
                                   strategy: str,
                                   env_name: str) -> FAISS:
        """
        Create a filtered FAISS vector store based on strategy and environment.

        This method is equivalent to the original filtered_vectorstore() function.
        """
        strat2filter = {
            'task_similarity': 'task', 'step_similarity': 'step',
            'reflection_similarity': 'reflection', 'thought_similarity': 'thought',
            'action_similarity': 'action'
        }

        if strategy == 'random':
            subset_docs = list(filter(
                lambda doc: doc.metadata['type'] == strat2filter['task_similarity'] and
                           doc.metadata['env_name'] == env_name,
                self.docs
            ))
        else:
            subset_docs = list(filter(
                lambda doc: doc.metadata['type'] == strat2filter[strategy] and
                           doc.metadata['env_name'] == env_name,
                self.docs
            ))

        # Special filtering for webshop
        if self.benchmark_name == 'webshop':
            filtered_subset_docs = []
            for doc in subset_docs:
                trajectory = self.combined_history[doc.metadata['task']][0].trajectory
                if 'Observation: Invalid action!' not in trajectory and \
                   len(trajectory.split('Observation: You have clicked')) >= 3:
                    filtered_subset_docs.append(doc)
        else:
            filtered_subset_docs = subset_docs

        return FAISS.from_documents(filtered_subset_docs, self.embedder)

    def retrieve_topk_documents(self,
                               queries: Dict[str, str],
                               query_type: str,
                               current_task: str = None) -> List[str]:
        """
        Retrieve top-k most similar documents using FAISS similarity search.

        This method is equivalent to the original topk_docs() function.
        """
        # Retrieve enough fewshots, filtering the ones that are too long
        fewshot_docs = self.vectorstore.similarity_search(
            queries[query_type],
            k=self.num_fewshots * self.buffer_retrieve_ratio
        )

        if self.fewshot_strategy == 'random':
            random.shuffle(fewshot_docs)

        fewshots = []
        current_tasks = set()

        def fewshot_doc_token_count(fewshot_doc):
            return self.token_counter(self.combined_history[fewshot_doc.metadata['task']][0].trajectory)

        # Apply reranking strategy
        if self.reranker == 'none' or (self.reranker == 'thought' and queries['thought'] == ''):
            fewshot_docs = list(fewshot_docs)
        elif self.reranker == 'len':
            fewshot_docs = list(sorted(fewshot_docs, key=fewshot_doc_token_count, reverse=True))
        elif self.reranker == 'thought' and queries['thought'] != '':
            fewshot_tasks = set([doc.metadata['task'] for doc in fewshot_docs])
            subset_docs = list(filter(
                lambda doc: doc.metadata['type'] == 'thought' and
                           doc.metadata['env_name'] == fewshot_docs[0].metadata['env_name'] and
                           doc.metadata['task'] in fewshot_tasks,
                list(self.docs)
            ))
            fewshot_docs = sorted(subset_docs, key=lambda doc:
                cosine(self.embedder.embed_query(doc.page_content),
                      self.embedder.embed_query(queries['thought'])))
        elif self.reranker == 'task':
            fewshot_tasks = set([doc.metadata['task'] for doc in fewshot_docs])
            subset_docs = list(filter(
                lambda doc: doc.metadata['type'] == 'thought' and
                           doc.metadata['env_name'] == fewshot_docs[0].metadata['env_name'] and
                           doc.metadata['task'] in fewshot_tasks,
                list(self.docs)
            ))
            fewshot_docs = sorted(subset_docs, key=lambda doc:
                cosine(self.embedder.embed_query(doc.page_content),
                      self.embedder.embed_query(queries['task'])))
        else:
            raise NotImplementedError

        # Filter and collect fewshots
        for fewshot_doc in fewshot_docs:
            idx, shortest_fewshot = sorted(
                enumerate([traj.trajectory for traj in self.combined_history[fewshot_doc.metadata['task']]]),
                key=lambda x: len(x[1])
            )[0]

            # Skip if fewshot is too long, same as current task, or duplicate
            if self.token_counter(shortest_fewshot) > self.max_fewshot_tokens or \
               (current_task and current_task == fewshot_doc.metadata['task']) or \
               fewshot_doc.metadata['task'] in current_tasks:
                continue

            current_tasks.add(fewshot_doc.metadata['task'])
            # Add task prefix to fewshot (matching original behavior)
            full_fewshot = self.combined_history[fewshot_doc.metadata['task']][idx].task + '\n' + shortest_fewshot
            fewshots.append(full_fewshot)

            if len(fewshots) == self.num_fewshots:
                break

        return fewshots

