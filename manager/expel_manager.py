import random
import re
from typing import List, Dict, Callable, Union, Any, Tuple
from functools import partial
from copy import deepcopy

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import numpy as np
import openai

from manager.base import BaseManager
from utils import random_divide_list, save_trajectories_log
from memory import Trajectory
from agent.reflect import Count

class ExpelManager(BaseManager):
    """
    Concrete implementation of insights extraction and rule management for ExpeL agents.

    This class handles the complete Training Phase 2 workflow including critique generation,
    rule parsing, and rule updating operations based on success/failure trajectory analysis.
    """

    def __init__(
        self,
        agent,  # Reference to the agent instance for accessing its attributes
        system_critique_instructions: Dict[str, str],
        human_critiques: Dict[str, PromptTemplate],
        rule_template: PromptTemplate,
        max_num_rules: Union[int, str],
        critique_truncate_strategy: str,
        success_critique_num: int,
        critique_summary_suffix: str,
        benchmark_name: str,
        testing: bool = False,
    ):
        """
        Initialize the ExpelManager with critique generation and rule management components.

        Args:
            agent: Reference to the agent instance
            system_critique_instructions: System instructions for different critique types
            human_critiques: Human instruction templates for critique generation
            rule_template: Template for formatting rules
            max_num_rules: Maximum number of rules to maintain
            critique_truncate_strategy: Strategy for truncating long critiques
            success_critique_num: Number of successful trajectories per critique batch
            critique_summary_suffix: Suffix text for critique summaries
            benchmark_name: Name of the benchmark being used
            testing: Whether in testing mode
        """
        self.agent = agent
        self.system_critique_instructions = system_critique_instructions
        self.human_critiques = human_critiques
        self.rule_template = rule_template
        self.max_num_rules = max_num_rules
        self.critique_truncate_strategy = critique_truncate_strategy
        self.success_critique_num = success_critique_num
        self.critique_summary_suffix = critique_summary_suffix
        self.benchmark_name = benchmark_name
        self.testing = testing

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

        This method implements the complete Training Phase 2 workflow:
        1. Compare success/failure trajectories to generate critiques
        2. Extract actionable rules from critiques
        3. Update rule set with ADD/EDIT/REMOVE/AGREE operations
        4. Format final rule set for evaluation phase
        """
        if load_cache_fold is not None:
            self.agent.rules = '\n'.join([f'{i}. {item}' for i, item in enumerate(self.agent.cache_rules.get(load_cache_fold, []), 1)])
            return

        def extend_rules(rule_items: List[str], success_history: str = None, fail_history: str = None, task: str = None, reflections: List[str] = None) -> List[str]:
            llm_output: str = self.prompt_critique(
                success_history=success_history,
                fail_history=fail_history,
                existing_rules=rule_items,
                reflections=reflections,
                task=task,
            )
            parsed_operations = self.parse_rules(llm_output)

            # update the rule_items with counter
            self.agent.rule_items_with_count = self.update_rules(self.agent.rule_items_with_count, parsed_operations, list_full = self.max_num_rules+5 <= len(self.agent.rule_items_with_count))

            new_ordered_rules_str = [rule[0] for rule in self.agent.rule_items_with_count]
            return new_ordered_rules_str, llm_output

        # Shuffling the rules into a pool
        resume_flag = fail_resume_flag = loaded_dict is None
        if resume_flag:
            self.agent.rule_items = []
            self.agent.rule_items_with_count: List[tuple(str, int)] = []
        agent_dicts = []
        if loaded_log is None:
            all_logs = '################ Compare Critiques ################\n'
        else:
            all_logs = loaded_log
        for training_id in training_ids:
            training_task = self.agent.idx2task[training_id]
            if (loaded_dict is not None and loaded_dict['critique_summary_section'] == 'compare' and \
                loaded_dict['critique_summary_idx'][0] == training_id):
                resume_flag = True
                # if there are still failed tasks to do, then dont continue, otherwise do the next idx's critiques
                if len(self.agent.failed_trial_history[training_task]) - 1 <= loaded_dict['critique_summary_idx'][1]:
                    fail_resume_flag = True
                    continue
            elif not resume_flag:
                continue
            if self.agent.succeeded_trial_history[training_task] != []:
                # if first time critiquing the task
                for traj in self.agent.succeeded_trial_history[training_task]:
                    success_history = traj.trajectory.strip()
                    # forming critiques by comparing successful and failed trials
                    for e, fail_history in enumerate(self.agent.failed_trial_history[training_task]):
                        if fail_resume_flag:
                            pass
                        elif e <= loaded_dict['critique_summary_idx'][1]:
                            continue
                        fail_resume_flag = True
                        self.agent.rule_items, llm_output = extend_rules(self.agent.rule_items, success_history, fail_history.trajectory.strip(), training_task)
                        all_logs += training_task + '\n' + success_history + '\n' + fail_history.trajectory.strip() + f'\n-------\n{llm_output}\n-------\n' +'\n- ' + '\n- '.join([str(r) + " {" + str(c) + "}" for r, c in self.agent.rule_items_with_count]) + '\n\n'
                        if saving_dict:
                            save_dict = {k: deepcopy(v) for k, v in self.agent.__dict__.items() if type(v) in [list, set, str, bool, int, dict, Count] and k not in ['openai_api_key', 'llm']}
                            save_dict['critique_summary_section'] = 'compare'
                            save_dict['critique_summary_idx'] = (training_id, e)
                            save_dict['critique_summary_fold'] = cache_fold if cache_fold is not None else 0
                            save_dict['critique_summary_log'] = all_logs
                            save_dict['eval_idx_list'] = eval_idx_list
                            agent_dicts.append(save_dict)
                            save_trajectories_log(path=logging_dir, log=all_logs, dicts=agent_dicts, run_name=run_name, save_true_log=False)

        # SUCCESS
        if loaded_log is None or loaded_dict['critique_summary_section'] in ['compare']:
            all_logs += '\n\n################ SUCCESS CRITIQUES ################\n'
        else:
            all_logs = loaded_log
        if loaded_dict is None or loaded_dict['critique_summary_section'] == 'compare':
            for training_id in training_ids:
                all_success = []
                for idx, task in enumerate(self.agent.succeeded_trial_history):
                    if idx in training_ids and len(self.agent.succeeded_trial_history[task]) > 0:
                        all_success.append((task, self.agent.succeeded_trial_history[task][0].trajectory))
                all_success = random_divide_list(all_success, self.success_critique_num)
        else:
            all_success = loaded_dict['critique_summary_all_success']
        for success_chunk in all_success:
            if (loaded_dict is not None and loaded_dict['critique_summary_section'] == 'success' and \
                loaded_dict['critique_summary_idx'] == success_chunk):
                resume_flag = True
                continue
            elif not resume_flag:
                continue
            success_trials = '\n\n'.join([self.agent.remove_task_suffix(task) + '\n' + trajectory for task, trajectory in success_chunk])
            self.agent.rule_items, llm_output = extend_rules(self.agent.rule_items, success_trials.strip(), None)
            all_logs += success_trials.strip() + f'\n-------\n{llm_output}\n-------' + '\n- ' + '\n- '.join([str(r) + " {" + str(c) + "}" for r, c in self.agent.rule_items_with_count]) + '\n\n'
            if saving_dict:
                save_dict = {k: deepcopy(v) for k, v in self.agent.__dict__.items() if type(v) in [list, set, str, bool, int, dict, Count] and k not in ['openai_api_key', 'llm']}
                save_dict['critique_summary_all_success'] = all_success
                save_dict['critique_summary_idx'] = success_chunk
                save_dict['critique_summary_section'] = 'success'
                save_dict['critique_summary_fold'] = cache_fold if cache_fold is not None else 0
                save_dict['critique_summary_log'] = all_logs
                save_dict['eval_idx_list'] = eval_idx_list
                agent_dicts.append(save_dict)
                save_trajectories_log(path=logging_dir, log=all_logs, dicts=agent_dicts, run_name=run_name, save_true_log=False)

        # numbered list format
        self.agent.rules = '\n'.join([f"{i}. {item}" for i, item in enumerate(self.agent.rule_items, 1)])
        if cache_fold is not None:
            self.agent.cache_rules[cache_fold] = list(self.agent.rule_items)
        return all_logs

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
        """
        critique_history = self.agent.constructor.collapse_prompts(
            self.build_critique_prompt(success_history, fail_history, existing_rules, task if task is None else self.agent.remove_task_suffix(task), reflections)
        )
        print("\n###################################\n")
        if self.testing:
            print('###################################')
            for prompt in critique_history:
                self.agent.print_message(prompt, self.agent.token_counter)
            return input()
        # just use the base llm for critiques
        try:
            returns = self.agent.llm(critique_history, replace_newline=False)
        except openai.BadRequestError:
            returns = self.agent.long_context_llm(critique_history, replace_newline=False)
        for i, m in enumerate(critique_history):
            self.agent.print_message(m)
            if i == len(critique_history) - 1:
                print(returns)
        return returns

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
        """
        critique_history = []
        if reflections is not None:
            critique_type = 'all_reflection'
        elif fail_history is not None and success_history is not None:
            critique_type = 'compare'
        elif fail_history is None and success_history is not None:
            critique_type = 'all_success'
        elif fail_history is not None and success_history is None:
            critique_type = 'all_fail'
        if existing_rules is not None:
            critique_type += '_existing_rules'
        if existing_rules == []:
            existing_rules = ['']

        # system prompt
        critique_history.extend(self.agent.system_prompt.format_messages(
            instruction=self.system_critique_instructions[critique_type].format(
                fewshots=[],
            ),
            ai_name='an advanced reasoning agent that can critique past task trajectories of youself' if existing_rules is None \
                else 'an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories',
        ))
        # task_prompt
        human_format_dict = dict(instruction='',)
        if critique_type == 'compare':
            human_format_dict['task'] = task
        if fail_history is not None:
            human_format_dict['fail_history'] = fail_history
            human_format_dict['task'] = task
        if success_history is not None:
            human_format_dict['success_history'] = success_history
        if reflections is not None:
            human_format_dict['reflections_list'] = '- ' + '\n- '.join(reflections)
        if existing_rules is not None:
            human_format_dict['existing_rules'] = '\n'.join([f'{i}. {r}' for i, r in enumerate(existing_rules, 1)])
        human_critique_summary_message = self.human_critiques[critique_type].format_messages(**human_format_dict)[0]
        critique_summary_suffix = self.critique_summary_suffix['full'] if self.max_num_rules <= len(self.agent.rule_items_with_count) else self.critique_summary_suffix['not_full']
        human_critique_summary_message.content = human_critique_summary_message.content + critique_summary_suffix
        critique_history.append(human_critique_summary_message)
        return critique_history

    def task_critique(
        self,
        task: str,
        return_log: bool = False
    ) -> Union[None, str]:
        """
        Generate critiques for a specific task by comparing success/failure trajectories.
        """
        # only critique if the task has success
        if task not in self.agent.critiques:
            self.agent.critiques[task] = []
        if return_log:
            log = ''
        if self.agent.succeeded_trial_history[task] != []:
            # if first time critiquing the task
            for traj in self.agent.succeeded_trial_history[task]:
                success_history = traj.trajectory.strip()
                # forming critiques by comparing successful and failed trials
                for fail_history in self.agent.failed_trial_history[task]:
                    critiques: str = self.prompt_critique(
                        success_history=success_history,
                        fail_history=fail_history.trajectory.lstrip(),
                    )
                    if return_log:
                        log += success_history + '\n' + fail_history.trajectory.strip() + '\n' + critiques + '\n\n'
                    critiques: List[str] = critiques.split('\n- ' if not self.testing else '\\n- ')
                    self.agent.critiques[task].extend(critiques)
                pattern = r"\s*\([^()]*\)"
                self.agent.critiques[task] = [re.sub(pattern, '', critique).strip().strip('- ') for critique in self.agent.critiques[task]]
                # removing empty critique
                self.agent.critiques[task] = [critique for critique in self.agent.critiques[task] if critique != '']

        if return_log:
            return log

    def success_critique(
        self,
        training_ids: List[int]
    ) -> None:
        """
        Generate critiques from successful trajectories only.
        """
        # make sure to only take the training ids, assuming theres only one success trajectory per task
        all_success = []
        for task in self.agent.succeeded_trial_history:
            idx = self.agent.task2idx[task]
            if idx in training_ids and len(self.agent.succeeded_trial_history[task]) > 0:
                all_success.append((self.agent.remove_task_suffix(task), self.agent.succeeded_trial_history[task][0].trajectory))
        all_success = random_divide_list(all_success, self.success_critique_num)
        # refresh the success critiques
        self.agent.all_success_critiques = {}
        for success_chunk in all_success:
            success_trials = '\n\n'.join([task + '\n' + trajectory for task, trajectory in success_chunk])
            critiques: str = self.prompt_critique(success_history=success_trials.strip(), fail_history=None)
            critiques: List[str] = critiques.split('\n- ' if not self.testing else '\\n- ')
            key = '\n'.join([task for task, _ in success_chunk])
            self.agent.all_success_critiques[key] = critiques
            pattern = r"\s*\([^()]*\)"
            self.agent.all_success_critiques[key] = [re.sub(pattern, '', critique).strip().strip('- ') for critique in self.agent.all_success_critiques[key]]
            # removing empty critique
            self.agent.all_success_critiques[key] = [critique for critique in self.agent.all_success_critiques[key] if critique != '']

    def failure_critique(self) -> None:
        """
        Generate critiques from multiple failed trajectories for the same task.
        """
        self.agent.all_fail_critiques = {}
        for task, failed_trajectories in self.agent.failed_trial_history.items():
            # only critiquing if the task has failed more than once
            if len(failed_trajectories) > 1:
                failed_trials = '\n\n'.join([traj.trajectory for traj in failed_trajectories])
                if self.agent.token_counter(failed_trials) > 13000:
                    print('TRUNCATING FAILED TRIALS')
                    if self.critique_truncate_strategy == 'random':
                        idx = np.random.choice(range(len(failed_trajectories)), size=len(failed_trajectories) - 1, replace=False)
                        failed_trials = '\n\n'.join([traj.trajectory for i, traj in enumerate(failed_trajectories) if i in idx])
                    elif self.critique_truncate_strategy == 'longest':
                        filtered_idx = max(range(len(failed_trajectories)), key=lambda i: self.agent.token_counter(failed_trajectories[i].trajectory))
                        failed_trials = '\n\n'.join([traj.trajectory for i, traj in enumerate(failed_trajectories) if i != filtered_idx])
                    elif self.critique_truncate_strategy == 'shortest':
                        filtered_idx = min(range(len(failed_trajectories)), key=lambda i: self.agent.token_counter(failed_trajectories[i].trajectory))
                        failed_trials = '\n\n'.join([traj.trajectory for i, traj in enumerate(failed_trajectories) if i != filtered_idx])
                    else:
                        raise NotImplementedError
                    critiques: str = self.prompt_critique(success_history=None, fail_history=failed_trials.strip(), task=task)
                    critiques: List[str] = critiques.split('\n- ' if not self.testing else '\\n- ')
                    self.agent.all_fail_critiques[task] = critiques
                    pattern = r"\s*\([^()]*\)"
                    self.agent.all_fail_critiques[task] = [re.sub(pattern, '', critique).strip().strip('- ') for critique in self.agent.all_fail_critiques[task]]
                    # removing empty critique
                    self.agent.all_fail_critiques[task] = [critique for critique in self.agent.all_fail_critiques[task] if critique != '']
            else:
                self.agent.all_fail_critiques[task] = []

    def parse_rules(self, llm_text: str) -> List[Tuple[str, str]]:
        """
        Parse LLM output to extract rule operations (ADD/EDIT/REMOVE/AGREE).
        """
        pattern = r'((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)'
        matches = re.findall(pattern, llm_text)

        res = []
        banned_words = ['ADD', 'AGREE', 'EDIT']
        for operation, text in matches:
            text = text.strip()
            if text != '' and not any([w in text for w in banned_words]) and text.endswith('.'):
            # if text is not empty
            # if text doesn't contain banned words (avoid weird formatting cases from llm)
            # if text ends with a period (avoid cut off sentences from llm)
                if 'ADD' in operation:
                    res.append(('ADD', text))
                else:
                    res.append((operation.strip(), text))
        return(res)

    def update_rules(
        self,
        rules: List[Tuple[str, int]],
        operations: List[Tuple[str, str]],
        list_full: bool = False
    ) -> List[Tuple[str, int]]:
        """
        Update rule list based on parsed operations.
        """
        # remove problematic operations
        delete_indices = []
        for i in range(len(operations)):
            operation, operation_rule_text = operations[i]
            operation_type = operation.split(' ')[0]
            rule_num = int(operation.split(' ')[1]) if ' ' in operation else None

            if operation_type == 'ADD':
                if self._is_existing_rule(rules, operation_rule_text): # if new rule_text is an existing rule ('in')
                    delete_indices.append(i)
            else:
                if operation_type == 'EDIT':
                    if self._is_existing_rule(rules, operation_rule_text): # if rule is matching ('in') existing rule, change it to AGREE
                        rule_num = self._retrieve_rule_index(rules, (operation, operation_rule_text))
                        operations[i] = (f'AGREE {rule_num+1}', rules[rule_num][0])
                    elif (rule_num is None) or (rule_num > len(rules)):   # if rule doesn't exist, remove
                        delete_indices.append(i)

                elif operation_type == 'REMOVE' or operation_type == 'AGREE':
                    if not self._is_existing_rule(rules, operation_rule_text): # if new operation_rule_text is not an existing rule
                        delete_indices.append(i)

        operations = [operations[i] for i in range(len(operations)) if i not in delete_indices] # remove problematic operations

        for op in ['REMOVE', 'AGREE', 'EDIT', 'ADD']: # Order is important
            for i in range(len(operations)):
                operation, operation_rule_text = operations[i]
                operation_type = operation.split(' ')[0]
                if operation_type != op:
                    continue

                if operation_type == 'REMOVE': # remove rule: -1
                    rule_index = self._retrieve_rule_index(rules, (operation, operation_rule_text)) # if rule_num doesn't match but text does
                    remove_strength = 3 if list_full else 1
                    rules[rule_index] = (rules[rule_index][0], rules[rule_index][1]-remove_strength) # -1 (-3 if list full) to the counter
                elif operation_type == 'AGREE': # agree with rule: +1
                    rule_index = self._retrieve_rule_index(rules, (operation, operation_rule_text)) # if rule_num doesn't match but text does
                    rules[rule_index] = (rules[rule_index][0], rules[rule_index][1]+1) # +1 to the counter
                elif operation_type == 'EDIT': # edit the rule: +1 // NEED TO BE AFTER REMOVE AND AGREE
                    rule_index = int(operation.split(' ')[1])-1
                    rules[rule_index] = (operation_rule_text, rules[rule_index][1]+1) # +1 to the counter
                elif operation_type == 'ADD': # add new rule: +2
                    rules.append((operation_rule_text, 2))
        rules = [rules[i] for i in range(len(rules)) if rules[i][1] > 0] # remove rules when counter reach 0
        rules.sort(key=lambda x: x[1], reverse=True)

        return rules

    def _retrieve_rule_index(self, rules: List[Tuple[str, int]], operation: Tuple[str, str]) -> int:
        """Helper method to retrieve rule index."""
        operation_rule_text = operation[1]
        for i in range(len(rules)):
            if rules[i][0] in operation_rule_text:
                return i

    def _is_existing_rule(self, rules: List[Tuple[str, int]], operation_rule_text: str) -> bool:
        """Helper method to check if rule exists."""
        for i in range(len(rules)):
            if rules[i][0] in operation_rule_text:
                return True
        return False