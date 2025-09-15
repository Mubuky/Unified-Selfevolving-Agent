from typing import List, Callable, Tuple, Dict, Any, Union
from functools import partial
from copy import deepcopy

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import ChatMessage
import openai

from envs import BaseEnv
from agent import BaseAgent
from prompts.templates.human import (
    human_instruction_fewshot_message_prompt,
    human_task_message_prompt,
)
from utils import print_message, token_counter
from constructor import ExpelConstructor

class ReactAgent(BaseAgent):
    """
    A Generic ReAct Agent.
    """
    def __init__(self,
                 name: str,
                 system_instruction: Union[str, Dict[str, str]],
                 human_instruction: Callable,
                 fewshots: Union[List[str], Dict[str, List[str]]],
                 system_prompt: Callable,
                 env: BaseEnv,
                 llm: str,
                 llm_builder: Callable,
                 openai_api_key: str,
                 tasks: List[Dict[str, Any]],
                 max_steps: int,
                 llm_parser: Callable,
                 observation_formatter: Callable,
                 testing: bool = False,
                 task_idx: int = 0,
                 benchmark_name = None,
                 base_url: str = None,
                 *args,
                 **kwargs,
                 ) -> None:
        self.benchmark_name = benchmark_name
        self.name = name
        self.tasks = tasks
        self.task_idx = task_idx
        self.all_system_instruction = system_instruction
        self.human_instruction = human_instruction
        self.human_instruction_kwargs = {'max_steps': max_steps}
        self.all_fewshots = fewshots
        self.system_prompt = system_prompt
        self.prompt_history = []
        self.testing = testing
        self.max_steps = max_steps
        self.llm_parser = llm_parser
        self.observation_formatter = observation_formatter
        self._last_observation_history = None

        self.llm = llm_builder(llm_name=llm, openai_api_key=openai_api_key, long_ver=False, base_url=base_url)
        self.long_context_llm = llm_builder(llm_name=llm, openai_api_key=openai_api_key, long_ver=True, base_url=base_url)
        del openai_api_key
        self.token_counter = partial(token_counter, llm=llm, tokenizer=getattr(self.llm, 'tokenizer', None))

        self.env = env(**self.tasks[self.task_idx]['env_kwargs'], max_steps=self.max_steps)
        self.env.reset()
        self.task = self.tasks[self.task_idx]['task']

        # Update dynamic components first to initialize system_instruction
        self.update_dynamic_prompt_components()

        # Initialize prompt constructor after system_instruction is set
        self.constructor = ExpelConstructor(
            benchmark_name=benchmark_name,
            system_instruction=self.system_instruction,
            human_instruction=self.human_instruction,
            system_prompt=self.system_prompt,
            ai_name=self.name,
            human_instruction_kwargs=self.human_instruction_kwargs
        )

        # Now safe to reset and build prompt
        self.reset()
        self.truncated, self.reward, self.terminated = False, False, False
        self.print_message = partial(print_message, testing=testing)

        self.success, self.fail, self.halted = 0, 0, 0
        self.long_pass = None

    def is_success(self) -> bool:
        return self.env.success_fn()

    def set_env(self, task_kwargs: Dict[str, Any], max_steps: int):
        self.env.__init__(**task_kwargs, max_steps=max_steps)

    def run(self, reset: bool = True, *args, **kwargs) -> None:
        if reset:
            self.env.reset()
            self.reset()

        while not (self.is_truncated() or self.is_terminated()):
            self.step()

    def step(self) -> None:
        message, message_type, others = self.llm_parser(self.prompt_agent(), self.curr_step, False)
        self.prompt_history.append(message)
        self.print_message(message)

        thought_num = 1
        # loops while in thinking mode
        while message_type == 'thought':
            thought_num += 1
            message, message_type, others = self.llm_parser(self.prompt_agent(), self.curr_step, False)
            self.prompt_history.append(message)
            self.print_message(message)

            if thought_num > 2:
                if message_type == 'thought':
                    others['action'] = 'N/A'
                break

        # Observe
        observation, self.reward, self.terminated, self.truncated, _ = self.env.step(others['action'])
        if others['action'] == 'N/A' and thought_num > 2:
            observation = "You are thinking too many times without taking action."
        observation_history, operation = self.observation_formatter(observation, step=self.curr_step)
        if operation == 'append':
            self.prompt_history.append(observation_history)
        elif operation == 'replace':
            for message in self.prompt_history:
                if self._last_observation_history.content in message.content:
                    message.content = message.content.replace(self._last_observation_history.content, observation_history.content)
                    break
            self._last_observation_history = deepcopy(observation_history)        
        self.print_message(observation_history)

        BaseAgent.after_step(self)

        self.prompt_history = self.collapse_prompts(self.prompt_history)

        self.curr_step += 1

    def prompt_agent(self) -> str:
        self.prompt_history = self.collapse_prompts(self.prompt_history)
        self.update_dynamic_prompt_components()
        prompt_history = self.collapse_prompts(self.prompt_history)
        if self.testing:
            print('###################################')
            for prompt in prompt_history:
                self.print_message(prompt, self.token_counter)
            return input()
        try:
            return self.llm(prompt_history, stop=['\n', '\n\n'])
        except openai.BadRequestError:
            while self.long_pass is None:
                res = input('Changing to long context LLM. Press Enter to continue.\n')
                if res == 'pass':
                    self.long_pass = True
                elif res != '':
                    continue
                break

            return self.long_context_llm(prompt_history, stop=['\n', '\n\n'])

    def _build_fewshot_prompt(
        self,
        fewshots: List[str],
        prompt_history: List[ChatMessage],
        instruction_prompt: PromptTemplate,
        instruction_prompt_kwargs: Dict[str, Any],
        prompt_type: str,
    ) -> str:
        if human_instruction_fewshot_message_prompt is not None and instruction_prompt is not None:
            prompt_history.append(
                human_instruction_fewshot_message_prompt('message_style_kwargs').format_messages(
                    instruction=instruction_prompt.format_messages(
                        **instruction_prompt_kwargs)[0].content,
                    fewshots='\n\n'.join(fewshots)
                )[0]
            )

    def _build_agent_prompt(self) -> None:
        # Build system and fewshot prompts first
        system_messages = self.constructor.build_system_prompt()
        fewshot_messages = self.constructor.build_fewshot_prompt(self.fewshots)

        self.prompt_history = system_messages + fewshot_messages
        self.prompt_history = self.constructor.collapse_prompts(self.prompt_history)
        self.log_idx = len(self.prompt_history)

        # Insert before task (for rules in ExpelAgent)
        self.insert_before_task_prompt()

        # Add task description
        task_messages = self.constructor.build_task_prompt(self.remove_task_suffix(self.task))
        self.prompt_history.extend(task_messages)

        # Insert after task
        self.insert_after_task_prompt()

        # Final collapse
        self.prompt_history = self.constructor.collapse_prompts(self.prompt_history)
        self.pretask_idx = len(self.prompt_history)
        return self.prompt_history

    def reset(self, *args, **kwargs) -> None:
        self.prompt_history = []
        self.update_dynamic_prompt_components(reset=True)
        self.curr_step = 1
        self._build_agent_prompt()

    def insert_after_task_prompt(self) -> None:
        return

    def job_not_done(self) -> bool:
        return self.task_idx < len(self.tasks)

    def next_task(self):
        self.task_idx += 1
        # if there are more tasks, reset the env and the agent
        if self.job_not_done():
            self.task = self.tasks[self.task_idx]['task']
            self.set_env(self.tasks[self.task_idx]['env_kwargs'], max_steps=self.max_steps)
            self.env.reset()
            self.reset()

    def reset_stats(self) -> None:
        self.success = 0
        self.fail = 0
        self.halted = 0

    def update_stats(self) -> None:
        if not self.is_success() and self.is_truncated():
            self.halted += 1
        else:
            if self.reward:
                self.success += 1
            else:
                self.fail += 1

    def get_stats(self) -> Tuple[int, int, int]:
        return self.success, self.fail, self.halted

    def collapse_prompts(self, prompt_history: List[ChatMessage]) -> List[ChatMessage]:
        """Use constructor's collapse_prompts method"""
        return self.constructor.collapse_prompts(prompt_history)

    def update_dynamic_prompt_components(self, reset: bool = False):
        #####################
        # Updating fewshots #
        #####################
        if isinstance(self.all_fewshots, dict):
            self.fewshots = self.all_fewshots[self.env.env_name]
        elif isinstance(self.all_fewshots, list):
            self.fewshots = self.all_fewshots

        #########################
        # Updating instructions #
        #########################
        if isinstance(self.all_system_instruction, str):
            self.system_instruction = self.all_system_instruction
        elif isinstance(self.all_system_instruction, dict):
            self.system_instruction = self.all_system_instruction[self.env.env_name]
        # if system gives instruction, then human instruction is empty
        self.human_instruction_kwargs['instruction'] = ''
        self.num_fewshots = len(self.fewshots)

        # Update constructor if it exists
        if hasattr(self, 'constructor'):
            self.constructor.system_instruction = self.system_instruction
            self.constructor.human_instruction_kwargs = self.human_instruction_kwargs

    def load_checkpoint(self, loaded_dict: Dict[str, Any], no_load_list: List['str'] = []) -> None:
        for k, v in loaded_dict.items():
            if k in no_load_list:
                continue
            setattr(self, k, v)
        # following attributes are not saved in pickle but correctely initialized back: ['rule_template', 'truncate_strategy', 'embedder', 'retriever_cls', 'manual', 'reflection_task_prompt', 'message_splitter', 'identifier', 'message_step_splitter', 'format_reflections', 'formatted_reflection', 'human_instruction', 'system_prompt', 'llm_parser', 'observation_formatter', 'env', 'print_message', 'llm', 'long_context_llm', 'token_counter']
