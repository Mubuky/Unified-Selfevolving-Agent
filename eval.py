from typing import List
import getpass
import hydra
from omegaconf import DictConfig
from pathlib import Path
import os
from copy import deepcopy
from functools import partial
import dotenv
dotenv.load_dotenv()

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
from memory import (
    EMBEDDERS,
    RETRIEVERS,
)
from models import LLM_CLS
from utils import get_fewshot_max_tokens, split_logs_by_task, plot_trial_stats, alfworld_results_per_env_name_log, get_webshop_mean_score
from storage import ExpelStorage


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg : DictConfig) -> None:
    if cfg.testing:
        openai_api_key = 'NO_KEY_FOR_TESTING'
        base_url = None
    else:
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        base_url = os.environ.get('BASE_URL')    
    LOG_PATH = Path('/'.join([cfg.log_dir, cfg.benchmark.name, cfg.agent_type]))
    SAVE_PATH = LOG_PATH / 'eval'
    SAVE_PATH.mkdir(exist_ok=True)

    # Initialize storage system
    storage = ExpelStorage(cfg)

    print(f"{SAVE_PATH}/{cfg.run_name}.pkl")
    
    # Overwriting confirmation
    if not cfg.resume and os.path.exists(f"{SAVE_PATH}/{cfg.run_name}.pkl") and cfg.run_name != 'test':
        while True:
            res = input(f"Are you sure to overwrite '{cfg.run_name}'? (Y/N)\n").lower()
            if res == 'n':
                exit(0)
            elif res == 'y':
                break

    # Load trajectory checkpoint
    if cfg.resume:
        # Resume from evaluation checkpoint
        out = storage.load_evaluation_checkpoint(cfg.load_run_name)
    else:
        # Load insights from training phase 2 for fresh evaluation
        out = storage.load_insights(cfg.load_run_name)
    dicts = out['dicts']
    log = out['log'] if cfg.resume else f'### EVALUATION MODE ###\n{str(cfg)}\n'
    true_log = out['true_log'] if cfg.resume else f'### EVALUATION MODE ###\n{str(cfg)}\n'

    # Load training tasks for rule creation (from training data range)
    training_tasks = INIT_TASKS_FN[cfg.benchmark.name](cfg, mode='train')
    num_training_tasks = len(training_tasks)

    # Load evaluation tasks (from eval data range)
    eval_tasks = INIT_TASKS_FN[cfg.benchmark.name](cfg, mode='eval')

    # Resume logic - start from task index if resuming
    starting_idx = dicts[-1].get('starting_idx', 0) if len(dicts) > 0 else 0

    react_agent = AGENT[cfg.agent_type](
        name=cfg.ai_name,
        system_instruction=SYSTEM_INSTRUCTION[cfg.benchmark.name],
        human_instruction=HUMAN_INSTRUCTION[cfg.benchmark.name],
        tasks=eval_tasks,
        fewshots=FEWSHOTS[cfg.benchmark.name],
        system_prompt=system_message_prompt,
        env=ENVS[cfg.benchmark.name],
        max_steps=cfg.benchmark.max_steps,
        openai_api_key=openai_api_key,
        base_url=base_url,
        llm=cfg.agent.llm,
        llm_builder=LLM_CLS,
        reflection_fewshots=REFLECTION_FEWSHOTS[cfg.benchmark.name],
        reflection_task_prompt=HUMAN_REFLECTION_INSTRUCTION[cfg.benchmark.name],
        reflection_system_instruction=SYSTEM_REFLECTION_INSTRUCTION[cfg.benchmark.name],
        reflection_system_prompt=SYSTEM_INSTRUCTION[cfg.benchmark.name],
        max_relfection_depth=cfg.agent.max_reflection_depth if 'max_reflection_depth' in cfg.agent.keys() else 0,
        system_critique_instructions=SYSTEM_CRITIQUE_INSTRUCTION[cfg.benchmark.name],
        human_critiques=HUMAN_CRITIQUES,
        max_num_rules=cfg.agent.max_num_rules if 'max_num_rules' in cfg.agent.keys() else 0,
        rule_template=RULE_TEMPLATE[cfg.benchmark.name],
        truncate_strategy=cfg.agent.truncate_strategy if 'truncate_strategy' in cfg.agent.keys() else None,
        llm_parser=LLM_PARSER[cfg.benchmark.name],
        observation_formatter=OBSERVATION_FORMATTER[cfg.benchmark.name],
        embedder=EMBEDDERS(cfg.agent.retrieval_kwargs.embedder_type),
        embedder_path=cfg.agent.retrieval_kwargs.embedder_path,
        step_stripper=STEP_STRIPPER[cfg.benchmark.name],
        retriever_cls=RETRIEVERS(cfg.agent.retrieval_kwargs.retriever_type),
        message_splitter=CYCLER[cfg.benchmark.name],
        identifier=STEP_IDENTIFIER[cfg.benchmark.name],
        message_step_splitter=partial(STEP_CYCLER, benchmark=cfg.benchmark.name),
        reflection_prefix=REFLECTION_PREFIX[cfg.benchmark.name],
        previous_trials_formatter=PREVIOUS_TRIALS_FORMATTER[cfg.benchmark.name],
        success_critique_num=cfg.agent.success_critique_num,
        fewshot_strategy=cfg.agent.fewshot_strategy,
        benchmark_name=cfg.benchmark.name,
        reranker=cfg.agent.retrieval_kwargs.reranker,
        buffer_retrieve_ratio=cfg.agent.retrieval_kwargs.buffer_retrieve_ratio,
        critique_truncate_strategy=cfg.agent.critique_truncate_strategy,
        critique_summary_suffix=CRITIQUE_SUMMARY_SUFFIX,
        testing=cfg.testing,
        task_idx=starting_idx,
        max_fewshot_tokens=get_fewshot_max_tokens(cfg.benchmark.name) if cfg.agent.retrieval_kwargs.max_fewshot_tokens == 'auto' else cfg.agent.retrieval_kwargs.max_fewshot_tokens,
    )

    if len(dicts) > 0:
        no_load_list = ['ai_message', 'message_type_format', 'max_num_rules', 'testing', 'human_critiques', 'system_critique_instructions', 'fewshot_strategy', 'success', 'halted', 'fail', 'task_idx', 'prompt_history', 'critique_truncate_strategy', 'success_critique_num', 'reflection_fewshots', 'reflection_system_prompt', 'reflection_prefix', 'reflection_prompt_history', 'reflections', 'previous_trial', 'perform_reflection', 'increment_task', 'reflection_system_kwargs', 'prepend_human_instruction', 'name', 'tasks', 'human_instruction_kwargs', 'all_system_instruction', 'all_fewshots', 'max_steps', 'ordered_summary', 'fewshots', 'system_instruction', 'num_fewshots', 'curr_step', 'log_idx', 'pretask_idx', 'reflect_interaction_idx', 'truncated', 'reward', 'terminated', 'autoregressive_model_instruction', 'failed_training_task_idx', '_train', 'task',
        'starting_idx', 'rule_template', 'max_fewshot_tokens', 'buffer_retrieve_ratio']
        react_agent.load_checkpoint(dicts[-1], no_load_list=no_load_list)
        # resetting task_idx
        react_agent.task = react_agent.tasks[starting_idx]['task']
        react_agent.reset()

    react_agent.eval()
    react_agent.no_rules = cfg.no_rules

    print(f'*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n\nWe are using the following model: {react_agent.llm.model_name}\n\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
    true_log += str(react_agent.llm.llm) + '\n'

    # Create rules using training data (if not disabled)
    if not cfg.no_rules:
        training_ids = list(range(num_training_tasks))
        react_agent.create_rules(
            training_ids,
            cache_fold=None,
            load_cache_fold=0 if cfg.load_cache_rules else None,
        )

    # Evaluate on each task in evaluation set
    for eval_idx in range(starting_idx, len(eval_tasks)):
        prefix = f"#######################################\nTASK {eval_idx}\n"
        prefix += react_agent.remove_task_suffix(react_agent.tasks[eval_idx]['task']) + '\n'
        print(prefix)

        react_agent.run(mode='eval', eval_idx=eval_idx)

        # logging
        react_agent.update_stats()
        log += prefix + react_agent.log_history(include_task=False) + '\n\n'
        true_log += prefix + react_agent.log_history(include_all=True, include_task=False) + '\n\n'

        # Save checkpoint
        eval_dict = {k: deepcopy(v) for k, v in react_agent.__dict__.items() if type(v) in [list, set, str, bool, int, dict]}
        eval_dict.update({
            'starting_idx': eval_idx + 1,  # Next task to start from if resuming
        })
        dicts.append(eval_dict)
        storage.save_evaluation_results(
            run_name=cfg.run_name,
            agent_dict=eval_dict,
            log=log,
            true_log=true_log
        )

    # logging to files
    success, fail, halted = react_agent.get_stats()
    log += f"########################################\nEND TRIAL\nTrial summary: Success: {success}/{success + fail + halted}, Fail: {fail}/{success + fail + halted}, Halted: {halted}/{success + fail + halted}"
    true_log += f"########################################\nEND TRIAL\nTrial summary: Success: {success}/{success + fail + halted}, Fail: {fail}/{success + fail + halted}, Halted: {halted}/{success + fail + halted}"

    print(f'Finished. Success: {success}, Fail: {fail}, Halted: {halted}')

    parsed_result = split_logs_by_task(text=log, num_tasks=len(react_agent.tasks))
    reflection_results = plot_trial_stats(parsed_result=parsed_result, benchmark=cfg.benchmark.name, max_trials=1, save_path=f"{LOG_PATH}/{cfg.run_name}_logs_stats.png")

    results = ', '.join([f"{k}: {v}" for k, v in reflection_results.items()]) + '\n'
    if cfg.benchmark.name == 'alfworld':
        results += str(alfworld_results_per_env_name_log(log, len(react_agent.tasks), 1))
    elif cfg.benchmark.name == 'webshop':
        results += str(get_webshop_mean_score(log, len(react_agent.tasks), 1))
    log += f'\n\n{results}\n########################################'
    true_log += f'\n\n{results}\n########################################'
    print(results)

    storage.save_evaluation_results(
        run_name=cfg.run_name,
        agent_dict=dicts[-1] if dicts else {},
        log=log,
        true_log=true_log
    )

if __name__ == "__main__":
    main()
