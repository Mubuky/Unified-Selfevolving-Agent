import getpass
import hydra
from omegaconf import DictConfig
from pathlib import Path
import os
from copy import deepcopy
from functools import partial
import dotenv
from tqdm import tqdm
dotenv.load_dotenv()

from common import AgentFactory
from utils import save_trajectories_log, load_trajectories_log, plot_trial_stats, split_logs_by_task, alfworld_results_per_env_name, get_webshop_mean_scores, get_fewshot_max_tokens
from storage import ExpelStorage
from agent.reflect import Count

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg : DictConfig) -> None:
    openai_api_key, base_url = AgentFactory.get_api_credentials(cfg)
    LOG_PATH = AgentFactory.get_log_path(cfg)
    LOG_PATH.mkdir(parents=True, exist_ok=True)

    storage = AgentFactory.initialize_storage(cfg)

    # Load trajectory checkpoint, init as empty if not exist
    checkpoint_exists = storage.exists(cfg.run_name, 'train')

    # Enhanced auto_resume support
    auto_resume = getattr(cfg, 'auto_resume', False)

    if cfg.resume:
        if checkpoint_exists:
            print(f"Loading checkpoint from {storage.get_run_path(cfg.run_name, 'train')}")
            out = storage.load_training_phase1(cfg.run_name)
        else:
            print(f"Resume requested but no checkpoint found, starting fresh")
            out = {'log': '', 'dicts': [], 'true_log': f'{str(cfg)}'}
    else:
        AgentFactory.handle_auto_resume_confirmation(cfg, checkpoint_exists)
        out = {'log': '', 'dicts': [], 'true_log': f'{str(cfg)}'}
    log, dicts, true_log = out['log'], out['dicts'], out['true_log']

    task_idx = dicts[-1]['task_idx'] if len(dicts) > 0 else 0
    react_agent = AgentFactory.create_agent(
        cfg=cfg,
        openai_api_key=openai_api_key,
        base_url=base_url,
        mode='train',
        task_idx=task_idx
    )

    if len(dicts) > 0:
        print(f"📊 Incremental training: Resuming from task_idx {dicts[-1].get('task_idx', 0)}")
        print(f"📈 Training with {len(react_agent.tasks)} total tasks ({len(react_agent.tasks) - dicts[-1].get('task_idx', 0)} new tasks)")
        print(f"🔍 Before load_checkpoint: agent.task_idx = {react_agent.task_idx}")
        print(f"🔍 Checkpoint contains task_idx = {dicts[-1].get('task_idx', 'NOT FOUND')}")
        react_agent.load_checkpoint(loaded_dict=dicts[-1], no_load_list=AgentFactory.get_training_phase_no_load_list())
        print(f"🔍 After load_checkpoint: agent.task_idx = {react_agent.task_idx}")
        print(f"🔍 Agent will start training from task {react_agent.task_idx} to {len(react_agent.tasks)-1}")
        print(f"🔍 First task to train: {react_agent.tasks[react_agent.task_idx]['task'][:80] if react_agent.task_idx < len(react_agent.tasks) else 'NO MORE TASKS'}")
        if 'eval_idx_list' in dicts[-1]:
            react_agent.eval_idx_list = dicts[-1]['eval_idx_list']

    print(f"""*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

You are using the following language model: {react_agent.llm.model_name}

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*""")

    # Create progress bar
    progress_bar = tqdm(
        total=len(react_agent.tasks),
        initial=react_agent.task_idx,
        desc="Training",
        unit="task"
    )

    while react_agent.job_not_done():
        prefix = f"#######################################\nTASK {react_agent.task_idx}"
        if cfg.agent_type in ['reflection', 'expel']:
            prefix += f' Reflection {react_agent.reflection_counter.count}\n\n'
        else:
            prefix += '\n\n'

        print(prefix + react_agent.remove_task_suffix(react_agent.task)) # remove_task_suffix used for alfworld

        react_agent.run(mode='train')

        #############################################
        ### Update & Save trajectory logs + dicts ###
        #############################################
        react_agent.update_stats()
        log += prefix + react_agent.log_history() + '\n\n'
        true_log += prefix + react_agent.log_history(include_all=True) + '\n\n'

        # next task - only update progress bar if we actually moved to next task
        if cfg.agent_type in ['reflection', 'expel']:
            # For reflection-based agents, only update progress when truly moving to next task
            task_incremented = react_agent.next_task()
            if task_incremented:
                progress_bar.update(1)
        else:
            # For basic react agent, always update progress (no reflection mechanism)
            react_agent.next_task()
            progress_bar.update(1)

        dicts.append({k: deepcopy(v) for k, v in react_agent.__dict__.items() if type(v) in [list, set, str, bool, int, dict, Count] and k not in ['openai_api_key', 'llm']}) # not saving complicated objects

        # Save experience checkpoint using storage system
        storage.save_experience(
            run_name=cfg.run_name,
            agent_dict=dicts[-1],  # Save the latest agent state
            log=log,
            true_log=true_log
        )
        #############################################

    progress_bar.close()

    ######################################
    ### Final Log & Save stats + PRINT ###
    ######################################
    success, fail, halted = react_agent.get_stats()
    log += f"########################################\nEND TRIAL\nTrial summary: Success: {success}/{success + fail + halted}, Fail: {fail}/{success + fail + halted}, Halted: {halted}/{success + fail + halted}"
    true_log += f"########################################\nEND TRIAL\nTrial summary: Success: {success}/{success + fail + halted}, Fail: {fail}/{success + fail + halted}, Halted: {halted}/{success + fail + halted}"
    print(f'Finished. Success: {success}, Fail: {fail}, Halted: {halted}')

    parsed_result = split_logs_by_task(text=log, num_tasks=len(react_agent.tasks))
    reflection_results = plot_trial_stats(parsed_result=parsed_result, benchmark=cfg.benchmark.name, max_trials=cfg.agent.max_reflection_depth + 1, save_path=f"{LOG_PATH}/{cfg.run_name}_logs_stats.png")

    results = ', '.join([f"{k}: {v}" for k, v in reflection_results.items()]) + '\n'
    if cfg.benchmark.name == 'alfworld':
        results += str(alfworld_results_per_env_name(dicts[-1]))
    elif cfg.benchmark.name == 'webshop':
        results += str(get_webshop_mean_scores(log, len(react_agent.tasks), cfg.agent.max_reflection_depth + 1))
    log += f'\n\n{results}\n#######################################'
    true_log += f'\n\n{results}\n#######################################'
    print(results)

    # Save final experience results using storage system
    storage.save_experience(
        run_name=cfg.run_name,
        agent_dict=dicts[-1] if dicts else {},  # Save the final agent state
        log=log,
        true_log=true_log
    )

    log, dicts, true_log = '', [], ''
    react_agent.reset_stats()
    ################################

if __name__ == "__main__":
    main()