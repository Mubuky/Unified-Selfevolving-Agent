import getpass
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os
from copy import deepcopy
from functools import partial
import dotenv
from tqdm import tqdm
import random
dotenv.load_dotenv()

from common import AgentFactory
from utils import save_trajectories_log, load_trajectories_log, plot_trial_stats, split_logs_by_task, alfworld_results_per_env_name, get_webshop_mean_scores, get_fewshot_max_tokens, shuffled_chunks, get_split_eval_idx_list
from storage import ExpelStorage
from agent.reflect import Count
from langchain_openai import ChatOpenAI
from envs import INIT_TASKS_FN


def prepare_phase_2_config(cfg: DictConfig) -> DictConfig:
    """
    Prepare configuration for Training Phase 2 (Insights Extraction).
    Adds the necessary parameters from insight_extraction.yaml.
    """
    # Create a copy to avoid modifying the original config
    phase2_cfg = OmegaConf.create(dict(cfg))

    # Add insights extraction specific parameters
    phase2_cfg.load_run_name = cfg.run_name  # Critical for data handoff
    phase2_cfg.folded = True
    phase2_cfg.resume_fold = -1
    phase2_cfg.seed = 42

    return phase2_cfg


def run_training_phase_1(cfg: DictConfig) -> None:
    """
    Execute Training Phase 1: Experience Collection.
    This is the exact logic from the original train.py.
    """
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
        desc="Training Phase 1",
        unit="task"
    )

    # Determine starting task index
    start_task_idx = react_agent.task_idx

    # Outer loop: Task-level iteration
    for expected_task_idx in range(start_task_idx, len(react_agent.tasks)):
        print(f"Starting Task {expected_task_idx}")

        # Inner loop: Reflection step-level iteration (preserves original logic)
        while react_agent.job_not_done() and react_agent.task_idx == expected_task_idx:
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
                    print(f"Task {expected_task_idx} completed")
                    break  # Exit inner loop when task is completed
            else:
                # For basic react agent, always update progress (no reflection mechanism)
                react_agent.next_task()
                progress_bar.update(1)
                print(f"Task {expected_task_idx} completed")
                break  # Exit inner loop after single step for basic agents

            # Note: Checkpoint saving moved to task level (after for loop iteration)

        # Task-level checkpoint: Save state after each completed task
        if react_agent.task_idx > expected_task_idx or not react_agent.job_not_done():
            # Task completed - save the final state
            task_completion_state = {k: deepcopy(v) for k, v in react_agent.__dict__.items() if type(v) in [list, set, str, bool, int, dict, Count] and k not in ['openai_api_key', 'llm']}
            dicts.append(task_completion_state)

            storage.save_experience(
                run_name=cfg.run_name,
                agent_dict=task_completion_state,
                log=log,
                true_log=true_log
            )

            print(f"Task {expected_task_idx} checkpoint saved (task_idx: {react_agent.task_idx})")

        # Check if all tasks are completed
        if not react_agent.job_not_done():
            break

    progress_bar.close()

    ######################################
    ### Final Log & Save stats + PRINT ###
    ######################################
    success, fail, halted = react_agent.get_stats()
    log += f"########################################\nEND TRIAL\nTrial summary: Success: {success}/{success + fail + halted}, Fail: {fail}/{success + fail + halted}, Halted: {halted}/{success + fail + halted}"
    true_log += f"########################################\nEND TRIAL\nTrial summary: Success: {success}/{success + fail + halted}, Fail: {fail}/{success + fail + halted}, Halted: {halted}/{success + fail + halted}"
    print(f'Training Phase 1 Finished. Success: {success}, Fail: {fail}, Halted: {halted}')

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


def run_training_phase_2(cfg: DictConfig) -> None:
    """
    Execute Training Phase 2: Insights Extraction.
    This is the exact logic from the original insight_extraction.py.
    """
    openai_api_key, base_url = AgentFactory.get_api_credentials(cfg)
    LOG_PATH = AgentFactory.get_log_path(cfg)
    SAVE_PATH = LOG_PATH / 'extracted_insights'
    SAVE_PATH.mkdir(exist_ok=True)

    storage = AgentFactory.initialize_storage(cfg)

    # Enhanced auto_resume support for insight extraction
    auto_resume = getattr(cfg, 'auto_resume', False)
    checkpoint_path = f"{SAVE_PATH}/{cfg.run_name}.pkl"
    checkpoint_exists = os.path.exists(checkpoint_path)

    # Overwriting confirmation (skip if auto_resume is enabled)
    if not cfg.resume and checkpoint_exists and cfg.run_name != 'test' and not auto_resume:
        while True:
            res = input(f"Are you sure to overwrite '{cfg.run_name}'? (Y/N)\n").lower()
            if res == 'n':
                exit(0)
            elif res == 'y':
                break
    elif not cfg.resume and checkpoint_exists and auto_resume:
        print(f"Auto-resume mode: Existing insight checkpoint will be overwritten at {checkpoint_path}")
    if cfg.resume and cfg.resume_fold < 0:
        print('Specify a fold to resume when resuming a run! (resume_fold=X)')
        exit(1)
    # Load training phase 1 data for insights extraction
    if cfg.resume and cfg.resume_fold > -1:
        # If resuming from a specific fold, load insights checkpoint
        out = storage.load_insights(cfg.load_run_name)
    else:
        # Normal case: load experience data
        out = storage.load_experience(cfg.load_run_name)
    dicts = out['dicts']
    log = out['log'] if cfg.resume else ''

    cfg.folded = True
    react_agent = AgentFactory.create_agent(
        cfg=cfg,
        openai_api_key=openai_api_key,
        base_url=base_url,
        mode='train',
        use_direct_max_fewshot_tokens=True
    )

    print(f'Loading agent from {LOG_PATH}')
    react_agent.load_checkpoint(dicts[-1], no_load_list=AgentFactory.get_insights_extraction_no_load_list())

    random.seed(cfg.seed)
    num_training_tasks = len(INIT_TASKS_FN[cfg.benchmark.name](cfg, mode='train'))
    if not cfg.resume:
        resume = False
    else:
        resume = 'eval_idx_list' in dicts[-1]
    eval_idx_list = dicts[-1].get('eval_idx_list', get_split_eval_idx_list(dicts[-1], cfg.benchmark.eval_configs.k_folds))

    print(f'eval_idx_list: {eval_idx_list}')

    # If no cross-validation (empty eval_idx_list), use all data for training
    if not eval_idx_list:
        print("No cross-validation: using all data for training")
        training_ids = set(range(num_training_tasks))
        log += '################## NO CROSS-VALIDATION - FULL DATA TRAINING ##################\n'
        log += react_agent.manager.create_rules(
            training_ids=list(training_ids),
            cache_fold=None,
            logging_dir=str(SAVE_PATH),
            run_name=cfg.run_name,
            loaded_dict=dicts[-1] if resume else None,
            loaded_log=dicts[-1].get('critique_summary_log', '') if resume else None,
            eval_idx_list=[],
            saving_dict=True,
        )
    else:
        # Original cross-validation logic
        starting_fold = dicts[-1]['starting_fold'] = dicts[-1].get('critique_summary_fold', 0)

        resume_starting_fold = starting_fold
        critique_summary_log = dicts[-1].get('critique_summary_log', '')

        for k, eval_idxs in enumerate(eval_idx_list):
            if k < starting_fold:
                continue
            training_ids = set(range(num_training_tasks)) - set(eval_idxs)
            (SAVE_PATH / f"fold_{k}").mkdir(exist_ok=True)
            log += f'################## FOLD {k} ##################\n'
            log += react_agent.manager.create_rules(
                training_ids=list(training_ids),
                cache_fold=k,
                logging_dir=str(SAVE_PATH / f"fold_{k}"),
                run_name=cfg.run_name,
                loaded_dict=dicts[-1] if resume and resume_starting_fold == starting_fold else None,
                loaded_log=critique_summary_log if resume and resume_starting_fold == starting_fold else None,
                eval_idx_list=eval_idx_list,
                saving_dict=True,
            )
            starting_fold += 1

    save_dict = {k: v for k, v in react_agent.__dict__.items() if type(v) in [list, set, str, bool, int, dict, Count] and k not in ['openai_api_key', 'llm']}
    if cfg.folded:
        save_dict['eval_idx_list'] = eval_idx_list
    dicts.append(save_dict)

    # Save insights extraction results using storage system
    storage.save_insights(
        run_name=cfg.run_name,
        agent_dict=save_dict,
        log=log,
        original_run_name=cfg.load_run_name  # Reference to original training run
    )

    print("Training Phase 2 (Insights Extraction) completed successfully!")


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """
    Main function for unified two-phase training.

    Configuration options:
    - run_phase_1: Whether to run Training Phase 1 (default: True)
    - run_phase_2: Whether to run Training Phase 2 (default: True)

    When both phases are enabled, they run sequentially with automatic data handoff.
    """
    # Phase control parameters (can be added to config files)
    run_phase_1 = cfg.get('run_phase_1', True)
    run_phase_2 = cfg.get('run_phase_2', True)

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                        ExpeL Two-Phase Training                      ║
║                                                                      ║
║  Phase 1: Experience Collection    -> {run_phase_1}                 ║
║  Phase 2: Insights Extraction      -> {run_phase_2}                 ║
║                                                                      ║
║  Run Name: {cfg.run_name:<50}   ║
║  Benchmark: {cfg.benchmark.name:<49}  ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    if run_phase_1:
        print("\n" + "="*70)
        print("🚀 Starting Training Phase 1: Experience Collection")
        print("="*70)
        run_training_phase_1(cfg)
        print("\n✅ Training Phase 1 completed successfully!")

    if run_phase_2:
        print("\n" + "="*70)
        print("🧠 Starting Training Phase 2: Insights Extraction")
        print("="*70)

        # Prepare configuration for Phase 2
        phase2_cfg = prepare_phase_2_config(cfg)
        run_training_phase_2(phase2_cfg)
        print("\n✅ Training Phase 2 completed successfully!")

    if run_phase_1 and run_phase_2:
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                   🎉 Two-Phase Training Complete! 🎉                ║
║                                                                      ║
║  ✅ Phase 1: Experience Collection - DONE                          ║
║  ✅ Phase 2: Insights Extraction   - DONE                          ║
║                                                                      ║
║  Your agent is now ready for evaluation with the extracted insights!║
║                                                                      ║
║  Next step: Run evaluation with the insights:                       ║
║  python eval.py run_name={cfg.run_name}                           ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
    elif run_phase_1:
        print(f"""
✅ Training Phase 1 completed!
   To continue with insights extraction, run:
   python train.py run_name={cfg.run_name} run_phase_1=false run_phase_2=true
        """)
    elif run_phase_2:
        print(f"""
✅ Training Phase 2 completed!
   Your agent is now ready for evaluation with the extracted insights!
        """)


if __name__ == "__main__":
    main()