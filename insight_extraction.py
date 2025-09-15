import getpass
import hydra
from omegaconf import DictConfig
from pathlib import Path
from functools import partial
import os
import random

from common import AgentFactory
from langchain_openai import ChatOpenAI
from envs import INIT_TASKS_FN
from utils import save_trajectories_log, load_trajectories_log, shuffled_chunks, get_split_eval_idx_list
from storage import ExpelStorage
from agent.reflect import Count

from dotenv import load_dotenv
load_dotenv()


@hydra.main(version_base=None, config_path="configs", config_name="insight_extraction")
def main(cfg : DictConfig) -> None:
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
        log += react_agent.create_rules(
            list(training_ids),
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
            log += react_agent.create_rules(
                list(training_ids),
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

if __name__ == "__main__":
    main()
