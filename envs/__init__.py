import json
import joblib
import random

from .base import BaseEnv
from .hotpotqa.hotpotqa import QAEnv
# from .fever.fever import FeverEnv
from .alfworld.alfworld import AlfworldEnv
# from .webshop.webshop import WebshopEnv
from utils import get_env_name_from_gamefile

# Taken from ReAct Github
idxs = list(range(7405))
random.Random(233).shuffle(idxs)

def get_data_range(cfg, mode='train'):
    """Get data range based on mode and configuration."""
    if hasattr(cfg.benchmark, 'data_split') and cfg.benchmark.data_split:
        if mode == 'eval':
            data_range = cfg.benchmark.data_split.eval_range
        else:  # mode == 'train' or default
            data_range = cfg.benchmark.data_split.train_range
        print(f"[DEBUG] Using data_split config for {mode} mode: {data_range}")
        return data_range
    else:
        # Fallback to original logic for backwards compatibility
        print(f"[DEBUG] No data_split config found for {mode} mode, using fallback")
        return None

INIT_TASKS_FN = dict(
    hotpotqa=lambda cfg, mode='train': (lambda data: [
        {
        'task': f'{cfg.benchmark.task_prefix}{row["question"]}',
        'env_kwargs': {
            'question': row['question'],
            'key': row['answer'],
        },
        'env_name': 'hotpotqa',
    } for row in data])(json.load(open(cfg.benchmark.task_file, "r"))),
    # # 100 tasks for fever
    # fever=lambda cfg, mode='train': [{
    #     'task': cfg.benchmark.task_prefix + FeverEnv(idx).reset().replace('Claim: ', ''),
    #     'env_kwargs': {
    #         'idx': idx,
    #     },
    #     'env_name': 'fever',
    # } for idx in idxs[:100]],
    alfworld=lambda cfg, mode='train': (lambda all_tasks, data_range: (
        lambda selected_tasks: (
            print(f"[DEBUG] ALFWorld {mode} mode: Total tasks loaded: {len(all_tasks)}, Selected range: {data_range}, Final task count: {len(selected_tasks)}") or selected_tasks
        )([
            {
            'task': f'{cfg.benchmark.task_prefix}{row["goal"]}',
            'env_kwargs': {
                'config': cfg.benchmark,
                "gamefile": row["gamefile"],
            },
            'env_name': get_env_name_from_gamefile(row['gamefile'])
            } for row in (all_tasks[data_range[0]:data_range[1]]
                         if data_range is not None
                         else (all_tasks[:cfg.benchmark.dataset.num_train_games]
                              if cfg.benchmark.dataset.num_train_games > 0
                              else all_tasks))
        ])
    )())(json.load(open(cfg.benchmark.task_file, "r")), get_data_range(cfg, mode)),
    # webshop=lambda cfg, mode='train': [
    #     {
    #     'task': f'{cfg.benchmark.task_prefix}{row["task"]}',
    #     'env_kwargs': {
    #         'session_idx': row["session_idx"],
    #     },
    #     'env_name': 'webshop'
    #     } for row in json.load(open(cfg.benchmark.task_file, "r"))
    # ],
)

# ENVS = dict(hotpotqa=QAEnv, fever=FeverEnv, alfworld=AlfworldEnv, webshop=WebshopEnv)
ENVS = dict(hotpotqa=QAEnv, alfworld=AlfworldEnv)
