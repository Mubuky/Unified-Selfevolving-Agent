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

INIT_TASKS_FN = dict(
    hotpotqa=lambda cfg: (lambda data: [
        {
        'task': f'{cfg.benchmark.task_prefix}{row["question"]}',
        'env_kwargs': {
            'question': row['question'],
            'key': row['answer'],
        },
        'env_name': 'hotpotqa',
    } for row in data])(json.load(open(cfg.benchmark.task_file, "r"))),
    # # 100 tasks for fever
    # fever=lambda cfg: [{
    #     'task': cfg.benchmark.task_prefix + FeverEnv(idx).reset().replace('Claim: ', ''),
    #     'env_kwargs': {
    #         'idx': idx,
    #     },
    #     'env_name': 'fever',
    # } for idx in idxs[:100]],
    alfworld=lambda cfg: (lambda all_tasks: [
        {
        'task': f'{cfg.benchmark.task_prefix}{row["goal"]}',
        'env_kwargs': {
            'config': cfg.benchmark,
            "gamefile": row["gamefile"],
        },
        'env_name': get_env_name_from_gamefile(row['gamefile'])
        } for row in (all_tasks[:cfg.benchmark.dataset.num_train_games] 
                     if cfg.benchmark.dataset.num_train_games > 0 
                     else all_tasks)
    ])(json.load(open(cfg.benchmark.task_file, "r"))),
    # webshop=lambda cfg: [
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
