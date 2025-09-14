import json
import joblib
import random

from .base import BaseEnv
from .hotpotqa.hotpotqa import QAEnv
# from .fever.fever import FeverEnv
from .alfworld.alfworld import AlfworldEnv
# from .webshop.webshop import WebshopEnv
from utils import get_env_name_from_gamefile

# Import the new dataset classes
from datasets import ExpelDataset

# Taken from ReAct Github
idxs = list(range(7405))
random.Random(233).shuffle(idxs)

def _create_dataset_instance(cfg):
    """Create and cache ExpelDataset instance."""
    if not hasattr(_create_dataset_instance, 'cache'):
        _create_dataset_instance.cache = {}

    # Use a simple key for caching (benchmark name and task file)
    cache_key = f"{cfg.benchmark.name}:{cfg.benchmark.task_file}"

    if cache_key not in _create_dataset_instance.cache:
        _create_dataset_instance.cache[cache_key] = ExpelDataset(cfg)

    return _create_dataset_instance.cache[cache_key]

def _load_tasks_with_dataset(cfg, mode='train'):
    """Load tasks using ExpelDataset class."""
    dataset = _create_dataset_instance(cfg)
    return dataset.load_tasks(mode)

INIT_TASKS_FN = dict(
    hotpotqa=_load_tasks_with_dataset,
    fever=_load_tasks_with_dataset,
    alfworld=_load_tasks_with_dataset,
    webshop=_load_tasks_with_dataset,
)

# ENVS = dict(hotpotqa=QAEnv, fever=FeverEnv, alfworld=AlfworldEnv, webshop=WebshopEnv)
ENVS = dict(hotpotqa=QAEnv, alfworld=AlfworldEnv)
