# env_utils.py
import random
import logging
import numpy as np
import torch
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from contextlib import contextmanager
import os

def permanent_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def set_seed(seed):
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.Formatter.converter = lambda *args: (datetime.now(UTC) - timedelta(hours=2)).timetuple()
    return logging.getLogger()


@contextmanager
def NoLoggerWarnings():
    from gym import logger
    logger.set_level(logger.ERROR)
    try:
        yield
    finally:
        logger.set_level(logger.INFO)


def get_train_val_env(env_class, config: dict):
    val_env = None
    if config.env.name == 'frozenlake':
        env = env_class(size=config.env.size, p=config.env.p)
    elif config.env.name == 'bandit':
        env = env_class(n_arms=config.env.n_arms)
    elif config.env.name == 'two_armed_bandit':
        lo_name, hi_name = config.env.low_risk_name, config.env.high_risk_name
        lo_val_name = config.env.low_risk_name if getattr(config.env, 'low_risk_val_name', None) is None else config.env.low_risk_val_name
        hi_val_name = config.env.high_risk_name if getattr(config.env, 'high_risk_val_name', None) is None else config.env.high_risk_val_name
        env = env_class(low_risk_name=lo_name, high_risk_name=hi_name)
        val_env = env_class(low_risk_name=lo_val_name, high_risk_name=hi_val_name)
        print(f"[INFO] val_env low_risk_name: {val_env.low_risk_name}, high_risk_name: {val_env.high_risk_name}")
        if val_env.low_risk_name is None or val_env.high_risk_name is None:
            print("[WARNING] val_env arm are None, falling back to not create val_env")
            val_env = None
    elif config.env.name == 'sokoban' or config.env.name == 'sokoban_dual':
        env_dict = config.env
        if isinstance(env_dict, dict) and 'env_kwargs' in env_dict:
            env_kwargs = dict(env_dict['env_kwargs'])
        else:
            try:
                env_kwargs = dict(env_dict.env_kwargs)
            except AttributeError:
                env_kwargs = dict(env_dict)
        # 合成dim_room
        if 'dim_x' in env_kwargs and 'dim_y' in env_kwargs:
            env_kwargs['dim_room'] = (env_kwargs.pop('dim_x'), env_kwargs.pop('dim_y'))
        print("DEBUG env_kwargs before filter:", env_kwargs)
        # 只保留SokobanEnv需要的参数
        allowed_keys = {'dim_room', 'max_steps', 'num_boxes', 'search_depth', 'dual_agent'}
        filtered_kwargs = {k: v for k, v in env_kwargs.items() if k in allowed_keys}
        print("DEBUG filtered_kwargs:", filtered_kwargs)
        env = env_class(**filtered_kwargs)
    elif config.env.name == 'countdown':
        env = env_class(parquet_path=config.env.train_path)
        val_env = env_class(parquet_path=config.env.val_path)
    else:
        raise ValueError(f"Environment {config.env.name} not supported")
    return env, val_env