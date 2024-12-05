import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from myosuite.utils import gym
import isaacgym
from modexenvs.inhand_manipulation import *
import torch
import argparse
from datacollector.data_collector import InHandManipulationDataCollector
import time
from omegaconf import OmegaConf
from tqdm import tqdm

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def collect(args):
    cfg = OmegaConf.load(args.cfg)
    env = gym.make(cfg.task)
    explore_policy = SAC.load(path=cfg.policy_path)
    vec = VecNormalize.load(cfg.vec_norm_path, DummyVecEnv([lambda: env]))
    data_collector = InHandManipulationDataCollector(env=env, cfg=cfg)
    curr_t = data_collector.t
    env.reset()
    for _ in tqdm(range(curr_t, data_collector.eps_len)):
        o = env.get_obs()
        o = vec.normalize_obs(o)
        action, _ = explore_policy.predict(o, deterministic=False)  # deterministic=False to explore
        o, r, _, _, info = data_collector.step(action)
    data_collector.save()
    env.close()
 
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg', type=str, default='datacollector/cfg/inhand_manipulation/object_hold.yaml', help='the path of configuration of data collector')
	args = parser.parse_args()
	t0 = time.time()
	collect(args)
	t1 = time.time()
	print(f'------------time: {t1-t0} s-----------')