import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import isaacgym
import modexenvs
from modexenvs import *
import torch
import argparse
from datacollector.data_collector import IsaacHandDataCollector, SeqIsaacHandDataCollector, MyoHandDataCollector, SeqMyoHandDataCollector
import time
from omegaconf import OmegaConf
from tqdm import tqdm

def collect(args):
	cfg = OmegaConf.load(args.cfg)
	if 'Myo' in cfg.task:
		if cfg.task == 'MyoHand':
			env = MyoHand(seed=0)
		else:
			raise NotImplementedError(f'{cfg.task} not implemented !!!')
		if cfg.collector == 'MyoHandDataCollector':
			data_collector = MyoHandDataCollector(env=env, cfg=cfg)
		elif cfg.collector == 'SeqMyoHandDataCollector':
			data_collector = SeqMyoHandDataCollector(env=env, cfg=cfg)
		else:
			raise NotImplementedError(f'{cfg.collector} not implemented !!!')
		
		curr_t = data_collector.t
		for _ in tqdm(range(curr_t, data_collector.eps_len)):
			# random exploration here
			# you can replace this with your own policy
			# env.mj_render()
			random_actions = env.action_space.sample()
			data_collector.step(random_actions)
		data_collector.save()
		data_collector.env.close()
	else:
		num_envs = cfg.num_envs # default: 5000 for train, 50 for validation

		env = modexenvs.make(
			seed=1, 
			task=cfg.task, 
			num_envs=num_envs, 
			sim_device=cfg.device,
			rl_device=cfg.device,
			graphics_device_id=0,
			# headless=True,
		)
		if cfg.collector == 'IsaacHandDataCollector':
			data_collector = IsaacHandDataCollector(env=env, cfg=cfg)
		elif cfg.collector == 'SeqIsaacHandDataCollector':
			data_collector = SeqIsaacHandDataCollector(env=env, cfg=cfg)
		else:
			raise NotImplementedError(f'{cfg.collector} not implemented !!!')
		
		curr_t = data_collector.t
		for _ in tqdm(range(curr_t, data_collector.eps_len)):
			# random exploration here
			# you can replace this with your own policy
			random_actions = 2.0 * torch.rand((num_envs,) + env.action_space.shape, device = 'cuda:0') - 1.0
			data_collector.step(random_actions, cfg.ctrl_args)
		data_collector.save()
 
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg', type=str, default='datacollector/cfg/Allegro.yaml', help='the path of configuration of data collector')
	args = parser.parse_args()
	t0 = time.time()
	collect(args)
	t1 = time.time()
	print(f'------------time: {t1-t0} s-----------')