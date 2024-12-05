import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import isaacgym
import modexenvs
from modexenvs import *
import torch
import argparse

def run(args):
	if 'Myo' in args.task:
		if args.task == 'MyoHand':
			env = MyoHand(seed=0)
			print('='*10)
			print("Observation space is", env.observation_space)
			print("Action space is", env.action_space)
			print('='*10)
			for _ in range(1000):
				env.reset()
				action = env.action_space.sample()
				for _ in range(1000):
					env.mj_render()
					env.step(action) # take a random action
			env.close()
		else:
			raise NotImplementedError(f'{args.task} not implemented !!!')
	else:
		num_envs = args.num_envs
		env = modexenvs.make(
			seed=0, 
			task=args.task, 
			num_envs=num_envs, 
			sim_device=args.device,
			rl_device=args.device,
			graphics_device_id=0,
		)
		print('='*10)
		print("Observation space is", env.observation_space)
		print("Action space is", env.action_space)
		print('='*10)
		obs = env.reset()
		for _ in range(300):
			random_action = .0 * torch.rand((num_envs,) + env.action_space.shape, device = 'cuda:0') - 0.5
			for i in range(100):
				env.step(random_action)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--task', type=str, default='AllegroHand',help='tasks')
	parser.add_argument('--num_envs', type=int, default=100, help='the number of created environments')
	parser.add_argument('--device', type=str, default='cuda:0', help='the device where we create env and collect data')
	args = parser.parse_args()
	run(args)