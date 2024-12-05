import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import isaacgym
import modexenvs
from modexenvs import *

import torch
import torch.optim
import torch.nn.functional as F
from omegaconf import OmegaConf
from models import Trainer

import os
import argparse
from fastprogress import progress_bar

import scipy.stats as stats
import numpy as np

from functools import partial

from datacollector.utils import compute_poses_wrt_root, compute_poses_wrt_world

import time


task = 'Robotiq3Finger'
target_positions = torch.from_numpy(np.load('data/hand_data/Robotiq3Finger/Robotiq_random_val_Jul_23_21:28:16_2024/hand_states.npy').reshape(-1, 3, 3)).to('cuda:0')
target_actions = torch.from_numpy(np.load('data/hand_data/Robotiq3Finger/Robotiq_random_val_Jul_23_21:28:16_2024/actions.npy').reshape(-1, 11)).to('cuda:0')


np.random.seed(0)
envs = modexenvs.make(
    seed=0, 
    task=task, 
    num_envs=100, 
    sim_device='cuda:0',
    rl_device='cuda:0',
    graphics_device_id=0,
    headless=True,
)

reach_error = 0
successes = 0
plan_times = 0

for num in range(100):
    target_action = target_actions[num]
    target_position = target_positions[num]

    #  rs planning
    start_t = time.time()
    actions = 2.0 * torch.rand((1000, 100,) + envs.action_space.shape, device = 'cuda:0') - 1.0
    dists = []
    bar = progress_bar(range(1000))
    for i in bar:
        envs.reset()
        for j in range(200):
            envs.step(actions[i])

        finger_tips_idxs = envs.fingertip_handles
        fingertip_pose = envs.rigid_body_states[:, finger_tips_idxs][:, :, :7]
        root_base_pose = envs.root_state_tensor[:,:7]
        root_base_pose = root_base_pose.unsqueeze(1).repeat(1, fingertip_pose.shape[1], 1).view(-1,7)
        fingertip_pose = fingertip_pose.view(-1,7)
        
        # converse fingertip pose to hand root coordinate
        fingertip_pose_base = compute_poses_wrt_root(fingertip_pose, root_base_pose).view(envs.num_envs, -1, 7)
        
        tip_offset = torch.Tensor(envs.tip_offset+[0,0,0,1]).to(fingertip_pose_base.device)
        tip_offset = tip_offset.expand_as(fingertip_pose)
        fingertip_pos_base = compute_poses_wrt_world(tip_offset.reshape(-1,7), fingertip_pose_base.reshape(-1, 7)).reshape(envs.num_envs, -1, 7)[...,:3]

        pos_loss = torch.norm(fingertip_pos_base[0] - target_position, dim=-1).mean(0)/5
        dists.append(pos_loss)
    dists = torch.stack(dists)
    best_idx = torch.argmin(dists)
    plan_time = time.time() - start_t
    pos_dist = dists[best_idx]

    
    reach_error += pos_dist
    success = pos_dist < 0.006
    successes += int(success)
    plan_times += plan_time
    print(f'----------{num}----------')
    print(f'success: {success}, reach_error: {pos_loss}, planning time: {plan_time}')
    print('---------------------')
    envs.reset_buf = torch.ones(envs.num_envs, device=envs.device, dtype=torch.long)
        
avg_reach_error = reach_error / 100.
avg_success = successes / 100.
avg_plan_time = plan_times / 100.
print(f'----------Done!!----------')
print(f'averaged success: {avg_success}, reach_error: {avg_reach_error}, planning time: {avg_plan_time}')