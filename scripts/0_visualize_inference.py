import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import isaacgym
import modexenvs
from modexenvs import *
import torch
import numpy as np
from datacollector.utils import compute_poses_wrt_world
# import matplotlib.pyplot as plt
import os
import argparse
from isaacgym import gymutil
from isaacgym import gymapi

from models import Trainer
from omegaconf import OmegaConf

import mujoco

def visualize_targets(envs, fingertip_position):
    '''Visualize the target poses of fingertips as spheres
    '''
    envs.gym.clear_lines(envs.viewer)
    num_envs, fingertip_num, _ = fingertip_position.shape
    sphere_geoms = [gymutil.WireframeSphereGeometry(0.012, 12, 12, color=(1, 0, 0)),
                    gymutil.WireframeSphereGeometry(0.012, 12, 12, color=(0, 1, 0)),
                    gymutil.WireframeSphereGeometry(0.012, 12, 12, color=(0, 0, 1)),
                    gymutil.WireframeSphereGeometry(0.012, 12, 12, color=(1, 1, 0)),
                    gymutil.WireframeSphereGeometry(0.012, 12, 12, color=(1, 0, 1)),
                    ]
    
    root_base_pose = envs.root_state_tensor[:,:7]
    root_base_pose = root_base_pose.unsqueeze(1).repeat(1, fingertip_num, 1).reshape(-1,7)
    fingertip_rot = torch.Tensor([0,0,0,1]).to(fingertip_position.device).expand(num_envs, fingertip_num, 4)
    fingertip_pose = torch.cat([fingertip_position, fingertip_rot], dim=-1)
    fingertip_pose = compute_poses_wrt_world(fingertip_pose.reshape(-1,7), root_base_pose).reshape(num_envs, fingertip_num, -1)
    for i in range(num_envs):
        for j in range(fingertip_num):
            pos = fingertip_pose[i][j][:3]
            quat = fingertip_pose[i][j][3:7]
            p = gymapi.Vec3(pos[0], pos[1], pos[2])
            r = gymapi.Quat()
            r.x, r.y, r.z, r.w = quat[0], quat[1], quat[2], quat[3]
            gymutil.draw_lines(sphere_geoms[j], envs.gym, envs.viewer, envs.envs[i], gymapi.Transform(p=p, r=r))
            
    return fingertip_pose

def visualize_targets_mjc(env, fingertip_position):
    ## TODO: finish target visualization part of MyoHand

    ## Method 1: modify the position of sites in XML file
    # target_names = ['THtip_target', 'IFtip_target', 'MFtip_target','RFtip_target', 'LFtip_target']
    
    ## Method 2: use mujoco to create and delete geometries
    with env.sim.renderer._window.lock():
        for i in range(fingertip_position.shape[0]):
            env.sim.renderer._window._scn.ngeom+=1
            mujoco.mjv_initGeom(
                env.sim.renderer._window._scn.geoms[env.sim.renderer._window._scn.ngeom-1],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([1,1,1]),
                pos=fingertip_position[i],
                mat=np.eye(3).flatten(),
                rgba=np.array([0,1,0,1])
        )
    env.sim.renderer._window.sync()

def visualize_inference(args):
    cfg = OmegaConf.load(args.cfg_path)
    if args.model == 'mlp':
        model = Trainer(cfg)
    else:
        AssertionError(f'{args.model} not implemented yet !')
        
    model.load_model(args.model_path)
    model.net.eval()
    model.net.to(args.device)

    if 'Myo' in args.task:
        if args.task == 'MyoHand':
            envs = MyoHand(seed=128)
        else:
            raise NotImplementedError(f'{args.task} not implemented !!!')

        for _ in range(10):
            envs.reset()
            envs.mj_render()
            random_actions = envs.action_space.sample()
            pred_poses = model.net(torch.Tensor(random_actions).to(args.device)).view(-1, 3).detach().cpu().numpy()
            visualize_targets_mjc(envs, pred_poses)
            for _ in range(500):
                envs.mj_render()
                envs.step(random_actions)
            
    else: 
        num_envs = args.num_envs 

        envs = modexenvs.make(
            seed=0, 
            task=args.task, 
            num_envs=num_envs, 
            sim_device=args.device,
            rl_device=args.device,
            graphics_device_id=0,
        )

        for _ in range(10):
            random_actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = args.device) - 1.0
            pred_poses = model.net(random_actions).view(num_envs, -1, 3)
            _ = visualize_targets(envs, pred_poses)
            for _ in range(500):
                envs.step(random_actions)
            envs.reset_buf = torch.ones(envs.num_envs, device=envs.device, dtype=torch.long)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='ShadowHand',help='tasks')
    parser.add_argument('--num_envs', type=int, default=1, help='the number of created environments')
    parser.add_argument('--cfg_path', type=str, default='models/cfg/hand_model_train/shadowhand_forward_model.yaml', help='the path of configuration of data collector')
    parser.add_argument('--model', type=str, default='mlp', help='type of the model')
    parser.add_argument('--model_path', type=str, default='logs/models/shadowhand_forward_model_Jul_23_14:35:39_2024/ckpt-1.pt', help='the path of trained model')
    parser.add_argument('--device', type=str, default='cuda:0', help='the device where we create envs and collect data')
    args = parser.parse_args()
    
    visualize_inference(args)
        

 