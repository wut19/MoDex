import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import isaacgym
import modexenvs
import torch
import numpy as np
from datacollector.utils import compute_poses_wrt_world
import os
import argparse
from isaacgym import gymutil
from isaacgym import gymapi

def visualize_data(args):
    data_dir = args.data_path
    actions = np.load(os.path.join(data_dir, 'actions.npy'))
    fingertip_poses = np.load(os.path.join(data_dir, 'fingertip_poses.npy'))
    print(actions.shape, fingertip_poses.shape)

    num_envs, eps_len, fingertip_num, _ = fingertip_poses.shape 

    actions = torch.from_numpy(actions).to(device='cuda:0')
    fingertip_poses_data = torch.from_numpy(fingertip_poses).to(device='cuda:0')

    envs = modexenvs.make(
        seed=0, 
        task=args.task, 
        num_envs=num_envs, 
        sim_device=args.device,
        rl_device=args.device,
        graphics_device_id=0,
    )
    print("Observation space is", envs.observation_space)
    print("Action space is", envs.action_space)

    obs = envs.reset()

    for i in range(eps_len):
        fingertip_pose = fingertip_poses_data[:, i]
        _ = visualize_targets(envs, fingertip_pose)
        action = actions[:,i]
        for _ in range(100):
            envs.step(action)
        envs.reset_buf = torch.ones(envs.num_envs, device=envs.device, dtype=torch.long)
        

    # fingertip_poses_sim = []
    # for i in range(eps_len):
    #     action = actions[:, i]
    #     envs.step(action)
    #     fingertip_pose_sim = envs.rigid_body_states[:, envs.fingertip_handles][:, :, :7]
    #     root_base_pose = envs.root_state_tensor[:,:7]
    #     root_base_pose = root_base_pose.unsqueeze(1).repeat(1, fingertip_pose_sim.shape[1], 1).view(-1,7)
    #     fingertip_pose_sim = fingertip_pose_sim.view(-1,7)
    #     fingertip_pose_sim = compute_poses_wrt_root(fingertip_pose_sim, root_base_pose).view(num_envs, 1, -1, 7)
    #     fingertip_poses_sim.append(fingertip_pose_sim)
        
    # fingertip_poses_sim = torch.concat(fingertip_poses_sim,dim=1)
    # print(fingertip_poses_sim.shape, fingertip_poses_data.shape)

    # # calculate error between simulated fingertip poses and collected fingertip poses
    # error = torch.norm(fingertip_poses_sim-fingertip_poses_data,dim=-1).mean(-1).mean(0)
    # plt.plot(error.cpu().numpy())
    # plt.savefig('error.png')
    # plt.show()

def visualize_targets(envs, fingertip_pose):
    '''Visualize the target poses of fingertips as balls
    '''
    envs.gym.clear_lines(envs.viewer)
    num_envs, fingertip_num, _ = fingertip_pose.shape
    sphere_geom = gymutil.WireframeSphereGeometry(0.012, 12, 12, color=(1, 1, 0))
    
    root_base_pose = envs.root_state_tensor[:,:7]
    root_base_pose = root_base_pose.unsqueeze(1).repeat(1, fingertip_num, 1).reshape(-1,7)
    fingertip_pose = compute_poses_wrt_world(fingertip_pose.reshape(-1,7), root_base_pose).reshape(num_envs, fingertip_num, -1)
    for i in range(num_envs):
        for j in range(fingertip_num):
            pos = fingertip_pose[i][j][:3]
            quat = fingertip_pose[i][j][3:7]
            p = gymapi.Vec3(pos[0], pos[1], pos[2])
            r = gymapi.Quat()
            r.x, r.y, r.z, r.w = quat[0], quat[1], quat[2], quat[3]
            gymutil.draw_lines(sphere_geom, envs.gym, envs.viewer, envs.envs[i], gymapi.Transform(p=p, r=r))
            
    return fingertip_pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='task name')
    parser.add_argument('--data_path', type=str, help='path of data')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to work on')
    args = parser.parse_args()
    visualize_data(args)