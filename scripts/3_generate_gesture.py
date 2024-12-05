import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import isaacgym
import gym
import modexenvs
from modexenvs import *

import torch
import torch.optim
import torch.nn.functional as F
from omegaconf import OmegaConf
from models import Trainer

from isaacgym import gymutil
from isaacgym import gymapi

import os
import argparse
from fastprogress import progress_bar

import scipy.stats as stats
import numpy as np

from functools import partial

from datacollector.utils import compute_poses_wrt_root, compute_poses_wrt_world

import time
import cv2

'''
    Generate gestures with a shadowhand / allegro hand
'''
def normalize(x):
    #  B x N

    # # first version
    # if x.abs().max()>1:
    #     x_min, x_max = x.min(), x.max()
    #     x = (x - x_min) / (x_max - x_min + 1e-10)
    #     return 2*x - 1
    # return x

    # # second version
    return 2 * torch.sigmoid(x) - 1

def unnormalize(x):
    return torch.logit((x + 1) / 2)

########## Gesture cost functions ##########
def gesture_test(fingertip_position):
    ''' num_envs (1) x finger_num x 3
    '''
    loss1 = - fingertip_position[:,:, 0].mean(-1)
    loss = loss1
    return loss

def gesture_ok(fingertip_position):
    ''' num_envs (1) x finger_num x 3
        for shadowhand and allegro
    '''
    # ##### for shadowhand
    # o_loss1 = torch.norm(fingertip_position[:,0] - fingertip_position[:,4], dim=-1)
    # k_loss1 =  -fingertip_position[:,1:4, 2].mean(dim=-1)

    # k_loss2 = 0
    # for i in range(1, fingertip_position.shape[1]-2):
    #     k_loss2 += torch.norm(fingertip_position[:,i] - fingertip_position[:,i+1], dim=-1).mean()

    ####### for allegro
    o_loss1 = torch.norm(fingertip_position[:,0] - fingertip_position[:,3], dim=-1)
    k_loss1 =  -fingertip_position[:,1:3, 2].mean(dim=-1)
    
    loss = o_loss1 + 1. * k_loss1
    # loss = o_loss1 + 1. * k_loss1 + 0.1 * loss2

    return loss

def gesture_ok_myo(fingertip_position):
    ''' num_envs (1) x finger_num x 3
        only for myohand
    '''
    k_loss1 =  fingertip_position[:,2:, 1].mean(dim=-1)
    o_loss1 = torch.norm(fingertip_position[:,0] - fingertip_position[:,1], dim=-1)
    loss = o_loss1 + 1. * k_loss1
    return loss

def gesture_scissors(fingertip_position):
    ''' num_envs (1) x finger_num x 3
        for shadowhand and allegro
    '''
    # ##### for shadowhand
    # loss1 = (torch.norm(fingertip_position[:,2] - fingertip_position[:,3], dim=-1) + \
    #         torch.norm(fingertip_position[:,3] - fingertip_position[:,4], dim=-1) + \
    #         torch.norm(fingertip_position[:,4] - fingertip_position[:,2], dim=-1)) / 3
    # loss2 = - torch.norm(fingertip_position[:,0] - fingertip_position[:,1], dim=-1)
    # loss3 = - fingertip_position[:,:2, 2].mean(-1)

    ###### for allegro
    loss1 = torch.norm(fingertip_position[:,0] - fingertip_position[:,1], dim=-1)
    loss2 = - torch.norm(fingertip_position[:,2] - fingertip_position[:,3], dim=-1)
    loss3 = - fingertip_position[:,2:, 2].mean(-1)

    loss = loss1 + 0.1 * loss2 + 1. * loss3
    # loss = loss1 + 1. * loss3

    return loss

def gesture_scissors_myo(fingertip_position):
    ''' num_envs (1) x finger_num x 3
        only for myohand
    '''
    loss1 = (torch.norm(fingertip_position[:,0] - fingertip_position[:,3], dim=-1) + \
            torch.norm(fingertip_position[:,3] - fingertip_position[:,4], dim=-1) + \
            torch.norm(fingertip_position[:,4] - fingertip_position[:,0], dim=-1)) / 3
    loss2 = - torch.norm(fingertip_position[:,1] - fingertip_position[:,2], dim=-1)
    loss3 = fingertip_position[:,1:3, 1].mean(-1)
    loss = loss1 + 0.1 * loss2 + 1. * loss3
    # loss = loss1 + 1. * loss3
    return loss

def gesture_rockandroll(fingertip_position):
    ''' num_envs (1) x finger_num x 3
        for shadowhand and allegro
    '''
    # ######## for shadowhand
    # loss1 = (torch.norm(fingertip_position[:,1] - fingertip_position[:,2], dim=-1).mean() + \
    #         torch.norm(fingertip_position[:,2] - fingertip_position[:,4], dim=-1).mean() + \
    #         torch.norm(fingertip_position[:,4] - fingertip_position[:,1], dim=-1).mean()) / 3
    # loss2 = - (fingertip_position[:,0, 2] + fingertip_position[:,3,2])/2

    # loss3 =  - torch.norm(fingertip_position[:,0] - fingertip_position[:,3], dim=-1)
    
    ###### for allegro
    loss1 = torch.norm(fingertip_position[:,0] - fingertip_position[:,2], dim=-1)
    loss2 = - (fingertip_position[:,1, 2] + fingertip_position[:,3,2])/2
    loss3 =  - torch.norm(fingertip_position[:,1] - fingertip_position[:,3], dim=-1)

    loss = loss1 + 1. * loss2 + 0.1 * loss3

    return loss

def gesture_rockandroll_myo(fingertip_position):
    ''' num_envs (1) x finger_num x 3
        only for myohand
    '''
    loss1 = (torch.norm(fingertip_position[:,0] - fingertip_position[:,2], dim=-1).mean() + \
            torch.norm(fingertip_position[:,2] - fingertip_position[:,3], dim=-1).mean() + \
            torch.norm(fingertip_position[:,3] - fingertip_position[:,0], dim=-1).mean()) / 3
    loss2 = (fingertip_position[:,1, 1] + fingertip_position[:,4, 1])/2

    loss3 =  - torch.norm(fingertip_position[:,1] - fingertip_position[:,4], dim=-1)
    loss3 = 0
    loss = 1 * loss1 + 1. * loss2 + 0.1 * loss3
    return loss

def gesture_good(fingertip_position):
    ''' num_envs (1) x finger_num x 3
        for shadowhand
        # for allegro
    '''
    loss1 = -fingertip_position[:,4,0]
    loss2 = fingertip_position[:,0:4, 2].mean(-1)
    loss3  =0
    # loss3 =  fingertip_position[:,3,0]

    # loss1 = -fingertip_position[:,0,1]
    # loss2 = fingertip_position[:,1:, 2].mean(-1)
    loss3 = 0

    loss = loss1 + 1. * loss2 + 1. * loss3
    return loss

def gesture_good_myo(fingertip_position):
    ''' num_envs (1) x finger_num x 3
        only for myohand
    '''
    loss1 = - fingertip_position[:,0,2]
    loss2 = - fingertip_position[:,1:, 1].mean(-1)
    loss3 = fingertip_position[:,1:, 0].mean(-1)

    # loss3 =  - torch.norm(fingertip_position[:,1] - fingertip_position[:,4], dim=-1)

    loss = loss1 + 1. * loss2 + 0.3 * loss3
    return loss

def gesture_six(fingertip_position):
    ''' num_envs (1) x finger_num x 3
        for shadowhand 
        # for allegro
    '''
    loss1 = -fingertip_position[:,4,0]
    loss2 = -fingertip_position[:,3,2]
    loss3 = fingertip_position[:,:3, 2].mean(-1)

    # loss1 = -fingertip_position[:,0,1]
    # loss2 = fingertip_position[:,2:, 2].mean(-1)
    # loss3 = fingertip_position[:,1, 0]

    loss = loss1 + 1. * loss2 + 1. * loss3
    return loss

def gesture_six_myo(fingertip_position):
    ''' num_envs (1) x finger_num x 3
        only for myohand
    '''
    loss1 = -fingertip_position[:,0,2]
    loss2 = - fingertip_position[:,1:4, 1].mean(-1)
    loss3 = fingertip_position[:,4, 1]
    loss4 = fingertip_position[:,1:4, 0].mean(-1)

    loss = loss1 + 1. * loss2 + 1. * loss3 + 0.5 * loss4
    return loss

def gesture_gun(fingertip_position):
    ''' num_envs (1) x finger_num x 3
        for shadowhand
        # for allegro
    '''
    loss1 = -fingertip_position[:,4,0]
    loss2 = -fingertip_position[:,:2,2].mean(-1)
    loss3 = fingertip_position[:,2:4, 2].mean(-1)

    # loss1 = -fingertip_position[:,0,1]
    # loss2 = -fingertip_position[:,2:, 2].mean(-1)
    # loss3 = fingertip_position[:,1, 2]

    loss = loss1 + 1. * loss2 + 1. * loss3
    return loss

def gesture_gun_myo(fingertip_position):
    ''' num_envs (1) x finger_num x 3
        only for myohand
    '''
    loss1 = -fingertip_position[:,0,2]
    loss2 = fingertip_position[:,1:3, 1].mean(-1)
    loss3 = - fingertip_position[:,3:, 1].mean(-1)
    loss4 = fingertip_position[:,3:,0].mean(-1)

    loss = loss1 + 1. * loss2 + 1. * loss3 + 0.5 * loss4
    return loss

def gesture_target_position(fingertip_position, target_position):
    ''' num_envs/num_samples (1) x finger_num x 3
    '''
    return  torch.norm(fingertip_position - target_position.unsqueeze(0), dim=-1).mean(-1) # (num_envs/samples, )

# several new generated examples by GPT-4
def preprocess(fingertip_positions):
    fingertip_position_thumb_z = fingertip_positions[:,4,2].clone()
    fingertip_positions[:,4,2] = fingertip_positions[:,4,0]
    fingertip_positions[:,4,0] = fingertip_position_thumb_z
    return fingertip_positions

def gesture_fist(fingertip_position):
    ''' force a gesture "fist"
       fingertip_position: num_envs (1) x finger_num x 3   
    '''
    fingertip_position = preprocess(fingertip_position)
    loss = fingertip_position[:,:,2].mean(dim=-1) # force all fingers curl towards palm
    return loss

def gesture_four(fingertip_position):
    ''' force a gesture "four"
       fingertip_position: num_envs (1) x finger_num x 3   
    '''
    fingertip_position = preprocess(fingertip_position)
    loss1 = -fingertip_position[:,4,2]	# force thumb to be straight
    loss2 = -fingertip_position[:,:3,2].mean(-1) # force index finger, middle finger and ring finger to be straight
    loss3 = fingertip_position[:,3, 2].mean(-1) # force little finger to curl towards palm
    loss = loss1 + 1. * loss2 + 1. * loss3
    return loss

def gesture_peace(fingertip_position):
    ''' force a gesture "peace"
       fingertip_position: num_envs (1) x finger_num x 3   
    '''
    loss1 = -fingertip_position[:,:2,2].mean(-1) # force index finger and middle finger to be straight
    loss2 = fingertip_position[:,2:4, 2].mean(-1) # force ring finger and little finger to curl towards palm
    loss3 = - torch.norm(fingertip_position[:,0] - fingertip_position[:,1], dim=-1) # force index finger and middle finger stay away
    loss = loss1 + 1. * loss2 + 0.1 * loss3 # 0.1 for loss3 for weak encouraing of forcing index finger and middle finger stay away
    return loss

gestures_isg = {
    'ok': gesture_ok,
    'scissors': gesture_scissors,
    'rockandroll': gesture_rockandroll,
    'good': gesture_good,
    'six': gesture_six,
    'gun': gesture_gun,
    'targetpos': gesture_target_position,
    'fist': gesture_fist,
    'four': gesture_four,
    'peace': gesture_peace,
}

gestures_myo = {
    'ok': gesture_ok_myo,
    'scissors': gesture_scissors_myo,
    'good': gesture_good_myo,
    'six': gesture_six_myo,
    'gun': gesture_gun_myo,
    'rockandroll':gesture_rockandroll_myo,
}

def sgd_optimization(model, loss_func, init_action):
    action = unnormalize(init_action.detach().clone())
    action.requires_grad = True

    lr = 1e-2
    iters = 2000 # 2000 for shadowhand, 10000 for myohand 
    optimizer = torch.optim.Adam(params=[action], lr=lr)

    pb = progress_bar(range(iters))
    for i in pb:
        pred_positions = model.mlp(normalize(action))
        loss = loss_func(pred_positions.reshape(1, -1, 3)).mean()
        optimizer.zero_grad()
        loss.backward()
        # print(action.grad)
        optimizer.step()
        pb.comment = f'loss: {loss}'
        # print(action.detach())
        
    return normalize(action).detach()

def cem_optimization(model, loss_func, init_action):
    input_lower_bounds = -1
    input_upper_bounds = 1
    epsilon = 1e-4
    iters = 300
    sample_num = 800
    num_elites = 20
    alpha = 0.2
    mean = init_action.detach().clone().cpu()
    var = torch.ones_like(mean) * 1. # depend on RMSE
    # X = stats.truncnorm(-1, 1,
    #                         loc=torch.zeros_like(mean),
    #                         scale=torch.ones_like(mean))
    X = stats.norm(loc=torch.zeros_like(mean), scale=torch.ones_like(mean))
    opt_count = 0

    while (opt_count < iters) and torch.max(var) > epsilon:
        # constrained
        lb_dist = mean - input_lower_bounds
        ub_dist = input_upper_bounds - mean
        constrained_var = torch.minimum(torch.minimum(torch.square(lb_dist), torch.square(ub_dist)), var)
        
        # sample
        action_samples = torch.from_numpy(X.rvs(size=(sample_num,)+ init_action.shape)) * torch.sqrt(constrained_var) + mean    # samp_num x action_dim
        action_samples = torch.clamp(action_samples, min=-1, max=1)

        # calculate cost
        with torch.no_grad():
            action_samples = action_samples.float()
            pred_positions = model.net(action_samples.to(args.device))
            # print(pred_positions.reshape(sample_num, -1, 3)[0])
            costs = loss_func(pred_positions.reshape(sample_num, -1, 3)).cpu()
        
        elites = action_samples[torch.argsort(costs)][:num_elites]

        new_mean = torch.mean(elites, axis=0)
        new_var = torch.var(elites, axis=0)

        mean = alpha * mean + (1. - alpha) * new_mean
        var = alpha * var + (1. - alpha) * new_var

        opt_count += 1
        # print(f'iter:{opt_count} max_var: {torch.max(var)}')
    
    return mean.detach()

def batch_sgd_optimization(model, loss_func, init_action):
    input_lower_bounds = -1
    input_upper_bounds = 1
    mean = init_action.detach().clone().cpu()
    var = torch.ones_like(mean)
    batch_num = 32

    X = stats.truncnorm(-1, 1,
                            loc=torch.zeros_like(mean),
                            scale=torch.ones_like(mean))
    
    lb_dist = mean - input_lower_bounds
    ub_dist = input_upper_bounds - mean
    constrained_var = torch.minimum(torch.minimum(torch.square(lb_dist), torch.square(ub_dist)), var)
    
    # sample
    action_samples = torch.from_numpy(X.rvs(size=(batch_num,)+ init_action.shape)) * torch.sqrt(constrained_var) + mean
    
    actions = action_samples.detach().clone().float().to('cuda:0')
    actions = unnormalize(actions)
    actions.requires_grad = True
    lr = 1e-2
    iters = 1000 # 2000 for shadowhand, 10000 for myohand
    optimizer = torch.optim.Adam(params=[actions], lr=lr)
    pb = progress_bar(range(iters))
    for i in pb:
        pred_positions = model.net(normalize(actions))
        loss = loss_func(pred_positions.reshape(batch_num, -1, 3)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pb.comment = f'loss: {loss}'
        # print(action.detach())

    actions = actions.detach()
    with torch.no_grad():
        
        pred_positions = model.net(normalize(actions))
        loss = loss_func(pred_positions.reshape(batch_num, -1, 3))
        action = actions[torch.argsort(loss)][0]
    return normalize(action)

def random_shooting_optimization(model, loss_func, init_action):
    input_lower_bounds = -1
    input_upper_bounds = 1

    batch_num = 10000

    actions = torch.rand((batch_num,) + init_action.shape) * 2 - 1
    actions = actions.to(model.device)

    with torch.no_grad():
        pred_positions = model.net(actions)
        losses = loss_func(pred_positions.reshape(batch_num, -1, 3))  
    action = actions[torch.argsort(losses)][0]
    return action

def generate_gesture(args, envs, target_position=None):
    gestures = gestures_myo if 'Myo' in args.task else gestures_isg
    gesture_type = args.gesture
    print(gesture_type)
    assert gesture_type in list(gestures.keys()), "Gesture not implemented !!!"
    if gesture_type == 'targetpos':
        if target_position is not None:
            full_loss_func = gestures.get(args.gesture)
            loss_func = partial(full_loss_func, target_position=target_position)
        else: 
            raise RuntimeError('No target position!')
    else:
        loss_func = gestures.get(args.gesture)

    cfg = OmegaConf.load(args.cfg_path)
    if args.model == 'mlp':
        model = Trainer(cfg)
    else:
        AssertionError(f'{args.model} not implemented yet !')
    model.load_model(args.model_path)
    model.net.eval()
    for name, param in model.net.named_parameters():
        param.requires_grad = False

    if args.use_inverse_model and target_position is not None:
        print('Using inverse model to accelerate planning!!!')
        inv_cfg = OmegaConf.load(args.inv_cfg_path)
        if args.inv_model == 'mlp':
            inv_model = Trainer(inv_cfg)
        else:
            AssertionError(f'{args.inv_model} not implemented yet !')
        inv_model.load_model(args.inv_model_path)
        inv_model.net.eval()
        
        with torch.no_grad():
            init_action = inv_model.net(target_position.reshape(-1,))
    else:
        init_action = torch.zeros(envs.action_space.shape, device=args.device)
    time0 = time.time()
    if args.opt_alg == 'sgd':
        action = sgd_optimization(model,loss_func, init_action)
    elif args.opt_alg == 'cem':
        action = cem_optimization(model,loss_func, init_action)
    elif args.opt_alg == 'batch_sgd':
        print('Using batch sgd optimization!!!')
        action = batch_sgd_optimization(model,loss_func, init_action)
    elif args.opt_alg == 'rs':
        action = random_shooting_optimization(model, loss_func, init_action)
    time1 = time.time()
    if 'Myo' in args.task:
        return action.detach().cpu().numpy()
    else:
        return action.expand((envs.num_envs,) + envs.action_space.shape).to(args.device), time1-time0

def pose_gesture(args):
    if 'Myo' in args.task:
        if args.task == 'MyoHand':
            envs = MyoHand(seed=128)
            actions = generate_gesture(args, envs)
            envs.reset()
            for _ in range(200):
                envs.step(actions)
        else:
            raise NotImplementedError(f'{args.task} not implemented !!!')
    else:
        envs = modexenvs.make(
            seed=0, 
            task=args.task, 
            num_envs=args.num_envs, 
            sim_device=args.device,
            rl_device=args.device,
            graphics_device_id=0,
        )
        actions,_ = generate_gesture(args, envs)
    
        for _ in range(3):
            for _ in range(100):
                envs.step(actions)
            envs.reset_buf = torch.ones(envs.num_envs, device=envs.device, dtype=torch.long)

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

def pose_target_gesture(args, headless=True):
    np.random.seed(0)
    envs = modexenvs.make(
        seed=0, 
        task=args.task, 
        num_envs=args.num_envs, 
        sim_device=args.device,
        rl_device=args.device,
        graphics_device_id=0,
        # virtual_screen_capture=True,
	    # force_render=False,
    )
#     envs.is_vector_env = True
#     envs = gym.wrappers.RecordVideo(
# 	envs,
# 	"./videos",
#     step_trigger=lambda step: step % 200 == 0, # record the videos every 10000 steps
# 	video_length=199 # for each video record up to 100 steps
# )
    # target_positions = torch.from_numpy(np.load('data/hand_data/AllegroHand/AllegroHand_random_val_Jul_23_14:27:56_2024/hand_states.npy').reshape(-1, 4, 3)).to(args.device)
    # target_actions = torch.from_numpy(np.load('data/hand_data/AllegroHand/AllegroHand_random_val_Jul_23_14:27:56_2024/actions.npy').reshape(-1, 16)).to(args.device)
    
    # target_positions = torch.from_numpy(np.load('data/hand_data/Robotiq3Finger/Robotiq_random_val_Jul_23_21:28:16_2024/hand_states.npy').reshape(-1, 3, 3)).to(args.device)
    # target_actions = torch.from_numpy(np.load('data/hand_data/Robotiq3Finger/Robotiq_random_val_Jul_23_21:28:16_2024/actions.npy').reshape(-1, 11)).to(args.device)
    
    target_positions = torch.from_numpy(np.load('data/hand_data/ShadowHand/ShadowHand_random_Jun_27_09:38:10_2024/hand_states.npy').reshape(-1, 5, 3)).to(args.device)
    target_actions = torch.from_numpy(np.load('data/hand_data/ShadowHand/ShadowHand_random_Jun_27_09:38:10_2024/actions.npy').reshape(-1, 20)).to(args.device)
    
    reach_error = 0
    successes = 0
    plan_times = 0
    path = './results/reach/allegro_qs_hight_res'
    os.makedirs(path, exist_ok=True)
    for i in range(50):
        # idx = np.random.randint(0, target_positions.shape[0])
        target_action = target_actions[i]
        target_position = target_positions[i]
        actions, plan_time = generate_gesture(args, envs, target_position=target_position)

        action_loss = F.mse_loss(actions[0], target_action).item()
        video_path = f'{path}/video_{i}.mp4'
        video = []
        for _ in range(200):
            envs.step(actions)
            visualize_targets(envs, target_position.reshape(1, -1, 3))
            rgb = envs.gym.get_camera_image(envs.sim, envs.envs[0], envs.camera_handles[0][0], gymapi.IMAGE_COLOR).reshape(2160, 3840, 4)[...,:3][..., [2,1,0]]
            video.append(rgb)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (3840,2160))
        for j in range(len(video)):
            data = video[j]
            out.write(data)
        out.release()
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
        reach_error += pos_loss
        success = pos_loss < 0.006
        successes += int(success)
        plan_times += plan_time
        print(f'----------{i}----------')
        print(f'success: {success}, reach_error: {pos_loss}, planning time: {plan_time}')
        print('---------------------')
        envs.reset_buf = torch.ones(envs.num_envs, device=envs.device, dtype=torch.long)
    
    avg_reach_error = reach_error / 100.
    avg_success = successes / 100.
    avg_plan_time = plan_times / 100.
    print(f'----------Done!!----------')
    print(f'averaged success: {avg_success}, reach_error: {avg_reach_error}, planning time: {avg_plan_time}')

    # for _ in range(3):
    #     for _ in range(500):
    #         envs.step(actions)
    #     envs.reset_buf = torch.ones(envs.num_envs, device=envs.device, dtype=torch.long)

if __name__ == "__main__":
    # # Myohand
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--task', type=str, default='MyoHand',help='tasks')
    # parser.add_argument('--num_envs', type=int, default=1, help='the number of created environments')
    # parser.add_argument('--model', type=str, default='mlp', help='type of the model')
    # parser.add_argument('--cfg_path', type=str, default='models/cfg/hand_model_train/myohand_forward_model.yaml', help='the path of configuration of data collector')
    # parser.add_argument('--model_path', type=str, default='logs/models/myohand_forward_model_Jun_20_14:23:01_2024/ckpt270.pt',help='the path of trained model')
    # parser.add_argument('--use_inverse_model', action="store_true", help='whether use inverse model to generate initial action')
    # parser.add_argument('--inv_model',type=str, default='mlp', help='type of inverse model')
    # parser.add_argument('--inv_cfg_path', type=str, default='models/cfg/inverse_model_train/robotiq_inverse_model.yaml',help='configuration file path of inverse model')
    # parser.add_argument('--inv_model_path', type=str, default='logs/models/robotiq_inverse_model_Jul_23_21:35:19_2024/ckpt-1.pt', help='the path of inverse model')
    # parser.add_argument('--device', type=str, default='cuda:0', help='the device where we create envs and collect data')
    # parser.add_argument('--opt_alg',type=str, default='batch_sgd', help='algorithm used for optimization')
    # parser.add_argument('--gesture', type=str, default='ok', help='the gesture to generate')

    # AllegroHand
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='AllegroHand',help='tasks')
    parser.add_argument('--num_envs', type=int, default=1, help='the number of created environments')
    parser.add_argument('--model', type=str, default='mlp', help='type of the model')
    parser.add_argument('--cfg_path', type=str, default='models/cfg/hand_model_train/allegro_forward_model.yaml', help='the path of configuration of data collector')
    parser.add_argument('--model_path', type=str, default='logs/models/allegro_forward_model_Jul_23_14:33:28_2024/ckpt-1.pt',help='the path of trained model')
    parser.add_argument('--use_inverse_model', action="store_true", help='whether use inverse model to generate initial action')
    parser.add_argument('--inv_model',type=str, default='mlp', help='type of inverse model')
    parser.add_argument('--inv_cfg_path', type=str, default='models/cfg/inverse_model_train/robotiq_inverse_model.yaml',help='configuration file path of inverse model')
    parser.add_argument('--inv_model_path', type=str, default='logs/models/robotiq_inverse_model_Jul_23_21:35:19_2024/ckpt-1.pt', help='the path of inverse model')
    parser.add_argument('--device', type=str, default='cuda:0', help='the device where we create envs and collect data')
    parser.add_argument('--opt_alg',type=str, default='cem', help='algorithm used for optimization')
    parser.add_argument('--gesture', type=str, default='ok', help='the gesture to generate')

    # # Robotiq3Finger
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--task', type=str, default='Robotiq3Finger',help='tasks')
    # parser.add_argument('--num_envs', type=int, default=1, help='the number of created environments')
    # parser.add_argument('--model', type=str, default='mlp', help='type of the model')
    # parser.add_argument('--cfg_path', type=str, default='models/cfg/hand_model_train/robotiq_forward_model.yaml', help='the path of configuration of data collector')
    # parser.add_argument('--model_path', type=str, default='logs/models/robotiq_forward_model_Jul_23_21:33:27_2024/ckpt-1.pt',help='the path of trained model')
    # parser.add_argument('--use_inverse_model', action="store_true", help='whether use inverse model to generate initial action')
    # parser.add_argument('--inv_model',type=str, default='mlp', help='type of inverse model')
    # parser.add_argument('--inv_cfg_path', type=str, default='models/cfg/inverse_model_train/robotiq_inverse_model.yaml',help='configuration file path of inverse model')
    # parser.add_argument('--inv_model_path', type=str, default='logs/models/robotiq_inverse_model_Jul_23_21:35:19_2024/ckpt-1.pt', help='the path of inverse model')
    # parser.add_argument('--device', type=str, default='cuda:0', help='the device where we create envs and collect data')
    # parser.add_argument('--opt_alg',type=str, default='cem', help='algorithm used for optimization')
    # parser.add_argument('--gesture', type=str, default='ok', help='the gesture to generate')

    # # ShadowHand
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--task', type=str, default='ShadowHand',help='tasks')
    # parser.add_argument('--num_envs', type=int, default=1, help='the number of created environments')
    # parser.add_argument('--model', type=str, default='mlp', help='type of the model')
    # parser.add_argument('--cfg_path', type=str, default='models/cfg/hand_model_train/shadowhand_forward_model.yaml', help='the path of configuration of data collector')
    # parser.add_argument('--model_path', type=str, default='logs/models/shadowhand_forward_model_Jul_23_14:35:39_2024/ckpt-1.pt',help='the path of trained model')
    # parser.add_argument('--use_inverse_model', action="store_true", help='whether use inverse model to generate initial action')
    # parser.add_argument('--inv_model',type=str, default='mlp', help='type of inverse model')
    # parser.add_argument('--inv_cfg_path', type=str, default='models/cfg/inverse_model_train/robotiq_inverse_model.yaml',help='configuration file path of inverse model')
    # parser.add_argument('--inv_model_path', type=str, default='logs/models/robotiq_inverse_model_Jul_23_21:35:19_2024/ckpt-1.pt', help='the path of inverse model')
    # parser.add_argument('--device', type=str, default='cuda:0', help='the device where we create envs and collect data')
    # parser.add_argument('--opt_alg',type=str, default='cem', help='algorithm used for optimization')
    # parser.add_argument('--gesture', type=str, default='ok', help='the gesture to generate')
    
    args = parser.parse_args()

    pose_gesture(args)
    # pose_target_gesture(args)