import torch
import numpy as np
from scipy import stats
from tqdm import tqdm
from datacollector.normalization import MyoHandNorm, MyoHandSeqNorm, InhandNorm
from datacollector.utils import compute_poses_wrt_root, compute_poses_wrt_world
import os
from isaacgym import gymutil
from isaacgym import gymapi
import cv2
import time
from myosuite.utils.quat_math import euler2quat, quat2euler
from myosuite.utils.vector_math import calculate_cosine

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

class CEMMJFixed:
    def __init__(self, 
                 env,
                 forward_model, 
                 inverse_model = None,
                 input_lower_bounds = -1, 
                 input_upper_bounds = 1, 
                 epsilon = 1e-3, 
                 max_iters = 100, 
                 sample_num = 100, 
                 num_elites = 20, 
                 alpha = 0.1,
                 device = 'cuda:1',
                 verbose = False,
    ):
        self.env = env
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.input_lower_bounds = input_lower_bounds
        self.input_upper_bounds = input_upper_bounds
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.sample_num = sample_num
        self.num_elites = num_elites
        self.alpha = alpha
        self.device = device
        self.verbose = verbose
        self.norm_m = MyoHandNorm.mean
        self.norm_s = MyoHandNorm.std

    def plan(self, mean, var, cost_func):
        X = stats.norm(loc=np.zeros_like(mean), scale=np.ones_like(mean))

        opt_count = 0
        bar = tqdm(range(self.max_iters)) if self.verbose else range(self.max_iters)
        for opt_count in bar:
            if np.max(var) < self.epsilon:
                if self.verbose:
                    print('----------- Converged! -----------')
                break
            # constrained
            lb_dist = mean - self.input_lower_bounds
            ub_dist = self.input_upper_bounds - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist), np.square(ub_dist)), var)

            # sample
            action_samples = X.rvs(size=(self.sample_num,)+ mean.shape) * np.sqrt(constrained_var) + mean    # samp_num x action_dim
            action_samples = np.clip(action_samples, a_min=self.input_lower_bounds, a_max=self.input_upper_bounds)

            # calculate cost
            with torch.no_grad():
                action_samples = torch.from_numpy(action_samples).float()
                pred_positions = self.forward_model.net(action_samples.to(self.device)).cpu()
                pred_positions = (pred_positions * self.norm_s + self.norm_m)
                costs = cost_func(pred_positions.numpy())

            elites = action_samples[np.argsort(costs)][:self.num_elites].numpy()

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1. - self.alpha) * new_mean
            var = self.alpha * var + (1. - self.alpha) * new_var

            opt_count += 1
            # print(f'iter:{opt_count} max_var: {np.max(var)}')

        return mean
    
    def step(self):
        self.idx = self.env.np_random.integers(0, self.env.reach_space.shape[0])
        target_pos = self.env.reach_space[self.idx]
        if self.inverse_model is not None:
            # TODO: improve the use of inverse model
            target_pos_ = (torch.from_numpy(target_pos.reshape(15,)).float().to(self.device) - self.norm_m.reshape(15,).to(self.device)) / self.norm_s.reshape(15,).to(self.device)
            mean = self.inverse_model.net(target_pos_).detach().cpu().numpy()
            vars = np.ones_like(mean) * 0.45
        else:
            mean = np.zeros_like(self.env.action_space.sample())
            vars = np.ones_like(mean)
        cost_func = lambda x: np.linalg.norm(x - target_pos.reshape(1, -1), axis=-1)
        time0 = time.time()
        action = self.plan(mean, vars, cost_func)
        time1 = time.time()
        return action, time1-time0

    def eval(self, env_val, actions, render=False, save_path=None):
        env_val.reset(idx=self.idx)
        if render:
                env_val.render_offscreen()
        for _ in range(100):
            _, r, _, _, info = env_val.step(actions)
            if render:
                env_val.render_offscreen()
        if render:
            env_val.write_video(out_file=save_path, duration=2, size=(640,480))
        return info['solved'], r, info
    
class CEMMJSeq:
    def __init__(self, 
                 env,
                 forward_model, 
                 inverse_model = None,
                 input_lower_bounds = -1, 
                 input_upper_bounds = 1, 
                 epsilon = 1e-2, 
                 max_iters = 100, 
                 sample_num = 100, 
                 num_elites = 20, 
                 max_timesteps = 50,
                 alpha = 0.2,
                 gamma = 0.95,
                 device = 'cuda:1',
                 verbose = False,
    ):
        self.env = env
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.input_lower_bounds = input_lower_bounds
        self.input_upper_bounds = input_upper_bounds
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.sample_num = sample_num
        self.num_elites = num_elites
        self.max_timesteps = max_timesteps
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.verbose = verbose
        self.norm_m = MyoHandSeqNorm.mean
        self.norm_s = MyoHandSeqNorm.std

    def plan(self, init_state, mean, var, cost_func):
        """
            Plan the action sequence
            init_state (state_dim): initial state of the environment
            mean (max_timesteps x action_dim): initial mean of the action
            var (max_timesteps x action_dim): initial variance of the action
            cost_func (function): loss function to minimize
        """
        X = stats.norm(loc=np.zeros_like(mean), scale=np.ones_like(mean))
        init_state = init_state.reshape(1,-1).repeat(self.sample_num, axis=0) # expand to sample_num x state_dim

        opt_count = 0
        bar = tqdm(range(self.max_iters)) if self.verbose else range(self.max_iters)
        for opt_count in bar:
            if np.max(var) < self.epsilon:
                if self.verbose:
                    print('----------- Converged! -----------')
                break
            # constrained
            lb_dist = mean - self.input_lower_bounds
            ub_dist = self.input_upper_bounds - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist), np.square(ub_dist)), var)

            # sample
            action_samples = X.rvs(size=(self.sample_num,)+ mean.shape) * np.sqrt(constrained_var) + mean    # samp_num x max_timesteps x action_dim
            action_samples = np.clip(action_samples, a_min=self.input_lower_bounds, a_max=self.input_upper_bounds)

            # calculate cost
            states = []
            cur_state = (torch.from_numpy(init_state).float().to(self.device) - self.norm_m.to(self.device)) / self.norm_s.to(self.device)
            with torch.no_grad():
                for i in range(self.max_timesteps):
                    action = torch.from_numpy(action_samples[:, i]).float().to(self.device) # sample_num x action_dim
                    cur_state = self.forward_model.net(torch.concat([cur_state, action], dim=-1))
                    cur_state_ = (cur_state.cpu() * self.norm_s + self.norm_m)
                    states.append(cur_state_)
                states = torch.stack(states, dim=1)
                costs = cost_func(states.numpy())

            elites = action_samples[np.argsort(costs)][:self.num_elites]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1. - self.alpha) * new_mean
            var = self.alpha * var + (1. - self.alpha) * new_var

            opt_count += 1
            # print(f'iter:{opt_count} max_var: {np.max(var)}')
        print(opt_count)
        return mean, var
    
    def get_env_obs(self):
        return self.env.get_obs_dict(self.env.sim)['tip_pos']
    
    def step(self):
        self.idx = self.env.np_random.integers(0, self.env.reach_space.shape[0])
        self.env.reset(idx=self.idx)
        target_pos = self.env.reach_space[self.idx]
        state = self.get_env_obs()

        # cost function
        def cost_func(states, gamma=self.gamma, target_pos=target_pos):
            """
                states (sample_num x max_timesteps x state_dim): state sequence
                gamma (float): discount factor
            """
            target_pos = target_pos.reshape(1, 1, -1)
            dist = np.linalg.norm(states - target_pos, axis=-1)
            weights = gamma ** np.arange(self.max_timesteps)
            costs = np.sum(dist * weights, axis=-1)
            return costs
        
        # planning
        actions = []
        done = False
        t = 0
        max_act_steps = 10

        if self.inverse_model is not None:
            target_pos_ = (torch.from_numpy(target_pos.reshape(15,)).float().to(self.device) - self.norm_m.reshape(15,).to(self.device)) / self.norm_s.reshape(15,).to(self.device)
            state_ = (torch.from_numpy(state).float().to(self.device) - self.norm_m.to(self.device)) / self.norm_s.to(self.device) 
            state_ = state_.reshape(15,)
            with torch.no_grad():
                mean = self.inverse_model.net(torch.concat([state_, target_pos_], dim=-1)).detach().cpu().numpy().reshape(1,-1).repeat(self.max_timesteps, axis=0)
            var = np.ones_like(mean) * 0.45
        else:
            mean = np.zeros_like(self.env.action_space.sample()).reshape(1,-1).repeat(self.max_timesteps, axis=0)
            var = np.ones_like(mean)

        while not done and t < max_act_steps:
            mean, var = self.plan(state, mean, var, cost_func)
            action = mean[0]
            _, _, _, _, info = self.env.step(action)
            state = self.get_env_obs()
            done = info['solved']
            if self.inverse_model is not None:
                mean = np.concatenate([mean[1:], np.zeros_like(mean[-1:])], axis=0)
                var = np.concatenate([var[1:] + 0.1, np.ones_like(var[-1:])], axis=0)
            else:
                mean = np.concatenate([mean[1:], np.zeros_like(mean[-1:])], axis=0)
                var = np.concatenate([var[1:] + 0.1, np.ones_like(var[-1:])], axis=0)
            actions.append(action)
            t += 1
        actions = np.stack(actions, axis=0)
        return actions

    def eval(self, env_val, actions, render=False, save_path=None):
        env_val.reset(idx=self.idx)
        if render:
                env_val.render_offscreen()
        for i in range(actions.shape[0]):
            for _ in range(10):
                _, r, _, _, info = env_val.step(actions[i])
                if render:
                    env_val.render_offscreen()
        if render:
            env_val.write_video(out_file=save_path, duration=2, size=(640,480))
        return info['solved'], r, info

class CEMIsaacSeq:
    def __init__(self, 
                 env,
                 env_name,
                 forward_model, 
                 inverse_model = None,
                 init_var = 1.,
                 input_lower_bounds = -1, 
                 input_upper_bounds = 1, 
                 epsilon = 1e-2, 
                 max_iters = 100, 
                 sample_num = 100, 
                 num_elites = 20, 
                 max_timesteps = 50,
                 alpha = 0.2,
                 gamma = 0.95,
                 device = 'cuda:0',
                 verbose = False,
                 reach_space_paths = {},
    ):
        self.env = env
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.input_lower_bounds = input_lower_bounds
        self.input_upper_bounds = input_upper_bounds
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.sample_num = sample_num
        self.num_elites = num_elites
        self.max_timesteps = max_timesteps
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.verbose = verbose
        self.init_var = init_var
        
        if env_name == 'ShadowHand':
            self.finger_num = 5
        elif env_name == 'AllegroHand':
            self.finger_num = 4
        elif env_name == 'Robotiq3Finger':
            self.finger_num = 3
        else:
            raise ValueError("Invalid env_name")
        
        self.reach_space = np.load(os.path.join(reach_space_paths[env_name], 'hand_states.npy'))
        self.reach_space = self.reach_space.reshape(-1, self.finger_num*3)


    def plan(self, init_state: torch.Tensor, mean, var, cost_func):
        """
            Plan the action sequence
            init_state (state_dim): initial state of the environment
            mean (max_timesteps x action_dim): initial mean of the action
            var (max_timesteps x action_dim): initial variance of the action
            cost_func (function): loss function to minimize
        """
        X = stats.norm(loc=torch.zeros_like(mean), scale=torch.ones_like(mean))
        init_state = init_state.reshape(1,-1).repeat(self.sample_num, 1) # expand to sample_num x state_dim

        opt_count = 0
        bar = tqdm(range(self.max_iters)) if self.verbose else range(self.max_iters)
        for opt_count in bar:
            if torch.max(var) < self.epsilon:
                if self.verbose:
                    print('----------- Converged! -----------')
                break
            # constrained
            lb_dist = mean - self.input_lower_bounds
            ub_dist = self.input_upper_bounds - mean
            constrained_var = torch.minimum(torch.minimum(torch.square(lb_dist), torch.square(ub_dist)), var)

            # sample
            action_samples = torch.from_numpy(X.rvs(size=(self.sample_num,)+ mean.shape)) * torch.sqrt(constrained_var) + mean    # samp_num x max_timesteps x action_dim
            action_samples = torch.clamp(action_samples, min=self.input_lower_bounds, max=self.input_upper_bounds)

            # calculate cost
            states = []
            cur_state = init_state
            with torch.no_grad():
                for i in range(self.max_timesteps):
                    action = action_samples[:, i].float().to(self.device) # sample_num x action_dim
                    cur_state = self.forward_model.net(torch.concat([cur_state, action], dim=-1))
                    states.append(cur_state)
                states = torch.stack(states, dim=1)
                costs = cost_func(states)

            elites = action_samples[torch.argsort(costs).cpu()][:self.num_elites]

            new_mean = torch.mean(elites, dim=0)
            new_var = torch.var(elites, dim=0)

            mean = self.alpha * mean + (1. - self.alpha) * new_mean
            var = self.alpha * var + (1. - self.alpha) * new_var

            opt_count += 1
            # print(f'iter:{opt_count} max_var: {np.max(var)}')

        return mean, var
    
    def get_env_obs(self):
        finger_tips_idxs = self.env.fingertip_handles
        fingertip_pose = self.env.rigid_body_states[:, finger_tips_idxs][:, :, :7]
        root_base_pose = self.env.root_state_tensor[:,:7]
        root_base_pose = root_base_pose.unsqueeze(1).repeat(1, fingertip_pose.shape[1], 1).view(-1,7)
        fingertip_pose = fingertip_pose.view(-1,7)

        # converse fingertip pose to hand root coordinate
        fingertip_pose_base = compute_poses_wrt_root(fingertip_pose, root_base_pose).view(self.env.num_envs, -1, 7)
        
        tip_offset = torch.Tensor(self.env.tip_offset+[0,0,0,1]).to(fingertip_pose_base.device)
        tip_offset = tip_offset.expand_as(fingertip_pose)
        fingertip_pos_base = compute_poses_wrt_world(tip_offset.reshape(-1,7), fingertip_pose_base.reshape(-1, 7)).reshape(self.env.num_envs, -1, 7)[...,:3]
        return fingertip_pos_base[0].reshape(self.finger_num*3, )

    def step(self, **args):
        self.idx = np.random.randint(0, self.reach_space.shape[0])
        self.env.reset_buf = torch.ones(self.env.num_envs, device=self.env.device, dtype=torch.long)
        target_pos = torch.from_numpy(self.reach_space[self.idx]).float().to(self.device)
        state = self.get_env_obs()

        # cost function
        def cost_func(states, gamma=self.gamma, target_pos=target_pos):
            """
                states (sample_num x max_timesteps x state_dim): state sequence
                gamma (float): discount factor
            """
            target_pos = target_pos.reshape(1, 1, -1)
            dist = torch.norm(states - target_pos, dim=-1)
            weights = gamma ** torch.arange(self.max_timesteps, device=self.device)
            costs = torch.sum(dist * weights, dim=-1)
            return costs
        
        # planning
        actions = []
        done = False
        t = 0
        max_act_steps = 10
        target_pos_ = target_pos.reshape(self.finger_num*3,)
        if self.inverse_model is not None:
            state_ = state.reshape(self.finger_num*3,)
            with torch.no_grad():
                mean = self.inverse_model.net(torch.concat([state_, target_pos_], dim=-1)).reshape(1,-1).repeat(self.max_timesteps, 1).cpu()
            var = torch.ones_like(mean) * self.init_var
        else:
            mean = torch.zeros(self.env.action_space.shape).reshape(1,-1).repeat(self.max_timesteps, 1)
            var = torch.ones_like(mean)

        # video = []

        while not done and t < max_act_steps:
            mean, var = self.plan(state, mean, var, cost_func)
            action = mean[0]
            action = action.reshape(1, -1).repeat(self.env.num_envs, 1)
            for _ in range(10):
                self.env.step(action.float())
                visualize_targets(self.env, target_pos.reshape(1, self.finger_num, 3))
                # if args.get('video_path', None) is not None:
                #     video.append(self.env.gym.get_camera_image(self.env.sim, self.env.envs[0], self.env.camera_handles[0][0], gymapi.IMAGE_COLOR).reshape(480, 640, 4)[...,:3][..., [2,1,0]])
            state = self.get_env_obs()
            done = True if torch.norm(state.reshape(self.finger_num*3, ) - target_pos_)/5 < 0.008 else False
            if self.inverse_model is not None:
                mean = torch.concat([mean[1:], torch.zeros_like(mean[-1:])], dim=0)
                var = torch.concat([var[1:] + 0.1, torch.ones_like(var[-1:])], dim=0)
            else:
                mean = torch.concat([mean[1:], torch.zeros_like(mean[-1:])], dim=0)
                var = torch.concat([var[1:] + 0.1, torch.ones_like(var[-1:])], dim=0)
            actions.append(action)
            t += 1
        if t < max_act_steps+5:
            for _ in range((max_act_steps+5-t)*10):
                self.env.step(action.float())
                visualize_targets(self.env, target_pos.reshape(1, self.finger_num, 3))
                # if args.get('video_path', None) is not None:
                #     video.append(self.env.gym.get_camera_image(self.env.sim, self.env.envs[0], self.env.camera_handles[0][0], gymapi.IMAGE_COLOR).reshape(480, 640, 4)[...,:3][..., [2,1,0]])
        actions = torch.stack(actions, dim=0)
        
        # out = cv2.VideoWriter(args.get('video_path'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,480))
        # for j in range(len(video)):
        #     data = video[j]
        #     out.write(data)
        # out.release()
        return actions, done, torch.norm(state.reshape(self.finger_num*3, ) - target_pos_)/5

    # def eval(self, actions):
    #     self.env.reset(idx=self.idx)
    #     for i in range(actions.shape[0]):
    #         _, r, _, _, info = self.env.step(actions[i])
    #     return info['solved'], r, info

class CEMInhand:
    def __init__(self, 
                 env,
                 forward_model, 
                 input_lower_bounds = -1, 
                 input_upper_bounds = 1, 
                 epsilon = 1e-2, 
                 max_iters = 3, 
                 sample_num = 800, 
                 num_elites = 20, 
                 max_timesteps = 50,
                 alpha = 0.2,
                 gamma = 0.95,
                 device = 'cuda:0',
                 verbose = False,
    ):
        self.env = env
        self.forward_model = forward_model
        self.input_lower_bounds = input_lower_bounds
        self.input_upper_bounds = input_upper_bounds
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.sample_num = sample_num
        self.num_elites = num_elites
        self.max_timesteps = max_timesteps
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.verbose = verbose
        self.hand_mean = InhandNorm.hand_mean
        self.hand_std = InhandNorm.hand_std
        self.obj_mean = InhandNorm.obj_mean
        self.obj_std = InhandNorm.obj_std

    def plan(self, init_state, mean, var, cost_func):
        """
            Plan the action sequence
            init_state (state_dim): initial state of the environment
            mean (max_timesteps x action_dim): initial mean of the action
            var (max_timesteps x action_dim): initial variance of the action
            cost_func (function): loss function to minimize
        """
        X = stats.norm(loc=np.zeros_like(mean), scale=np.ones_like(mean))
        init_state = init_state.reshape(1,-1).repeat(self.sample_num, axis=0) # expand to sample_num x state_dim

        opt_count = 0
        bar = tqdm(range(self.max_iters)) if self.verbose else range(self.max_iters)
        for opt_count in bar:
            if np.max(var) < self.epsilon:
                if self.verbose:
                    print('----------- Converged! -----------')
                break
            # constrained
            lb_dist = mean - self.input_lower_bounds
            ub_dist = self.input_upper_bounds - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist), np.square(ub_dist)), var)

            # sample
            action_samples = X.rvs(size=(self.sample_num,)+ mean.shape) * np.sqrt(constrained_var) + mean    # samp_num x max_timesteps x action_dim
            action_samples = np.clip(action_samples, a_min=self.input_lower_bounds, a_max=self.input_upper_bounds)

            # calculate cost
            states = []
            init_hand_state = (torch.from_numpy(init_state).float().to(self.device)[...,:15] - self.hand_mean.to(self.device)) / self.hand_std.to(self.device)
            init_obj_state = (torch.from_numpy(init_state).float().to(self.device)[...,15:] - self.obj_mean.to(self.device)) / self.obj_std.to(self.device)
            cur_state = torch.cat([init_hand_state, init_obj_state], dim=-1)
            with torch.no_grad():
                for i in range(self.max_timesteps):
                    action = torch.from_numpy(action_samples[:, i]).float().to(self.device) # sample_num x action_dim
                    cur_state = self.forward_model.net(torch.concat([cur_state, action], dim=-1))
                    cur_state_obj = (cur_state.cpu()[..., 15:] * self.obj_std + self.obj_mean)
                    states.append(cur_state_obj)
                states = torch.stack(states, dim=1)
                costs = cost_func(states.numpy())

            elites = action_samples[np.argsort(costs)][:self.num_elites]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1. - self.alpha) * new_mean
            var = self.alpha * var + (1. - self.alpha) * new_var

            opt_count += 1
            # print(f'iter:{opt_count} max_var: {np.max(var)}')

        return mean, var
    
    def get_env_obs(self):
        if 'obj_rot' in self.env.obs_keys:
            return np.concatenate([self.env.get_obs_dict(self.env.sim)['tip_pos'], self.env.get_obs_dict(self.env.sim)['obj_pos'], self.env.get_obs_dict(self.env.sim)['obj_rot']])
        else:
            return np.concatenate([self.env.get_obs_dict(self.env.sim)['tip_pos'], self.env.get_obs_dict(self.env.sim)['obj_pos']])
    
    def step(self):
        self.env.reset()
        self.target = self.env.sim.model.site_pos[self.env.goal_sid]
        state = self.get_env_obs()

        # cost function
        def cost_func(states, gamma=self.gamma, target=self.target):
            """
                states (sample_num x max_timesteps x state_dim): state sequence
                gamma (float): discount factor
            """
            target = target.reshape(1, 1, -1)
            dist = np.linalg.norm(states - target, axis=-1)
            weights = gamma ** np.arange(self.max_timesteps)
            costs = np.sum(dist * weights, axis=-1)
            return costs
        
        # planning
        actions = []
        done = False
        t = 0
        max_act_steps = 50 # 65, 50

        mean = np.zeros_like(self.env.action_space.sample()).reshape(1,-1).repeat(self.max_timesteps, axis=0)
        var = np.ones_like(mean)
        error = 1000
        while not done and t < max_act_steps:
            mean, var = self.plan(state, mean, var, cost_func)
            action = mean[0]
            _, r, _, _, info = self.env.step(action)
            state = self.get_env_obs()
            err = np.linalg.norm(state[-3:] - self.target)
            if err < error:
                error = err
            done = err < 0.01
            mean = np.concatenate([mean[1:], np.zeros_like(mean[-1:])], axis=0)
            var = np.concatenate([var[1:] + 0.1, np.ones_like(var[-1:])], axis=0)
            actions.append(action)
            t += 1
        actions = np.stack(actions, axis=0)
        return actions, error, done

    def eval(self, actions, render=False, output_path=None):
        self.env.reset(pos=self.env.target_pos, size=self.env.size)
        for i in range(actions.shape[0]):
            if render:
                self.env.render_offscreen()
            _, r, _, _, info = self.env.step(actions[i])
        if render:
            self.env.write_video(out_file=output_path, duration=2, size=(1080,720))
        return info['solved'], r, info

class MPCRollout:
    def __init__(self, 
                 env,
                 dyn_model, 
                 input_lower_bounds = -1, 
                 input_upper_bounds = 1, 
                 max_iters = 3, 
                 sample_num = 800, 
                 num_elites = 20, 
                 alpha = 0.2,
                 gamma = 0.98,
                 plan_step = 10,
                 device = 'cuda:0',
                 verbose = False,
    ):
        self.env = env
        self.dyn_model = dyn_model
        self.input_lower_bounds = input_lower_bounds
        self.input_upper_bounds = input_upper_bounds
        self.max_iters = max_iters
        self.sample_num = sample_num
        self.num_elites = num_elites
        self.alpha = alpha
        self.gamma = gamma
        self.plan_step = plan_step
        self.device = device
        self.verbose = verbose

    def plan(self, init_state, mean, var, cost_func):
        """
            Plan the action sequence
            init_state (state_dim): initial state of the environment
            mean (plan_step x action_dim): initial mean of the action
            var (plan_step x action_dim): initial variance of the action
            cost_func (function): loss function to minimize
        """
        X = stats.norm(loc=np.zeros_like(mean), scale=np.ones_like(mean))
        init_state = init_state.reshape(1,-1).repeat(self.sample_num, axis=0) # expand to sample_num x state_dim

        opt_count = 0
        bar = tqdm(range(self.max_iters)) if self.verbose else range(self.max_iters)
        for opt_count in bar:
            # constrained
            lb_dist = mean - self.input_lower_bounds
            ub_dist = self.input_upper_bounds - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist), np.square(ub_dist)), var)

            # sample
            action_samples = X.rvs(size=(self.sample_num,)+ mean.shape) * np.sqrt(constrained_var) + mean    # samp_num x plan_step x action_dim
            action_samples = np.clip(action_samples, a_min=self.input_lower_bounds, a_max=self.input_upper_bounds)

            # calculate cost
            states = []
            cur_state = torch.from_numpy(init_state).float().to(self.device)
            actions = torch.from_numpy(action_samples).float().to(self.device)  # sample_num x plan_step x action_dim
            with torch.no_grad():
                states = self.dyn_model.predict(cur_state, actions) # sample_num x plan_step x state_dim
                obj_states = states[..., 15:].cpu().numpy()
                obj_states = (obj_states * self.dyn_model.data_processor.obj_data_std + self.dyn_model.data_processor.obj_data_mean)
                actions_np = actions.cpu().numpy()
                costs = cost_func(obj_states, actions_np)

            elites = action_samples[np.argsort(costs)][:self.num_elites]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1. - self.alpha) * new_mean
            var = self.alpha * var + (1. - self.alpha) * new_var

            opt_count += 1
            # print(f'iter:{opt_count} max_var: {np.max(var)}')

        return mean, var
    
    def predict(self, obs):
        # # object hold
        # self.target = self.env.sim.model.site_pos[self.env.goal_sid]

        # reorientation
        target_pos = self.env.get_obs_dict(self.env.sim)['obj_des_pos']
        target_rot = self.env.get_obs_dict(self.env.sim)['obj_des_rot']
        self.target = np.concatenate([target_pos, target_rot])
        
        ###### cost functions ###### 
        # # object hold
        # def cost_func(states, actions, gamma=self.gamma, target=self.target):
        #     """
        #         states (sample_num x plan_step x state_dim): state sequence
        #         gamma (float): discount factor
        #     """
        #     target = target.reshape(1, 1, -1)
        #     dist = np.linalg.norm(states - target, axis=-1)
        #     weights = gamma ** np.arange(self.plan_step).reshape(1, -1)
        #     goal_dist = np.sum(dist * weights, axis=-1) / np.sum(weights)
        #     bonus = - (1.*(goal_dist<2*0.01) + 1.*(goal_dist<0.01))
        #     penalty = goal_dist > 0.3
        #     action_mag = np.linalg.norm(actions, axis=-1)/self.env.sim.model.na if self.env.sim.model.na !=0 else 0
        #     action_mag = np.sum(action_mag * weights, axis=-1) / np.sum(weights)
        #     costs = 100 * goal_dist + 4 * bonus + 10 * penalty
        #     return costs

        # reorientation
        # def cost_func(states, actions, gamma=self.gamma, target=self.target):
        #     """
        #         states (sample_num x plan_step x state_dim): state sequence
        #         gamma (float): discount factor
        #     """
        #     target = target.reshape(1, 1, -1)
        #     pos_err = np.linalg.norm(states[...,:3] - target[...,:3], axis=-1)
        #     rot_err = np.sum(states[...,3:] * target[...,3:], axis=-1) / np.linalg.norm(states[...,3:],axis=-1) / np.linalg.norm(target[...,3:],axis=-1)
        #     weights = gamma ** np.arange(self.plan_step).reshape(1, -1)
        #     pos_err = np.sum(pos_err * weights, axis=-1) / np.sum(weights)
        #     rot_err = np.sum(rot_err * weights, axis=-1) / np.sum(weights)
        #     action_mag = np.linalg.norm(actions, axis=-1)/self.env.sim.model.na if self.env.sim.model.na !=0 else 0
        #     action_mag = np.sum(action_mag * weights, axis=-1) / np.sum(weights)

        #     cost_pos_align = 1.* pos_err
        #     cost_rot_align = - rot_err
        #     bonus = - (1.*(rot_err > 0.9)*(pos_err<0.075) + 5.0*(rot_err > 0.95)*(pos_err<0.075))
        #     penalty = (pos_err > 0.075)
        #     costs = cost_pos_align + cost_rot_align + 10. * bonus + 5. * penalty + 5. * action_mag
        #     return costs
        def cost_func(states, actions, gamma=self.gamma, target=self.target):
            """
                states (sample_num x plan_step x state_dim): state sequence
                gamma (float): discount factor
            """
            target = target.reshape(1, 1, -1)
            pos_err = np.linalg.norm(states[...,:3] - target[...,:3], axis=-1)
            rot_err = np.sum(states[...,3:] * target[...,3:], axis=-1) / np.linalg.norm(states[...,3:],axis=-1) / np.linalg.norm(target[...,3:],axis=-1)
            weights = gamma ** np.arange(self.plan_step).reshape(1, -1)
            action_mag = np.linalg.norm(actions, axis=-1)/self.env.sim.model.na if self.env.sim.model.na !=0 else 0

            cost_pos_align = 1.* pos_err
            cost_pos_align = 0
            cost_rot_align = - rot_err
            bonus = - (1.*(rot_err > 0.9)*(pos_err<0.075) + 5.0*(rot_err > 0.95)*(pos_err<0.075))
            bonus = 0
            action_mag = 0
            penalty = (pos_err > 0.075)
            costs = cost_pos_align + cost_rot_align + 10. * bonus + 5. * penalty + 5. * action_mag
            costs = np.sum(costs * weights, axis=-1)
            return costs
        
        # planning
        obs = self.dyn_model.data_processor.get_normalized_data(obs)
        cur_hand_state, cur_obj_state = obs['hand_cur_data'], obs['obj_cur_data']
        init_state = np.concatenate([cur_hand_state, cur_obj_state])
        mean = np.zeros_like(self.env.action_space.sample()).reshape(1,-1).repeat(self.plan_step, axis=0)
        var = np.ones_like(mean)
        mean, var = self.plan(init_state, mean, var, cost_func)
        action = mean[0]
        return action