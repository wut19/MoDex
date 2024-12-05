import torch
import numpy as np
from scipy import stats
from fastprogress.fastprogress import progress_bar
from datacollector.normalization import MyoHandNorm, MyoHandSeqNorm

from datacollector.utils import compute_poses_wrt_root, compute_poses_wrt_world
import os

class BGDMJFixed:
    def __init__(self,
                 env, 
                 forward_model,
                 inverse_model = None,
                 input_lower_bounds = -1,
                 input_upper_bounds = 1,
                 batchsize = 32,
                 max_iters = 100,
                 learning_rate = 1e-2,
                 device = 'cuda:0',
    ):
        self.env = env
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.input_lower_bounds = input_lower_bounds
        self.input_upper_bounds = input_upper_bounds
        self.batchsize = batchsize
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.device = device
        self.norm_m = MyoHandNorm.mean
        self.norm_s = MyoHandNorm.std

    def optimize(self, mean, var, loss_func):
        X = stats.truncnorm(-1, 1,
                            loc=np.zeros_like(mean),
                            scale=np.ones_like(mean))
        
        lb_dist = mean - self.input_lower_bounds
        ub_dist = self.input_upper_bounds - mean
        constrained_var = np.minimum(np.minimum(np.square(lb_dist), np.square(ub_dist)), var)
        
        # sample
        actions = torch.from_numpy(X.rvs(size=(self.batchsize,)+ mean.shape) * np.sqrt(constrained_var) + mean)
        actions = torch.clamp(actions, min=self.input_lower_bounds, max=self.input_upper_bounds)
        actions = unnormalize(actions).float().to(self.device)
        actions.requires_grad = True
        optimizer = torch.optim.Adam(params=[actions], lr=self.learning_rate)
        
        # optimize
        pb = progress_bar(range(self.max_iters))
        for _ in pb:
            pred_positions = self.forward_model.net(normalize(actions))
            pred_positions = pred_positions * self.norm_s.to(self.device) + self.norm_m.to(self.device)
            loss = loss_func(pred_positions).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pb.comment = f'loss: {loss}'
        
        # choose the best action
        actions = actions.detach()
        with torch.no_grad():
            
            pred_positions = self.forward_model.net(normalize(actions))
            pred_positions = pred_positions * self.norm_s.to(self.device) + self.norm_m.to(self.device)
            loss = loss_func(pred_positions)
            action = actions[torch.argsort(loss)][0]
        return normalize(action).cpu().numpy()
    
    def step(self):
        self.idx = self.env.np_random.integers(0, self.env.reach_space.shape[0])
        target_pos = self.env.reach_space[self.idx]
        if self.inverse_model is not None:
            # TODO: improve the use of inverse model
            mean = self.inverse_model.net(torch.from_numpy(target_pos).float().to(self.device)).cpu().numpy()
            var = np.ones_like(mean)
        else:
            mean = np.zeros_like(self.env.action_space.sample())
            var = np.ones_like(mean)
        loss_func = lambda x: torch.norm(x - torch.from_numpy(target_pos).reshape(1, -1).to(self.device), dim=-1)
        action = self.optimize(mean, var, loss_func)
        return action
    
    def eval(self, action):
        self.env.reset(idx=self.idx)
        _, r, _, _, info = self.env.step(action)
        return info['solved'], r, info
    
class BGDMJSeq:
    def __init__(self,
                 env, 
                 forward_model,
                 inverse_model = None,
                 input_lower_bounds = -1,
                 input_upper_bounds = 1,
                 batchsize = 32,
                 max_iters = 100,
                 max_timesteps = 10,
                 learning_rate = 1e-2,
                 gamma=0.95,
                 device = 'cuda:0',
    ):
        self.env = env
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.input_lower_bounds = input_lower_bounds
        self.input_upper_bounds = input_upper_bounds
        self.batchsize = batchsize
        self.max_iters = max_iters
        self.max_timesteps = max_timesteps
        self.learning_rate = learning_rate
        self.device = device
        self.norm_m = MyoHandSeqNorm.mean
        self.norm_s = MyoHandSeqNorm.std
        self.gamma = gamma

    def optimize(self, mean, var, init_state, loss_func):
        # mean: seq x action_dim
        X = stats.truncnorm(-1, 1,
                            loc=np.zeros_like(mean),
                            scale=np.ones_like(mean))
        
        lb_dist = mean - self.input_lower_bounds
        ub_dist = self.input_upper_bounds - mean
        constrained_var = np.minimum(np.minimum(np.square(lb_dist), np.square(ub_dist)), var)
        
        # sample
        actions = torch.from_numpy(X.rvs(size=(self.batchsize,)+ mean.shape) * np.sqrt(constrained_var) + mean)
        actions = torch.clamp(actions, min=self.input_lower_bounds, max=self.input_upper_bounds)
        actions = unnormalize(actions).float().to(self.device)
        actions.requires_grad = True
        optimizer = torch.optim.Adam(params=[actions], lr=self.learning_rate)
        
        init_state = init_state.reshape(1,-1).repeat(self.batchsize, axis=0) # expand to sample_num x state_dim
        
        # optimize
        pb = progress_bar(range(self.max_iters))
        init_state = (torch.from_numpy(init_state).float().to(self.device) - self.norm_m.to(self.device)) / self.norm_s.to(self.device)
        for _ in pb:
            states = []
            cur_state = init_state
            for t in range(self.max_timesteps):
                cur_state = self.forward_model.net(torch.cat([cur_state, normalize(actions[:, t])], dim=-1))
                cur_state_ = cur_state * self.norm_s.to(self.device) + self.norm_m.to(self.device)
                states.append(cur_state_)
            loss = loss_func(torch.stack(states, dim=1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pb.comment = f'loss: {loss}'

        # choose the best action
        actions = actions.detach()
        cur_state = init_state
        with torch.no_grad():
            states = []
            for t in range(self.max_timesteps):
                cur_state = self.forward_model.net(torch.cat([cur_state, normalize(actions[:, t])], dim=-1))
                cur_state_ = cur_state * self.norm_s.to(self.device) + self.norm_m.to(self.device)
                states.append(cur_state_)
            loss = loss_func(torch.stack(states, dim=1))
            action = actions[torch.argmin(loss)]
        return normalize(action).cpu().numpy()
    
    def get_env_obs(self):
        return self.env.get_obs_dict(self.env.sim)['tip_pos']
    
    def step(self):
        self.idx = self.env.np_random.integers(0, self.env.reach_space.shape[0])
        self.env.reset(idx=self.idx)
        target_pos = self.env.reach_space[self.idx]
        state = self.get_env_obs()

        if self.inverse_model is not None:
            # TODO: improve the use of inverse model
            mean = self.inverse_model.net(torch.from_numpy(target_pos).float().to(self.device)).cpu().numpy()
            var = np.ones_like(mean)
        else:
            mean = np.zeros_like(self.env.action_space.sample()).reshape(1,-1).repeat(self.max_timesteps, axis=0)
            var = np.ones_like(mean)
        
        def cost_func(states, gamma=self.gamma, target_pos=torch.from_numpy(target_pos).float().to(self.device)):
            """
                states (sample_num x max_timesteps x state_dim): state sequence
                gamma (float): discount factor
            """
            target_pos = target_pos.reshape(1, 1, -1)
            dist = torch.norm(states - target_pos, dim=-1)
            weights = gamma ** torch.arange(self.max_timesteps, device=self.device)
            costs = torch.sum(dist * weights, dim=-1)
            return costs
        
        actions = []
        done = False
        t = 0
        max_act_steps = 10
        while not done and t < max_act_steps:
            action = self.optimize(mean, var, state, cost_func)
            _, _, _, _, info = self.env.step(action[0])
            state = self.get_env_obs()
            done = info['solved']
            mean = np.concatenate([action[1:], np.zeros_like(action[-1:])], axis=0)
            var = np.ones_like(mean) * 0.2
            actions.append(action[0])
            t += 1
        actions = np.stack(actions, axis=0)
        return actions
    
    def eval(self, actions):
        self.env.reset(idx=self.idx)
        for i in range(actions.shape[0]):
            _, r, _, _, info = self.env.step(actions[i])
        return info['solved'], r, info
    

class BGDIsaacSeq:
    def __init__(self,
                 env, 
                 env_name,
                 forward_model,
                 inverse_model = None,
                 input_lower_bounds = -1,
                 input_upper_bounds = 1,
                 batchsize = 32,
                 max_iters = 100,
                 max_timesteps = 10,
                 learning_rate = 1e-2,
                 gamma=0.95,
                 device = 'cuda:0',
                 reach_space_paths = {},
    ):
        self.env = env
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.input_lower_bounds = input_lower_bounds
        self.input_upper_bounds = input_upper_bounds
        self.batchsize = batchsize
        self.max_iters = max_iters
        self.max_timesteps = max_timesteps
        self.learning_rate = learning_rate
        self.device = device
        self.gamma = gamma

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

    def optimize(self, mean, var, init_state, loss_func):
        # mean: seq x action_dim
        X = stats.truncnorm(-1, 1,
                            loc=torch.zeros_like(mean),
                            scale=torch.ones_like(mean))
        
        lb_dist = mean - self.input_lower_bounds
        ub_dist = self.input_upper_bounds - mean
        constrained_var = torch.minimum(torch.minimum(torch.square(lb_dist), torch.square(ub_dist)), var)
        
        # sample
        actions = torch.from_numpy(X.rvs(size=(self.batchsize,)+ mean.shape)) * torch.sqrt(constrained_var) + mean
        actions = torch.clamp(actions, min=self.input_lower_bounds, max=self.input_upper_bounds)
        actions = unnormalize(actions).float().to(self.device)
        actions.requires_grad = True
        optimizer = torch.optim.Adam(params=[actions], lr=self.learning_rate)
        
        init_state = init_state.reshape(1,-1).repeat(self.batchsize, 1) # expand to sample_num x state_dim
        
        # optimize
        pb = progress_bar(range(self.max_iters))
        init_state = init_state.float().to(self.device)
        for _ in pb:
            states = []
            cur_state = init_state
            for t in range(self.max_timesteps):
                cur_state = self.forward_model.net(torch.cat([cur_state, normalize(actions[:, t])], dim=-1))
                states.append(cur_state)
            loss = loss_func(torch.stack(states, dim=1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pb.comment = f'loss: {loss}'
        # choose the best action
        actions = actions.detach()
        cur_state = init_state
        with torch.no_grad():
            states = []
            for t in range(self.max_timesteps):
                cur_state = self.forward_model.net(torch.cat([cur_state, normalize(actions[:, t])], dim=-1))
                states.append(cur_state)
            loss = loss_func(torch.stack(states, dim=1))
            action = actions[torch.argmin(loss)]
        return normalize(action).cpu()

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
    
    def step(self):
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

        while not done and t < max_act_steps:
            action_ = self.optimize(mean, var, state, cost_func)
            action = action_[0]
            action = action.reshape(1, -1).repeat(self.env.num_envs, 1)
            for _ in range(10):
                self.env.step(action.float())
            state = self.get_env_obs()
            done = True if torch.norm(state.reshape(self.finger_num*3, ) - target_pos_)/5 < 0.008 else False
            mean = torch.concat([action_[1:], torch.zeros_like(action_[-1:])], dim=0)
            var = torch.ones_like(mean) * 0.3
            actions.append(action)
            t += 1
        actions = torch.stack(actions, dim=0)
        return actions, done, torch.norm(state.reshape(self.finger_num*3, ) - target_pos_)/5
        

def normalize(x):
    #  B x N
    return 2 * torch.sigmoid(x) - 1

def unnormalize(x):
    return torch.logit((x + 1) / 2)
