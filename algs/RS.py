import torch
import numpy as np
from tqdm import tqdm
from datacollector.normalization import MyoHandNorm, MyoHandSeqNorm
from datacollector.utils import compute_poses_wrt_root, compute_poses_wrt_world
import os

class RandomShootingMJFixed:
    def __init__(self, env, model=None, num_shoots=100, action_space=None):
        self.env = env
        self.model = model
        self.num_shoots = num_shoots
        self.action_space = action_space
        self.norm_m = MyoHandNorm.mean
        self.norm_s = MyoHandNorm.std
    
    def gen_samples(self):
        if self.action_space is None:
            self.action_space = self.env.action_space
        return np.stack([self.action_space.sample() for _ in range(self.num_shoots)], axis=0)
    
    def env_step(self, samples):
        if self.model is not None:
            target_pos = self.env.reach_space[self.idx]
            with torch.no_grad():
                pred_pos = self.model.net(torch.from_numpy(samples).float().to(self.model.device))
            pred_pos = (pred_pos * self.norm_s.to(self.model.device) + self.norm_m.to(self.model.device)).detach().cpu().numpy()
            r = -np.linalg.norm(target_pos.reshape(1, 15) - pred_pos, axis=-1)
            return r
        else:
            rewards = []
            for sample in tqdm(samples):
                self.env.reset(idx=self.idx)
                _, r, _, _, _ = self.env.step(sample)
            rewards.append(r)
        return rewards
    
    def step(self):
        self.idx = self.env.np_random.integers(0, self.env.reach_space.shape[0])
        samples = self.gen_samples()
        rewards = self.env_step(samples)
        best_idx = np.argmax(rewards)
        best_reward = rewards[best_idx]
        return samples[best_idx], best_reward
    
    def eval(self, sample):
        
        self.env.reset(idx=self.idx)
        _, r, _, _, info = self.env.step(sample)
        return info['solved'], r, info

class RandomShootingMJSeq:
    def __init__(self, env, model=None, num_shoots=100, num_steps=10, action_space=None) -> None:
        self.env = env
        self.model = model
        self.num_shoots = num_shoots
        self.num_steps = num_steps
        self.action_space = action_space
        self.device = model.device if model is not None else torch.device('cpu')
        self.norm_m = MyoHandSeqNorm.mean
        self.norm_s = MyoHandSeqNorm.std
    
    def gen_samples(self):
        if self.action_space is None:
            self.action_space = self.env.action_space
        action_seqs = []
        for _ in range(self.num_shoots):
            action_seq = np.stack([self.action_space.sample() for _ in range(self.num_steps)], axis=0)
            action_seqs.append(action_seq)
        action_seqs = np.stack(action_seqs, axis=0)
        return action_seqs

    def get_env_obs(self):
        return self.env.get_obs_dict(self.env.sim)['tip_pos']

    def env_step(self, samples):
        if self.model is not None:
            samples = torch.from_numpy(samples).float().to(self.device)
            target_pos = torch.from_numpy(self.env.reach_space[self.idx]).float().to(self.device)
            cur_state = self.get_env_obs().reshape(1,-1).repeat(self.num_shoots, axis=0) # expand to sample_num x state_dim
            cur_state= (torch.from_numpy(cur_state).float().to(self.device) - self.norm_m.to(self.device)) / self.norm_s.to(self.device)
            with torch.no_grad():
                for t in range(self.num_steps):
                    cur_state = self.model.net(torch.cat([cur_state, samples[:, t]], dim=-1))
            cur_state_ = cur_state * self.norm_s.to(self.device) + self.norm_m.to(self.device)
            r = -torch.norm(target_pos.reshape(1,15) - cur_state_, dim=-1).cpu().numpy()
            return r
        else:
            rewards = []
            for sample in tqdm(samples):
                self.env.reset(self.idx)
                r = 0
                for action in sample:
                    _, r, _, _, _ = self.env.step(action)
                rewards.append(r)
            return rewards
    
    def step(self):
        self.idx = self.env.np_random.integers(0, self.env.reach_space.shape[0])
        samples = self.gen_samples()
        rewards = self.env_step(samples)
        best_idx = np.argmax(rewards)
        best_reward = rewards[best_idx]
        return samples[best_idx], best_reward
    
    def eval(self, actions):
        self.env.reset(idx=self.idx)
        for i in range(actions.shape[0]):
            _, r, _, _, info = self.env.step(actions[i])
        return info['solved'], r, info


class RandomShootingIsaacSeq:
    def __init__(self, env, env_name, model=None, num_shoots=100, num_steps=10, action_space=None, reach_space_paths={}) -> None:
        self.env = env
        self.model = model
        self.num_shoots = num_shoots
        self.num_steps = num_steps
        self.action_space = action_space
        self.device = model.device if model is not None else torch.device('cpu')
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

    def gen_samples(self):
        if self.action_space is None:
            self.action_space = self.env.action_space
        action_seqs = []
        for _ in range(self.num_shoots):
            action_seq = torch.stack([2.*torch.rand(self.action_space.shape).reshape(-1,)-1 for _ in range(self.num_steps)], dim=0)
            action_seqs.append(action_seq)
        action_seqs = torch.stack(action_seqs, dim=0)
        return action_seqs

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

    def env_step(self, samples):
        samples = samples.float().to(self.device)
        target_pos = torch.from_numpy(self.reach_space[self.idx]).float().to(self.device)
        
        if self.model is not None:
            cur_state = self.get_env_obs().reshape(1,-1).repeat(self.num_shoots, 1) # expand to sample_num x state_dim
            cur_state= cur_state.float().to(self.device) 
            with torch.no_grad():
                for t in range(self.num_steps):
                    cur_state = self.model.net(torch.cat([cur_state, samples[:, t]], dim=-1))
            r = -torch.norm(target_pos.reshape(1,self.finger_num*3) - cur_state, dim=-1).cpu()
            return r
        else:
            rewards = []
            for sample in tqdm(samples):
                self.env.reset_buf = torch.ones(self.env.num_envs, device=self.env.device, dtype=torch.long)
                for action in sample:
                    for _ in range(10):
                        self.env.step(action)
                cur_state = self.get_env_obs()
                r = -torch.norm(target_pos - cur_state, dim=-1).cpu()
                rewards.append(r)
            return rewards
    
    def step(self):
        self.idx = np.random.randint(0, self.reach_space.shape[0])
        samples = self.gen_samples()
        rewards = self.env_step(samples)
        best_idx = np.argmax(rewards)
        best_reward = rewards[best_idx]
        return samples[best_idx], best_reward
    
    def eval(self, actions):
        self.env.reset_buf = torch.ones(self.env.num_envs, device=self.env.device, dtype=torch.long)
        target_pos = torch.from_numpy(self.reach_space[self.idx]).float().to(self.device)
        for i in range(actions.shape[0]):
            for _ in range(10):
                self.env.step(actions[i].reshape(1,-1).repeat(self.env.num_envs, 1))
        cur_state = self.get_env_obs()
        dist = torch.norm(target_pos - cur_state, dim=-1).cpu() / 5
        solved = dist < 0.008
        return solved, dist