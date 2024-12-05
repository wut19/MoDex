import abc
from abc import ABC
import torch 
import os
import time
import numpy as np
from .utils import quaternion_to_matrix, matrix_to_quaternion, convert_pos_quat_to_mat, compute_poses_wrt_world
from scipy.spatial.transform import Rotation as R
import mujoco as mj

class DataCollector(ABC):
    def __init__(self, cfg, env) -> None:
        self.env = env
        self.cfg = cfg

    @abc.abstractmethod
    def collect(self):
        """Collect current data
        Returns:
            Collections
        """
    
    @abc.abstractmethod    
    def allocate_buffers(self):
        """Allocate torch buffers for data collection
        """        
        
    @abc.abstractmethod
    def save(self, save_path):
        """Save collected data
        """
    
    def step(self, action):
        """Step physical environment and collect data

        Args:
            actions: actions to apply
        """
        self.env.step(action)
        self.collect(action)
        
# ========================================
# IsaacHandDataCollectors
# ========================================

class IsaacHandDataCollector(DataCollector):
    ''' Data collector for isaac gym environments
    '''
    def __init__(self, env, cfg) -> None:
        super().__init__(cfg, env)
        self.task_name = cfg.task
        self.num_data = cfg.num_data
        self.eps_len = int(self.num_data / cfg.num_envs)
        self.device = cfg.device
        self.interval = cfg.interval
        self.save_path = cfg.save_path
        self.action_steps = cfg.action_steps
        self.t = 0
        self.steps = 0
        self.name = cfg.name

        # parse data type
        self.data_type = cfg.data_type
        if cfg.data_type == 'fingertip_pos':
            self.data_dim = [self.env.num_fingertips, 3]
        elif cfg.data_type == 'fingertip_pose':
            self.data_dim = [self.env.num_fingertips, 7]
        elif cfg.data_type == 'rigid_body_pos':
            self.data_dim = [self.env.num_bodies, 3]
        elif cfg.data_type == 'rigid_body_pose':
            self.data_dim = [self.env.num_bodies, 7]
        else:
            raise NotImplementedError("Data type not implemented!!!")
        
        self.allocate_buffers()
    
    def allocate_buffers(self):
        '''Initialize torch.Tensor for data collection, use self.deivce to store the data, may be differet from the device used for running the simulation
        '''
        self.hand_states = torch.zeros(
            (self.env.num_envs, self.eps_len, *self.data_dim), device=self.device, dtype=torch.float)
        
        self.actions = torch.zeros(
            (self.env.num_envs, self.eps_len, self.env.num_actions), device=self.device, dtype=torch.float)
        
    def collect(self, action):
        '''Collect each action-state pairs for model learning
        '''
        if self.data_type == 'fingertip_pos':
            state_idxs = self.env.fingertip_handles
            states = self.env.rigid_body_states[:, state_idxs][:, :, :3]
        elif self.data_type == 'fingertip_pose':
            state_idxs = self.env.fingertip_handles
            states = self.env.rigid_body_states[:, state_idxs][:, :, :7]
        elif self.data_type == 'rigid_body_pos':
            states = self.env.rigid_body_states[:, :, :3]
        elif self.data_type == 'rigid_body_pose':
            states = self.env.rigid_body_states[:, :, :7]
        else:
            raise NotImplementedError('Data type not implemented!!!')
        
        if 'pos' in self.data_type or 'pose' in self.data_type:
            states = self.process_states_coords(states)     

        self.hand_states[:, self.t] = states.to(self.device)
        self.actions[:, self.t] = action.to(self.device)
    
    def process_states_coords(self, states):
        state_dim = states.shape[-1]
        if state_dim == 3:
            states = torch.cat([states, torch.Tensor([[[0,0,0,1]]]).expand(states.shape[0], states.shape[1], 4).to(states.device)], dim=-1)

        root_base_pose = self.env.root_state_tensor[:,:7]
        root_base_pose = root_base_pose.unsqueeze(1).repeat(1, states.shape[1], 1).view(-1,7)
        states = states.view(-1,7)

        # converse pose to hand root coordinate
        states_base = self.compute_poses_wrt_root(states, root_base_pose).view(self.env.num_envs, -1, 7)
        if 'fingertip' in self.data_type:
            tip_offset = torch.Tensor(self.env.tip_offset+[0,0,0,1]).to(states_base.device)
            tip_offset = tip_offset.expand_as(states)
            states_base = compute_poses_wrt_world(tip_offset.reshape(-1,7), states_base.reshape(-1, 7)).reshape(self.env.num_envs, -1, 7)
        
        if state_dim == 3:
            states_base = states_base[:,:,:3]

        return states_base
        
    def step(self, action, args):
        """Step physical environment and collect data. Use arg "action_steps" and "interval" to prevent collect too similar data due to moving average control.

        Args:
            actions: actions to apply
            args: args on control signal decay and broken dim
        """
        # # deal with some biological or mechanical phenomena
        # control signal decay
        action_original = action.clone()
        if args.ctrl_decay > 0:
            assert args.ctrl_decay < 1, 'ctrl decay shound less than 1'
            action = action * (1 - args.ctrl_decay)

        if args.broken_dim >= 0:
            action[:, args.broken_dim] *= 0.

        for _ in range(self.action_steps):
            self.env.step(action)
        
        if self.steps % self.interval == 0:
            self.collect(action_original)
            self.t += 1
        self.steps += 1
        self.env.reset_buf = torch.ones(self.env.num_envs, device=self.env.device, dtype=torch.long)
        
    def save(self):
        '''save collected data to .npy file
        '''
        cur_time = time.strftime('_%b_%d_%H:%M:%S_%Y', time.localtime())
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.save_path, self.task_name, self.name + cur_time)
        os.makedirs(save_path,exist_ok=True)
        np.save(os.path.join(save_path, 'hand_states.npy'), self.hand_states.cpu().numpy())
        np.save(os.path.join(save_path, 'actions.npy'), self.actions.cpu().numpy())
        return save_path
        
    def compute_poses_wrt_root(self, object_pose, root_pose):
        assert object_pose.shape == root_pose.shape, f"Different shapes of object poses and root_poses !!! {object_pose.shape},{root_pose.shape}"
        object_pos = object_pose[:, 0:3]
        object_rot = object_pose[:, 3:7]

        root_pos = root_pose[:, 0:3]
        root_quat_xyzw = root_pose[:, 3:7]

        root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]   
        R_W_P = quaternion_to_matrix(root_quat_wxyz)

        T_W_P = torch.eye(4).repeat(R_W_P.shape[0], 1, 1).to(R_W_P.device)
        T_W_P[:, 0:3, 0:3] = R_W_P
        T_W_P[:, 0:3, 3] = root_pos

        object_quat_xyzw = object_rot
        object_quat_wxyz = object_quat_xyzw[:, [3, 0, 1, 2]]
        R_W_O = quaternion_to_matrix(object_quat_wxyz)

        T_W_O = torch.eye(4).repeat(R_W_O.shape[0], 1, 1).to(R_W_O.device)
        T_W_O[:, 0:3, 0:3] = R_W_O
        T_W_O[:, 0:3, 3] = object_pos

        relative_pose = torch.matmul(torch.inverse(T_W_P), T_W_O)

        relative_translation = relative_pose[:, 0:3, 3]
        relative_quat_wxyz = matrix_to_quaternion(relative_pose[:, 0:3, 0:3])

        relative_quat_xyzw = relative_quat_wxyz[:, [1, 2, 3, 0]]

        object_pos_wrt_root = relative_translation
        object_quat_wrt_root = relative_quat_xyzw

        object_pose_wrt_root = torch.cat((object_pos_wrt_root, object_quat_wrt_root), axis=-1)
        
        return object_pose_wrt_root
    
class IsaacHandDataCollectorV2(IsaacHandDataCollector):
    def __init__(self, cfg, env) -> None:
        self.full_state_dim = [env.num_bodies, 7]
        super().__init__(cfg=cfg, env=env)

    def allocate_buffers(self):
        super().allocate_buffers()
        self.full_states = torch.zeros(
            (self.env.num_envs, self.eps_len, *self.full_state_dim), device=self.device, dtype=torch.float)

    def collect(self, action):
        super().collect(action)
        full_state = self.env.rigid_body_states[:, :, :7]
        full_state = self.process_states_coords(full_state)
        self.full_states[:, self.t] = full_state

    def save(self):
        save_path = super().save()
        np.save(os.path.join(save_path, 'full_states.npy'), self.full_states.cpu().numpy())
        return save_path

class SeqIsaacHandDataCollector(DataCollector):
    def __init__(self, env, cfg) -> None:
        super().__init__(cfg, env)
        self.task_name = cfg.task
        self.num_data = cfg.num_data
        self.episode_len = cfg.episode_length
        self.eps_len_per_env = int(self.num_data / cfg.num_envs)
        self.device = cfg.device
        self.interval = cfg.interval
        self.save_path = cfg.save_path
        self.action_steps = cfg.action_steps
        self.t = 0
        self.steps = 0
        self.name = cfg.name

        # parse data type
        self.data_type = cfg.data_type
        if cfg.data_type == 'fingertip_pos':
            self.data_dim = [self.env.num_fingertips, 3]
        elif cfg.data_type == 'fingertip_pose':
            self.data_dim = [self.env.num_fingertips, 7]
        else:
            raise NotImplementedError("Data type not implemented!!!")

        self.allocate_buffers()

    def allocate_buffers(self):
        self.cur_hand_states = torch.zeros(
            (self.env.num_envs, self.eps_len_per_env, *self.data_dim), device=self.device, dtype=torch.float)
        
        self.next_hand_states = torch.zeros(
            (self.env.num_envs, self.eps_len_per_env, *self.data_dim), device=self.device, dtype=torch.float)
        
        self.actions = torch.zeros(
            (self.env.num_envs, self.eps_len_per_env, self.env.num_actions), device=self.device, dtype=torch.float)

    def get_states(self):
        if self.data_type == 'fingertip_pos':
            states = self.env.rigid_body_states[:, self.env.fingertip_handles][:, :, :3]
        elif self.data_type == 'fingertip_pose':
            states = self.env.rigid_body_states[:, self.env.fingertip_handles][:, :, :7]
        
        if 'pos' in self.data_type or 'pose' in self.data_type:
            states = self.process_states_coords(states)

        return states
    
    def collect(self, cur_states, action):
        states = self.get_states()
        
        self.cur_hand_states[:, self.t] = cur_states
        self.next_hand_states[:, self.t] = states
        self.actions[:, self.t] = action

    def process_states_coords(self, states):
        state_dim = states.shape[-1]
        if state_dim == 3:
            states = torch.cat([states, torch.Tensor([[[0,0,0,1]]]).expand(states.shape[0], states.shape[1], 4).to(states.device)], dim=-1)

        root_base_pose = self.env.root_state_tensor[:,:7]
        root_base_pose = root_base_pose.unsqueeze(1).repeat(1, states.shape[1], 1).view(-1,7)
        states = states.view(-1,7)

        # converse pose to hand root coordinate
        states_base = self.compute_poses_wrt_root(states, root_base_pose).view(self.env.num_envs, -1, 7)
        if 'fingertip' in self.data_type:
            tip_offset = torch.Tensor(self.env.tip_offset+[0,0,0,1]).to(states_base.device)
            tip_offset = tip_offset.expand_as(states)
            states_base = compute_poses_wrt_world(tip_offset.reshape(-1,7), states_base.reshape(-1, 7)).reshape(self.env.num_envs, -1, 7)
        
        if state_dim == 3:
            states_base = states_base[:,:,:3]

        return states_base
    
    def step(self, action, args):
        """Step physical environment and collect data. Use arg "action_steps" and "interval" to prevent collect too similar data due to moving average control.

        Args:
            actions: actions to apply
            args: args on control signal decay and broken dim
        """
        cur_states = self.get_states()
        
        # # deal with some biological or mechanical phenomena
        # control signal decay
        action_original = action.clone()
        if args.ctrl_decay > 0:
            assert args.ctrl_decay < 1, 'ctrl decay shound less than 1'
            action = action * (1 - args.ctrl_decay)

        if args.broken_dim >= 0:
            action[:, args.broken_dim] *= 0.

        for _ in range(self.action_steps):
            self.env.step(action)
        
        if self.steps % self.interval == 0:
            self.collect(cur_states, action_original)
            self.t += 1
        self.steps += 1

        if self.t % self.episode_len == 0:
            self.env.reset_buf = torch.ones(self.env.num_envs, device=self.env.device, dtype=torch.long)
    
    def save(self):
        cur_time = time.strftime('_%b_%d_%H:%M:%S_%Y', time.localtime())
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.save_path, self.task_name, self.name + cur_time)
        os.makedirs(save_path,exist_ok=True)
        np.save(os.path.join(save_path, 'cur_hand_states.npy'), self.cur_hand_states.cpu().numpy())
        np.save(os.path.join(save_path, 'next_hand_states.npy'), self.next_hand_states.cpu().numpy())
        np.save(os.path.join(save_path, 'actions.npy'), self.actions.cpu().numpy())
        return save_path

    def compute_poses_wrt_root(self, object_pose, root_pose):
        assert object_pose.shape == root_pose.shape, f"Different shapes of object poses and root_poses !!! {object_pose.shape},{root_pose.shape}"
        object_pos = object_pose[:, 0:3]
        object_rot = object_pose[:, 3:7]

        root_pos = root_pose[:, 0:3]
        root_quat_xyzw = root_pose[:, 3:7]

        root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]   
        R_W_P = quaternion_to_matrix(root_quat_wxyz)

        T_W_P = torch.eye(4).repeat(R_W_P.shape[0], 1, 1).to(R_W_P.device)
        T_W_P[:, 0:3, 0:3] = R_W_P
        T_W_P[:, 0:3, 3] = root_pos

        object_quat_xyzw = object_rot
        object_quat_wxyz = object_quat_xyzw[:, [3, 0, 1, 2]]
        R_W_O = quaternion_to_matrix(object_quat_wxyz)

        T_W_O = torch.eye(4).repeat(R_W_O.shape[0], 1, 1).to(R_W_O.device)
        T_W_O[:, 0:3, 0:3] = R_W_O
        T_W_O[:, 0:3, 3] = object_pos

        relative_pose = torch.matmul(torch.inverse(T_W_P), T_W_O)

        relative_translation = relative_pose[:, 0:3, 3]
        relative_quat_wxyz = matrix_to_quaternion(relative_pose[:, 0:3, 0:3])

        relative_quat_xyzw = relative_quat_wxyz[:, [1, 2, 3, 0]]

        object_pos_wrt_root = relative_translation
        object_quat_wrt_root = relative_quat_xyzw

        object_pose_wrt_root = torch.cat((object_pos_wrt_root, object_quat_wrt_root), axis=-1)
        
        return object_pose_wrt_root
# ========================================
# MyoHandDataCollectors
# ========================================
class MyoHandDataCollector(DataCollector):
    ''' Data collector for myosuite environments.
    '''
    def __init__(self, env, cfg) -> None:
        super().__init__(cfg, env)
        self.task_name = cfg.task
        self.num_data = cfg.num_data
        self.eps_len = int(self.num_data / 1)   # mysuite is not good for parallel environments
        self.device = cfg.device
        self.interval = cfg.interval
        self.save_path = cfg.save_path
        self.action_steps = cfg.action_steps
        self.t = 0
        self.steps = 0
        self.name = cfg.name

        # parse data type
        self.data_type = cfg.data_type
        if cfg.data_type == 'fingertip_pos':
            self.data_dim = [self.env.num_fingertips, 3]
        elif cfg.data_type == 'fingertip_pose':
            self.data_dim = [self.env.num_fingertips, 7]
        else:
            raise NotImplementedError("Data type not implemented!!!")

        self.allocate_buffers()
    
    def allocate_buffers(self):
        '''Initialize np.array for data collection, use cpu to store the data
        '''
        self.hand_states = np.zeros(
            (self.eps_len, *self.data_dim), dtype=np.float64)
        
        self.actions = np.zeros(
            (self.eps_len, *self.env.action_space.shape), dtype=np.float64)
        
    def collect(self, action):
        '''Collect each action-fingertip/rigid body poses(in root base coordinate) pairs for model learning
        '''
        if self.data_type == 'fingertip_pos':
            states = self.env.get_obs_dict(self.env.sim)['tip_pose'].reshape(-1, 12)[:,:3]
        elif self.data_type == 'fingertip_pose':
            states = self.env.get_obs_dict(self.env.sim)['tip_pose'].reshape(-1, 12)
            mats = states[:,3:].reshape(-1,3,3)
            quats = []
            for i in range(mats.shape[0]):
                quat = R.from_matrix(mats[i]).as_quat()
                quats.append(quat)
            quats = np.stack(quats, 0)
            states = np.concatenate([states[:,:3], quats], axis=-1)
        
        # TODO: introduce a root site to make the target pose wrt world
        # if 'pos' in self.data_type or 'pose' in self.data_type:
        #     states = self.process_states_coords(states)
        
        self.hand_states[self.t] = states
        self.actions[self.t] = action
    
    def process_states_coords(self, states):
        state_dim = states.shape[-1]
        if state_dim == 3:
            states = np.concatenate([states, np.tile(np.array([0.,0.,0.,1.]), (states.shape[0], 1))], axis=-1)
        
        root_base_id = self.env.sim.model.site_name2id(self.env.base_name)
        root_base_pose = np.concatenate([self.env.sim.model.site_pos[root_base_id], self.env.sim.model.site_quat[root_base_id]])
        root_base_pose = np.tile(root_base_pose, (states.shape[0], 1))
        states = states.reshape(-1,7)

        # converse pose to hand root coordinate
        states_base = self.compute_poses_wrt_root(states, root_base_pose)
        
        if state_dim == 3:
            states_base = states_base[:, :3]

        return states_base

    def step(self, action):
        """Step physical environment and collect data. Use arg "action_steps" and "interval" to prevent collect too similar data due to moving average control.

        Args:
            actions: actions to apply
            biological phenomena is added in env instead of here
        """
        self.env.reset()

        for _ in range(self.action_steps):
            # self.env.mj_render()
            self.env.step(action)
            
        
        if self.steps % self.interval == 0:
            self.collect(action)
            self.t += 1
        self.steps += 1
        
        
    def save(self):
        '''save collected data to .npy file
        '''
        cur_time = time.strftime('_%b_%d_%H:%M:%S_%Y', time.localtime())
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.save_path, self.task_name, self.name + cur_time)
        os.makedirs(save_path,exist_ok=True)
        np.save(os.path.join(save_path, 'hand_states.npy'), self.hand_states)
        np.save(os.path.join(save_path, 'actions.npy'), self.actions)

    def compute_poses_wrt_root(self, object_pose, root_pose):
        assert object_pose.shape == root_pose.shape, f"Different shapes of object poses and root_poses !!! {object_pose.shape},{root_pose.shape}"
        object_pos = torch.from_numpy(object_pose[:, 0:3])
        object_rot = torch.from_numpy(object_pose[:, 3:7])
        root_pose = torch.from_numpy(root_pose)

        root_pos = root_pose[:, 0:3]
        root_quat_wxyz = root_pose[:, 3:7]
  
        R_W_P = quaternion_to_matrix(root_quat_wxyz)

        T_W_P = torch.eye(4).repeat(R_W_P.shape[0], 1, 1).to(R_W_P.device)
        T_W_P[:, 0:3, 0:3] = R_W_P
        T_W_P[:, 0:3, 3] = root_pos

        object_quat_wxyz = object_rot
        R_W_O = quaternion_to_matrix(object_quat_wxyz)

        T_W_O = torch.eye(4).repeat(R_W_O.shape[0], 1, 1).to(R_W_O.device)
        T_W_O[:, 0:3, 0:3] = R_W_O
        T_W_O[:, 0:3, 3] = object_pos

        relative_pose = torch.matmul(torch.inverse(T_W_P), T_W_O)

        relative_translation = relative_pose[:, 0:3, 3]
        relative_quat_wxyz = matrix_to_quaternion(relative_pose[:, 0:3, 0:3])

        relative_quat_xyzw = relative_quat_wxyz[:, [1, 2, 3, 0]]

        object_pos_wrt_root = relative_translation
        object_quat_wrt_root = relative_quat_xyzw

        object_pose_wrt_root = torch.cat((object_pos_wrt_root, object_quat_wrt_root), axis=-1)
        
        return object_pose_wrt_root.numpy()
    
class SeqMyoHandDataCollector(DataCollector):
    ''' Sequential Data collector for myosuite environments.
    '''
    def __init__(self, env, cfg) -> None:
        super().__init__(cfg, env)
        self.task_name = cfg.task
        self.num_data = cfg.num_data
        self.episode_len = cfg.episode_length
        self.eps_len = int(self.num_data / 1)
        self.device = cfg.device
        self.interval = cfg.interval
        self.save_path = cfg.save_path
        self.action_steps = cfg.action_steps
        self.t = 0
        self.steps = 0
        self.name = cfg.name

        # parse data type
        self.data_type = cfg.data_type
        if cfg.data_type == 'fingertip_pos':
            self.data_dim = [self.env.num_fingertips, 3]
        elif cfg.data_type == 'fingertip_pose':
            self.data_dim = [self.env.num_fingertips, 7]
        else:
            raise NotImplementedError("Data type not implemented!!!")

        self.allocate_buffers()
    
    def allocate_buffers(self):
        '''Initialize np.array for data collection, use cpu to store the data
        '''
        self.cur_hand_states = np.zeros(
            (self.eps_len, *self.data_dim), dtype=np.float64)
        
        self.next_hand_states = np.zeros(
            (self.eps_len, *self.data_dim), dtype=np.float64)
        
        self.actions = np.zeros(
            (self.eps_len, *self.env.action_space.shape), dtype=np.float64)

    def get_states(self):
        if self.data_type == 'fingertip_pos':
            states = self.env.get_obs_dict(self.env.sim)['tip_pose'].reshape(-1, 12)[:,:3]
        elif self.data_type == 'fingertip_pose':
            states = self.env.get_obs_dict(self.env.sim)['tip_pose'].reshape(-1, 12)
            mats = states[:,3:].reshape(-1,3,3)
            quats = []
            for i in range(mats.shape[0]):
                quat = R.from_matrix(mats[i]).as_quat()
                quats.append(quat)
            quats = np.stack(quats, 0)
            states = np.concatenate([states[:,:3], quats], axis=-1)
        return states

    def collect(self, cur_states, action):
        '''Collect each action-fingertip/rigid body poses(in root base coordinate) pairs for model learning
        '''
        states = self.get_states()
        # TODO: introduce a root site to make the target pose wrt world
        # if 'pos' in self.data_type or 'pose' in self.data_type:
        #     states = self.process_states_coords(states)
        
        self.cur_hand_states[self.t] = cur_states
        self.next_hand_states[self.t] = states
        self.actions[self.t] = action
    
    def process_states_coords(self, states):
        state_dim = states.shape[-1]
        if state_dim == 3:
            states = np.concatenate([states, np.tile(np.array([0.,0.,0.,1.]), (states.shape[0], 1))], axis=-1)
        
        root_base_id = self.env.sim.model.site_name2id(self.env.base_name)
        root_base_pose = np.concatenate([self.env.sim.model.site_pos[root_base_id], self.env.sim.model.site_quat[root_base_id]])
        root_base_pose = np.tile(root_base_pose, (states.shape[0], 1))
        states = states.reshape(-1,7)

        # converse pose to hand root coordinate
        states_base = self.compute_poses_wrt_root(states, root_base_pose)
        
        if state_dim == 3:
            states_base = states_base[:, :3]

        return states_base

    def step(self, action):
        """Step physical environment and collect data. Use arg "action_steps" and "interval" to prevent collect too similar data due to moving average control.

        Args:
            actions: actions to apply
            biological phenomena is added in env instead of here
        """
        # self.env.mj_render()
        if self.t % self.episode_len == 0:
            self.env.reset()
        cur_states = self.get_states()
        
        for _ in range(self.action_steps):
            self.env.step(action)
        
        if self.steps % self.interval == 0:
            self.collect(cur_states, action)
            self.t += 1
        self.steps += 1

        
    def save(self):
        '''save collected data to .npy file
        '''
        cur_time = time.strftime('_%b_%d_%H:%M:%S_%Y', time.localtime())
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.save_path, self.task_name, self.name + cur_time)
        os.makedirs(save_path,exist_ok=True)
        np.save(os.path.join(save_path, 'cur_hand_states.npy'), self.cur_hand_states)
        np.save(os.path.join(save_path, 'actions.npy'), self.actions)
        np.save(os.path.join(save_path, 'next_hand_states.npy'), self.next_hand_states)

    def compute_poses_wrt_root(self, object_pose, root_pose):
        assert object_pose.shape == root_pose.shape, f"Different shapes of object poses and root_poses !!! {object_pose.shape},{root_pose.shape}"
        object_pos = torch.from_numpy(object_pose[:, 0:3])
        object_rot = torch.from_numpy(object_pose[:, 3:7])
        root_pose = torch.from_numpy(root_pose)

        root_pos = root_pose[:, 0:3]
        root_quat_wxyz = root_pose[:, 3:7]
  
        R_W_P = quaternion_to_matrix(root_quat_wxyz)

        T_W_P = torch.eye(4).repeat(R_W_P.shape[0], 1, 1).to(R_W_P.device)
        T_W_P[:, 0:3, 0:3] = R_W_P
        T_W_P[:, 0:3, 3] = root_pos

        object_quat_wxyz = object_rot
        R_W_O = quaternion_to_matrix(object_quat_wxyz)

        T_W_O = torch.eye(4).repeat(R_W_O.shape[0], 1, 1).to(R_W_O.device)
        T_W_O[:, 0:3, 0:3] = R_W_O
        T_W_O[:, 0:3, 3] = object_pos

        relative_pose = torch.matmul(torch.inverse(T_W_P), T_W_O)

        relative_translation = relative_pose[:, 0:3, 3]
        relative_quat_wxyz = matrix_to_quaternion(relative_pose[:, 0:3, 0:3])

        relative_quat_xyzw = relative_quat_wxyz[:, [1, 2, 3, 0]]

        object_pos_wrt_root = relative_translation
        object_quat_wrt_root = relative_quat_xyzw

        object_pose_wrt_root = torch.cat((object_pos_wrt_root, object_quat_wrt_root), axis=-1)
        
        return object_pose_wrt_root.numpy()
    
class InHandManipulationDataCollector(DataCollector):
    def __init__(self, env, cfg) -> None:
        super().__init__(cfg, env)
        self.task_name = cfg.task
        self.num_data = cfg.num_data
        self.episode_len = cfg.episode_length
        self.eps_len = int(self.num_data / 1)
        self.device = cfg.device
        self.interval = cfg.interval
        self.save_path = cfg.save_path
        self.action_steps = cfg.action_steps
        self.t = 0
        self.steps = 0
        self.name = cfg.name

        # parse data type
        self.data_type = cfg.data_type
        if cfg.data_type == 'fingertip_pos':
            self.data_dim = [5, 3]
        else:
            raise NotImplementedError("Data type not implemented!!!")
        
        self.obj_data_type = cfg.obj_data_type
        if cfg.obj_data_type == 'obj_pos':
            self.obj_data_dim = [3]
        elif cfg.obj_data_type == 'obj_pose':
            self.obj_data_dim = [7]
        else:
            raise NotImplementedError("Object Data type not implemented!!!")
        
        self.allocate_buffers()
    
    def allocate_buffers(self):
        '''Initialize np.array for data collection, use cpu to store the data
        '''
        self.cur_hand_states = np.zeros(
            (self.eps_len, *self.data_dim), dtype=np.float64)
        
        self.next_hand_states = np.zeros(
            (self.eps_len, *self.data_dim), dtype=np.float64)
        
        self.cur_obj_states = np.zeros(
            (self.eps_len, *self.obj_data_dim), dtype=np.float64)
        
        self.next_obj_states = np.zeros(
            (self.eps_len, *self.obj_data_dim), dtype=np.float64)
        
        self.actions = np.zeros(
            (self.eps_len, *self.env.action_space.shape), dtype=np.float64)
        
    def get_states(self):
        if self.data_type == 'fingertip_pos':
            hand_states = self.env.get_obs_dict(self.env.sim)['tip_pos'].reshape(-1, 3)
        elif self.data_type == 'fingertip_pose':
            raise NotImplementedError("Fingertip pose not implemented!!!")

        if self.obj_data_type == 'obj_pos':
            obj_states = self.env.get_obs_dict(self.env.sim)['obj_pos']
        elif self.obj_data_type == 'obj_pose':
            obj_pos = self.env.get_obs_dict(self.env.sim)['obj_pos']
            obj_rot = self.env.get_obs_dict(self.env.sim)['obj_rot']
            obj_states = np.concatenate([obj_pos, obj_rot], axis=-1)
        return hand_states, obj_states
    
    def collect(self, cur_hand_states, cur_obj_states, action):
        next_hand_states, next_obj_states = self.get_states()

        self.cur_hand_states[self.t] = cur_hand_states
        self.cur_obj_states[self.t] = cur_obj_states
        self.next_hand_states[self.t] = next_hand_states
        self.next_obj_states[self.t] = next_obj_states 
        self.actions[self.t] = action

    def step(self, action):
        """Step physical environment and collect data. Use arg "action_steps" and "interval" to prevent collect too similar data due to moving average control.

        Args:
            actions: actions to apply
            biological phenomena is added in env instead of here
        """
        # self.env.mj_render()
        if self.t % self.episode_len == 0:
            self.env.reset()
        cur_hand_states, cur_obj_states = self.get_states()
        
        for _ in range(self.action_steps):
            o, r, t, d, info = self.env.step(action)
        
        if self.steps % self.interval == 0:
            self.collect(cur_hand_states, cur_obj_states, action)
            self.t += 1
        self.steps += 1
        return o, r, t, d, info

    def save(self):
        '''save collected data to .npy file
        '''
        cur_time = time.strftime('_%b_%d_%H:%M:%S_%Y', time.localtime())
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.save_path, self.name + cur_time)
        os.makedirs(save_path,exist_ok=True)
        np.save(os.path.join(save_path, 'cur_hand_states.npy'), self.cur_hand_states)
        np.save(os.path.join(save_path, 'next_hand_states.npy'), self.next_hand_states)
        np.save(os.path.join(save_path, 'cur_obj_states.npy'), self.cur_obj_states)
        np.save(os.path.join(save_path, 'next_obj_states.npy'), self.next_obj_states)
        np.save(os.path.join(save_path, 'actions.npy'), self.actions)

class InHandDynamicDataCollector(DataCollector):
    def __init__(self, env, eps_num, eps_len, split, data_type='fingertip_pos', obj_data_type='obj_pos') -> None:
        super().__init__(cfg=None, env=env)
        self.eps_num = eps_num
        self.eps_min_len = eps_len
        self.eps_count = 0
        self.step_count = 0
        self.split = split

        # parse data type
        self.data_type = data_type
        if data_type == 'fingertip_pos':
            self.data_dim = [5, 3]
        else:
            raise NotImplementedError("Data type not implemented!!!")
        
        self.obj_data_type = obj_data_type
        if obj_data_type == 'obj_pos':
            self.obj_data_dim = [3]
        elif obj_data_type == 'obj_pose':
            self.obj_data_dim = [7]
        elif obj_data_type == 'obj_pos_rot':
            self.obj_data_dim = [6]
        else:
            raise NotImplementedError("Object Data type not implemented!!!")
        
        self.allocate_buffers()
    
    def allocate_buffers(self):
        '''Initialize np.array for data collection, use cpu to store the data
        '''
        self.cur_hand_states = []
        
        self.next_hand_states = []
        
        self.cur_obj_states = []
        
        self.next_obj_states = []
        
        self.actions = []
        
        self.reward = []

        self.solved = []
        
    def reset(self, eps_num=None, eps_len=None, split=None):
        self.eps_count = 0
        self.step_count = 0
        if eps_num is not None:
            self.eps_num = eps_num
        if eps_len is not None:
            self.eps_min_len = eps_len
        if split is not None:
            self.split = split

        self.allocate_buffers()
        
    def get_states(self):
        if self.data_type == 'fingertip_pos':
            hand_states = self.env.get_obs_dict(self.env.sim)['tip_pos'].reshape(-1, 3)
        elif self.data_type == 'fingertip_pose':
            raise NotImplementedError("Fingertip pose not implemented!!!")

        if self.obj_data_type == 'obj_pos':
            obj_states = self.env.get_obs_dict(self.env.sim)['obj_pos']
        elif self.obj_data_type == 'obj_pose':
            obj_pos = self.env.get_obs_dict(self.env.sim)['obj_pos']
            obj_rot = self.env.get_obs_dict(self.env.sim)['obj_rot']
            obj_states = np.concatenate([obj_pos, obj_rot], axis=-1)
        elif self.obj_data_type == 'obj_pos_rot':
            obj_pos = self.env.get_obs_dict(self.env.sim)['obj_pos']
            obj_rot = self.env.get_obs_dict(self.env.sim)['obj_rot']
            obj_states = np.concatenate([obj_pos, obj_rot], axis=-1)
        return hand_states, obj_states
    
    def collect(self, cur_hand_states, cur_obj_states, action, r , solved):
        next_hand_states, next_obj_states = self.get_states()

        self.cur_hand_states[self.eps_count].append(cur_hand_states)
        self.cur_obj_states[self.eps_count].append(cur_obj_states)
        self.next_hand_states[self.eps_count].append(next_hand_states)
        self.next_obj_states[self.eps_count].append(next_obj_states)
        self.actions[self.eps_count].append(action)
        self.reward[self.eps_count].append(r)
        self.solved[self.eps_count].append(solved)

    def collect_rollout(self, policy):
        """

        Args:
            actions: actions to apply
            biological phenomena is added in env instead of here
        """
        while self.eps_count < self.eps_num:
            self.env.reset()
            self.cur_hand_states.append([])
            self.next_hand_states.append([])
            self.cur_obj_states.append([])
            self.next_obj_states.append([])
            self.actions.append([])
            self.reward.append([])
            self.solved.append([])
            done = False
            while not done:
                cur_hand_states, cur_obj_states = self.get_states()
                obs = {'hand_cur_data': cur_hand_states, 'obj_cur_data': cur_obj_states}
                action = policy.predict(obs)
                o, r, t, d, info = self.env.step(action)
                self.collect(cur_hand_states, cur_obj_states, action, r, info['solved'])
                self.step_count += 1
                done = d or t
            if self.step_count < self.eps_min_len:
                self.cur_hand_states.pop()
                self.next_hand_states.pop()
                self.cur_obj_states.pop()
                self.next_obj_states.pop()
                self.actions.pop()
                self.reward.pop()
                self.solved.pop()
            else:
                self.cur_hand_states[self.eps_count] = np.array(self.cur_hand_states[self.eps_count])
                self.next_hand_states[self.eps_count] = np.array(self.next_hand_states[self.eps_count])
                self.cur_obj_states[self.eps_count] = np.array(self.cur_obj_states[self.eps_count])
                self.next_obj_states[self.eps_count] = np.array(self.next_obj_states[self.eps_count])
                self.actions[self.eps_count] = np.array(self.actions[self.eps_count])
                self.reward[self.eps_count] = np.array(self.reward[self.eps_count])
                self.solved[self.eps_count] = np.array(self.solved[self.eps_count])
                self.eps_count += 1
            self.step_count = 0
        
        train_transdata = {'hand_cur_data': self.cur_hand_states[:int(self.eps_num*self.split)], 
                     'hand_next_data': self.next_hand_states[:int(self.eps_num*self.split)],
                     'obj_cur_data': self.cur_obj_states[:int(self.eps_num*self.split)],
                     'obj_next_data': self.next_obj_states[:int(self.eps_num*self.split)],
                     'action_data': self.actions[:int(self.eps_num*self.split)],
                     'reward': self.reward[:int(self.eps_num*self.split)],
                     'solved': self.solved[:int(self.eps_num*self.split)],
        }
        val_transdata = {'hand_cur_data': self.cur_hand_states[int(self.eps_num*self.split):], 
                     'hand_next_data': self.next_hand_states[int(self.eps_num*self.split):],
                     'obj_cur_data': self.cur_obj_states[int(self.eps_num*self.split):],
                     'obj_next_data': self.next_obj_states[int(self.eps_num*self.split):],
                     'action_data': self.actions[int(self.eps_num*self.split):],
                     'reward': self.reward[int(self.eps_num*self.split):],
                     'solved': self.solved[int(self.eps_num*self.split):],
        }
        return train_transdata ,val_transdata
    
    def save(self, save_path):
        pass