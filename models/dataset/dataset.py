import numpy as np
import os
import torch

from torch.utils.data import Dataset, DataLoader
from datacollector.normalization import MyoHandNorm, MyoHandSeqNorm, InhandNorm

# class HandStateDataset(Dataset):
#     def 

class PoseDataset(Dataset):
    def __init__(self, data_path, device) -> None:
        # self.device = device
        self.data = torch.from_numpy(np.load(os.path.join(data_path, 'actions.npy')))
        self.labels = torch.from_numpy(np.load(os.path.join(data_path, 'hand_states.npy')))
        num_envs, length, finger_num, _ = self.labels.shape
        self.len = num_envs * length
        
        self.data = self.data.view(self.len, -1)        # action
        self.labels = self.labels.view(self.len, -1)    # fingertip pose
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    
class PositionDataset(Dataset):
    def __init__(self, data_path, normalize, device) -> None:
        # self.device = device
        self.data = torch.from_numpy(np.load(os.path.join(data_path, 'actions.npy'))).float()
        self.labels = torch.from_numpy(np.load(os.path.join(data_path, 'hand_states.npy'))).float()
        if self.labels.ndim == 4:
            num_envs, length, finger_num, _ = self.labels.shape
            self.len = num_envs * length
        else:
            length, finger_num, _ = self.labels.shape 
            self.len = length
        self.data = self.data.view(self.len, -1)        # action
        self.labels = self.labels[...,:3].reshape(self.len, 3*finger_num)    # fingertip position
        
        # normalization for fingertip position
        if normalize:
            self.mean = MyoHandNorm.mean
            self.std = MyoHandNorm.std
            self.labels = (self.labels - self.mean) / self.std

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
class ActionDataset(Dataset):
    def __init__(self, data_path, normalize, device) -> None:
        # self.device = device
        self.data = torch.from_numpy(np.load(os.path.join(data_path, 'hand_states.npy'))).float()
        self.labels = torch.from_numpy(np.load(os.path.join(data_path, 'actions.npy'))).float()
        if self.data.ndim == 4:
            num_envs, length, finger_num, _ = self.data.shape
            self.len = num_envs * length
        else:
            length, finger_num, _ = self.data.shape 
            self.len = length
        self.labels = self.labels.view(self.len, -1)        # action
        self.data = self.data[...,:3].reshape(self.len, 3*finger_num)    # fingertip position
        
        # # normalization for fingertip position
        if normalize:
            self.mean = MyoHandNorm.mean
            self.std = MyoHandNorm.std
            self.data = (self.data - self.mean) / self.std 

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]  

class TransitionDataset(Dataset):
    def __init__(self, data_path, device) -> None:
        # self.device = device
        self.cur_states = torch.from_numpy(np.load(os.path.join(data_path, 'cur_hand_states.npy'))).float()
        self.actions = torch.from_numpy(np.load(os.path.join(data_path, 'actions.npy'))).float()
        self.next_states = torch.from_numpy(np.load(os.path.join(data_path, 'next_hand_states.npy'))).float()
        if self.cur_states.ndim == 4:
            num_envs, length, finger_num, _ = self.cur_states.shape
            self.len = num_envs * length
        else:
            length, finger_num, _ = self.cur_states.shape 
            self.len = length
        self.cur_states = self.cur_states.view(self.len, -1)        # action
        self.actions = self.actions.view(self.len, -1)        # action
        self.next_states = self.next_states.view(self.len, -1)    # fingertip pose

        # normalization for fingertip position
        self.mean = MyoHandSeqNorm.mean
        self.std = MyoHandSeqNorm.std
        self.cur_states = (self.cur_states - self.mean) / self.std
        self.next_states = (self.next_states - self.mean) / self.std

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return torch.cat((self.cur_states[index], self.actions[index]), dim=-1), self.next_states[index]
    
class SeqTransDataset(Dataset):
    def __init__(self, data_path, episode_length=50, seq_len=10, device='cuda') -> None:
        # self.device = device
        self.ep_len = episode_length
        self.seq_len = seq_len
        self.cur_states = torch.from_numpy(np.load(os.path.join(data_path, 'cur_hand_states.npy'))).float()
        self.actions = torch.from_numpy(np.load(os.path.join(data_path, 'actions.npy'))).float()
        self.next_states = torch.from_numpy(np.load(os.path.join(data_path, 'next_hand_states.npy'))).float()
        if self.cur_states.ndim == 4:
            num_envs, length, finger_num, _ = self.cur_states.shape
            length = num_envs * length
        else:
            length, finger_num, _ = self.cur_states.shape
         
        self.cur_states = self.cur_states.view(length//episode_length, episode_length, -1)        # action
        self.actions = self.actions.view(length//episode_length, episode_length, -1)        # action
        self.next_states = self.next_states.view(length//episode_length, episode_length, -1)    # fingertip pose

        # normalization for fingertip position
        self.mean = MyoHandSeqNorm.mean
        self.std = MyoHandSeqNorm.std
        self.cur_states = (self.cur_states - self.mean) / self.std
        self.next_states = (self.next_states - self.mean) / self.std

        self.len = (episode_length - seq_len + 1) * (length//episode_length)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        ep_idx = index // (self.ep_len - self.seq_len + 1)
        seq_idx = index % (self.ep_len - self.seq_len + 1)
        return (self.cur_states[ep_idx, seq_idx], self.actions[ep_idx, seq_idx: seq_idx+self.seq_len]), self.next_states[ep_idx, seq_idx: seq_idx+self.seq_len]

class InhandTransDataset(Dataset):
    def __init__(self, data_path, episode_length=50, seq_len=10, device='cuda') -> None:
        # self.device = device
        self.ep_len = episode_length
        self.seq_len = seq_len
        self.cur_hand_states = torch.from_numpy(np.load(os.path.join(data_path, 'cur_hand_states.npy'))).float()
        self.actions = torch.from_numpy(np.load(os.path.join(data_path, 'actions.npy'))).float()
        self.next_hand_states = torch.from_numpy(np.load(os.path.join(data_path, 'next_hand_states.npy'))).float()
        self.cur_obj_states = torch.from_numpy(np.load(os.path.join(data_path, 'cur_obj_states.npy'))).float()
        self.next_obj_states = torch.from_numpy(np.load(os.path.join(data_path, 'next_obj_states.npy'))).float()

        if self.cur_hand_states.ndim == 4:
            num_envs, length, finger_num, _ = self.cur_hand_states.shape
            length = num_envs * length
        else:
            length, finger_num, _ = self.cur_hand_states.shape
         
        self.cur_hand_states = self.cur_hand_states.view(length//episode_length, episode_length, -1)        # action
        self.actions = self.actions.view(length//episode_length, episode_length, -1)        # action
        self.next_hand_states = self.next_hand_states.view(length//episode_length, episode_length, -1)    # fingertip pose
        self.cur_obj_states = self.cur_obj_states.view(length//episode_length, episode_length, -1)   
        self.next_obj_states = self.next_obj_states.view(length//episode_length, episode_length, -1)
        
        # normalization for fingertip position
        self.hand_mean = InhandNorm.hand_mean
        self.hand_std = InhandNorm.hand_std
        self.obj_mean = InhandNorm.obj_mean
        self.obj_std = InhandNorm.obj_std
        self.cur_hand_states = (self.cur_hand_states - self.hand_mean) / self.hand_std
        self.next_hand_states = (self.next_hand_states - self.hand_mean) / self.hand_std
        self.cur_obj_states = (self.cur_obj_states - self.obj_mean) / self.obj_std
        self.next_obj_states = (self.next_obj_states - self.obj_mean) / self.obj_std

        self.len = (episode_length - seq_len + 1) * (length//episode_length)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        ep_idx = index // (self.ep_len - self.seq_len + 1)
        seq_idx = index % (self.ep_len - self.seq_len + 1)
        return (self.cur_hand_states[ep_idx, seq_idx], self.cur_obj_states[ep_idx, seq_idx], self.actions[ep_idx, seq_idx: seq_idx+self.seq_len]), (self.next_hand_states[ep_idx, seq_idx: seq_idx+self.seq_len], self.next_obj_states[ep_idx, seq_idx: seq_idx+self.seq_len])

class InhandDynamicDataset(Dataset):
    def __init__(self, trans_data, seq_len) -> None:
        super().__init__()
        self.obj_cur_data = trans_data['obj_cur_data']
        self.hand_cur_data = trans_data['hand_cur_data']
        self.obj_next_data = trans_data['obj_next_data']
        self.hand_next_data = trans_data['hand_next_data']
        self.action_data = trans_data['action_data']
        
        self.seq_len = seq_len
        self.eps_num = len(self.obj_cur_data)
        self.total_len = 0
        self.accu_eps_lens = []

        for i in range(self.eps_num):
            self.total_len += self.obj_cur_data[i].shape[0] - seq_len + 1
            self.accu_eps_lens.append(self.total_len)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        for i, accu_len in enumerate(self.accu_eps_lens):
            if index < accu_len:
                ep_idx = i
                break
        seq_idx = index - self.accu_eps_lens[ep_idx-1] if ep_idx > 0 else index
        return (torch.from_numpy(self.hand_cur_data[ep_idx][seq_idx]).reshape(-1,), torch.from_numpy(self.obj_cur_data[ep_idx][seq_idx]).reshape(-1,), torch.from_numpy(self.action_data[ep_idx][seq_idx: seq_idx+self.seq_len])), (torch.from_numpy(self.hand_next_data[ep_idx][seq_idx: seq_idx+self.seq_len]).reshape(self.seq_len, -1), torch.from_numpy(self.obj_next_data[ep_idx][seq_idx: seq_idx+self.seq_len]).reshape(self.seq_len, -1))



# class DataProcessor:
#     def __init__(self, params=None) -> None:
#         self.params = params

#     def preprocess_data(self, transdata: dict):
#         eps_num, eps_len = transdata['obj_cur_data'].shape[0], transdata['obj_cur_data'].shape[1]
#         obj_cur_data = transdata['obj_cur_data'].reshape(eps_num*eps_len, -1)
#         hand_cur_data = transdata['hand_cur_data'].reshape(eps_num*eps_len, -1)
#         obj_next_data = transdata['obj_next_data'].reshape(eps_num*eps_len, -1)
#         hand_next_data = transdata['hand_next_data'].reshape(eps_num*eps_len, -1)
#         action_data = transdata['action_data'].reshape(eps_num*eps_len, -1)
       

#         # normalization
#         self.obj_data_mean = np.concatenate([obj_cur_data, obj_next_data], axis=0).mean(axis=0, keepdims=True)
#         self.obj_data_std = np.concatenate([obj_cur_data, obj_next_data], axis=0).std(axis=0, keepdims=True)
#         self.hand_data_mean = np.concatenate([hand_cur_data, hand_next_data], axis=0).mean(axis=0, keepdims=True)
#         self.hand_data_std = np.concatenate([hand_cur_data, hand_next_data], axis=0).std(axis=0, keepdims=True)

#         obj_cur_data = (obj_cur_data - self.obj_data_mean) / self.obj_data_std
#         obj_next_data = (obj_next_data - self.obj_data_mean) / self.obj_data_std
#         hand_cur_data = (hand_cur_data - self.hand_data_mean) / self.hand_data_std
#         hand_next_data = (hand_next_data - self.hand_data_mean) / self.hand_data_std

#         return {
#             'obj_cur_data': obj_cur_data.reshape(eps_num, eps_len, -1),
#             'hand_cur_data': hand_cur_data.reshape(eps_num, eps_len, -1),
#             'obj_next_data': obj_next_data.reshape(eps_num, eps_len, -1),
#             'hand_next_data': hand_next_data.reshape(eps_num, eps_len, -1),
#             'action_data': action_data.reshape(eps_num, eps_len, -1)
#         }
    
#     def get_normalized_data(self):
#         return {
#             'obj_data_mean': self.obj_data_mean,
#             'obj_data_std': self.obj_data_std,
#             'hand_data_mean': self.hand_data_mean,
#             'hand_data_std': self.hand_data_std
#         }

#     def combine_data(self, transdata1: dict, transdata2: dict):
#         transdata = {}
#         for key in transdata1.keys():
#             transdata[key] = np.concatenate([transdata1[key], transdata2[key]], axis=0)
#         return transdata
    
#     def get_normalized_data(self, cur_obs):
#         obj_cur_data = cur_obs['obj_cur_data'].reshape(-1,)
#         hand_cur_data = cur_obs['hand_cur_data'].reshape(-1,)
#         obj_cur_data = (obj_cur_data - self.obj_data_mean.squeeze()) / self.obj_data_std.squeeze()
#         hand_cur_data = (hand_cur_data - self.hand_data_mean.squeeze()) / self.hand_data_std.squeeze()
#         return {
#             'obj_cur_data': obj_cur_data,
#             'hand_cur_data': hand_cur_data,
#         }

#     def get_normalized_dataset(self, transdata):
#         eps_num, eps_len = transdata['obj_cur_data'].shape[0], transdata['obj_cur_data'].shape[1]
#         obj_cur_data = transdata['obj_cur_data'].reshape(eps_num*eps_len, -1)
#         hand_cur_data = transdata['hand_cur_data'].reshape(eps_num*eps_len, -1)
#         obj_next_data = transdata['obj_next_data'].reshape(eps_num*eps_len, -1)
#         hand_next_data = transdata['hand_next_data'].reshape(eps_num*eps_len, -1)
#         action_data = transdata['action_data'].reshape(eps_num*eps_len, -1)

#         obj_cur_data = (obj_cur_data - self.obj_data_mean) / self.obj_data_std
#         obj_next_data = (obj_next_data - self.obj_data_mean) / self.obj_data_std
#         hand_cur_data = (hand_cur_data - self.hand_data_mean) / self.hand_data_std
#         hand_next_data = (hand_next_data - self.hand_data_mean) / self.hand_data_std

#         return {
#             'obj_cur_data': obj_cur_data.reshape(eps_num, eps_len, -1),
#             'hand_cur_data': hand_cur_data.reshape(eps_num, eps_len, -1),
#             'obj_next_data': obj_next_data.reshape(eps_num, eps_len, -1),
#             'hand_next_data': hand_next_data.reshape(eps_num, eps_len, -1),
#             'action_data': action_data.reshape(eps_num, eps_len, -1),
#         }


class DataProcessor:
    def __init__(self, params=None) -> None:
        self.params = params

    def preprocess_data(self, transdata: dict):
        eps_num = len(transdata['obj_cur_data'])
        obj_cur_data = transdata['obj_cur_data'].copy()
        hand_cur_data = transdata['hand_cur_data'].copy()
        obj_next_data = transdata['obj_next_data'].copy()
        hand_next_data = transdata['hand_next_data'].copy()
        action_data = transdata['action_data'].copy()
        reward_data = transdata['reward'].copy()

        for i in range(eps_num):
            hand_cur_data[i] = hand_cur_data[i].reshape(-1, 15)
            hand_next_data[i] = hand_next_data[i].reshape(-1, 15)

        # normalization
        self.obj_data_mean = np.concatenate(obj_cur_data, axis=0).mean(axis=0, keepdims=True)
        self.obj_data_std = np.concatenate(obj_cur_data, axis=0).std(axis=0, keepdims=True)
        self.hand_data_mean = np.concatenate(hand_cur_data, axis=0).mean(axis=0, keepdims=True)
        self.hand_data_std = np.concatenate(hand_cur_data, axis=0).std(axis=0, keepdims=True)

        # # filter data
        # if eps_num < self.params['min_eps_num']:
        #     obj_cur_data = obj_cur_data * (self.params['min_eps_num'] // eps_num + 1)
        #     hand_cur_data = hand_cur_data * (self.params['min_eps_num'] // eps_num + 1)
        #     obj_next_data = obj_next_data * (self.params['min_eps_num'] // eps_num + 1)
        #     hand_next_data = hand_next_data * (self.params['min_eps_num'] // eps_num + 1)
        #     action_data = action_data * (self.params['min_eps_num'] // eps_num + 1)
        # elif eps_num > self.params['max_eps_num']:
        #     eps_neg_rewards = [-reward_data[i].sum() for i in range(eps_num)]
        #     eps_idxs = np.argsort(eps_neg_rewards)
        #     obj_cur_data = [obj_cur_data[i] for i in eps_idxs[:self.params['max_eps_num']]]
        #     hand_cur_data = [hand_cur_data[i] for i in eps_idxs[:self.params['max_eps_num']]]
        #     obj_next_data = [obj_next_data[i] for i in eps_idxs[:self.params['max_eps_num']]]
        #     hand_next_data = [hand_next_data[i] for i in eps_idxs[:self.params['max_eps_num']]]
        #     action_data = [action_data[i] for i in eps_idxs[:self.params['max_eps_num']]]
        
        # # filter data, version 2
        # if eps_num < self.params['train_eps_num']:
        #     obj_cur_data = (obj_cur_data * (self.params['train_eps_num'] // eps_num + 1))[:self.params['train_eps_num']]
        #     hand_cur_data = (hand_cur_data * (self.params['train_eps_num'] // eps_num + 1))[:self.params['train_eps_num']]
        #     obj_next_data = (obj_next_data * (self.params['train_eps_num'] // eps_num + 1))[:self.params['train_eps_num']]
        #     hand_next_data = (hand_next_data * (self.params['train_eps_num'] // eps_num + 1))[:self.params['train_eps_num']]
        #     action_data = (action_data * (self.params['train_eps_num'] // eps_num + 1))[:self.params['train_eps_num']]
        # else:
        #     num_used = self.params['train_eps_num']
        #     eps_neg_rewards = [-reward_data[i].sum() for i in range(eps_num)]
        #     eps_idxs = np.argsort(eps_neg_rewards)[:num_used]
        #     obj_cur_data = [obj_cur_data[i] for i in eps_idxs[:num_used]]
        #     hand_cur_data = [hand_cur_data[i] for i in eps_idxs[:num_used]]
        #     obj_next_data = [obj_next_data[i] for i in eps_idxs[:num_used]]
        #     hand_next_data = [hand_next_data[i] for i in eps_idxs[:num_used]]
        #     action_data = [action_data[i] for i in eps_idxs[:num_used]]

        eps_num = len(obj_cur_data)
        for ep in range(eps_num):
            obj_cur_data[ep] = (obj_cur_data[ep] - self.obj_data_mean) / self.obj_data_std
            obj_next_data[ep] = (obj_next_data[ep] - self.obj_data_mean) / self.obj_data_std
            hand_cur_data[ep] = (hand_cur_data[ep] - self.hand_data_mean) / self.hand_data_std
            hand_next_data[ep] = (hand_next_data[ep] - self.hand_data_mean) / self.hand_data_std

        return {
            'obj_cur_data': obj_cur_data,
            'hand_cur_data': hand_cur_data,
            'obj_next_data': obj_next_data,
            'hand_next_data': hand_next_data,
            'action_data': action_data
        }
    
    def get_normalized_data(self):
        return {
            'obj_data_mean': self.obj_data_mean,
            'obj_data_std': self.obj_data_std,
            'hand_data_mean': self.hand_data_mean,
            'hand_data_std': self.hand_data_std
        }

    def combine_data(self, transdata1: dict, transdata2: dict):
        transdata = {}
        for key in transdata1.keys():
            transdata[key] = transdata1[key] + transdata2[key]
        return transdata
    
    def get_normalized_data(self, cur_obs):
        obj_cur_data = cur_obs['obj_cur_data'].reshape(-1,)
        hand_cur_data = cur_obs['hand_cur_data'].reshape(-1,)
        obj_cur_data = (obj_cur_data - self.obj_data_mean.squeeze()) / self.obj_data_std.squeeze()
        hand_cur_data = (hand_cur_data - self.hand_data_mean.squeeze()) / self.hand_data_std.squeeze()
        return {
            'obj_cur_data': obj_cur_data,
            'hand_cur_data': hand_cur_data,
        }

    def get_normalized_dataset(self, transdata, params):
        eps_num = len(transdata['obj_cur_data'])
        obj_cur_data = transdata['obj_cur_data'].copy()
        hand_cur_data = transdata['hand_cur_data'].copy()
        obj_next_data = transdata['obj_next_data'].copy()
        hand_next_data = transdata['hand_next_data'].copy()
        action_data = transdata['action_data'].copy()
        reward_data = transdata['reward'].copy()

        for i in range(eps_num):
            hand_cur_data[i] = hand_cur_data[i].reshape(-1, 15)
            hand_next_data[i] = hand_next_data[i].reshape(-1, 15)
       
        # # filter data
        # if eps_num < params['min_eps_num']:
        #     obj_cur_data = obj_cur_data * (params['min_eps_num'] // eps_num + 1)
        #     hand_cur_data = hand_cur_data * (params['min_eps_num'] // eps_num + 1)
        #     obj_next_data = obj_next_data * (params['min_eps_num'] // eps_num + 1)
        #     hand_next_data = hand_next_data * (params['min_eps_num'] // eps_num + 1)
        #     action_data = action_data * (params['min_eps_num'] // eps_num + 1)
        # elif eps_num > params['max_eps_num']:
        #     eps_neg_rewards = [-reward_data[i].sum() for i in range(eps_num)]
        #     eps_idxs = np.argsort(eps_neg_rewards)
        #     obj_cur_data = [obj_cur_data[i] for i in eps_idxs[:params['max_eps_num']]]
        #     hand_cur_data = [hand_cur_data[i] for i in eps_idxs[:params['max_eps_num']]]
        #     obj_next_data = [obj_next_data[i] for i in eps_idxs[:params['max_eps_num']]]
        #     hand_next_data = [hand_next_data[i] for i in eps_idxs[:params['max_eps_num']]]
        #     action_data = [action_data[i] for i in eps_idxs[:params['max_eps_num']]]
        
        # # filter data, version 2
        # if eps_num < params['val_eps_num']:
        #     obj_cur_data = (obj_cur_data * (params['val_eps_num'] // eps_num + 1))[:params['val_eps_num']]
        #     hand_cur_data = (hand_cur_data * (params['val_eps_num'] // eps_num + 1))[:params['val_eps_num']]
        #     obj_next_data = (obj_next_data * (params['val_eps_num'] // eps_num + 1))[:params['val_eps_num']]
        #     hand_next_data = (hand_next_data * (params['val_eps_num'] // eps_num + 1))[:params['val_eps_num']]
        #     action_data = (action_data * (params['val_eps_num'] // eps_num + 1))[:params['val_eps_num']]
        # else:
        #     num_used = params['val_eps_num']
        #     eps_neg_rewards = [-reward_data[i].sum() for i in range(eps_num)]
        #     eps_idxs = np.argsort(eps_neg_rewards)
        #     obj_cur_data = [obj_cur_data[i] for i in eps_idxs[:num_used]]
        #     hand_cur_data = [hand_cur_data[i] for i in eps_idxs[:num_used]]
        #     obj_next_data = [obj_next_data[i] for i in eps_idxs[:num_used]]
        #     hand_next_data = [hand_next_data[i] for i in eps_idxs[:num_used]]
        #     action_data = [action_data[i] for i in eps_idxs[:num_used]]

        eps_num  = len(obj_cur_data)
        for ep in range(eps_num):
            obj_cur_data[ep] = (obj_cur_data[ep] - self.obj_data_mean) / self.obj_data_std
            obj_next_data[ep] = (obj_next_data[ep] - self.obj_data_mean) / self.obj_data_std
            hand_cur_data[ep] = (hand_cur_data[ep] - self.hand_data_mean) / self.hand_data_std
            hand_next_data[ep] = (hand_next_data[ep] - self.hand_data_mean) / self.hand_data_std

        return {
            'obj_cur_data': obj_cur_data,
            'hand_cur_data': hand_cur_data,
            'obj_next_data': obj_next_data,
            'hand_next_data': hand_next_data,
            'action_data': action_data
        }



class InverseTransitionDataset(Dataset):
    def __init__(self, data_path, device) -> None:
        # self.device = device
        self.cur_states = torch.from_numpy(np.load(os.path.join(data_path, 'cur_hand_states.npy'))).float()
        self.actions = torch.from_numpy(np.load(os.path.join(data_path, 'actions.npy'))).float()
        self.next_states = torch.from_numpy(np.load(os.path.join(data_path, 'next_hand_states.npy'))).float()
        if self.cur_states.ndim == 4:
            num_envs, length, finger_num, _ = self.cur_states.shape
            self.len = num_envs * length
        else:
            length, finger_num, _ = self.cur_states.shape 
            self.len = length
        self.cur_states = self.cur_states.view(self.len, -1)        # action
        self.actions = self.actions.view(self.len, -1)        # action
        self.next_states = self.next_states.view(self.len, -1)    # fingertip pose

        # normalization for fingertip position
        self.mean = MyoHandSeqNorm.mean
        self.std = MyoHandSeqNorm.std
        self.cur_states = (self.cur_states - self.mean) / self.std
        self.next_states = (self.next_states - self.mean) / self.std

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return torch.cat((self.cur_states[index], self.next_states[index]), dim=-1), self.actions[index]

class IsaacSeqTransDataset(Dataset):
    def __init__(self, data_path, episode_length=50, seq_len=10, device='cuda') -> None:
        # self.device = device
        self.ep_len = episode_length
        self.seq_len = seq_len
        self.cur_states = torch.from_numpy(np.load(os.path.join(data_path, 'cur_hand_states.npy'))).float()
        self.actions = torch.from_numpy(np.load(os.path.join(data_path, 'actions.npy'))).float()
        self.next_states = torch.from_numpy(np.load(os.path.join(data_path, 'next_hand_states.npy'))).float()
        if self.cur_states.ndim == 4:
            num_envs, length, finger_num, _ = self.cur_states.shape
            length = num_envs * length
        else:
            length, finger_num, _ = self.cur_states.shape
         
        self.cur_states = self.cur_states.view(length//episode_length, episode_length, -1)        # action
        self.actions = self.actions.view(length//episode_length, episode_length, -1)        # action
        self.next_states = self.next_states.view(length//episode_length, episode_length, -1)    # fingertip pose

        self.len = (episode_length - seq_len + 1) * (length//episode_length)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        ep_idx = index // (self.ep_len - self.seq_len + 1)
        seq_idx = index % (self.ep_len - self.seq_len + 1)
        return (self.cur_states[ep_idx, seq_idx], self.actions[ep_idx, seq_idx: seq_idx+self.seq_len]), self.next_states[ep_idx, seq_idx: seq_idx+self.seq_len]

class IsaacInverseTransitionDataset(Dataset):
    def __init__(self, data_path, device) -> None:
        # self.device = device
        self.cur_states = torch.from_numpy(np.load(os.path.join(data_path, 'cur_hand_states.npy'))).float()
        self.actions = torch.from_numpy(np.load(os.path.join(data_path, 'actions.npy'))).float()
        self.next_states = torch.from_numpy(np.load(os.path.join(data_path, 'next_hand_states.npy'))).float()
        if self.cur_states.ndim == 4:
            num_envs, length, finger_num, _ = self.cur_states.shape
            self.len = num_envs * length
        else:
            length, finger_num, _ = self.cur_states.shape 
            self.len = length
        self.cur_states = self.cur_states.view(self.len, -1)        # action
        self.actions = self.actions.view(self.len, -1)        # action
        self.next_states = self.next_states.view(self.len, -1)    # fingertip pose

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return torch.cat((self.cur_states[index], self.next_states[index]), dim=-1), self.actions[index]



if __name__ == '__main__':
    dataset = PoseDataset('data/hand_data/MyoHand/myohand_sequence_train_Jul_28_20:00:24_2024')
    print(dataset[0].shape)
    # data = torch.from_numpy(np.load(os.path.join('/home/wutong/dexterity/G-Dex/data/train_allegro_Jan_25_17:00:47_2024', 'actions.npy')))
    # labels = torch.from_numpy(np.load(os.path.join('/home/wutong/dexterity/G-Dex/data/train_allegro_Jan_25_17:00:47_2024', 'fingertip_poses.npy')))
    dataloader = DataLoader(dataset, batch_size=32)