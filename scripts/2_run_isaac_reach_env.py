import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import isaacgym
import gym
import modexenvs
from modexenvs import *
from models import Trainer

from algs.RS import RandomShootingIsaacSeq
from algs.CEM import CEMIsaacSeq
from algs.BGD import BGDIsaacSeq

from omegaconf import OmegaConf
import time

import numpy as np

# set random seed
np.random.seed(0)

reach_space_paths = {
    'ShadowHand': 'data_/hand_data/ShadowHand/ShadowHand_random_Jun_27_11:20:56_2024',
    'AllegroHand': 'data_/hand_data/AllegroHand/AllegroHand_random_train_Jul_23_14:27:07_2024',
    'Robotiq3Finger': 'data_/hand_data/Robotiq3Finger/Robotiq_random_train_Jul_23_21:24:03_2024',
}

# task = 'ShadowHand' 
# task = 'AllegroHand'
task = 'Robotiq3Finger'
exp_num = 50
success = 0
reach_error = 0
t = 0

###### Sequence Experiments #######
env = modexenvs.make(
        seed=0, 
        task=task, 
        num_envs=1, 
        sim_device='cuda:0',
        rl_device='cuda:0',
        graphics_device_id=0,
        # virtual_screen_capture=True,
        # force_render=False,
    )

# env.is_vector_env = True
# env = gym.wrappers.RecordVideo(
#         env,
#         "./results/reach/robotiq_seq",
#         step_trigger=lambda step: step % 150 == 0, # record the videos every 10000 steps
#         video_length=149 # for each video record up to 100 steps
# )


# Model-based planning
# instance and load the model
# cfg = OmegaConf.load('models/cfg/hand_model_train/shadowhand_seq_forward_model.yaml')
# cfg = OmegaConf.load('models/cfg/hand_model_train/allegro_seq_forward_model.yaml')
cfg = OmegaConf.load('models/cfg/hand_model_train/robotiq_seq_forward_model.yaml')
forward_model = Trainer(cfg)
# forward_model.load_model('logs/models/shadowhand_seq_forward_model_Jul_31_15:12:42_2024/ckpt-1.pt')
# forward_model.load_model('logs/models/allegro_seq_forward_model_Jul_31_16:08:14_2024/ckpt-1.pt')
forward_model.load_model('logs/models/robotiq_seq_forward_model_Jul_31_16:26:35_2024/ckpt-1.pt')
forward_model.net.eval()

# # instance and load the inverse model
# cfg = OmegaConf.load('models/cfg/inverse_model_train/shadowhand_seq_inverse_model.yaml')
# # cfg = OmegaConf.load('models/cfg/inverse_model_train/allegro_seq_inverse_model.yaml')
# # cfg = OmegaConf.load('models/cfg/inverse_model_train/robotiq_seq_inverse_model.yaml')
# inverse_model = Trainer(cfg)
# inverse_model.load_model('logs/models/shadowhand_seq_inverse_model_Jul_31_15:33:39_2024/ckpt-1.pt')
# init_var = 0.5
# # inverse_model.load_model('logs/models/allegro_seq_inverse_model_Jul_31_16:11:48_2024/ckpt-1.pt')
# # init_var = 0.25
# # inverse_model.load_model('logs/models/robotiq_seq_inverse_model_Jul_31_16:28:16_2024/ckpt-1.pt')
# # init_var = 0.4
# inverse_model.net.eval()


# # Random shooting policy
# rs_policy = RandomShootingIsaacSeq(env, env_name=task, model=forward_model, num_shoots=1000, num_steps=10, reach_space_paths=reach_space_paths)
# for i in range(exp_num):
#     start_t = time.time()
#     actions, _ = rs_policy.step()
#     end_t = time.time()
#     solved, reach_err = rs_policy.eval(actions)
#     reach_error += reach_err
#     print("="*5, f' solved: {solved}, reach_error: {reach_err}, planning time: {end_t-start_t}', "="*5)
#     t += end_t-start_t
#     if solved:
#         success += 1

# # BGD
# bgd_policy = BGDIsaacSeq(
#                 env, 
#                 env_name=task,
#                 forward_model=forward_model, 
#                 max_timesteps=10,
#                 max_iters=100,
#                 learning_rate=0.01,      
#                reach_space_paths=reach_space_paths,          
# )

# for i in range(exp_num):
#     start_t = time.time()
#     _, solved, reach_err = bgd_policy.step()
#     end_t = time.time()
#     reach_error += reach_err
#     print("="*5, f' solved: {solved}, reach_error: {reach_err}, planning time: {end_t-start_t}', "="*5)
#     t += end_t-start_t
#     if solved:
#         success += 1

# CEM
cem_policy = CEMIsaacSeq(
                env, 
                env_name=task,
                forward_model=forward_model, 
                # inverse_model=inverse_model, 
                # init_var=init_var,
                sample_num=100, 
                num_elites=20, 
                max_timesteps=10,
                verbose=False,
                reach_space_paths=reach_space_paths,
)

# path = 'results/reach/robotiq_seq'
# os.makedirs(path, exist_ok=True)

for i in range(exp_num):
    start_t = time.time()
    _, solved, reach_err = cem_policy.step(video_path=None)
    end_t = time.time()
    reach_error += reach_err
    print("="*5, f' solved: {solved}, reach_error: {reach_err}, planning time: {end_t-start_t}', "="*5)
    t += end_t-start_t
    if solved:
        success += 1


print('-'*10)
print(f"Success rate: {success}/{exp_num}")
print(f'Average planning time: {t/exp_num}')
print(f'Average reach error: {reach_error/exp_num}')
print('-'*10)
