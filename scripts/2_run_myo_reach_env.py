import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import isaacgym
import myosuite
from myosuite.utils import gym
from modexenvs.reach_envs import *
from models import Trainer

from algs.RS import RandomShootingMJFixed, RandomShootingMJSeq
from stable_baselines3 import PPO
from algs.CEM import CEMMJFixed, CEMMJSeq
from algs.BGD import BGDMJFixed, BGDMJSeq

from omegaconf import OmegaConf
import time

# set random seed
np.random.seed(0)

####### One Step Experiments #######

# env = gym.make('myoHandReachOneStep')
# env_val = gym.make('myoHandReachOneStep')
# o = env.reset()


# exp_num = 100
# success = 0
# reach_error = 0
# t = 0

# # # Random shooting policy
# # for _ in range(exp_num):
# #     rs_policy = RandomShootingMJFixed(env, num_shoots=100)
# #     start_t = time.time()
# #     action, r = rs_policy.step()
# #     end_t = time.time()
# #     # print("Best action: ", action)
# #     solved, r, info = rs_policy.eval(action)
# #     reach_err = np.linalg.norm(info['obs_dict']['reach_err'])/5
# #     reach_error += reach_err
# #     print("="*5, f' solved: {solved}, reward: {r}, reach_error: {reach_err}, planning time: {end_t-start_t}', "="*5)
# #     t += end_t-start_t
# #     if solved:
# #         success += 1

# # Model-free Reinforcement learning policy
# # model = PPO.load('logs/ppo/PPO_1/ppo_reach.pth', print_system_info=True)
# # for i in range(exp_num):
# #     idx = env.np_random.integers(0, env.reach_space.shape[0])
# #     obs, _ = env.reset(idx=idx)
# #     start_t = time.time()
# #     action, _states = model.predict(obs, deterministic=True)
# #     end_t = time.time()
# #     _, reward, _, _, info = env.step(action)
# #     solved = info['solved']
# #     reach_err = np.linalg.norm(info['obs_dict']['reach_err'])/5
# #     reach_error += reach_err
# #     t += end_t-start_t
# #     print("="*5, f' solved: {solved}, reward: {reward}, reach_error: {reach_err}, planning time: {end_t-start_t}', "="*5)
# #     if solved:
# #         success += 1

# # # # Model-based Reinforcement learning policy
# # # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/ppo/")
# # # model.learn(total_timesteps=1000000)

# # Model-based Planning 
# # instance and load the model
# cfg = OmegaConf.load('models/cfg/hand_model_train/myohand_forward_model.yaml')
# forward_model = Trainer(cfg)
# forward_model.load_model('logs/models/myohand_forward_model_Jun_20_14:23:01_2024/ckpt-1.pt')
# forward_model.net.eval()
# forward_model.net.to('cuda:1')

# # if need, load the inverse model
# cfg = OmegaConf.load('models/cfg/inverse_model_train/myohand_inverse_model.yaml')
# inverse_model = Trainer(cfg)
# inverse_model.load_model('logs/models/myohand_inverse_model_Jul_18_17:00:16_2024/ckpt-1.pt')
# inverse_model.net.to('cuda:1')

# # # Random shooting  
# # for _ in range(exp_num):
# #     rs_policy = RandomShootingMJFixed(env, model=forward_model, num_shoots=10000)
# #     start_t = time.time()
# #     action, r = rs_policy.step()
# #     end_t = time.time()
# #     # print("Best action: ", action)
# #     solved, r, info = rs_policy.eval(action)
# #     reach_err = np.linalg.norm(info['obs_dict']['reach_err'])/5
# #     reach_error += reach_err
# #     print("="*5, f' solved: {solved}, reward: {r}, reach_error: {reach_err}, planning time: {end_t-start_t}', "="*5)
# #     t += end_t-start_t
# #     if solved:
# #         success += 1

# # CEM
# cem_policy = CEMMJFixed(
#                 env, 
#                 forward_model=forward_model, 
#                 inverse_model=inverse_model,
#                 max_iters=2,
#                 sample_num=800, 
#                 num_elites=20, 
#                 verbose=False
# )

# # path = 'results/reach/myohand_qs'
# # os.makedirs(path, exist_ok=True)

# for i in range(exp_num):
#     # planning and evaluation
#     start_t = time.time()
#     action, plan_time = cem_policy.step()
#     end_t = time.time()
#     # solved, r, info = cem_policy.eval(env_val,action, render=True, save_path=os.path.join(path,f'{i}.mp4'))
#     solved, r, info = cem_policy.eval(env_val,action)
#     # print(info)
#     reach_err = np.linalg.norm(info['obs_dict']['reach_err'])/5
#     reach_error += reach_err
#     print("="*5, f' solved: {solved}, reward: {r}, reach_error: {reach_err}, planning time: {plan_time}', "="*5)
#     t += end_t-start_t
#     if solved:
#         success += 1

# # # BGD
# # bgd_policy = BGDMJFixed(env, forward_model=forward_model, learning_rate=1e-2, max_iters=100)
# # for _ in range(exp_num):
# #     # planning and evaluation
# #     start_t = time.time()
# #     action = bgd_policy.step()
# #     end_t = time.time()
# #     solved, r, info = bgd_policy.eval(action)
# #     reach_err = np.linalg.norm(info['obs_dict']['reach_err'])/5
# #     reach_error += reach_err
# #     print("="*5, f' solved: {solved}, reward: {r}, reach_error: {reach_err}, planning time: {end_t-start_t}', "="*5)
# #     t += end_t-start_t
# #     if solved:
# #         success += 1


# print('-'*10)
# print(f"Success rate: {success}/{exp_num}")
# print(f'Average planning time: {t/exp_num}')
# print(f'Average reach error: {reach_error/exp_num}')
# print('-'*10)



# env.close()

###### Sequence Experiments #######
env = gym.make('myoHandReachSequence')
env_val = gym.make('myoHandReachSequenceVal')
o = env.reset()

exp_num = 100
success = 0
reach_error = 0
t = 0

# # # Random shooting policy
# # rs_policy = RandomShootingMJSeq(env, num_shoots=500, num_steps=50)
# # start_t = time.time()
# # action, r = rs_policy.step()
# # end_t = time.time()
# # # print("Best action: ", action)
# # solved, r = rs_policy.eval(action)
# # print(solved, r)
# # t = end_t-start_t
# # print(f'time: {t}')

# # # Model-free Reinforcement learning policy
# # model = PPO.load('logs/ppo_seq/PPO_2/ppo_reach.pth', print_system_info=True)
# # for i in range(exp_num):
# #     idx = env.np_random.integers(0, env.reach_space.shape[0])
# #     obs, _ = env.reset(idx=idx)
# #     start_t = time.time()
# #     action, _states = model.predict(obs, deterministic=True)
# #     end_t = time.time()
# #     _, reward, _, _, info = env.step(action)
# #     solved = info['solved']
# #     reach_err = np.linalg.norm(info['obs_dict']['reach_err'])/5
# #     reach_error += reach_err
# #     t += end_t-start_t
# #     print("="*5, f' solved: {solved}, reward: {reward}, reach_error: {reach_err}, planning time: {end_t-start_t}', "="*5)
# #     if solved:
# #         success += 1

# Model-based planning
# instance and load the model
cfg = OmegaConf.load('models/cfg/hand_model_train/myohand_seq_forward_model.yaml')
forward_model = Trainer(cfg)
forward_model.load_model('logs/models/myohand_seq_forward_model_Jul_29_15:01:25_2024/ckpt-1.pt')
forward_model.net.eval()
forward_model.net.to('cuda:1')

# instance and load inverse model
cfg = OmegaConf.load('models/cfg/inverse_model_train/myohand_seq_inverse_model.yaml')
inverse_model = Trainer(cfg)
inverse_model.load_model('logs/models/myohand_seq_inverse_model_Jul_30_20:56:24_2024/ckpt-1.pt')
inverse_model.net.eval()
inverse_model.net.to('cuda:1')

# # Random shooting policy
# rs_policy = RandomShootingMJSeq(env, model=forward_model, num_shoots=50000, num_steps=10)
# for i in range(exp_num):
#     start_t = time.time()
#     action, _ = rs_policy.step()
#     end_t = time.time()
#     solved, r, info = rs_policy.eval(action)
#     # print(info)
#     reach_err = np.linalg.norm(info['obs_dict']['reach_err'])/5
#     reach_error += reach_err
#     print("="*5, f' solved: {solved}, reward: {r}, reach_error: {reach_err}, planning time: {end_t-start_t}', "="*5)
#     t += end_t-start_t
#     if solved:
#         success += 1

# # BGD
# bgd_policy = BGDMJSeq(env, forward_model=forward_model, max_timesteps=10)

# for i in range(exp_num):
#     start_t = time.time()
#     action = bgd_policy.step()
#     end_t = time.time()
#     solved, r, info = bgd_policy.eval(action)
#     # print(info)
#     reach_err = np.linalg.norm(info['obs_dict']['reach_err'])/5
#     reach_error += reach_err
#     print("="*5, f' solved: {solved}, reward: {r}, reach_error: {reach_err}, planning time: {end_t-start_t}', "="*5)
#     t += end_t-start_t
#     if solved:
#         success += 1

# CEM
cem_policy = CEMMJSeq(
                env, 
                forward_model=forward_model, 
                inverse_model=inverse_model, 
                sample_num=800, 
                num_elites=20, 
                max_iters=2,
                max_timesteps=10,
                # verbose=True,
)
# path = 'results/reach/myohand'
# os.makedirs(path, exist_ok=True)
for i in range(exp_num):
    start_t = time.time()
    action = cem_policy.step()
    end_t = time.time()
    # solved, r, info = cem_policy.eval(env_val, action,render=True, save_path=os.path.join(path,f'{i}.mp4'))
    solved, r, info = cem_policy.eval(env_val, action)
    # print(info)
    reach_err = np.linalg.norm(info['obs_dict']['reach_err'])/5
    reach_error += reach_err
    print("="*5, f' solved: {solved}, reward: {r}, reach_error: {reach_err}, planning time: {end_t-start_t}', "="*5)
    t += end_t-start_t
    if solved:
        success += 1

print('-'*10)
print(f"Success rate: {success}/{exp_num}")
print(f'Average planning time: {t/exp_num}')
print(f'Average reach error: {reach_error/exp_num}')
print('-'*10)

env.close()