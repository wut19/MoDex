import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from myosuite.utils import gym
import isaacgym
from modexenvs.inhand_manipulation import *

from algs.CEM import MPCRollout

from models import NNDynModel
from models.dataset.dataset import InhandDynamicDataset, DataProcessor

from omegaconf import OmegaConf
import pickle

class MeanStd:

    def __init__(self):
        self.obj_mean = None
        self.obj_std = None
        self.hand_mean = None
        self.hand_std = None


def eval(visualize=True):
    exp_num = 500

    # cfg_file = 'models/cfg/inhand_mani/reorientationtarget2hand.yaml'
    # pickle_file = 'inhand_logs/reorientation_hand_no_freeze_1/training_data/normalization_data_6.pickle'
    # dyn_file = 'inhand_logs/reorientation_hand_no_freeze_1/models/dynamics_model_iter6.pt'

    # cfg_file = 'models/cfg/inhand_mani/reorientation8_hand.yaml'
    # pickle_file = 'inhand_logs/reorientation8_hand_no_freeze/training_data/normalization_data_18.pickle'
    # dyn_file = 'inhand_logs/reorientation8_hand_no_freeze/models/dynamics_model_iter18.pt'

    cfg_file = 'models/cfg/inhand_mani/reorientation100_hand.yaml'
    pickle_file = 'inhand_logs/reorientation100_hand_no_freeze_Aug_27_13:37:17_2024/training_data/normalization_data_25.pickle'
    dyn_file = 'inhand_logs/reorientation100_hand_no_freeze_Aug_27_13:37:17_2024/models/dynamics_model_iter25.pt'

    cfg = OmegaConf.load(cfg_file)

    obj_data_type = cfg.obj_data_type
    env_name = cfg.env_name
    seq_len = cfg.seq_len

    env = gym.make(env_name)
    ## load data processor for observation normalization
    params = {'train_eps_num': 400}
    data_processor = DataProcessor(params)
    with open(pickle_file,'rb') as f:
        mean_std = pickle.load(f)
    data_processor.obj_data_mean = mean_std.obj_mean
    data_processor.obj_data_std = mean_std.obj_std
    data_processor.hand_data_mean = mean_std.hand_mean
    data_processor.hand_data_std = mean_std.hand_std

    ## load dynamics model
    
    dyn_model = NNDynModel(cfg, data_processor)
    dyn_model.load_model(dyn_file)
    dyn_model.net.eval()
    mpc_rollout = MPCRollout(env, dyn_model, plan_step = cfg.horizon, max_iters=cfg.max_iters, sample_num=cfg.sample_num, num_elites=cfg.num_elites, alpha=cfg.alpha)

    out_dir = './results/inhand_mani/reorient_100_hand'
    os.makedirs(out_dir, exist_ok=True)
    for i in range(exp_num):
        env.reset()
        done  = False
        solved = False
        while not done:
            obj_pos = env.get_obs_dict(env.sim)['obj_pos']
            obj_rot = env.get_obs_dict(env.sim)['obj_rot']
            obj_state = np.concatenate([obj_pos, obj_rot], axis=-1)
            hand_state = env.get_obs_dict(env.sim)['tip_pos'].reshape(-1,3)
            obs = {'obj_cur_data': obj_state, 'hand_cur_data': hand_state}
            action = mpc_rollout.predict(obs)
            _, r, d, t, info = env.step(action)
            env.render_offscreen()
            done = d or t or info['solved']
            sovled = info['solved']
        if sovled:
            env.write_video(f'{out_dir}/{i}_solved.mp4')
        else:
            env.write_video(f'{out_dir}/{i}_failed.mp4')

if __name__ == "__main__":
    eval()