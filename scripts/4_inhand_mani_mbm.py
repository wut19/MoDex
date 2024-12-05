import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from myosuite.utils import gym
import isaacgym
from modexenvs.inhand_manipulation import *

import numpy as np
import torch
import pickle
from omegaconf import OmegaConf

from algs.RandPolicy import RandPolicy
from algs.CEM import MPCRollout

from models import NNDynModel

from models.dataset.dataset import InhandDynamicDataset, DataProcessor
from datacollector.data_collector import InHandDynamicDataCollector

import matplotlib.pyplot as plt
import random

class MeanStd:

    def __init__(self):
        self.obj_mean = None
        self.obj_std = None
        self.hand_mean = None
        self.hand_std = None

class DataPerIter:

    def __init__(self):
        self.iter_num = 0
        self.rollouts_info = []
        self.train_rollouts_MPC = []
        self.val_rollouts_MPC = []

        self.normalization_data = MeanStd()

        self.eval_losses = []
        self.eval_errors = []
        self.training_numData = []
        self.rollouts_rewardsPerIter = []
        self.rollouts_successPerIter = []

class Loader:

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def load_initialData(self):

        rollouts_trainRand = pickle.load(
            open(self.save_dir + '/training_data/train_rollouts_rand.pickle', 'rb'))
        rollouts_valRand = pickle.load(
            open(self.save_dir + '/training_data/val_rollouts_rand.pickle',
                 'rb'))

        return rollouts_trainRand, rollouts_valRand

    def load_iter(self, iter_num):

        data_iteration = DataPerIter()

        #info from all MPC rollouts (from this iteration)
        data_iteration.rollouts_info = pickle.load(
            open(
                self.save_dir + '/saved_rollouts/rollouts_info_' + str(iter_num)
                + '.pickle', 'rb'))

        #on-policy data (used in conjunction w random data) to train the dynamics model at this iteration
        data_iteration.train_rollouts_onPol = pickle.load(
            open(
                self.save_dir + '/training_data/train_rollouts_onPol_iter' +
                str(iter_num) + '.pickle', 'rb'))
        data_iteration.val_rollouts_onPol = pickle.load(
            open(
                self.save_dir + '/training_data/val_rollouts_onPol_iter' +
                str(iter_num) + '.pickle', 'rb'))

        #mean/std info
        data_iteration.normalization_data = pickle.load(
            open(
                self.save_dir + '/training_data/normalization_data_' +
                str(iter_num) + '.pickle', 'rb'))

        #losses/rewards/scores/sample complexity (from all iterations thus far)
        data_iteration.training_losses = np.load(
            self.save_dir +
            '/losses/list_training_loss.npy').tolist()[:iter_num]
        data_iteration.training_numData = np.load(
            self.save_dir + '/datapoints_per_agg.npy').tolist()[:iter_num]
        data_iteration.rollouts_rewardsPerIter = np.load(
            self.save_dir + '/rollouts_rewardsPerIter.npy').tolist()[:iter_num]
        data_iteration.rollouts_scoresPerIter = np.load(
            self.save_dir + '/rollouts_scoresPerIter.npy').tolist()[:iter_num]

        return data_iteration

class Saver:

    def __init__(self, save_dir):

        ### init vars
        self.iter_num = -1
        self.save_dir = save_dir

        ### make directories for saving data
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir + '/losses', exist_ok=True)
        os.makedirs(self.save_dir + '/models', exist_ok=True)
        os.makedirs(self.save_dir + '/optims', exist_ok=True)
        os.makedirs(self.save_dir + '/saved_rollouts', exist_ok=True)
        os.makedirs(self.save_dir + '/training_data', exist_ok=True)

    def save_initialData(self, params, rollouts_trainRand, rollouts_valRand):

        pickle.dump(
            rollouts_trainRand,
            open(self.save_dir + '/training_data/train_rollouts_rand.pickle', 'wb'),
            pickle.HIGHEST_PROTOCOL)
        pickle.dump(
            rollouts_valRand,
            open(self.save_dir + '/training_data/val_rollouts_rand.pickle',
                 'wb'), pickle.HIGHEST_PROTOCOL)
        # pickle.dump(params, open(self.save_dir + '/params.pkl', 'wb'),
        #             pickle.HIGHEST_PROTOCOL)
        
    def save_model(self, model, iter_num):
        if self.iter_num==-1:
            raise ValueError("MUST SPECIFY ITER_NUM FOR SAVER...")

        torch.save(
            model.net.state_dict(),
            self.save_dir + '/models/dynamics_model_iter' + str(iter_num) +
            '.pt')
        
        torch.save(model.optimizer.state_dict(), self.save_dir + '/optims/dynamics_model_iter' + str(iter_num) +
            '.pt')

    ############################################################################################
    ##### The following 2 saves together represent a single "iteration"
    ########## (train model) + (collect new rollouts with that model) = a single "iteration"
    ############################################################################################

    def save_training_info(self, save_data):

        if self.iter_num==-1:
            raise ValueError("MUST SPECIFY ITER_NUM FOR SAVER...")

        #on-policy training data (used in conjunction w random training data) to train the dynamics model at this iteration
        pickle.dump(
            save_data.train_rollouts_MPC,
            open(
                self.save_dir + '/training_data/train_rollouts_MPC_iter' +
                str(self.iter_num) + '.pickle', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(
            save_data.val_rollouts_MPC,
            open(
                self.save_dir + '/training_data/val_rollouts_MPC_iter' + str(
                    self.iter_num) + '.pickle', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)

        #mean/std info
        pickle.dump(
            save_data.normalization_data,
            open(
                self.save_dir + '/training_data/normalization_data_' + str(
                    self.iter_num) + '.pickle', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)

        #losses and sample complexity
        np.save(self.save_dir + '/losses/list_eval_loss.npy',
                save_data.eval_losses)
        np.save(self.save_dir + '/datapoints_per_agg.npy',
                save_data.training_numData)

        # #train/val losses from model training
        # np.save(self.save_dir + '/losses/training_losses_iter' + str(self.iter_num)
        #         + '.npy', save_data.training_lists_to_save['training_loss_list'])
        # np.save(self.save_dir + '/losses/validation_losses_iter' + str(self.iter_num)
        #         + '.npy', save_data.training_lists_to_save['val_loss_list_rand'])
        # np.save(self.save_dir + '/losses/validation_losses_xaxis_iter' + str(self.iter_num)
        #         + '.npy', save_data.training_lists_to_save['val_loss_list_xaxis'])
        # np.save(self.save_dir + '/losses/validation_onPol_losses_iter' + str(self.iter_num)
        #         + '.npy', save_data.training_lists_to_save['val_loss_list_onPol'])
        # np.save(self.save_dir + '/losses/old_losses_iter' + str(self.iter_num)
        #         + '.npy', save_data.training_lists_to_save['rand_loss_list'])
        # np.save(self.save_dir + '/losses/new_losses_iter' + str(self.iter_num)
        #         + '.npy', save_data.training_lists_to_save['onPol_loss_list'])

    def save_rollout_info(self, save_data):

        if self.iter_num==-1:
            raise ValueError("MUST SPECIFY ITER_NUM FOR SAVER...")

        # #info from all MPC rollouts (from this iteration)
        # pickle.dump(
        #     save_data.rollouts_info,
        #     open(
        #         self.save_dir + '/saved_rollouts/rollouts_info_' + str(
        #             self.iter_num) + '.pickle', 'wb'),
        #     protocol=pickle.HIGHEST_PROTOCOL)

        #save rewards and scores (for rollouts from all iterations thus far)
        np.save(self.save_dir + '/rollouts_rewardsPerIter.npy',
                save_data.rollouts_rewardsPerIter)
        np.save(self.save_dir + '/rollouts_succssPerIter.npy',
                save_data.rollouts_successPerIter)

        #plot rewards and scores (for rollouts from all iterations thus far)
        rew = np.array(save_data.rollouts_rewardsPerIter)
        success = np.array(save_data.rollouts_successPerIter)

        plot_mean_std(rew[:, 0], rew[:, 1], self.save_dir + '/rewards_perIter')
        plot_mean_std(success[:, 0], success[:, 1],
                      self.save_dir + '/success_perIter')

        self.iter_num = -1

def plot_mean_std(mean_data, std_data, filename=None, label=None, newfig=True, color='b'):

    if newfig:
        fig, ax = plt.subplots(1, figsize=(5, 10))
        xvals = np.arange(len(mean_data))
        if label is None:
            ax.plot(xvals, mean_data, color=color)
        else:
            ax.plot(xvals, mean_data, color=color, label=label)
            ax.legend()
        ax.fill_between(
            xvals,
            mean_data - std_data,
            mean_data + std_data,
            color=color,
            alpha=0.25)
        if filename is not None:
            fig.savefig(filename + '.png', dpi=200, bbox_inches='tight')
    else:
        xvals = np.arange(len(mean_data))
        if label is None:
            plt.plot(xvals, mean_data, color=color)
        else:
            plt.plot(xvals, mean_data, color=color, label=label)
            plt.legend()
        plt.fill_between(
            xvals,
            mean_data - std_data,
            mean_data + std_data,
            color=color,
            alpha=0.25)
        if filename is not None:
            fig.savefig(filename + '.png', dpi=200, bbox_inches='tight')
    plt.close()

def set_all_seed(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    env.seed(seed)
    env.action_space.seed(seed)

def learn(cfg):

    # env_name = 'myoHandReorientTarget2Custom'
    # cfg_file = 'models/cfg/inhand_mani/reorientationtarget2.yaml'
    # obj_data_type = 'obj_pos_rot'

    # env_name = 'myoHandReorientTarget2Custom'
    # cfg_file = 'models/cfg/inhand_mani/reorientationtarget2hand.yaml'
    # obj_data_type = 'obj_pos_rot'

    # params
    seed = cfg.seed
    obj_data_type = cfg.obj_data_type
    env_name = cfg.env_name
    num_iters = cfg.num_iters
    rollout_min_len = cfg.rollout_min_len
    num_trajectory_per_iter = cfg.num_trajectory_per_iter
    num_rand_trajectory_per_iter = cfg.num_rand_trajectory_per_iter
    num_rand_rollouts = cfg.num_rand_rollouts
    seq_len = cfg.seq_len
    
    env = gym.make(env_name)

    # set seed
    set_all_seed(env, seed)

    # create random policy and data collector
    random_policy = RandPolicy(env)
    data_collector = InHandDynamicDataCollector(env, eps_num=num_trajectory_per_iter, eps_len=rollout_min_len, split=0.9, obj_data_type = obj_data_type) # TODO: modify the args
    data_collector_rand = InHandDynamicDataCollector(env, eps_num=num_rand_rollouts, eps_len=rollout_min_len, split=0.9, obj_data_type = obj_data_type) # TODO: modify the args

    # initialize dataprocesor 
    params = {'train_eps_num': 400}
    data_processor = DataProcessor(params)

    # start a new run
    print("\n#####################################")
    print("Collecting initial random rollouts..... ")
    print("#####################################\n")
    data_collector.reset()
    rollouts_randTrain, rollouts_randVal = data_collector_rand.collect_rollout(policy=random_policy)

    # MPC rollouts initialization
    rollouts_MPCTrain = rollouts_randTrain
    rollouts_MPCVal = rollouts_randVal

    # lists for logging
    evalloss_perIter = []
    error_perIter = []
    rew_perIter = []
    success_perIter = []
    traindata_perIter = []

    # initialize counter
    counter = 0

    # initialize model and mpc rollout
    dyn_model = NNDynModel(cfg, data_processor)
    dyn_model.prepare(cfg)
    mpc_rollout = MPCRollout(env, dyn_model, plan_step = cfg.horizon, max_iters=cfg.max_iters, sample_num=cfg.sample_num, num_elites=cfg.num_elites, alpha=cfg.alpha)

    # saver initialize
    saver = Saver(dyn_model.log_path)
    saver.save_initialData(None, rollouts_randTrain, rollouts_randVal)

    # training loop

    while counter < num_iters:
        # init vars for this iteration
        saver_data = DataPerIter()
        saver.iter_num = counter
        numData_train_MPC = 0
        numData_train_rand = 0
        numEps_train_MPC = len(rollouts_MPCTrain['action_data'])
        numEps_train_rand = len(rollouts_randTrain['action_data'])
        for eps in rollouts_MPCTrain['action_data']:
            numData_train_MPC += eps.shape[0]
        for eps in rollouts_randTrain['action_data']:
            numData_train_rand += eps.shape[0]

        # numData_train_MPC = rollouts_MPCTrain['action_data'].shape[0] * rollouts_MPCTrain['action_data'].shape[1]
        # numData_train_rand = rollouts_randTrain['action_data'].shape[0] * rollouts_randTrain['action_data'].shape[1]

        rollouts_allTrain = dyn_model.data_processor.combine_data(rollouts_MPCTrain, rollouts_randTrain)
        rollouts_allTrain_norm = dyn_model.data_processor.preprocess_data(rollouts_allTrain)
        # convert rollouts to dataset
        # rollouts_MPCTrain_norm = dyn_model.data_processor.get_normalized_dataset(rollouts_MPCTrain)
        params_MPCval = {'val_eps_num': 40}
        rollouts_MPCVal_norm = dyn_model.data_processor.get_normalized_dataset(rollouts_MPCVal, params_MPCval)

        dataset_allTrain = InhandDynamicDataset(rollouts_allTrain_norm , seq_len=seq_len)
        dataset_MPCVal = InhandDynamicDataset(rollouts_MPCVal_norm, seq_len=seq_len)
        dyn_model.get_data(dataset_allTrain, dataset_MPCVal, cfg)

        # train model
        print("\n#####################################")
        print("Training the dynamics model..... iteration ", counter)
        print("#####################################\n")
        print(f"    amount of random data: {numData_train_rand}, amount of random episodes: {numEps_train_rand}")
        print(f"    amount of onPol data: {numData_train_MPC}, amount of onPol episodes: {numEps_train_MPC}")

        loss, error = dyn_model.fit(epochs=cfg.epochs)

        # collect new rollouts with the model
        print("\n#####################################")
        print("performing on-policy MPC rollouts... iter ", counter)
        print("#####################################\n")
        
        data_collector.reset(eps_num=num_trajectory_per_iter)
        new_rollouts_MPCTrain, new_rollouts_MPCVal = data_collector.collect_rollout(policy=mpc_rollout)
        data_collector_rand.reset(eps_num=num_rand_trajectory_per_iter)
        new_rollouts_randTrain, new_rollouts_randVal = data_collector_rand.collect_rollout(policy=random_policy)

        # before combining, save the old data
        saver_data.train_rollouts_MPC = rollouts_MPCTrain
        saver_data.val_rollouts_MPC = rollouts_MPCVal
        mean_std = MeanStd()
        mean_std.obj_mean = dyn_model.data_processor.obj_data_mean
        mean_std.obj_std = dyn_model.data_processor.obj_data_std
        mean_std.hand_mean = dyn_model.data_processor.hand_data_mean
        mean_std.hand_std = dyn_model.data_processor.hand_data_std
        saver_data.normalization_data = mean_std

        # combine new rollouts with old rollouts
        rollouts_MPCTrain = dyn_model.data_processor.combine_data(rollouts_MPCTrain, new_rollouts_MPCTrain)
        rollouts_MPCVal = dyn_model.data_processor.combine_data(rollouts_MPCVal, new_rollouts_MPCVal)
        rollouts_randTrain = dyn_model.data_processor.combine_data(rollouts_randTrain, new_rollouts_randTrain)
        rollouts_randVal = dyn_model.data_processor.combine_data(rollouts_randVal, new_rollouts_randVal)

        # save outputs
        evalloss_perIter.append(loss)
        error_perIter.append(error)
        traindata_perIter.append(numData_train_rand + numData_train_MPC)
        saver_data.eval_losses = evalloss_perIter
        saver_data.eval_errors = error_perIter
        saver_data.training_numData = traindata_perIter
        
        saver.save_model(dyn_model, counter)
        saver.save_training_info(saver_data)

        # append onto rewards/successes
        new_rollouts_eps_reward = np.array([r.sum() for r in new_rollouts_MPCTrain['reward']])
        rew_perIter.append([new_rollouts_eps_reward.mean(), new_rollouts_eps_reward.std()])
        new_rollouts_eps_success = np.array([solved.max() for solved in new_rollouts_MPCTrain['solved']])
        success_perIter.append([new_rollouts_eps_success.mean(), 0.])
        print("\n#####################################")
        print(f"Reward: {new_rollouts_eps_reward.mean()}, Episode length: {numData_train_MPC/numEps_train_MPC}, Success Rate:{new_rollouts_eps_success.mean()}... iter {counter}"), 
        print("#####################################\n")

        # save rewards and success
        saver_data.rollouts_rewardsPerIter = rew_perIter
        saver_data.rollouts_successPerIter = success_perIter
        saver.save_rollout_info(saver_data)

        counter += 1

if __name__ == '__main__':
    # cfg = 'models/cfg/inhand_mani/reorientationtarget2.yaml'
    cfg = 'models/cfg/inhand_mani/reorientationtarget2hand.yaml'
    # cfg = 'models/cfg/inhand_mani/reorientation8.yaml'
    # cfg = 'models/cfg/inhand_mani/reorientation8_hand.yaml'
    # cfg = 'models/cfg/inhand_mani/reorientation100.yaml'
    # cfg = 'models/cfg/inhand_mani/reorientation100_hand.yaml'
    cfg = OmegaConf.load(cfg)
    learn(cfg)