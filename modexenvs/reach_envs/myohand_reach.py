import collections
from myosuite.utils import gym
import numpy as np
import torch

from myosuite.envs.myo.base_v0 import BaseV0
from datacollector.utils import compute_poses_wrt_world
from scipy.spatial.transform import Rotation as R
import os
import cv2

class MyoHandReach(BaseV0):

    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'tip_pos', 'reach_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "bonus": 4.0,
        "penalty": 50,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)

        self._setup(**kwargs)


    def _setup(self,
            target_reach_range:dict,
            reach_space:str,
            far_th = .35,
            near_th = .0125,
            init_cond = 'fixed',
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.far_th = far_th
        self.near_th = near_th
        self.init_cond = init_cond
        self.target_reach_range = target_reach_range
        self.reach_space = np.load(reach_space)

        self.num_fingertips = 5
        self.base_name = 'Elbow_PT_ECRL_site_ECRL_side'

        super()._setup(obs_keys=obs_keys,
                weighted_reward_keys=weighted_reward_keys,
                sites=self.target_reach_range.keys(),
                **kwargs,
                )
        self.viewer_setup(azimuth=180, distance=1, lookat=[-0.15,-0.5,1.3], render_actuator=True)


    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.sim.data.qvel[:].copy()*self.dt
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        # reach error
        self.obs_dict['tip_pos'] = np.array([])
        self.obs_dict['target_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            self.obs_dict['tip_pos'] = np.append(self.obs_dict['tip_pos'], self.sim.data.site_xpos[self.tip_sids[isite]].copy())
            self.obs_dict['target_pos'] = np.append(self.obs_dict['target_pos'], self.sim.data.site_xpos[self.target_sids[isite]].copy())
        self.obs_dict['reach_err'] = np.array(self.obs_dict['target_pos'])-np.array(self.obs_dict['tip_pos'])

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['qpos'] = sim.data.qpos[:].copy()
        obs_dict['qvel'] = sim.data.qvel[:].copy()*self.dt
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()

        # reach error
        obs_dict['tip_pos'] = np.array([])
        obs_dict['target_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            obs_dict['tip_pos'] = np.append(obs_dict['tip_pos'], sim.data.site_xpos[self.tip_sids[isite]].copy())
            obs_dict['target_pos'] = np.append(obs_dict['target_pos'], sim.data.site_xpos[self.target_sids[isite]].copy())
        obs_dict['reach_err'] = np.array(obs_dict['target_pos'])-np.array(obs_dict['tip_pos'])
        return obs_dict

    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        far_th = self.far_th*len(self.tip_sids) if np.squeeze(obs_dict['time'])>2*self.dt else np.inf
        near_th = len(self.tip_sids)*self.near_th
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   -1.*reach_dist),
            ('bonus',   1.*(reach_dist<2*near_th) + 1.*(reach_dist<near_th)),
            ('act_reg', -1.*act_mag),
            ('penalty', -1.*(reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.*reach_dist),
            ('solved',  reach_dist<near_th),
            ('done',    reach_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    # generate a valid target
    def generate_target_pose(self, idx=None, **kwargs):
        if self.reach_space is None:
            for site, span in self.target_reach_range.items():
                sid =  self.sim.model.site_name2id(site+'_target')
                self.sim.model.site_pos[sid] = self.np_random.uniform(low=span[0], high=span[1])
        else:
            # TODO: introduce a root site to make the target pose wrt world
            if idx is not None:
                target_pos_root = self.reach_space[idx]
            else:
                target_pos_root = self.reach_space[self.np_random.integers(0, self.reach_space.shape[0])]
            # target_pose_root = np.concatenate([target_pos_root, np.tile(np.array([0.,0.,0.,1.]), (target_pos_root.shape[0], 1))], axis=-1)
            # root_base_id = self.sim.model.site_name2id(self.base_name)
            # root_pose = np.concatenate([self.sim.model.site_pos[root_base_id], self.sim.model.site_quat[root_base_id]])
            # root_pose = np.tile(root_pose, (target_pose_root.shape[0], 1))
            # target_pose = compute_poses_wrt_world(torch.from_numpy(target_pose_root), torch.from_numpy(root_pose)).numpy()
            # target_pos = target_pose[:, :3]
            target_pos = target_pos_root

            for isite, site in enumerate(self.target_reach_range.keys()):
                sid =  self.sim.model.site_name2id(site+'_target')
                self.sim.model.site_pos[sid] = target_pos[isite, :3]
        self.sim.forward()



    def reset(self, **kwargs):
        self.generate_target_pose(**kwargs)
        self.robot.sync_sims(self.sim, self.sim_obsd)
        self.frames = []
        obs = super().reset(**kwargs)
        return obs
    
    def render_offscreen(self, width=640, height=480):
        rbg = self.sim.renderer.render_offscreen(width=width, height=height, camera_id=3)
        self.frames.append(rbg)

    def write_video(self, out_file, duration=2, size=(640,480)):
        fps = int(len(self.frames)/duration)

        out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for i in range(fps * duration):
            data = self.frames[i]
            out.write(data)
        out.release()

    # def step(self, action, steps=1, **kwargs):
    #     for _ in range(steps):
    #         obs, r, done, trunc, info = super().step(action, **kwargs)
    #     return obs, r, done, trunc, info