import collections
import numpy as np
from myosuite.utils import gym

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import euler2quat, mulQuat, negQuat, mat2quat
from myosuite.utils.vector_math import calculate_cosine
import cv2

class ProprioceptiveEnvV0(BaseV0):

    SAR_CREDIT = """\
    SAR: Generalization of Physiological Agility and Dexterity via Synergistic Action Representation
        Cameron Berg, Vittorio Caggiano*, Vikash Kumar*
        RSS-2023 | https://sites.google.com/view/sar-rl/
    """

    # DEFAULT_OBS_KEYS =['hand_jnt','obj_pos','obj_vel','obj_rot','obj_des_rot','obj_err_pos','obj_err_rot','act','mlen','mvel','mforce']
    DEFAULT_OBS_KEYS =['tip_pos','obj_pos', 'obj_rot', 'obj_des_rot']

    DEFAULT_RWD_KEYS_AND_WEIGHTS= {
        'pos_align':1.0,
        'rot_align':1.0,
        # 'act_reg':5.,
        'drop':5.0,
        # 'bonus':10.0
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
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.SAR_CREDIT)

        self._setup(**kwargs)
        self.viewer_setup(azimuth=180, distance=1, lookat=[-0.15,-0.5,1.3], render_actuator=True)


    def _setup(self,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):

        self.target_obj_bid = self.sim.model.body_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.eps_ball_sid = self.sim.model.site_name2id('eps_ball')

        self.success_indicator_sid = self.sim.model.site_name2id("success")

        self.obj_t_gid = self.sim.model.geom_name2id('top')
        self.obj_b_gid = self.sim.model.geom_name2id('bot')
        self.tar_t_gid = self.sim.model.geom_name2id('t_top')
        self.tar_b_gid = self.sim.model.geom_name2id('t_bot')

        self.pen_length = np.linalg.norm(self.sim.model.geom_pos[self.obj_t_gid] - self.sim.model.geom_pos[self.obj_b_gid])
        self.tar_length = np.linalg.norm(self.sim.model.geom_pos[self.tar_t_gid] - self.sim.model.geom_pos[self.tar_b_gid])

        self.sim.model.body_mass[self.obj_bid] *= 1.25
        super()._setup(obs_keys=obs_keys,
                        weighted_reward_keys=weighted_reward_keys,
                        **kwargs,
                    )
        self.tips = ['THtip', 'IFtip', 'MFtip', 'RFtip', 'LFtip']
        self.tip_sids = [self.sim.model.site_name2id(tip) for tip in self.tips]
        self.init_qpos[:-6] *= 0 # Use fully open as init pos
        self.init_qpos[0] = -1.5 # place palm up
        
        self.observation_space = gym.spaces.Box((-10)*np.ones(63), 10*np.ones(63), dtype=np.float32)


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_jnt'] = sim.data.qpos[:-6].copy()
        obs_dict['obj_pos'] = sim.data.body_xpos[self.obj_bid].copy()
        obs_dict['obj_des_pos'] = sim.data.site_xpos[self.eps_ball_sid].ravel()
        obs_dict['obj_vel'] = sim.data.qvel[-6:].copy()*self.dt
        obs_dict['obj_rot'] = (sim.data.geom_xpos[self.obj_t_gid] - sim.data.geom_xpos[self.obj_b_gid])/self.pen_length
        obs_dict['obj_des_rot'] = (sim.data.geom_xpos[self.tar_t_gid] - sim.data.geom_xpos[self.tar_b_gid])/self.tar_length
        obs_dict['obj_err_pos'] = obs_dict['obj_pos']-obs_dict['obj_des_pos']
        obs_dict['obj_err_rot'] = obs_dict['obj_rot']-obs_dict['obj_des_rot']
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
            obs_dict['mlen'] = sim.data.actuator_length[:].copy()
            obs_dict['mvel'] = sim.data.actuator_velocity[:].copy()
            obs_dict['mforce'] = sim.data.actuator_force[:].copy()
        # fingertip pos
        obs_dict['tip_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            obs_dict['tip_pos'] = np.append(obs_dict['tip_pos'], self.sim.data.site_xpos[self.tip_sids[isite]].copy())
        return obs_dict


    def get_reward_dict(self, obs_dict):
        pos_err = obs_dict['obj_err_pos']
        pos_align = np.linalg.norm(pos_err, axis=-1)
        rot_align = calculate_cosine(obs_dict['obj_rot'], obs_dict['obj_des_rot'])
        dropped = (pos_align > 0.075)
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('pos_align',   -1.*pos_align),
            ('rot_align',   rot_align),
            ('act_reg',     -1.*act_mag),
            ('drop',        -1.*dropped),
            ('bonus',       1.*(rot_align > 0.9)*(pos_align<0.075) + 5.0*(rot_align > 0.95)*(pos_align<0.075) ),
            # Must keys
            ('sparse',      -1.0*pos_align+rot_align),
            ('solved',      (rot_align > 0.95)*(~dropped)),
            ('done',        dropped),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        if list(self.sim.model.site_rgba[self.success_indicator_sid, :2]) != [0.0, 2.0]:
            self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([0, 2]) if rwd_dict['solved'] else np.array([2, 0])
        return rwd_dict
    
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = []
        return obs

    def render_offscreen(self, width=640, height=480):
        rbg = self.sim.renderer.render_offscreen(width=width, height=height, camera_id=3) # 2
        self.frames.append(rbg)

    def write_video(self, out_file, fps=30, size=(640,480)):

        out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for i in range(len(self.frames)):
            data = self.frames[i]
            out.write(data)
        out.release()

class Target2EnvV0(ProprioceptiveEnvV0):
    def reset(self, **kwargs):
        self.obj_gid = self.sim.model.geom_name2id('obj')
        self.tar_gid = self.sim.model.geom_name2id('target')

        self.obj_t_gid = self.sim.model.geom_name2id('top')
        self.obj_b_gid = self.sim.model.geom_name2id('bot')
        self.tar_t_gid = self.sim.model.geom_name2id('t_top')
        self.tar_b_gid = self.sim.model.geom_name2id('t_bot')

        size = [0.023, 0.023, 0.023]
        color = [0.996, 0.714, 0.596, 1.0]
        top_pos = np.array([0, 0, size[2]])
        bot_pos = np.array([0, 0, -size[2]])

        self.sim.model.geom_size[self.obj_gid] = size
        self.sim.model.geom_type[self.obj_gid] = 6
        self.sim.model.geom_rgba[self.obj_gid] = color
        self.sim.model.geom_pos[self.obj_t_gid] = top_pos
        self.sim.model.geom_pos[self.obj_b_gid] = bot_pos

        self.sim.model.body_mass[self.obj_bid] = 1.2

        self.sim.model.geom_size[self.tar_gid] = size
        self.sim.model.geom_type[self.tar_gid] = 6
        self.sim.model.geom_rgba[self.tar_gid] = color
        self.sim.model.geom_pos[self.tar_t_gid] = top_pos
        self.sim.model.geom_pos[self.tar_b_gid] = bot_pos
        
        desired_oriens = [np.array([0,0, np.pi/2]), np.array([0.,0.,-np.pi/2])]
        idx = self.np_random.choice([0,1])
        desired_orien = desired_oriens[idx]

        self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([2, 0])

        self.sim.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(**kwargs)
        self.frames = []
        return obs

class Geometries1EnvV0(ProprioceptiveEnvV0):
    def reset(self, **kwargs):
        self.obj_gid = self.sim.model.geom_name2id('obj')
        self.tar_gid = self.sim.model.geom_name2id('target')

        self.obj_t_gid = self.sim.model.geom_name2id('top')
        self.obj_b_gid = self.sim.model.geom_name2id('bot')
        self.tar_t_gid = self.sim.model.geom_name2id('t_top')
        self.tar_b_gid = self.sim.model.geom_name2id('t_bot')

        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-.8, high=1.2)

        self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([2, 0])

        self.sim.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(**kwargs)
        return obs
    
class Geometries8EnvV0(ProprioceptiveEnvV0):
    def reset(self, **kwargs):
        ellips = {0: [[0.011, 0.025, 0.025], [0.74792, 0.35159, 0.80154, 1.0]], 1: [[0.019, 0.040, 0.040], [0.23366, 0.67864, 0.53721, 1.0]]}

        box = {0: [[0.017, 0.017, 0.017], [0.42829, 0.76091, 0.4914, 1.0]], 1: [[0.023, 0.023, 0.023], [0.21995, 0.60938, 0.18821, 1.0]]}

        caps = {0: [[0.013, 0.025, 0.025], [0.2574, 0.6124, 0.5413, 1.0]], 1: [[0.019, 0.040, 0.040], [0.6466, 0.5044, 0.4588, 1.0]]}

        cyl = {0: [[0.013, 0.025, 0.025], [0.1647, 0.3373, 0.3611, 1.0]], 1: [[0.019, 0.040, 0.040], [0.4554, 0.208, 0.8395, 1.0]]}

        # randomize target
        self.obj_gid = self.sim.model.geom_name2id('obj')
        self.tar_gid = self.sim.model.geom_name2id('target')

        self.obj_t_gid = self.sim.model.geom_name2id('top')
        self.obj_b_gid = self.sim.model.geom_name2id('bot')
        self.tar_t_gid = self.sim.model.geom_name2id('t_top')
        self.tar_b_gid = self.sim.model.geom_name2id('t_bot')

        geom_type = self.np_random.choice([3,4,5,6])

        if geom_type == 3:
            ind = self.np_random.choice(range(len(caps)))
            size = caps[ind][0]
            color = caps[ind][1]
            top_pos = np.array([0, 0, 1.3*size[1]])
            bot_pos = np.array([0, 0, 1.3*-size[1]])
        elif geom_type == 4:
            ind = self.np_random.choice(range(len(ellips)))
            size = ellips[ind][0]
            color = ellips[ind][1]
            top_pos = np.array([0, 0, size[2]])
            bot_pos = np.array([0, 0, -size[2]])
        elif geom_type == 5:
            ind = self.np_random.choice(range(len(cyl)))
            size = cyl[ind][0]
            color = cyl[ind][1]
            top_pos = np.array([0, 0, size[1]])
            bot_pos = np.array([0, 0, -size[1]])
        else:
            ind = self.np_random.choice(range(len(box)))
            size = box[ind][0]
            color = box[ind][1]
            top_pos = np.array([0, 0, size[2]])
            bot_pos = np.array([0, 0, -size[2]])

        color = [0.996, 0.714, 0.596, 1.0]

        self.sim.model.geom_size[self.obj_gid] = size
        self.sim.model.geom_type[self.obj_gid] = geom_type
        self.sim.model.geom_rgba[self.obj_gid] = color
        self.sim.model.geom_pos[self.obj_t_gid] = top_pos
        self.sim.model.geom_pos[self.obj_b_gid] = bot_pos

        self.sim.model.body_mass[self.obj_bid] = 1.2

        self.sim.model.geom_size[self.tar_gid] = size
        self.sim.model.geom_type[self.tar_gid] = geom_type
        self.sim.model.geom_rgba[self.tar_gid] = color
        self.sim.model.geom_pos[self.tar_t_gid] = top_pos
        self.sim.model.geom_pos[self.tar_b_gid] = bot_pos

        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-.8, high=1.2)

        self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([2, 0])

        self.sim.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(**kwargs)
        return obs

class Geometries100EnvV0(ProprioceptiveEnvV0):
    def reset(self, **kwargs):
        ellips = {0: [[0.02843, 0.0256, 0.02902], [0.74792, 0.35159, 0.80154, 1.0]], 1: [[0.01057, 0.02655, 0.0328], [0.23366, 0.67864, 0.53721, 1.0]], 2: [[0.01126, 0.0273, 0.04264], [0.65043, 0.2313, 0.50699, 1.0]], 3: [[0.02641, 0.03524, 0.02831], [0.62441, 0.5802, 0.43566, 1.0]], 4: [[0.02804, 0.03722, 0.04313], [0.13892, 0.45695, 0.11598, 1.0]], 5: [[0.02305, 0.04456, 0.03709], [0.13836, 0.67832, 0.31776, 1.0]], 6: [[0.02332, 0.02673, 0.02606], [0.15742, 0.40625, 0.11557, 1.0]], 7: [[0.01247, 0.03233, 0.03759], [0.83, 0.24511, 0.30415, 1.0]], 8: [[0.02199, 0.029, 0.04484], [0.23557, 0.72447, 0.75669, 1.0]], 9: [[0.02674, 0.0428, 0.03764], [0.8393, 0.75063, 0.18226, 1.0]], 10: [[0.02278, 0.04006, 0.03556], [0.32785, 0.49373, 0.5858, 1.0]], 11: [[0.02392, 0.04095, 0.03467], [0.8965, 0.22427, 0.41412, 1.0]], 12: [[0.01928, 0.0348, 0.03044], [0.20289, 0.70564, 0.55928, 1.0]], 13: [[0.02388, 0.03644, 0.02817], [0.67021, 0.70081, 0.36769, 1.0]], 14: [[0.02739, 0.04338, 0.03457], [0.28136, 0.77765, 0.28719, 1.0]], 15: [[0.00962, 0.04047, 0.02614], [0.49566, 0.72634, 0.52086, 1.0]], 16: [[0.0163, 0.04443, 0.04326], [0.1731, 0.8899, 0.10808, 1.0]], 17: [[0.02417, 0.03157, 0.04038], [0.21701, 0.29525, 0.62152, 1.0]], 18: [[0.01927, 0.02814, 0.03786], [0.76065, 0.49735, 0.27818, 1.0]], 19: [[0.02477, 0.04456, 0.04493], [0.80959, 0.8233, 0.5421, 1.0]], 20: [[0.01656, 0.0291, 0.03996], [0.33429, 0.36101, 0.28275, 1.0]], 21: [[0.01763, 0.03877, 0.03636], [0.21692, 0.27226, 0.65917, 1.0]], 22: [[0.01915, 0.0346, 0.04245], [0.41227, 0.43577, 0.31358, 1.0]], 23: [[0.02485, 0.03324, 0.02881], [0.65375, 0.75452, 0.47755, 1.0]], 24: [[0.00856, 0.04185, 0.03749], [0.44235, 0.15332, 0.76038, 1.0]]}

        box = {0: [[0.02295, 0.02306, 0.02221], [0.42829, 0.76091, 0.4914, 1.0]], 1: [[0.02447, 0.0185, 0.02192], [0.21995, 0.20938, 0.48821, 1.0]], 2: [[0.01853, 0.01837, 0.01546], [0.76916, 0.26635, 0.16801, 1.0]], 3: [[0.01586, 0.02079, 0.022], [0.1423, 0.75584, 0.15317, 1.0]], 4: [[0.02293, 0.02116, 0.02255], [0.2323, 0.3077, 0.42493, 1.0]], 5: [[0.01542, 0.01651, 0.02381], [0.26685, 0.38304, 0.60438, 1.0]], 6: [[0.0186, 0.02402, 0.02333], [0.1624, 0.45497, 0.14676, 1.0]], 7: [[0.01782, 0.01584, 0.02208], [0.17459, 0.54131, 0.68087, 1.0]], 8: [[0.01907, 0.0195, 0.02161], [0.54092, 0.40078, 0.68101, 1.0]], 9: [[0.01751, 0.0211, 0.01864], [0.65326, 0.59045, 0.30555, 1.0]], 10: [[0.02258, 0.02334, 0.01856], [0.80835, 0.76567, 0.63477, 1.0]], 11: [[0.02195, 0.01617, 0.02438], [0.86178, 0.17993, 0.61248, 1.0]], 12: [[0.01627, 0.02254, 0.02073], [0.39115, 0.68792, 0.78923, 1.0]], 13: [[0.02364, 0.01946, 0.01777], [0.57742, 0.55447, 0.48724, 1.0]], 14: [[0.01754, 0.02463, 0.01549], [0.73537, 0.1708, 0.49452, 1.0]], 15: [[0.02394, 0.02382, 0.02387], [0.13934, 0.18804, 0.67206, 1.0]], 16: [[0.01997, 0.02372, 0.02032], [0.55577, 0.62793, 0.42524, 1.0]], 17: [[0.01741, 0.02316, 0.02203], [0.8914, 0.54996, 0.31562, 1.0]], 18: [[0.02032, 0.0217, 0.02432], [0.38009, 0.75075, 0.64515, 1.0]], 19: [[0.01961, 0.0248, 0.0176], [0.5283, 0.86192, 0.77579, 1.0]], 20: [[0.01906, 0.01999, 0.02399], [0.49911, 0.89906, 0.44505, 1.0]], 21: [[0.02472, 0.01826, 0.0151], [0.79248, 0.49588, 0.41427, 1.0]], 22: [[0.01636, 0.0158, 0.01958], [0.57198, 0.58271, 0.78801, 1.0]], 23: [[0.01542, 0.02434, 0.02237], [0.22467, 0.88589, 0.83947, 1.0]], 24: [[0.01731, 0.02185, 0.02019], [0.77122, 0.73006, 0.14257, 1.0]]}

        caps = {0: [[0.0162, 0.0422, 0.0484], [0.2574, 0.6124, 0.5413, 1.0]], 1: [[0.016, 0.0457, 0.0496], [0.6466, 0.5044, 0.4588, 1.0]], 2: [[0.0187, 0.0259, 0.0248], [0.3565, 0.7914, 0.1228, 1.0]], 3: [[0.0192, 0.0483, 0.0216], [0.1584, 0.208, 0.8504, 1.0]], 4: [[0.0213, 0.0218, 0.0481], [0.649, 0.3251, 0.4248, 1.0]], 5: [[0.0169, 0.0331, 0.0388], [0.1653, 0.1763, 0.5914, 1.0]], 6: [[0.0138, 0.0299, 0.0471], [0.2764, 0.297, 0.3948, 1.0]], 7: [[0.0194, 0.0252, 0.0419], [0.7098, 0.8251, 0.6623, 1.0]], 8: [[0.014, 0.0362, 0.0201], [0.4234, 0.7236, 0.8917, 1.0]], 9: [[0.0125, 0.029, 0.0298], [0.5309, 0.397, 0.5795, 1.0]], 10: [[0.0162, 0.0396, 0.0323], [0.5987, 0.6183, 0.2126, 1.0]], 11: [[0.019, 0.0365, 0.0421], [0.1012, 0.1079, 0.4266, 1.0]], 12: [[0.0143, 0.0228, 0.0255], [0.6569, 0.5273, 0.6765, 1.0]], 13: [[0.0147, 0.0391, 0.0369], [0.2067, 0.8218, 0.5556, 1.0]], 14: [[0.0192, 0.0324, 0.043], [0.1487, 0.4226, 0.5643, 1.0]], 15: [[0.0145, 0.0491, 0.0234], [0.6435, 0.8165, 0.5704, 1.0]], 16: [[0.013, 0.0458, 0.0457], [0.3264, 0.1895, 0.3025, 1.0]], 17: [[0.0187, 0.0219, 0.0434], [0.7298, 0.3122, 0.5842, 1.0]], 18: [[0.0198, 0.0276, 0.0238], [0.6896, 0.247, 0.3097, 1.0]], 19: [[0.0175, 0.0375, 0.0339], [0.8754, 0.7424, 0.8578, 1.0]], 20: [[0.0191, 0.049, 0.0472], [0.7943, 0.1871, 0.4875, 1.0]], 21: [[0.0145, 0.0425, 0.0356], [0.2251, 0.2459, 0.8375, 1.0]], 22: [[0.0134, 0.0291, 0.0379], [0.4273, 0.7564, 0.3796, 1.0]], 23: [[0.0185, 0.0445, 0.0454], [0.7564, 0.293, 0.7573, 1.0]], 24: [[0.0164, 0.041, 0.0328], [0.1576, 0.8699, 0.1782, 1.0]]}

        cyl = {0: [[0.0118, 0.044, 0.0265], [0.1647, 0.3373, 0.3611, 1.0]], 1: [[0.0189, 0.0316, 0.0415], [0.4554, 0.208, 0.8395, 1.0]], 2: [[0.0123, 0.0364, 0.0238], [0.8701, 0.7737, 0.4562, 1.0]], 3: [[0.0145, 0.0362, 0.032], [0.3982, 0.7781, 0.8505, 1.0]], 4: [[0.0146, 0.0306, 0.0283], [0.8632, 0.8666, 0.1886, 1.0]], 5: [[0.0155, 0.0237, 0.0383], [0.488, 0.3793, 0.6675, 1.0]], 6: [[0.0198, 0.0323, 0.03], [0.7845, 0.2764, 0.8887, 1.0]], 7: [[0.011, 0.0368, 0.0343], [0.5305, 0.607, 0.651, 1.0]], 8: [[0.0197, 0.0305, 0.0206], [0.1644, 0.8826, 0.1977, 1.0]], 9: [[0.0162, 0.0277, 0.0329], [0.7852, 0.6644, 0.6417, 1.0]], 10: [[0.0202, 0.0339, 0.0223], [0.3046, 0.4812, 0.4465, 1.0]], 11: [[0.0181, 0.0204, 0.0314], [0.5126, 0.7779, 0.4888, 1.0]], 12: [[0.0118, 0.0374, 0.0228], [0.7434, 0.7433, 0.8634, 1.0]], 13: [[0.0197, 0.0319, 0.0375], [0.5661, 0.2352, 0.3933, 1.0]], 14: [[0.0133, 0.0392, 0.0448], [0.8689, 0.5574, 0.1234, 1.0]], 15: [[0.0157, 0.0236, 0.0301], [0.6846, 0.8593, 0.2673, 1.0]], 16: [[0.0178, 0.0351, 0.0414], [0.1313, 0.8603, 0.8791, 1.0]], 17: [[0.0123, 0.024, 0.0399], [0.3679, 0.6363, 0.4691, 1.0]], 18: [[0.0158, 0.0254, 0.022], [0.2256, 0.1914, 0.8973, 1.0]], 19: [[0.0192, 0.021, 0.0355], [0.1923, 0.8149, 0.21, 1.0]], 20: [[0.011, 0.0266, 0.0261], [0.7158, 0.2207, 0.4712, 1.0]], 21: [[0.0121, 0.0369, 0.0226], [0.4497, 0.7305, 0.5118, 1.0]], 22: [[0.012, 0.0221, 0.0254], [0.6128, 0.4154, 0.2382, 1.0]], 23: [[0.0213, 0.0298, 0.0297], [0.8346, 0.4971, 0.3623, 1.0]], 24: [[0.0164, 0.0302, 0.025], [0.2496, 0.1486, 0.8377, 1.0]]}

        # randomize target
        self.obj_gid = self.sim.model.geom_name2id('obj')
        self.tar_gid = self.sim.model.geom_name2id('target')

        self.obj_t_gid = self.sim.model.geom_name2id('top')
        self.obj_b_gid = self.sim.model.geom_name2id('bot')
        self.tar_t_gid = self.sim.model.geom_name2id('t_top')
        self.tar_b_gid = self.sim.model.geom_name2id('t_bot')

        geom_type = self.np_random.choice([3,4,5,6])

        if geom_type == 3:
            size = caps[self.np_random.choice(range(len(caps)))][0]
            color = caps[self.np_random.choice(range(len(caps)))][1]
            top_pos = np.array([0, 0, 1.3*size[1]])
            bot_pos = np.array([0, 0, 1.3*-size[1]])
        elif geom_type == 4:
            size = ellips[self.np_random.choice(range(len(ellips)))][0]
            color = ellips[self.np_random.choice(range(len(ellips)))][1]
            top_pos = np.array([0, 0, size[2]])
            bot_pos = np.array([0, 0, -size[2]])
        elif geom_type == 5:
            size = cyl[self.np_random.choice(range(len(cyl)))][0]
            color = cyl[self.np_random.choice(range(len(cyl)))][1]
            top_pos = np.array([0, 0, size[1]])
            bot_pos = np.array([0, 0, -size[1]])
        else:
            size = box[self.np_random.choice(range(len(box)))][0]
            color = box[self.np_random.choice(range(len(box)))][1]
            top_pos = np.array([0, 0, size[2]])
            bot_pos = np.array([0, 0, -size[2]])

        color = [0.996, 0.714, 0.596, 1.0]

        self.sim.model.geom_size[self.obj_gid] = size
        self.sim.model.geom_type[self.obj_gid] = geom_type
        self.sim.model.geom_rgba[self.obj_gid] = color
        self.sim.model.geom_pos[self.obj_t_gid] = top_pos
        self.sim.model.geom_pos[self.obj_b_gid] = bot_pos

        self.sim.model.body_mass[self.obj_bid] = 1.2

        self.sim.model.geom_size[self.tar_gid] = size
        self.sim.model.geom_type[self.tar_gid] = geom_type
        self.sim.model.geom_rgba[self.tar_gid] = color
        self.sim.model.geom_pos[self.tar_t_gid] = top_pos
        self.sim.model.geom_pos[self.tar_b_gid] = bot_pos

        self.sim.model.geom_condim[self.obj_gid] = 3
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-.8, high=1.2)

        self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([2, 0])


        self.sim.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(**kwargs)
        return obs
