from myosuite.envs.myo.myobase import register_env_with_variants

import os
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(__file__))

register_env_with_variants(id='myoHandReachOneStep',
    entry_point='modexenvs.reach_envs.myohand_reach:MyoHandReach',
    max_episode_steps=1,
    kwargs={
        'model_path': curr_dir+'/../assets/mjcf/myosuite/myohand.xml',
        'seed': 0,
        'target_reach_range': {
            'THtip': ((-0.165-0.020, -0.537-0.040, 1.495-0.040), (-0.165+0.040, -0.537+0.020, 1.495+0.040)),
            'IFtip': ((-0.151-0.040, -0.547-0.020, 1.455-0.010), (-0.151+0.040, -0.547+0.020, 1.455+0.010)),
            'MFtip': ((-0.146-0.040, -0.547-0.020, 1.447-0.010), (-0.146+0.040, -0.547+0.020, 1.447+0.010)),
            'RFtip': ((-0.148-0.040, -0.543-0.020, 1.445-0.010), (-0.148+0.040, -0.543+0.020, 1.445+0.010)),
            'LFtip': ((-0.148-0.040, -0.528-0.020, 1.434-0.010), (-0.148+0.040, -0.528+0.020, 1.434+0.010)),
            },
        'reach_space': curr_dir+'/../../data/hand_data/MyoHand/myohand_random_pos_Jun_19_18:24:52_2024/hand_states.npy',
        'normalize_act': True,
        'far_th': 0.05,
        'near_th': 0.0125,
        'frame_skip': 5000,
    }
)

register_env_with_variants(id='myoHandReachOneStep',
    entry_point='modexenvs.reach_envs.myohand_reach:MyoHandReach',
    max_episode_steps=100,
    kwargs={
        'model_path': curr_dir+'/../assets/mjcf/myosuite/myohand.xml',
        'seed': 0,
        'target_reach_range': {
            'THtip': ((-0.165-0.020, -0.537-0.040, 1.495-0.040), (-0.165+0.040, -0.537+0.020, 1.495+0.040)),
            'IFtip': ((-0.151-0.040, -0.547-0.020, 1.455-0.010), (-0.151+0.040, -0.547+0.020, 1.455+0.010)),
            'MFtip': ((-0.146-0.040, -0.547-0.020, 1.447-0.010), (-0.146+0.040, -0.547+0.020, 1.447+0.010)),
            'RFtip': ((-0.148-0.040, -0.543-0.020, 1.445-0.010), (-0.148+0.040, -0.543+0.020, 1.445+0.010)),
            'LFtip': ((-0.148-0.040, -0.528-0.020, 1.434-0.010), (-0.148+0.040, -0.528+0.020, 1.434+0.010)),
            },
        'reach_space': curr_dir+'/../../data/hand_data/MyoHand/myohand_random_pos_Jun_19_18:24:52_2024/hand_states.npy',
        'normalize_act': True,
        'far_th': 0.05,
        'near_th': 0.0125,
        'frame_skip': 50,
    }
)

register_env_with_variants(id='myoHandReachSequence',
    entry_point='modexenvs.reach_envs.myohand_reach:MyoHandReach',
    max_episode_steps=50,
    kwargs={
        'model_path': curr_dir+'/../assets/mjcf/myosuite/myohand.xml',
        'seed': 0,
        'target_reach_range': {
            'THtip': ((-0.165-0.020, -0.537-0.040, 1.495-0.040), (-0.165+0.040, -0.537+0.020, 1.495+0.040)),
            'IFtip': ((-0.151-0.040, -0.547-0.020, 1.455-0.010), (-0.151+0.040, -0.547+0.020, 1.455+0.010)),
            'MFtip': ((-0.146-0.040, -0.547-0.020, 1.447-0.010), (-0.146+0.040, -0.547+0.020, 1.447+0.010)),
            'RFtip': ((-0.148-0.040, -0.543-0.020, 1.445-0.010), (-0.148+0.040, -0.543+0.020, 1.445+0.010)),
            'LFtip': ((-0.148-0.040, -0.528-0.020, 1.434-0.010), (-0.148+0.040, -0.528+0.020, 1.434+0.010)),
            },
        'reach_space': curr_dir+'/../../data/hand_data/MyoHand/myohand_random_pos_Jun_19_18:24:52_2024/hand_states.npy',
        'normalize_act': True,
        'far_th': 0.05,
        'near_th': 0.0125,
        'frame_skip': 100,
    }
)

register_env_with_variants(id='myoHandReachSequenceVal',
    entry_point='modexenvs.reach_envs.myohand_reach:MyoHandReach',
    max_episode_steps=500,
    kwargs={
        'model_path': curr_dir+'/../assets/mjcf/myosuite/myohand.xml',
        'seed': 0,
        'target_reach_range': {
            'THtip': ((-0.165-0.020, -0.537-0.040, 1.495-0.040), (-0.165+0.040, -0.537+0.020, 1.495+0.040)),
            'IFtip': ((-0.151-0.040, -0.547-0.020, 1.455-0.010), (-0.151+0.040, -0.547+0.020, 1.455+0.010)),
            'MFtip': ((-0.146-0.040, -0.547-0.020, 1.447-0.010), (-0.146+0.040, -0.547+0.020, 1.447+0.010)),
            'RFtip': ((-0.148-0.040, -0.543-0.020, 1.445-0.010), (-0.148+0.040, -0.543+0.020, 1.445+0.010)),
            'LFtip': ((-0.148-0.040, -0.528-0.020, 1.434-0.010), (-0.148+0.040, -0.528+0.020, 1.434+0.010)),
            },
        'reach_space': curr_dir+'/../../data/hand_data/MyoHand/myohand_random_pos_Jun_19_18:24:52_2024/hand_states.npy',
        'normalize_act': True,
        'far_th': 0.05,
        'near_th': 0.0125,
        'frame_skip': 10,
    }
)