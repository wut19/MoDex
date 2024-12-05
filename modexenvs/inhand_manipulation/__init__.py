from myosuite.envs.myo.myobase import register_env_with_variants

import os
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(__file__))

# Hold objects ==============================
register_env_with_variants(id='myoHandObjHoldFixedCustom',
        entry_point='modexenvs.inhand_manipulation.obj_hold:ObjHoldFixedEnvV0',
        max_episode_steps=75,
        kwargs={
            'model_path': curr_dir+'/../assets/mjcf/myosuite/hand/myohand_hold.xml',
            'normalize_act': True,
            'frame_skip': 10,
        }
    )
register_env_with_variants(id='myoHandObjHoldRandomCustom', # revisit
        entry_point='modexenvs.inhand_manipulation.obj_hold:ObjHoldRandomEnvV0',
        max_episode_steps=75,
        kwargs={
            'model_path': curr_dir+'/../assets/mjcf/myosuite/hand/myohand_hold.xml',
            'normalize_act': True,
            'frame_skip': 10,
        }
    )

# Pen twirl ==============================
register_env_with_variants(id='myoHandPenTwirlFixedCustom',
            entry_point='modexenvs.inhand_manipulation.pen:PenTwirlFixedEnvV0',
            max_episode_steps=50,
            kwargs={
                'model_path': curr_dir+'/../assets/mjcf/myosuite/hand/myohand_pen.xml',
                'normalize_act': True,
                'frame_skip': 5,
            }
    )
register_env_with_variants(id='myoHandPenTwirlRandomCustom',
        entry_point='modexenvs.inhand_manipulation.pen:PenTwirlRandomEnvV0',
        max_episode_steps=50,
        kwargs={
            'model_path': curr_dir+'/../assets/mjcf/myosuite/hand/myohand_pen.xml',
            'normalize_act': True,
            'frame_skip': 5,
        }
    )

# REORIENT: ==============================
register_env_with_variants(id='myoHandReorientTarget2Custom',
            entry_point='modexenvs.inhand_manipulation.reorient:Target2EnvV0',
            max_episode_steps=50,
            kwargs={
                'model_path': curr_dir+'/../assets/mjcf/myosuite/hand/myohand_sar.xml',
                'normalize_act': True,
                'frame_skip': 5,
            }
    )

register_env_with_variants(id='myoHandReorient8Custom',
            entry_point='modexenvs.inhand_manipulation.reorient:Geometries8EnvV0',
            max_episode_steps=50,
            kwargs={
                'model_path': curr_dir+'/../assets/mjcf/myosuite/hand/myohand_sar.xml',
                'normalize_act': True,
                'frame_skip': 5,
            }
    )

register_env_with_variants(id='myoHandReorient100Custom',
            entry_point='modexenvs.inhand_manipulation.reorient:Geometries100EnvV0',
            max_episode_steps=50,
            kwargs={
                'model_path': curr_dir+'/../assets/mjcf/myosuite/hand/myohand_sar.xml',
                'normalize_act': True,
                'frame_skip': 5,
            }
    )

# Baoding balls ==============================
register_env_with_variants(id='myoHandBaodingCustom',
        entry_point='modexenvs.inhand_manipulation.baoding:BaodingEnvV1',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../assets/mjcf/myosuite/hand/myohand_baoding.xml',
            'normalize_act': True,
            'goal_time_period': (5, 5),
            'goal_xrange': (0.025, 0.025),
            'goal_yrange': (0.028, 0.028),
        }
    )

