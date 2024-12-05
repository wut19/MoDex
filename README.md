# MoDex: Planning High-Dimensional Dexterous Control via Learning Neural Hand Models 🤌

## Installation
1. Create your environment with conda. 
2. Install [IsaacGym](https://developer.nvidia.com/isaac-gym) and follow the instruction to install and test the environment.
3. Clone [IssacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs)
    ```
    git clone https://github.com/isaac-sim/IsaacGymEnvs.git
    ```
    Then install the environment.
    ```
    pip install -e .    
    ```
4. Install [myosuite](https://github.com/MyoHub/myosuite)
    ```
    pip install -U myosuite
    ```
5. Install modexenvs
    ```
    cd modexenvs
    pip install -e.
    ```

## Usage
0. **Collect data through random exploration**
    ```
    python scripts/0_collect_data.py --cfg [config_path]
    ```

1. **Train hand model.** For quasi-static setting and inverse model training in sequential setting:
    ```
    python scripts/1_train.py --cfg [train_config_path]
    ```
    For forward model training in sequential setting to include multi-step loss:
    ```
    python scripts/1_train_seq_dynamics.py \
        --cfg [forward_train_config_path]
    ```

2. **Reach.** We split Isaac and Myosuite. Before planning the reach action/motion, modify the `reach_space_paths` in `scripts/2_run_isaac_reach_env.py` and `reach_space` in `modexenvs/reach_envs/__init__.py` to a random collected dataset which depicts the reachable space of the hand. Then modify the trained model path in above two scripts. Run
    ```
    python scripts/2_run_isaac_reach_env.py
    ```
    for Isaac Reach and 
    ```
    python scripts/2_run_myo_reach_env.py
    ```
    for Myosuite Reach.

3. **Gesture generation.** Run following command to generate gestures:
    ```
    python scripts/3_generate_gesture.py \
        --task  AllegroHand \
        --num_envs 1 \
        --cfg_path [path_to_forward_model_config] \
        --model_path [path_to_trained_forward_model] \
        --opt_alg cem \
        --gesture ok
    ```
    We provide several examples in this scripts. You can also generate through GPT by taking the prompt file: `LLM/system_prompt.txt` and then use the generated cost function for optimization.

4. **In-hand manipulation.** First, run 
    ```
    python scripts/0_collect_data.py \
        --cfg datacollector/cfg/SeqMyohand_.yaml
    ```
    to collect hand sequential data and run 
    ```
    python scripts/1_train_seq_dynamics.py \
        --cfg models/cfg/hand_model_train/myohand_seq_forward_model.yaml
    ```
    to learn a hand model. Then run
    ```
    python scripts/4_inhand_mani_mbm.py
    ```
    to learn a model-based method. Script `scripts/4_inhand_mani_mbm.py` is used for model-free reinforcement learning.