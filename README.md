# PoppyRL

**PoppyRL** is a project that implements Reinforcement Learning (RL) techniques on the Poppy humanoid robot platform.
We explore and benchmark two major RL algorithms enhanced with Hindsight Experience Replay (HER):

- Soft Actor-Critic (SAC) + HER

- Deep Deterministic Policy Gradient (DDPG) + HER

Our goal is to enable efficient and sample-effective learning of complex motor behaviors on Poppy for learning inverse kinematics.

# About Poppy 

[Poppy](https://www.poppy-project.org/en/) is an open-source robotic platform developed at INRIA, designed for research and education in humanoid robotics and bio-inspired robotics. It features a flexible and modular architecture suitable for a variety of robotic experiments.

# Project Structure
```
.
├── env.yml
├── poppy_urdf
│   ├── base_respondable.STL
│   ├── base.STL
│   ├── base_visual.STL
│   ├── bust_motors_respondable.STL
│   ├── bust_motors_visual.STL
│   ├── chest_respondable.STL
│   ├── chest_visual.STL
│   ├── head_respondable.STL
│   ├── head_visual.STL
│   ├── l_forearm_respondable.STL
│   ├── l_forearm_visual.STL
│   ├── l_shoulder_motor_respondable.STL
│   ├── l_shoulder_motor_visual.STL
│   ├── l_shoulder_respondable.STL
│   ├── l_shoulder_visual.STL
│   ├── l_upper_arm_respondable.STL
│   ├── l_upper_arm.STL
│   ├── l_upper_arm_visual.STL
│   ├── neck_respondable.STL
│   ├── neck_visual.STL
│   ├── r_forearm_respondable.STL
│   ├── r_forearm_visual.STL
│   ├── r_shoulder_motor_respondable.STL
│   ├── r_shoulder_motor_visual.STL
│   ├── r_shoulder_respondable.STL
│   ├── r_shoulder_visual.STL
│   ├── r_upper_arm_respondable (copy).STL
│   ├── r_upper_arm_respondable.STL
│   ├── r_upper_arm_visual.STL
│   ├── spine_respondable (copy).STL
│   ├── spine_respondable.STL
│   └── spine_visual.STL
└── src
    ├── inference_ddpg_her.py
    ├── poppy_sac_her.py
    ├── poppy_torso_ik_env.py
    ├── poppy_torso.urdf
    ├── popy_full_old_torch.py
    └── train_ddpg_her.py


```

# Algortihmns implemented

- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290)
  Entropy-regularized policy optimization for improved exploration.
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971) Actor-critic method for continuous action spaces.
- [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495)
  Sample-efficient strategy for sparse reward settings.

# Installation

1). Clone the repository: 

```
git clone https://github.com/AkshayArjun/PoppyRL.git

```
2). Install [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install)

3). Install dependencies: 

```
conda [name] create -f env.yml
```


  

