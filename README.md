# Real Robot Challenge 2022 

[![license](https://img.shields.io/badge/license-GPLv2-blue.svg)](https://opensource.org/licenses/GPL-2.0)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/ait-bsc.svg?logo=python&logoColor=FFE873)](https://pypi.org/project/ait-bsc/)

## About this chanllenge
The goal of [challenge in 2022](https://real-robot-challenge.com/) is to solve dexterous manipulation tasks with offline reinforcement learning (RL) or imitation learning. The participants are provided with datasets containing dozens of hours of robotic data and can evaluate their policies remotely on a cluster of real TriFinger robots.
Participants can tackle two tasks during the real-robot stage:
* Pushing a cube to a target location on the ground and
* lifting the cube to match a target position and orientation in the air.
Like last year’s challenge, the Real Robot Challenge III is featured in the [NeurIPS 2022 Competition Track](https://neurips.cc/Conferences/2022/CompetitionTrack).


![trifinger](trifingerpro_with_cube.jpg)

## Offline Dataset
We provide datasets for both tasks that have been created using an expert policy. The datasets are provided as part of the gym environments implemented in rrc_2022_datasets (see TriFingerDatasetEnv). The environments are compatible with the interface used by D4RL. Thus, they can be easily used in other frameworks such as D3RLPY. 

#### Dataset Description 

## Methodology 
### Offline RL 
There are 2 kinds of approaches in RL to solve this task, one is online RL(including on-policy & off-police methods), the other one is offline RL, which means the policy of actions is generate to mimic the behaviors of offline dataset. It should be noted that offline RL is surely a different notation from off-policy, in the offline setting, the agent no longer has the ability to interact with the environment and collect additional transitions using the behaviour policy. The learning algorithm is provided with a static dataset of fixed interaction, and must learn the best policy it can using this dataset.
We select 3 offline algorithms to solve this chanllenge, which are:
* **Standard BC (Bahavior Cloning)**
* **IQL (Implicit Q-Learning)**
* **TD3+BC (Twin Delayed DDPG)**

### Data augmentation by using discrete group symmetry

#### Morphorlogical Symmetry Analysis 
This fixed-based robot is symmetric w.r.t. rotations of space by $\theta=\frac{2\pi}{3}$ in the vertical axis. Therefore, its symmetry group is the cyclic group of order three ($\mathbb{G}=\mathbb{C_3}$ ).

#### Data augmentation by symmetry transformation 


## Evaluation

Example Policies
----------------

Here is the example commands for the pre-stage to show how you should set up for running the evaluation.  You use them to test the evaluation.

For the push task :

    $ python3 -m rrc_2022_datasets.evaluate_pre_stage push rrc2022.example.TorchPushPolicy --n-episodes=3 -v

For the lift task:

    $ python3 -m rrc_2022_datasets.evaluate_pre_stage lift rrc2022.example.TorchLiftPolicy --n-episodes=3 -v






 
