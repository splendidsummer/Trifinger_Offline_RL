import rrc_2022_datasets
import tqdm
import copy
import gym
import math
import numpy as np
import gc
from tqdm import tqdm
from copy import copy
from pathlib import Path
import argparse
from config import *


def truncate_raw_dataset(task, probs, obs_to_keep):
    if task == 'push':
        env_name = "trifinger-cube-push-sim-expert-v0"
    else:
        env_name = "trifinger-cube-lift-sim-expert-v0"

    env = gym.make(
        env_name,
        disable_env_checker=True,
        # flatten_obs=False,
        flatten_obs=True,
        # obs_to_keep=obs_to_keep,
        # visualization=True,  # enable visualization
    )
    dataset = env.get_dataset()
    truncate_dataset = {}

    obs, actions, rewards, timeouts = dataset['observations'],\
        dataset['actions'], dataset['rewards'], dataset['timeouts']

    truncate_dataset['observations'] = truncate(obs, probs)
    truncate_dataset['actions'] = truncate(actions, probs)
    truncate_dataset['rewards'] = truncate(rewards, probs)
    truncate_dataset['timeouts'] = truncate(timeouts, probs)
    file_name = task + '_' + str(int(probs*100)) + '_raw.npy'
    np.save(file_name, truncate_dataset)
    print(f'{file_name} saved successfully!!')


def rotation(ob, angle) -> np.array:
    rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]])

    rot_ob = np.dot(rot_mat, ob)
    return rot_ob


def build_perm3(degree):
    if degree == -120:
        perm = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    elif degree == -240:
        perm = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    return perm


def build_perm(degree):
    arr1 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    arr2 = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

    if degree == -120:
        perm1 = np.concatenate([arr2, arr2, arr1], axis=1)
        perm2 = np.concatenate([arr1, arr2, arr2], axis=1)
        perm3 = np.concatenate([arr2, arr1, arr2], axis=1)
        perm = np.concatenate([perm1, perm2, perm3], axis=0)

    elif degree == -240:
        perm1 = np.concatenate([arr2, arr1, arr2], axis=1)
        perm2 = np.concatenate([arr2, arr2, arr1], axis=1)
        perm3 = np.concatenate([arr1, arr2, arr2], axis=1)
        perm = np.concatenate([perm1, perm2, perm3], axis=0)

    return perm


def truncate(reps, probs):
    idx = np.arange(reps.shape[0])
    np.random.shuffle(idx)
    reps = reps[idx]
    # actions = acitons[idx]
    last_idx = int(idx.shape[0]*probs)
    reps = reps[: last_idx]
    # actions = actions[: last_idx]
    return reps


def build_aug_dataset(task, probs):
    file_name = 'dataset/' + task + '_' + str(int(probs*100)) + '_raw.npy'
    raw_dataset = np.load(file_name, allow_pickle=True).item()
    name = task + '_' + str(int(probs*100)) + '_aug.npy'
    save_path = Path('.') / 'dataset' / name
    aug_dataset = {}

    if task == 'push':
        dataset_rot120 = rot_push(raw_dataset, -120, save_path)
        dataset_rot240 = rot_push(raw_dataset, -240, save_path)
    elif task == 'lift':
        dataset_rot120 = rot_lift(raw_dataset, -120, save_path)
        dataset_rot240 = rot_lift(raw_dataset, -240, save_path)

    aug_dataset['observations'] = np.concatenate(
        [raw_dataset['observations'], dataset_rot120['observations'],
         dataset_rot240['observations']], axis=0)
    aug_dataset['actions'] = np.concatenate(
        [raw_dataset['actions'], dataset_rot120['actions'],
         dataset_rot240['actions']], axis=0)
    aug_dataset['rewards'] = np.concatenate(
        [raw_dataset['rewards'], dataset_rot120['rewards'],
         dataset_rot240['rewards']], axis=0)
    aug_dataset['timeouts'] = np.concatenate(
        [raw_dataset['timeouts'], dataset_rot120['timeouts'],
         dataset_rot240['timeouts']], axis=0)

    np.save(str(save_path), aug_dataset)
    print(str(save_path) + ' saved successfully!!!')
    print('Number of data samples: ' + str(aug_dataset['timeouts'].shape[0]))
    del(aug_dataset, dataset_rot120, dataset_rot240)
    gc.collect()
    # return dataset


def build_rot_dataset(task, probs, degree=-120):
    file_name = 'dataset/' + task + '_' + str(int(probs*100)) + '_raw.npy'
    raw_dataset = np.load(file_name, allow_pickle=True).item()
    name = task + '_' + str(int(probs*100)) + '_rot' + str(int(degree)) + '.npy'
    save_path = Path('.') / 'dataset' / name
    aug_dataset = {}

    if task == 'push':
        dataset_rot120 = rot_push(raw_dataset, -120, save_path)
        dataset_rot240 = rot_push(raw_dataset, -240, save_path)
    elif task == 'lift':
        dataset_rot120 = rot_lift(raw_dataset, -120, save_path)
        dataset_rot240 = rot_lift(raw_dataset, -240, save_path)

    aug_dataset['observations'] = np.concatenate(
        [raw_dataset['observations'], dataset_rot120['observations'],
         dataset_rot240['observations']], axis=0)
    aug_dataset['actions'] = np.concatenate(
        [raw_dataset['actions'], dataset_rot120['actions'],
         dataset_rot240['actions']], axis=0)
    aug_dataset['rewards'] = np.concatenate(
        [raw_dataset['rewards'], dataset_rot120['rewards'],
         dataset_rot240['rewards']], axis=0)
    aug_dataset['timeouts'] = np.concatenate(
        [raw_dataset['timeouts'], dataset_rot120['timeouts'],
         dataset_rot240['timeouts']], axis=0)

    np.save(str(save_path), aug_dataset)
    print(str(save_path) + ' saved successfully!!!')
    print('Number of data samples: ' + str(aug_dataset['timeouts'].shape[0]))
    del(aug_dataset, dataset_rot120, dataset_rot240)
    gc.collect()
    # return dataset


def rot_push(raw_dataset, degree, save_path=None):

    env = gym.make(
        "trifinger-cube-push-sim-expert-v0",
    )
    env.reset()
    angle = math.radians(degree)
    perm = build_perm(degree)
    perm3 = build_perm3(degree)

    with tqdm(total=raw_dataset['observations'].shape[0], desc="Augmenting the push task cw120") as pbar:

        for idx, obs in enumerate(raw_dataset['observations']):
            temp_obs = copy(obs)
            # observations
            # ag
            temp_obs[0:3] = rotation(obs[0:3], angle)
            # act
            temp_obs[3: 12] = np.dot(perm, temp_obs[3: 12])
            # g # what is the obs here
            temp_obs[12: 15] = rotation(obs[12: 15], angle)
            # confidence not change 15
            # delay not change 16

            # ag2
            temp_obs[17:20] = rotation(obs[17:20], angle)
            temp_obs[20:23] = rotation(obs[20:23], angle)
            temp_obs[23:26] = rotation(obs[23:26], angle)
            temp_obs[26:29] = rotation(obs[26:29], angle)
            temp_obs[29:32] = rotation(obs[29:32], angle)
            temp_obs[32:35] = rotation(obs[32:35], angle)
            temp_obs[35:38] = rotation(obs[35:38], angle)
            temp_obs[38:41] = rotation(obs[38:41], angle)

            # orientation and position
            # copy deep
            ag_key_points = copy(temp_obs[17:41]).reshape(8, 3)
            ag_pos = rrc_2022_datasets.utils.get_pose_from_keypoints(ag_key_points)[0]
            ag_ori = rrc_2022_datasets.utils.get_pose_from_keypoints(ag_key_points)[1]
            # quarterion sysmmetry can not reach by rotation
            temp_obs[41:45] = ag_ori
            temp_obs[45:48] = ag_pos

            # rob pos
            temp_obs[69: 78] = np.dot(perm, obs[69: 78])
            # 78 rob id not change 120
            # rob torque
            temp_obs[79: 88] = np.dot(perm, obs[79: 88])
            # rob vel
            temp_obs[88: 97] = np.dot(perm, obs[88: 97])
            # finger tip force
            temp_obs[48: 51] = np.dot(perm3, obs[48: 51])

            # finger tip pos and vel
            fingertip_position, fingertip_velocity = env.sim_env.platform.forward_kinematics(
                temp_obs[69: 78], temp_obs[88: 97]
            )
            fingertip_position = np.array(fingertip_position, dtype=np.float32)
            fingertip_velocity = np.array(fingertip_velocity, dtype=np.float32)
            temp_obs[51: 60] = fingertip_position.reshape(9, )
            temp_obs[60: 69] = fingertip_velocity.reshape(9, )

            raw_dataset['observations'][idx] = temp_obs
            raw_dataset['actions'][idx] = np.dot(perm, copy(raw_dataset['actions'][idx]))

            if idx % 1000 == 0:
                pbar.set_postfix()
                pbar.update(1000)

    # np.save(save_path, raw_dataset)
    # print('Scuuessfuly saved')

    # del raw_dataset
    # gc.collect()
    return raw_dataset


def rot_lift(raw_dataset, degree, save_path=None):
    env = gym.make(
        "trifinger-cube-lift-sim-expert-v0",
    )
    env.reset()
    angle = math.radians(degree)
    perm = build_perm(degree)
    perm3 = build_perm3(degree)

    with tqdm(total=raw_dataset['observations'].shape[0], desc="Augmenting the lift task cw120") as pbar:
        for idx, obs in enumerate(raw_dataset['observations']):

            temp_obs = copy(obs)
            # observations
            # ag
            temp_obs[0: 3] = rotation(obs[0: 3], angle)
            temp_obs[3: 6] = rotation(obs[3: 6], angle)
            temp_obs[6: 9] = rotation(obs[6: 9], angle)
            temp_obs[9: 12] = rotation(obs[9: 12], angle)
            temp_obs[12: 15] = rotation(obs[12: 15], angle)
            temp_obs[15: 18] = rotation(obs[15: 18], angle)
            temp_obs[18: 21] = rotation(obs[18: 21], angle)
            temp_obs[21: 24] = rotation(obs[21: 24], angle)

            # act
            temp_obs[24: 33] = np.dot(perm, copy(obs[24: 33]))

            # g
            temp_obs[33: 36] = rotation(obs[33: 36], angle)
            temp_obs[36: 39] = rotation(obs[36: 39], angle)
            temp_obs[39: 42] = rotation(obs[39: 42], angle)
            temp_obs[42: 45] = rotation(obs[42: 45], angle)
            temp_obs[45: 48] = rotation(obs[45: 48], angle)
            temp_obs[48: 51] = rotation(obs[48: 51], angle)
            temp_obs[51: 54] = rotation(obs[51: 54], angle)
            temp_obs[54: 57] = rotation(obs[54: 57], angle)

            # confidence not change 57
            # delay not change 58
            # ag2
            temp_obs[59:59 + 24] = temp_obs[0:0 + 24]

            # orientation and position
            ag_key_points = copy(temp_obs[0:0 + 24]).reshape(8, 3)
            ag_pos = rrc_2022_datasets.utils.get_pose_from_keypoints(ag_key_points)[0]
            ag_ori = rrc_2022_datasets.utils.get_pose_from_keypoints(ag_key_points)[1]
            temp_obs[83:83 + 4] = ag_ori
            temp_obs[87:87 + 3] = ag_pos

            # rob pos
            temp_obs[111: 120] = np.dot(perm, copy(obs[111: 120]))

            # rob id not change 120
            # rob torque
            temp_obs[121: 130] = np.dot(perm, copy(obs[121: 130]))

            # rob vel
            temp_obs[130: 139] = np.dot(perm, copy(obs[130: 139]))

            # finger tip force
            temp_obs[90: 93] = np.dot(perm3, copy(obs[90: 93]))

            # finger tip pos and vel --> input: torque + velocity
            fingertip_position, fingertip_velocity = env.sim_env.platform.forward_kinematics(
                temp_obs[111: 120], temp_obs[130: 139])

            fingertip_position = np.array(fingertip_position, dtype=np.float32)
            fingertip_velocity = np.array(fingertip_velocity, dtype=np.float32)

            temp_obs[93: 102] = fingertip_position.reshape(9, )
            temp_obs[102: 111] = fingertip_velocity.reshape(9, )

            raw_dataset['observations'][idx] = temp_obs

            # action
            raw_dataset['actions'][idx] = np.dot(perm, copy(raw_dataset['actions'][idx]))

            if idx % 100000 == 0:
                pbar.set_postfix()
                pbar.update(100000)

    # np.save(save_path, raw_dataset)
    # print('Scuuessfuly saved')

    # del raw_dataset
    # gc.collect()

    return raw_dataset


if __name__ == '__main__':
    # env_name = "trifinger-cube-push-sim-expert-v0"
    # env_name = "trifinger-cube-lift-sim-expert-v0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, choices=["push", "lift"],
                        help="Which task to evaluate ('push' or 'lift').", )
    parser.add_argument("--probs",  type=float, default=0.1,
                        help="Percentage of truncated data.",)
    parser.add_argument("--degree", type=int, default=-120, help="rotation degrees")
    parser.add_argument("--filtering", "-f", action="store_true", help="filter out unnecessary dims of data.")
    args = parser.parse_args()

    obs_to_keep = None
    if args.filtering:
        obs_to_keep = obs_to_keep

    # env = gym.make(
    #     env_name,
    #     disable_env_checker=True,
    #     # flatten_obs=False,
    #     flatten_obs=True,
    #     # obs_to_keep=obs_to_keep,
    #     # visualization=True,  # enable visualization
    # )

    # dataset = env.get_dataset()
    # np.save('dataset/lift_100_raw.npy', dataset)
    degree = -120    #
    # probs = 0.30
    # task = 'lift'
    # build_aug_dataset(task, probs)

    perm = build_perm(degree)
    print(perm)
    #
    # probs = 0.30
    # task = 'lift'
    # build_aug_dataset(task, probs)

    # aug_dataset = build_aug_dataset(dataset, probs, task=task, aug=aug)

    # degree1 = -120
    # degree2 = -240
    # save_path1 = 'data_aug1.npy'
    # save_path2 = 'data_aug2.npy'

    # rot_push(dataset, degree1, save_path1)
    # rot_push(dataset, degree2, save_path2)

    # dataset2 = np.load(save_path1, allow_pickle=True).item()
    # dataset3 = np.load(save_path2, allow_pickle=True).item()

    # obs1 = dataset['observations']
    # actions1 = dataset['actions']
    # rewards1 = dataset['rewards']
    # timeouts1 = dataset['timeouts']

    # obs2 = dataset2['observations']
    # actions2 = dataset2['actions']
    # rewards2 = dataset2['rewards']
    # timeouts2 = dataset2['timeouts']

    # obs3 = dataset3['observations']
    # actions3 = dataset3['actions']
    # rewards3 = dataset3['rewards'],
    # timeouts3 = dataset3['timeouts']

    # obs = np.concatenate([obs1, obs2], axis=0)
    # actions = np.concatenate([actions1, actions2], axis=0)
    # rewards = np.concatanate([rewards1, rewards2], axis=0)
    # timeouts = np.concatanate([timeouts1, timeouts2], axis=0)

    # obs = np.concatenate([obs1, obs2, obs3], axis=0)
    # actions = np.concatenate([actions1, actions2, actions3], axis=0)
    # rewards = np.concatanate([rewards1, rewards2, rewards3], axis=0)
    # timeouts = np.concatanate([timeouts1, timeouts2, timeouts3], axis=0)

