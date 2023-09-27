"""
Remarks:
    1. We are using the code modified base on the d3rlpy version 1.1.1
    2. 
"""
import d3rlpy
import gym
import seaborn as sns
import wandb
import os
import datetime
import pathlib
import logging, argparse
import numpy as np
from d3rlpy.dataset import MDPDataset
import rrc_2022_datasets
from rrc_2022_datasets import TriFingerDatasetEnv
from sklearn.model_selection import train_test_split
from d3rlpy.algos import BC, TD3PlusBC, IQL, CQL, AWAC, \
    BCQ, BEAR, CRR, PLAS, PLASWithPerturbation
import pickle
from d3rlpy.metrics.scorer import evaluate_on_environment, continuous_action_diff_scorer
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.preprocessing import action_scalers
import utils
from config import *
from escnn_model import *


def main(args):
    WANDB_CONFIG = {
        "task": args.task,
        "algorithm": args.algorithm,
        "n_epochs": args.n_epochs,
        "probs": args.probs,
        "dataset_type": args.dataset_type,
        'seed': args.seed,
        'actor_learning_rate': args.actor_learning_rate,
        'critic_learning_rate': args.critic_learning_rate,
        # 'alpha': args.alpha,
        'expectile': args.expectile,
        'n_critic': args.n_critics
    }
    # WANDB_CONFIG.update({'model_config': model_config})
    now = datetime.datetime.now()
    now = now.strftime('%Y%m%d%H%M%S')

    wandb.init(
        job_type='Data augmentation',
        project='Trifinger_' + args.task +'_Dataaugmentation',
        config=WANDB_CONFIG,
        sync_tensorboard=True,
        # entity='Symmetry_RL',
        name='train_' + args.task + '_' + args.algorithm + '_' + args.dataset_type + '_'
             + str(int(100*args.probs)) + '_' + str(args.n_epochs) + 'epochs' + now,
    )

    if args.task == "push":
        env_name = "trifinger-cube-push-sim-expert-v0"
    elif args.task == "lift":
        env_name = "trifinger-cube-lift-sim-expert-v0"
    else:
        print("Invalid task %s" % args.task)

    if args.algorithm == "bc":
        Model = BC
    elif args.algorithm == "td3+bc":
        Model = TD3PlusBC
    elif args.algorithm == "iql":
        Model = IQL
    elif args.algorithm == "cql":
        Model = CQL
    elif args.algorithm == "awac":
        Model = AWAC
    elif args.algorithm == "bcq":
        Model = BCQ
    elif args.algorithm == "crr":
        Model = CRR
    elif args.algorithm == "plas":
        Model = PLAS
    elif args.algorithm == "plaswp":
        Model = PLASWithPerturbation

    env = gym.make(
        env_name,
        disable_env_checker=True,
        # flatten_obs=False,
        flatten_obs=True,
        # obs_to_keep=obs_to_keep,
        # visualization=True,  # enable visualization
    )

    # dataset = env.get_dataset()

    if args.dataset_type == 'raw':
        file_name = 'dataset/' + args.task + '_' + str(int(args.probs*100)) + '_' + args.dataset_type + '.npy'
        dataset = np.load(file_name, allow_pickle=True).item()
    elif args.dataset_type == 'aug':
        if args.probs <= 0.30:
            file_name = 'dataset/' + args.task + '_' + str(int(args.probs * 100)) + '_' + args.dataset_type + '.npy'
            dataset = np.load(file_name, allow_pickle=True).item()
        else:
            file_name = 'dataset/' + args.task + '_' + str(int(args.probs*100)) + '_' + args.dataset_type + '.pck'
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)

    obs = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    timeouts = dataset['timeouts']
    dataset = MDPDataset(obs, actions, rewards, timeouts)
    valset = np.load('dataset/lift_valset.npy', allow_pickle=True).item()
    valset = MDPDataset(valset['observations'], valset['actions'], valset['rewards'], valset['timeouts'])

    # _, test_episodes = train_test_split(dataset, test_size=test_ratio)

    # Actor Encoder Factory: Output the mean value of the 9-dimensional action,
    #                        Output the standard deviation of the 9-dimensional action
    actor_encoder_factory = MainEncoderFactory(trifinger_actor_emlp_args)
    # equivariant encoder --> extract invariant features --> builing non-constrained NN block on top
    # inputs: concatenation of observation and actions
    critic_encoder_factory = MainEncoderFactory(trifinger_critic_emlp_args)
    # equivariant encoder --> extract invariant features --> builing non-constrained NN block on top
    # inputs: observations
    value_encoder_factory = MainEncoderFactory(trifinger_value_emlp_args)

    model = Model(
        # seed=args.seed,
        use_gpu=False,
        scaler='standard',
        action_scaler='min_max',
        actor_encoder_factory=actor_encoder_factory,
        critic_encoder_factory=critic_encoder_factory,
        value_encoder_factory=value_encoder_factory,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        n_critics=args.n_critics,
        expectile=args.expectile,
        # alpha=args.alpha,
    )

    # initialize neural networks with the given observation shape and action size.
    # this is not necessary when you directly call fit or fit_online method.
    model.build_with_dataset(dataset)
    # set environment in scorer function
    # evaluate algorithm on the environment
    evaluate_scorer = evaluate_on_environment(env)

    results = model.fit(
        dataset,
        eval_episodes=valset,
        n_epochs=args.n_epochs,
        scorers={
            'return': evaluate_scorer,
            'val_loss': continuous_action_diff_scorer
        }
    )
    results = [result[1] for result in results]
    for result in results:
        wandb.log({**result})

    print('results:  ', results)
    print('finishing fitting!!')

    # model_path = args.task + '_' + args.algorithm + '_' + args.dataset_type + '_' + str(int(100*args.probs)) + '_' + \
    #             str(args.n_epochs) + 'epochs' + str(int(args.seed)) + '.pt'

    # policy_path = args.task + '_' + args.algorithm + '_' + args.dataset_type + '_' + str(int(100*args.probs)) + '_' + \
    #              str(args.n_epochs) + 'epochs' + str(int(args.seed)) + '_policy.pt'

    # model.save_model(os.path.join(wandb.run.dir, model_path))
    # model.save_policy(os.path.join(wandb.run.dir, policy_path))

    # wandb.save(args.task + '_' + args.algorithm + '_' + args.dataset_type + '_' + str(int(100*args.probs)) + '_' +
    #            str(args.n_epochs) + 'epochs' + '.pt')
    # wandb.save(args.task + '_' + args.algorithm + '_' + args.dataset_type + '_' + str(int(100*args.probs)) + '_' +
    #            str(args.n_epochs) + 'epochs' + '_policy.pt')
    #


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, choices=["push", "lift"],
                        help="Which task to evaluate ('push' or 'lift').", )
    parser.add_argument("--algorithm", type=str, choices=
    ["bc", "td3+bc", "iql", "cql", "awac", "bcq", "bear", "crr", "plas", "plaswp"],
                        help="Which algorithm to train ('push' or 'lift').", )
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of episodes to run. Default: %(default)s", )
    parser.add_argument("--seed", type=int, default=168,
                        help="Random seed number!", )
    parser.add_argument("--probs",  type=float, default=1.0,
                        help="Percentage of truncated data.",)
    parser.add_argument("--dataset_type",  type=str, choices=["raw", "aug"],
                        help="Whether using raw dataset or using augmented dataset",)
    parser.add_argument("--actor_learning_rate",  type=float, default=0.0003,
                        help="Actor learning rate.",)
    parser.add_argument("--critic_learning_rate",  type=float, default=0.0003,
                        help="Critic learning rate.",)
    parser.add_argument("--n_critics", type=int, default=2,
                        help='the number of Q functions for ensemble')
    # parser.add_argument("--alpha",  type=float, default=2.5, help="the expectile value for value function training",)
    parser.add_argument("--expectile",  type=float, default=0.7, help="the expectile value for value function training",)
    # parser.add_argument("--policy_path", type=str, help="The path of trained model",)
    # parser.add_argument("--visualization", "-v", action="store_true",
    #     help="Enable visualization of environment.",)
    # parser.add_argument("--n_episodes", type=int, default=64,
    #     help="Number of episodes to run. Default: %(default)s",)
    # parser.add_argument("--output", type=pathlib.Path, metavar="FILENAME",
    #     help="Save results to a JSON file.",)

    args = parser.parse_args()
    main(args)

