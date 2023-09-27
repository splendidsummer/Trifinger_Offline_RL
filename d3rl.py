import d3rlpy
import gym
import pickle
import wandb
import datetime
import pathlib
import logging, argparse
from d3rlpy.dataset import MDPDataset
import rrc_2022_datasets
from rrc_2022_datasets import TriFingerDatasetEnv
from sklearn.model_selection import train_test_split
from d3rlpy.algos import BC, TD3PlusBC, IQL, CQL, AWAC, \
    BCQ, BEAR, CRR, PLAS, PLASWithPerturbation

from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.preprocessing import action_scalers
import utils
from config import *
import stable_baselines3 as sb3


def main(args):
    WANDB_CONFIG = {
        "task": args.task,
        "algorithm": args.algorithm,
        "n_epochs": args.n_epochs,
    }
    # WANDB_CONFIG.update({'model_config': model_config})
    now = datetime.datetime.now()
    now = now.strftime('%Y%m%d%H%M%S')

    wandb.init(
        job_type='Training',
        project=FEAT_PROJECT_NAME,
        config=WANDB_CONFIG,
        sync_tensorboard=True,
        # entity='Symmetry_RL',
        name='train_' + args.task + '_' + args.algorithm + '_'
             + str(args.n_epochs) + 'epochs' + now,
        # notes = 'some notes related',
        ####
    )

    obs_to_keep = utils.modify_obs_to_keep(args.task)

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
        seed=168,
        disable_env_checker=True,
        flatten_obs=True,
        # obs_to_keep=obs_to_keep,
        visualization=True,  # enable visualization
    )

    dataset = env.get_dataset()

    obs = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    timeouts = dataset['timeouts']

    ##################################################################
    # Preprocess data
    # 1. Delete features not relevant:
    #               1.1 robot_id
    #               1.2 ???
    # 2.Normalize is not appropriate for this case!!
    # 3. Transform key points to position&orientation
    ##################################################################

    dataset = MDPDataset(obs, actions, rewards, timeouts)
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    model = Model(
        use_gpu=False,
        scaler='standard',
        action_scaler='min_max',
    )

    # initialize neural networks with the given observation shape and action size.
    # this is not necessary when you directly call fit or fit_online method.
    model.build_with_dataset(dataset)
    # set environment in scorer function
    # evaluate algorithm on the environment
    evaluate_scorer = evaluate_on_environment(env)

    results = model.fit(
        dataset,
        eval_episodes=test_episodes,
        n_epochs=args.n_epochs,
        scorers={
            'enviroment': evaluate_scorer
        }
    )
    results = [result[1] for result in results]
    for result in results:
        wandb.log({**result})

    print('results:  ', results)
    print('finishing fitting!!')

    model.save_model(args.task + '_' + args.algorithm + '_' + str(args.n_epochs) + 'epochs.pt')
    model.save_policy(args.task + '_' + args.algorithm + '_' + str(args.n_epochs) + 'epochs' + '_policy.pt')
    wandb.save(args.task + '_' + args.algorithm + '_' + str(args.n_epochs) + 'epochs.pt')
    wandb.save(args.task + '_' + args.algorithm + '_' + str(args.n_epochs) + 'epochs' + '_policy.pt')


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
    # parser.add_argument("--policy_path", type=str, help="The path of trained model",)
    # parser.add_argument("--visualization", "-v", action="store_true",
    #     help="Enable visualization of environment.",)
    # parser.add_argument("--n_episodes", type=int, default=64,
    #     help="Number of episodes to run. Default: %(default)s",)
    # parser.add_argument("--output", type=pathlib.Path, metavar="FILENAME",
    #     help="Save results to a JSON file.",)
    args = parser.parse_args()
    main(args)
