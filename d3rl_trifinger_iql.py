import d3rlpy
import pickle
import gym
import wandb
import datetime
import logging, argparse
import numpy as np
import rrc_2022_datasets
from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import evaluate_on_environment, \
    continuous_action_diff_scorer
from d3rlpy.algos import IQL
from utils import *
from escnn_model import *


def main(args):
    WANDB_CONFIG = {
        "n_epochs": args.n_epochs,
        "dataset_type": args.dataset_type,
        'actor_learning_rate': args.actor_learning_rate,
        'critic_learning_rate': args.critic_learning_rate,
        'expectile': args.expectile,
        'n_critic': args.n_critics,
        'train_ratio': args.train_ratio,
        'test_ratio': args.test_ratio,
        'escnn': args.escnn,
    }

    now = datetime.datetime.now()
    now = now.strftime('%Y%m%d%H%M%S')

    wandb.init(
        project='Trifinger_Offline_Equivariant_IQL',
        config=WANDB_CONFIG,
        sync_tensorboard=True,
        # entity='Symmetry_RL',
        name=
            'escnn_' + str(wandb.escnn) + '_' +
            'train_' + str(WANDB_CONFIG['train_ratio']) + '_' +
            'test_' + str(WANDB_CONFIG['test_ratio']) + '_' +
            # 'augmentation_ ' + args.dataset_type + '_'
            str(args.n_epochs) + 'epochs' +
            now,
    )

    config = wandb.config
    env_name = "trifinger-cube-lift-sim-expert-v0"
    env = gym.make(
        env_name,
        disable_env_checker=True,
        flatten_obs=True,
        # obs_to_keep=obs_to_keep,
        # visualization=True,  # enable visualization
    )

    dataset = env.get_dataset()
    dataset, valset = train_test_split(dataset,
                                       train_size=config.train_ratio,
                                       test_size=config.test_ratio)

    actor_encoder_factory = TrifingerEnvEncoderFactory()
    critic_encoder_factory = TrifingerInvCriticEncoderFactory()
    value_encoder_factory = TrifingerInvValueEncoderFactory()
    reward_scaler = d3rlpy.preprocessing.StandardRewardScaler()

    iql = IQL(
        actor_encoder_factory=actor_encoder_factory,
        critic_encoder_factory=critic_encoder_factory,
        value_encoder_factory=value_encoder_factory,
        observation_scaler='standard',
        reward_scaler=reward_scaler,
        action_scaler='min_max',
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        n_critics=args.n_critics,
        expectile=args.expectile,
    ).create(device=None)

    iql.build_with_dataset(dataset)
    evaluate_scorer = evaluate_on_environment(env)

    results = iql.fit(
        dataset,
        eval_episodes=valset,
        n_epochs=config.n_epochs,
        scorers={
            'return': evaluate_scorer,
            'val_loss': continuous_action_diff_scorer
        }
    )

    results = [result[1] for result in results]
    for result in results:
        wandb.log({**result})

    print('results: ', results)
    print('finishing fitting!!')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of episodes to run. Default: %(default)s", )
    parser.add_argument("--probs",  type=float, default=1.0,
                        help="Percentage of truncated data.",)
    parser.add_argument("--dataset_type",  type=str, default='raw', choices=["raw", "aug"],
                        help="Whether using raw dataset or using augmented dataset",)
    parser.add_argument("--actor_learning_rate",  type=float, default=0.0003,
                        help="Actor learning rate.",)
    parser.add_argument("--critic_learning_rate",  type=float, default=0.0003,
                        help="Critic learning rate.",)
    parser.add_argument('--train_ratio', type=float, default=0.2,
                        help="Percentage of truncated data for training.",
                        )
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help="Percentage of truncated data for testing.",
                        )
    parser.add_argument("--n_critics", type=int, default=2,
                        help='the number of Q functions for ensemble')
    parser.add_argument("--expectile",  type=float, default=0.7, help="the expectile value for value function training",)
    parser.add_argument('--escnn', '-e', action='store_true')

    args = parser.parse_args()
    main(args)

