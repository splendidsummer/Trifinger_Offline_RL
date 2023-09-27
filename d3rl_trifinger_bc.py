"""
Remarks:
    1. We are using the code modified base on the d3rlpy version 1.1.1.
       Since we got all the non-equivariant baseline based on this version.
"""
import d3rlpy.preprocessing
import gym
import wandb
import datetime
import argparse
import rrc_2022_datasets
from d3rlpy.algos import BC
import pickle
from d3rlpy.metrics.scorer import evaluate_on_environment, continuous_action_diff_scorer
from utils import *
from escnn_model import *


def main(args):
    WANDB_CONFIG = {
        "n_epochs": args.n_epochs,
        "dataset_type": args.dataset_type,
        'learning_rate': args.critic_learning_rate,
        'n_critic': args.n_critics,
        'train_ratio': args.train_ratio,
        'test_ratio': args.test_ratio,
        'escnn': args.escnn,
    }

    now = datetime.datetime.now()
    now = now.strftime('%Y%m%d%H%M%S')

    wandb.init(
        project='Trifinger_Offline_Equivariant_BC',
        config=WANDB_CONFIG,
        sync_tensorboard=True,
        entity='unicore_upc_dl',
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

    ###################################################################################
    # Import the processed and augmented dataset based on the offline Trifinger datast
    ####################################################################################
    # if args.dataset_type == 'raw':
    #     file_name = 'dataset/' + args.task + '_' + str(int(args.probs*100)) + '_' + args.dataset_type + '.npy'
    #     dataset = np.load(file_name, allow_pickle=True).item()
    # elif args.dataset_type == 'aug':
    #     if args.probs <= 0.30:
    #         file_name = 'dataset/' + args.task + '_' + str(int(args.probs * 100)) + '_' + args.dataset_type + '.npy'
    #         dataset = np.load(file_name, allow_pickle=True).item()
    #     else:
    #         file_name = 'dataset/' + args.task + '_' + str(int(args.probs*100)) + '_' + args.dataset_type + '.pck'
    #         with open(file_name, 'rb') as f:
    #             dataset = pickle.load(f)
    #
    # obs = dataset['observations']
    # actions = dataset['actions']
    # rewards = dataset['rewards']
    # timeouts = dataset['timeouts']
    # dataset = MDPDataset(obs, actions, rewards, timeouts)
    # valset = np.load('dataset/lift_valset.npy', allow_pickle=True).item()
    # valset = MDPDataset(valset['observations'], valset['actions'], valset['rewards'], valset['timeouts'])
    # _, test_episodes = train_test_split(dataset, test_size=test_ratio)

    reward_scaler = d3rlpy.preprocessing.StandardRewardScaler()

    model = BC(
        use_gpu=False,
        scaler='standard',
        action_scaler='min_max',
        reward_scaler=reward_scaler,  # Maybe there is a problem
        learning_rate=config.learning_rate,
        n_critics=config.n_critics,
    )

    model.build_with_dataset(dataset)
    evaluate_scorer = evaluate_on_environment(env)

    results = model.fit(
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

    print('results:  ', results)
    print('finishing fitting!!')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of episodes to run. Default: %(default)s", )
    parser.add_argument("--probs",  type=float, default=1.0,
                        help="Percentage of truncated data.",)
    parser.add_argument("--dataset_type",  type=str, choices=["raw", "aug"],
                        help="Whether using raw dataset or using augmented dataset",)
    parser.add_argument("--learning_rate",  type=float, default=0.0003,
                        help="Policy learning rate.",)
    parser.add_argument("--n_critics", type=int, default=2,
                        help='the number of Q functions for ensemble')
    parser.add_argument('--train_ratio', type=float, default=0.2,
                        help="Percentage of truncated data for training.",
                        )
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help="Percentage of truncated data for testing.",
                        )
    parser.add_argument('--escnn', '-e', action='store_true')

    args = parser.parse_args()
    main(args)

