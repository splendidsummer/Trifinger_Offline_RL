import random
import os
import numpy as np
import torch
from typing_extensions import Protocol
from typing import Any, Dict, Iterator, List, Optional
import structlog
from tensorboardX import SummaryWriter
from datetime import datetime
import time
import json
from contextlib import contextmanager
from typing import (cast, List)
import d3rlpy
import math
from sklearn.model_selection import train_test_split
from scipy.spatial.transform import Rotation as R

LIFT_TASK_OBS_DIM = 139
PUSH_TASK_OBS_DIM = 97
ACTION_DIM = 9
MAX_ACTION = 0.397


def test_train_test_split():
    import numpy as np
    from sklearn.model_selection import train_test_split
    X, y = np.arange(10).reshape((5, 2)), range(5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.2, test_size=0.2, shuffle=False)
    print('Stop and debug')


def episodes_to_dataset(episodes):
    observations, actions, rewards, terminals = [], [], [], []
    for episode in episodes:
        observations += episode.observations.tolist()
        actions += episode.actions.tolist()
        rewards += episode.rewards.tolist()
        terminals += [0 for _ in range(len(episode.observations.tolist()) - 1)] + [1]

    observations, actions, rewards, terminals = np.array(observations), \
        np.array(actions, dtype=np.int32), np.array(rewards), np.array(terminals)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations, actions=actions, rewards=rewards, terminals=terminals
    )

    return dataset


def split_trifinger_dataset(dataset, train_ratio, test_ratio):
    train_episodes, test_episodes = train_test_split(dataset.episodes,
                                                     train_size=train_ratio,
                                                     test_size=test_ratio,
                                                     shuffle=False
                                                     )
    trainset = episodes_to_dataset(train_episodes)
    testset = episodes_to_dataset(test_episodes)

    return trainset, testset


def modify_obs_to_keep(task: str) -> dict:

    lift_obs_to_keep = {
        "robot_observation": {
            "position": True,   "velocity": True, "fingertip_force": True, "fingertip_position": True,
            "fingertip_velocity": True, "robot_id": False},
        "object_observation": {
            "position": True, "orientation": True, "keypoints": True,  "delay": False,
            "confidence": False},
        "desired_goal": {
            # "position": True,
            "keypoints": True},
        # "achieved_goal": {"position": False,  "keypoints": False}
    }

    push_obs_to_keep = {
        "robot_observation": {
            "position": True, "velocity": True, "fingertip_force": True, "fingertip_position": True,
            "fingertip_velocity": True, "robot_id": False},
        "object_observation": {
            "position": True, "orientation": True,
            # "keypoints": True,
            "delay": False,
            "confidence": False},
        "desired_goal": {
            "position": True,
            # "keypoints": True
            },
        # "achieved_goal": {"position": False,  "keypoints": False}

    }

    if task == 'push':
        return push_obs_to_keep
    elif task == 'lift':
        return lift_obs_to_keep


def device_handler(args):
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
    args.device = device
    return args


def directory_handler(args):
    if not args.save_path:
        proj_root_path = os.path.split(os.path.realpath(__file__))[0]
        args.save_path = f'{proj_root_path}/save'
    if os.path.split(args.save_path)[-1] != args.task:
        args.save_path = f'{args.save_path}/{args.task}'
        args.dataset_path = f'{args.save_path}/datasets'
        args.model_path = f'{args.save_path}/models'
        if not os.path.exists(args.dataset_path):
            os.makedirs(args.dataset_path)
        print('Exporting the datasets to the folder:', args.dataset_path)
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        print('Exporting the models to the folder:', args.model_path)
    return args


def rrc_task_handler(args):
    if args.task == 'real_lift_mix':
        args.task_type = 'lift'
        args.diff = 'mixed'
        args.obs_dim = 139
        args.action_dim = 9
        args.turns = 5
        args.confs = [1450, 1450, 1450, 1390]
    elif args.task == "real_lift_exp":
        args.task_type = 'lift'
        args.diff = 'expert'
        args.obs_dim = 139
        args.action_dim = 9
    elif args.task == 'real_push_mix':
        args.task_type = 'push'
        args.diff = 'mixed'
        args.obs_dim = 97
        args.action_dim = 9
        args.turns = 4
        args.confs = [730, 730, 500]
    elif args.task == "real_push_exp":
        args.task_type = 'push'
        args.diff = 'expert'
        args.obs_dim = 97
        args.action_dim = 9
    else:
        raise RuntimeError('Invalid input task')
    return args


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rot_augment(raw, angle):
    rotate_matrix = np.asarray([[math.cos(angle), -math.sin(angle)],
                                [math.sin(angle), math.cos(angle)]])
    return np.dot(rotate_matrix, raw)


class _SaveProtocol(Protocol):
    def save_model(self, fname: str) -> None:
        ...

# default json encoder for numpy objects


def default_json_encoder(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise ValueError(f"invalid object type: {type(obj)}")


LOG: structlog.BoundLogger = structlog.get_logger(__name__)


class TrainLogger:

    _experiment_name: str
    _logdir: str
    _save_metrics: bool
    _verbose: bool
    _metrics_buffer: Dict[str, List[float]]
    _params: Optional[Dict[str, float]]
    _writer: Optional[SummaryWriter]

    def __init__(
        self,
        experiment_name: str,
        tensorboard_dir: Optional[str] = None,
        save_metrics: bool = True,
        root_dir: str = "logs",
        verbose: bool = True,
        with_timestamp: bool = True,
    ):
        self._save_metrics = save_metrics
        self._verbose = verbose

        # add timestamp to prevent unintentional overwrites
        while True:
            if with_timestamp:
                date = datetime.now().strftime("%Y%m%d%H%M%S")
                self._experiment_name = experiment_name + "_" + date
            else:
                self._experiment_name = experiment_name

            if self._save_metrics:
                self._logdir = os.path.join(root_dir, self._experiment_name)
                if not os.path.exists(self._logdir):
                    os.makedirs(self._logdir)
                    LOG.info(f"Directory is created at {self._logdir}")
                    break
                if with_timestamp:
                    time.sleep(1.0)
                if os.path.exists(self._logdir):
                    LOG.warning(
                        f"You are saving another logger into {self._logdir}, this may cause unintentional overite")
                    break
            else:
                break

        self._metrics_buffer = {}

        if tensorboard_dir:
            tfboard_path = self._logdir
            self._writer = SummaryWriter(logdir=tfboard_path)
        else:
            self._writer = None

        self._params = None

    def add_params(self, params: Dict[str, Any]) -> None:
        assert self._params is None, "add_params can be called only once."

        if self._save_metrics:
            # save dictionary as json file
            params_path = os.path.join(self._logdir, "params.json")
            with open(params_path, "w") as f:
                json_str = json.dumps(
                    params, default=default_json_encoder, indent=2
                )
                f.write(json_str)

            if self._verbose:
                LOG.info(
                    f"Parameters are saved to {params_path}", params=params
                )
        elif self._verbose:
            LOG.info("Parameters", params=params)

        # remove non-scaler values for HParams
        self._params = {k: v for k, v in params.items() if np.isscalar(v)}

    def add_metric(self, name: str, value: float) -> None:
        if name not in self._metrics_buffer:
            self._metrics_buffer[name] = []
        self._metrics_buffer[name].append(value)

    def commit(self, epoch: int, step: int) -> Dict[str, float]:
        metrics = {}
        for name, buffer in self._metrics_buffer.items():

            metric = sum(buffer) / len(buffer)

            if self._save_metrics:
                path = os.path.join(self._logdir, f"{name}.csv")
                with open(path, "a") as f:
                    print(f"{epoch},{step},{metric}", file=f)

                if self._writer:
                    self._writer.add_scalar(f"metrics/{name}", metric, epoch)

            metrics[name] = metric

        if self._verbose:
            LOG.info(
                f"{self._experiment_name}: epoch={epoch} step={step}",
                epoch=epoch,
                step=step,
                metrics=metrics,
            )

        if self._params and self._writer:
            self._writer.add_hparams(
                self._params,
                metrics,
                name=self._experiment_name,
                global_step=epoch,
            )

        # initialize metrics buffer
        self._metrics_buffer = {}
        return metrics

    def save_model(self, epoch: int, algo: _SaveProtocol) -> None:
        if self._save_metrics:
            # save entire model
            model_path = os.path.join(self._logdir, f"model_{epoch}.pt")
            algo.save_model(model_path)
            LOG.info(f"Model parameters are saved to {model_path}")

    def close(self) -> None:
        if self._writer:
            self._writer.close()

    @contextmanager
    def measure_time(self, name: str) -> Iterator[None]:
        name = "time_" + name
        start = time.time()
        try:
            yield
        finally:
            self.add_metric(name, time.time() - start)

    @property
    def logdir(self) -> str:
        return self._logdir

    @property
    def experiment_name(self) -> str:
        return self._experiment_name


class Normlizor():
    def batch_norm(self, batch_obs):
        if self.mode == 'min_max':
            return (batch_obs[..., :] - self.min) / self.max_min
        elif self.mode == 'std':
            return (batch_obs[..., :] - self.mean) / (self.std)

    def norm(self, obs):
        if self.mode == 'min_max':
            return (obs - self.min) / self.max_min
        elif self.mode == 'std':
            return ((obs - self.mean) / (self.std))

    def noise_batch_norm(self, batch_obs, noise_type='gauss', noisy_ratio=3e-4):
        if noise_type == 'gauss':
            return self.norm(batch_obs) + np.random.normal(0, noisy_ratio, size=batch_obs.shape)
        elif noise_type == 'uniform':
            return self.norm(batch_obs) + np.random.uniform(-noisy_ratio, noisy_ratio, size=batch_obs.shape)

    def noise_batch(self, batch_obs, noise_type='gauss', noisy_ratio=3e-4):
        if noise_type == 'gauss':
            return batch_obs + np.random.normal(0, noisy_ratio, size=batch_obs.shape)
        elif noise_type == 'uniform':
            return batch_obs + np.random.uniform(-noisy_ratio, noisy_ratio, size=batch_obs.shape)

    def init_norm_with_params(self, path, mode):
        self.mode = mode
        params = np.load(path, allow_pickle=True).item()
        self.min = params['min']
        self.max = params['max']
        self.std = params['std']
        self.mean = params['mean']
        self.max_min = self.max - self.min

    def init_norm_with_dataset(self, raw_dataset, mode, name, save_path='default'):
        self.mode = mode
        raw_obs = raw_dataset['observations']
        min_list = raw_obs.min(axis=0)
        max_list = raw_obs.max(axis=0)
        mean_list = raw_obs.mean(axis=0)
        std_list = raw_obs.std(axis=0)

        params = {"min": min_list,
                  "max": max_list,
                  "mean": mean_list,
                  "std": std_list}

        self.min = min_list
        self.max = max_list
        self.std = std_list
        self.mean = mean_list
        self.max_min = self.max - self.min

        if save_path:
            if save_path == 'default':
                save_path = os.path.abspath(os.path.dirname(__file__))
            np.save(f'{save_path}/{name}.npy', params)

