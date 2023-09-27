from build_reps import load_trifinger_G
from typing import List, Tuple, Union
import d3rlpy.models.encoders
import torch
import torch.cuda
import logging
import escnn.group
from models import EMLP
from util_funcs import group_utils
from build_reps import *

log = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################################################################
# Setting the structure params of Trifinger for EMLP model
########################################################################
trifinger_gspace = escnn.gspaces.no_base_space(Trifinger_G)
trifinger_activation = escnn.nn.ReLU
trifinger_n_hidden_neurons = 256
trifinger_num_regular_field = int(np.ceil(trifinger_n_hidden_neurons / Trifinger_G.order()))
# Compute the observation space Isotypic Rep from the regular representation
# Define the observation space in the ISOTYPIC BASIS!
trifinger_rep_features_iso_basis = group_utils.isotypic_basis(Trifinger_G, trifinger_num_regular_field, prefix='ObsSpace')
trifinger_inv_encoder_out_type = FieldType(trifinger_gspace, [rep_iso for rep_iso in trifinger_rep_features_iso_basis.values()])

##################################################################
# Setting the hyparams for noraml NN after equivariant encoder
##################################################################
non_constrain_hidden_neurons = 256
non_constrain_hidden_layer = 1
non_constrain_activation = 'elu'  # or relu??

##################################################################
# Argments updated for the newest EMLP impl
#################################################################

categorical_emlp_args = {
    'in_type': policy_in_type,
    'out_type': categorical_prob_type,
    'activation': "ELU",
}

inv_encoder_emlp_args = {}

trifinger_actor_emlp_args = {
    'in_type': Trifinger_policy_in_type,
    'out_type': Trifinger_policy_out_type,
    'num_hidden_units': 256,
    'num_layers': 2,
    'activation': 'ELU',
}

trifinger_critic_emlp_args = {
    'in_type': Trifinger_value_in_type,
    'out_type': trifinger_inv_encoder_out_type,
    'num_hidden_units': trifinger_n_hidden_neurons,
    'num_layers': 2,
    'activation': 'ELU',
}

trifinger_value_emlp_args = {
    'in_type': Trifinger_policy_in_type,
    'out_type': trifinger_inv_encoder_out_type,
    'num_hidden_units': trifinger_n_hidden_neurons,
    'num_layers': 2,
    'activation': 'ELU',
}


class CartpoleInvEncoder(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.network = EMLP(**inv_encoder_emlp_args)
        # Discrete policy are modified inside d3rlpy

    def forward(self, inputs):
        outs = self.network(inputs).tensor
        return outs

    def get_feature_size(self):
        return self.feature_size


class CartpoleEnvEncoder(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.network = EMLP(**categorical_emlp_args)

    def forward(self, inputs):
        outs = self.network(inputs).tensor
        return outs

    def get_feature_size(self):
        return self.feature_size


class CartpoleInvEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = 'CartpoleInv'
    def __init__(self, feature_size=256):
        self.feature_size = feature_size
    def create(self, observation_shape):
        return CartpoleInvEncoder(self.feature_size)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}

    @staticmethod
    def get_type() -> str:
        return 'CartpoleInv'


class CartpoleEnvEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = 'CartpoleEnv'
    def __init__(self, feature_size=256):
        self.feature_size = feature_size
    def create(self, observation_shape):
        return CartpoleEnvEncoder(self.feature_size)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}

    @staticmethod
    def get_type() -> str:
        return 'CartpoleEnv'


class TrifingerEnvEncoder(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.network = EMLP(**trifinger_actor_emlp_args)

    def forward(self, inputs):
        inputs = group_utils.process_trifinger_obs(inputs)
        outs = self.network(inputs).tensor
        return outs

    def get_feature_size(self):
        return self.feature_size


class TrifingerEnvEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = 'CartpoleEnv'
    def __init__(self, feature_size=256):
        self.feature_size = feature_size
    def create(self, observation_shape):
        return TrifingerEnvEncoder(self.feature_size)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}

    @staticmethod
    def get_type() -> str:
        return 'CartpoleEnv'


class TrifingerInvCriticEncoder(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.network = EMLP(**trifinger_critic_emlp_args)
        self.normal_block = torch.nn.Sequential()
        for i in range(non_constrain_hidden_layer):
            if i == 0:
                self.normal_block.add_module(f'non-constrain-layer-{i}',
                                             torch.nn.Linear(len(trifinger_inv_encoder_out_type.irreps),
                                                            non_constrain_hidden_neurons))
            else:
                self.normal_block.add_module(f'non-constrain-layer-{i}',
                                             torch.nn.Linear(non_constrain_hidden_neurons,
                                                            non_constrain_hidden_neurons))

            self.normal_block.add_module(f'non-constrain-activation-{i}',
                                         torch.nn.ReLU(),
                                         # torch.nn.ELU(),
                                         )

        self.head = torch.nn.Linear(non_constrain_hidden_neurons, 1)

    def forward(self, inputs):
        """
        Args:
            inputs:  for critic the inputs are expected be the concatenation of 139 dimensional obs and 9 dimensional actions.
        Remarks:
            TODO: truncate the first 139 dimensional inputs, and the preprocess the truncated inputs according process_trifinger_obs func

        Returns:

        """
        obs = group_utils.process_trifinger_obs(inputs[:, :139])
        actions = inputs[:, 139:]
        inputs = torch.cat([obs, actions], dim=-1)
        # step 1. compute the output of the equivariant encoder
        outs = self.network(inputs)
        # step 2: extract the invariant features from the above output
        out_type = self.network.out_type
        inv_features = group_utils.compute_invariant_features(outs.tensor, out_type)
        # step 3: feed the invariant features into normal nn and extract more expressive feature.
        outs = self.normal_block(inv_features)
        # step 4: Compute the Q value from the last hidden layer
        outs = self.head(outs)
        return outs

    def get_feature_size(self):
        return self.feature_size


class TrifingerInvCriticEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = 'CartpoleEnv'
    def __init__(self, feature_size=256):
        self.feature_size = feature_size
    def create(self, observation_shape):
        return TrifingerInvCriticEncoder(self.feature_size)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}

    @staticmethod
    def get_type() -> str:
        return 'CartpoleEnv'


class TrifingerInvValueEncoder(torch.nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.network = EMLP(**trifinger_critic_emlp_args)

    def forward(self, inputs):
        inputs = group_utils.process_trifinger_obs(inputs)
        # TODO: proprecess the quaterion
        outs = self.network(inputs)
        out_type = self.network[-1].out_type
        return outs

    def get_feature_size(self):
        return self.feature_size


class TrifingerInvValueEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = 'CartpoleEnv'
    def __init__(self, feature_size=256):
        self.feature_size = feature_size
    def create(self, observation_shape):
        return TrifingerInvValueEncoder(self.feature_size)

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size}

    @staticmethod
    def get_type() -> str:
        return 'CartpoleEnv'


if __name__ == '__main__':
    # customized_encoder_factory = TrifingerEnvEncoderFactory()
    # trifinger_critic_encoder_factory = TrifingerInvCriticEncoderFactory()
    # trifinger_value_encoder_factory = TrifingerInvValueEncoderFactory()

    critic_encoder = TrifingerInvCriticEncoder(256)
    batch_ob_actions = torch.randn((4, 148))
    out = critic_encoder(batch_ob_actions)