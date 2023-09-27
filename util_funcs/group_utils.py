from escnn.group import Group, directsum, Representation
import torch
import functools
from collections import OrderedDict
import numpy as np
from escnn.group import Representation, directsum
from morpho_symm.groups.isotypic_decomposition import cplx_isotypic_decomposition
from scipy.spatial.transform import Rotation as R
from escnn.nn import FieldType, EquivariantModule


def quaternion2rot(quaternion: torch.Tensor) -> torch.Tensor:
    r = R.from_quat(quaternion)
    rot = torch.tensor(r.as_matrix())
    flatten_rot = torch.flatten(rot, start_dim=1, end_dim=-1)
    return flatten_rot


# Here we should design the process function in model forward call.
def process_trifinger_obs(batch_obs):
    transformed_ob_dim = 148  # to be confirmed
    batch_size = batch_obs.shape[0]
    transformed_obs = torch.zeros((batch_size, transformed_ob_dim), dtype=torch.float32)
    pos_ones = torch.ones((batch_size, 1))
    transformed_obs[:, :24] = batch_obs[:, :24]
    transformed_obs[:, 24:33] = batch_obs[:, 24:33]
    transformed_obs[:, 33: 57] = batch_obs[:, 33: 57]
    transformed_obs[:, 57: 58] = batch_obs[:, 57: 58]
    transformed_obs[:, 58: 59] = batch_obs[:, 58: 59]
    transformed_obs[:, 59: 83] = batch_obs[:, 59: 83]
    transformed_obs[:, 83: 92] = quaternion2rot(batch_obs[:, 83: 87])
    transformed_obs[:, 92: 96] = torch.cat([batch_obs[:, 87: 90], pos_ones], axis=-1)
    transformed_obs[:, 96: 99] = batch_obs[:, 90: 93]
    transformed_obs[:, 99: 103] = torch.cat([batch_obs[:, 93: 96], pos_ones], axis=-1)
    transformed_obs[:, 103: 107] = torch.cat([batch_obs[:, 96: 99], pos_ones], axis=-1)
    transformed_obs[:, 107: 111] = torch.cat([batch_obs[:, 99: 102], pos_ones], axis=-1)
    transformed_obs[:, 111: 120] = batch_obs[:, 102: 111]
    transformed_obs[:, 120: 129] = batch_obs[:, 111: 120]
    transformed_obs[:, 129:130] = batch_obs[:, 120: 121]
    transformed_obs[:, 130: 139] = batch_obs[:, 121: 130]
    transformed_obs[:, 139: 148] = batch_obs[:, 130: 139]

    return transformed_obs


# def identify_isotypic_spaces(rep: Representation) -> OrderedDict[tuple: Representation]:
def identify_isotypic_spaces(rep: Representation):
    """
    Identify the isotypic subspaces of a representation. See Isotypic Basis for more details (TODO).
    Args:
        rep (Representation): Input representation in any arbitrary basis.

    Returns: A `Representation` with a change of basis exposing an Isotypic Basis (a.k.a symmetry enabled basis).
        The instance of the representation contains an additional parameter `isotypic_subspaces` which is an
        `OrderedDict` of representations per each isotypic subspace. The keys are the active irreps' ids associated
        with each Isotypic subspace.
    """

    symm_group = rep.group
    potential_irreps = rep.group.irreps()
    isotypic_subspaces_indices = {irrep.id: [] for irrep in potential_irreps}
    for irrep in potential_irreps:
        for index, rep_irrep_id in enumerate(rep.irreps):
            if symm_group.irrep(*rep_irrep_id) == irrep:
                isotypic_subspaces_indices[rep_irrep_id].append(index)

    # Remove inactive Isotypic Spaces
    for irrep in potential_irreps:
        if len(isotypic_subspaces_indices[irrep.id]) == 0:
            del isotypic_subspaces_indices[irrep.id]

    # Each Isotypic Space will be indexed by the irrep it is associated with.
    active_isotypic_reps = {}
    for irrep_id, indices in isotypic_subspaces_indices.items():
        # if indices are not consecutive numbers raise an error
        if not np.all(np.diff(indices) == 1):
            raise NotImplementedError("TODO: Add permutations needed to handle this case")
        irrep = symm_group.irrep(*irrep_id)
        multiplicities = len(indices)
        active_isotypic_reps[irrep_id] = Representation(group=rep.group,
                                                        irreps=[irrep_id] * multiplicities,
                                                        name=f'IsoSubspace {irrep_id}',
                                                        change_of_basis=np.identity(irrep.size * multiplicities),
                                                        supported_nonlinearities=irrep.supported_nonlinearities
                                                        )

    # Impose canonical order on the Isotypic Subspaces.
    # If the trivial representation is active it will be the first Isotypic Subspace.
    # Then sort by dimension of the space from smallest to largest.
    ordered_isotypic_reps = OrderedDict(sorted(active_isotypic_reps.items(), key=lambda item: item[1].size))
    if symm_group.trivial_representation.id in ordered_isotypic_reps.keys():
        ordered_isotypic_reps.move_to_end(symm_group.trivial_representation.id, last=False)

    # Compute the decomposition of Real Irreps into Complex Irreps and store this information in irrep.attributes
    # cplx_irrep_i(g) = Q_re2cplx @ re_irrep_i(g) @ Q_re2cplx^-1`
    for irrep_id in ordered_isotypic_reps.keys():
        re_irrep = symm_group.irrep(*irrep_id)
        cplx_subreps, Q_re2cplx = cplx_isotypic_decomposition(symm_group, re_irrep)
        re_irrep.is_cplx_irrep = len(cplx_subreps) == 1
        symm_group.irrep(*irrep_id).attributes['cplx_irreps'] = cplx_subreps
        symm_group.irrep(*irrep_id).attributes['Q_re2cplx'] = Q_re2cplx

    new_rep = directsum(list(ordered_isotypic_reps.values()),
                        name=rep.name + '-Iso',
                        change_of_basis=None)  # TODO: Check for additional permutations

    iso_supported_nonlinearities = [iso_rep.supported_nonlinearities for iso_rep in ordered_isotypic_reps.values()]
    new_rep.supported_nonlinearities = functools.reduce(set.intersection, iso_supported_nonlinearities)
    new_rep.attributes['isotypic_reps'] = ordered_isotypic_reps
    return new_rep, rep.change_of_basis


def compute_invariant_features(x: torch.Tensor, field_type: FieldType) -> torch.Tensor:
    n_inv_features = len(field_type.irreps)
    # TODO: Ensure isotypic basis i.e irreps of the same type are consecutive to each other.
    inv_features = []
    for field_start, field_end, rep in zip(field_type.fields_start, field_type.fields_end, field_type.representations):
        # Each field here represents a representation of an Isotypic Subspace. This rep is only composed of a single
        # irrep type.
        x_field = x[..., field_start:field_end]
        num_G_stable_spaces = len(rep.irreps)  # Number of G-invariant features = multiplicity of irrep
        # Again this assumes we are already in an Isotypic basis
        assert len(np.unique(rep.irreps, axis=0)) == 1, "This only works for now on the Isotypic Basis"
        # This basis is useful because we can apply the norm in a vectorized way
        # Reshape features to [batchg, num_G_stable_spaces, num_features_per_G_stable_space]
        x_field_p = torch.reshape(x_field, (x_field.shape[0], num_G_stable_spaces, -1))
        # Compute G-invariant measures as the norm of the features in each G-stable space
        inv_field_features = torch.norm(x_field_p, dim=-1)
        # Append to the list of inv features
        inv_features.append(inv_field_features)
    # Concatenate all the invariant features
    inv_features = torch.cat(inv_features, dim=-1)
    assert inv_features.shape[-1] == n_inv_features, f"Expected {n_inv_features} got {inv_features.shape[-1]}"
    return inv_features


def isotypic_basis(group: Group, num_regular_fields: int, prefix=''):
    rep, _ = identify_isotypic_spaces(group.regular_representation)
    # Construct the obs state representation as a `num_regular_field` copies of the isotypic representation
    iso_reps = OrderedDict()
    for iso_irrep_id, reg_rep_iso in rep.attributes['isotypic_reps'].items():
        iso_reps[iso_irrep_id] = directsum([reg_rep_iso] * num_regular_fields,
                                           name=f"{prefix}_IsoSpace{iso_irrep_id}")
    return iso_reps


if __name__ == '__main__':
    # robot_name1 = 'Trifinger'  # or any of the robots in the library (see `/morpho_symm/cfg/robot`)
    # initialize(config_path="../cfg/supervised/robot", version_base='1.3')
    # trifinger_group_cfg = compose(config_name=f"{robot_name1.lower()}.yaml")
    # G, value_in_type, policy_in_type, value_out_type, policy_out_type = load_trifinger_G(trifinger_group_cfg)
    #
    # emlp_args = {
    #     'units': units,
    #     'activation': escnn.nn.ReLU,
    #     'in_type': policy_in_type,
    #     'out_type': policy_out_type,
    # }
    #
    # batch_size = 8
    # trifinger_state_dim = 157
    # inputs = torch.randn((batch_size, trifinger_state_dim))
    # trifinger_encoder = EMLPNoLast(**emlp_args)

    # out = trifinger_encoder(inputs)
    # encoder_out_type = trifinger_encoder.net[-1].out_type
    # inv_features = compute_invariant_features(out, encoder_out_type)

    quaternion_mat = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]], dtype=torch.float32)
    rot_mat = quaternion2rot(quaternion_mat)
    batch_obs = torch.randn((8, 139))
    trans_obs = process_trifinger_obs(batch_obs)
    print(trans_obs[0])
