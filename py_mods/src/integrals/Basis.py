from py_mods.src.integrals.CGTO import (
    S_GTO_mat,
    T_GTO_mat,
    V_GTO_mat,
    Eri_GTO_tensor,
    CGTOClass,
)
from dataclasses import dataclass
from typing import List
from numpy.typing import NDArray
import numpy as np


@dataclass
class BasisSetClass:
    CGTOs: List[CGTOClass]
    n_CGTOs: int
    matrix_indices: NDArray[np.int16]
    l_dims: NDArray[np.int16]
    r_atoms: NDArray[np.float64]
    q_atoms: NDArray[np.float64]


def construct_basis_from_lists(
    cgto_list: list[CGTOClass],
    r_atom_list: NDArray[np.float64],
    q_atom_list: NDArray[np.float64],
) -> BasisSetClass:
    """
    Constructs a BasisSetClass object from lists of CGTOs and atomic positions/charges.

    Parameters
    ----------
    cgto_list : list[CGTOClass]
        List of CGTOClass objects representing the basis functions.
    r_atom_list : NDArray[np.float64]
        Array of atomic positions (shape: (N_atoms, 3)).
    q_atom_list : NDArray[np.float64]
        Array of atomic charges (shape: (N_atoms,)).

    Returns
    -------
    BasisSetClass
        An instance of BasisSetClass containing the provided CGTOs and atomic data.
    """

    if len(r_atom_list) != len(q_atom_list):
        raise ValueError("Length of r_atom_list and q_atom_list must be the same.")

    projections = [basis.l_dim for basis in cgto_list]
    cgto_indices = [sum(projections[:i]) for i in range(len(cgto_list))]
    cgto_indices.append(projections[-1] + cgto_indices[-1])

    cgto_indices = np.array(cgto_indices, dtype=np.int16)

    basis_set = BasisSetClass(
        CGTOs=cgto_list,
        n_CGTOs=len(cgto_list),
        matrix_indices=cgto_indices,
        l_dims=np.array(projections, dtype=np.int16),
        r_atoms=np.array(r_atom_list, dtype=np.float64),
        q_atoms=np.array(q_atom_list, dtype=np.float64),
    )

    return basis_set


def S_basis_set(basis_set: BasisSetClass) -> NDArray[np.float64]:
    """
    Calculate overlap matrix of a given basis set.

    Parameters
    ----------
    basis_set : BasisSetClass
        Basis set.

    Returns
    -------
    NDArray[np.float64]
        Overlap matrix.
    """
    n_CGTOs = basis_set.n_CGTOs
    S_mat = np.zeros((basis_set.matrix_indices[-1], basis_set.matrix_indices[-1]))

    for i in range(n_CGTOs):
        for j in range(n_CGTOs):

            index_i = basis_set.matrix_indices[i]
            index_i_stop = basis_set.matrix_indices[i + 1]
            index_j = basis_set.matrix_indices[j]
            index_j_stop = basis_set.matrix_indices[j + 1]

            S_mat[index_i:index_i_stop, index_j:index_j_stop] = S_GTO_mat(
                basis_set.CGTOs[i], basis_set.CGTOs[j]
            )

    return S_mat


def V_basis_set(basis_set: BasisSetClass) -> NDArray[np.float64]:
    """
    Calculate potential matrix of a given basis set.

    Parameters
    ----------
    basis_set : BasisSetClass
        Basis set.

    Returns
    -------
    NDArray[np.float64]
        Potential matrix.
    """
    n_CGTOs = basis_set.n_CGTOs
    S_mat = np.zeros((basis_set.matrix_indices[-1], basis_set.matrix_indices[-1]))

    for i in range(n_CGTOs):
        for j in range(n_CGTOs):

            index_i = basis_set.matrix_indices[i]
            index_i_stop = basis_set.matrix_indices[i + 1]
            index_j = basis_set.matrix_indices[j]
            index_j_stop = basis_set.matrix_indices[j + 1]

            S_mat[index_i:index_i_stop, index_j:index_j_stop] = V_GTO_mat(
                basis_set.CGTOs[i],
                basis_set.CGTOs[j],
                basis_set.r_atoms,
                basis_set.q_atoms,
            )

    return S_mat


def T_basis_set(basis_set: BasisSetClass) -> NDArray[np.float64]:
    """
    Calculate kinetic matrix of a given basis set.

    Parameters
    ----------
    basis_set : BasisSetClass
        Basis set.

    Returns
    -------
    NDArray[np.float64]
        Kinetic matrix.
    """
    n_CGTOs = basis_set.n_CGTOs
    S_mat = np.zeros((basis_set.matrix_indices[-1], basis_set.matrix_indices[-1]))

    for i in range(n_CGTOs):
        for j in range(n_CGTOs):

            index_i = basis_set.matrix_indices[i]
            index_i_stop = basis_set.matrix_indices[i + 1]
            index_j = basis_set.matrix_indices[j]
            index_j_stop = basis_set.matrix_indices[j + 1]

            S_mat[index_i:index_i_stop, index_j:index_j_stop] = T_GTO_mat(
                basis_set.CGTOs[i], basis_set.CGTOs[j]
            )

    return S_mat


def eri_basis_set(basis_set: BasisSetClass) -> NDArray[np.float64]:
    """
    Calculate electron repulsion integral (ERI) tensor of a given basis set.

    Parameters
    ----------
    basis_set : BasisSetClass
        Basis set.

    Returns
    -------
    NDArray[np.float64]
        ERI tensor.
    """
    n_CGTOs = basis_set.n_CGTOs
    dim = basis_set.matrix_indices[-1]
    eri_tensor = np.zeros((dim, dim, dim, dim))

    for i in range(n_CGTOs):
        for j in range(n_CGTOs):
            for k in range(n_CGTOs):
                for l in range(n_CGTOs):

                    index_i = basis_set.matrix_indices[i]
                    index_i_stop = basis_set.matrix_indices[i + 1]
                    index_j = basis_set.matrix_indices[j]
                    index_j_stop = basis_set.matrix_indices[j + 1]
                    index_k = basis_set.matrix_indices[k]
                    index_k_stop = basis_set.matrix_indices[k + 1]
                    index_l = basis_set.matrix_indices[l]
                    index_l_stop = basis_set.matrix_indices[l + 1]

                    eri_tensor[
                        index_i:index_i_stop,
                        index_j:index_j_stop,
                        index_k:index_k_stop,
                        index_l:index_l_stop,
                    ] = Eri_GTO_tensor(
                        basis_set.CGTOs[i],
                        basis_set.CGTOs[j],
                        basis_set.CGTOs[k],
                        basis_set.CGTOs[l],
                    )

    return eri_tensor
