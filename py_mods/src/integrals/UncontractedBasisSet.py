from typing import List, Literal
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from py_mods.src.integrals.GTO import (
    GTO,
    create_normalized_GTO,
    S_ab_shell,
    T_ab_shell,
    g_abcd_shell,
)
from py_mods.src.integrals.internal.coulomb_utils import V_ab_Z_shell
from py_mods.src.SCF.plot_utilities import plot_map

# =============================================================================
#  Dataclasses
# =============================================================================


@dataclass
class UncontractedBasisSet:
    GTO_list: List[GTO]
    n_GTOs: int
    l_dims: NDArray[np.int16]
    start_indices: NDArray[np.int16]
    end_indices: NDArray[np.int16]
    n_mat_elem: int


@dataclass
class AtomSet:
    r_atoms: NDArray[np.float64]
    q_atoms: NDArray[np.float64]
    n_atoms: int


# =============================================================================
# Constructors
# =============================================================================


def create_UncontractedBasisSet(GTO_list: List[GTO]) -> UncontractedBasisSet:

    n_gtos = len(GTO_list)
    l_dims = np.array([len(gto.l_projections) for gto in GTO_list], dtype=np.int16)
    start_indices = np.zeros_like(l_dims, dtype=np.int16)
    end_indices = np.zeros_like(l_dims, dtype=np.int16)
    for i in range(n_gtos):
        start_indices[i] = sum(l_dims[:i])
        end_indices[i] = start_indices[i] + l_dims[i]

    n_mat_elem = int(np.sum(l_dims))

    return UncontractedBasisSet(
        GTO_list=GTO_list,
        n_GTOs=n_gtos,
        l_dims=l_dims,
        start_indices=start_indices,
        end_indices=end_indices,
        n_mat_elem=n_mat_elem,
    )


def create_AtomSet(atom_positions, atom_charges) -> AtomSet:
    r_atoms = np.array(atom_positions, dtype=np.float64)
    q_atoms = np.array(atom_charges, dtype=np.float64)
    n_atoms = len(q_atoms)
    return AtomSet(r_atoms=r_atoms, q_atoms=q_atoms, n_atoms=n_atoms)


# =============================================================================
# Matrix element evaluation
# =============================================================================


def S_UncontractedBasisSet(
    UBS: UncontractedBasisSet, symmetric: bool = True
) -> NDArray[np.float64]:
    n_mat_elem = UBS.n_mat_elem

    S = np.zeros((n_mat_elem, n_mat_elem), dtype=np.float64)
    n_gtos = UBS.n_GTOs

    for i in range(n_gtos):
        for j in range(i, n_gtos):
            i0 = int(UBS.start_indices[i])
            i1 = int(UBS.end_indices[i])
            j0 = int(UBS.start_indices[j])
            j1 = int(UBS.end_indices[j])

            S_block = S_ab_shell(UBS.GTO_list[i], UBS.GTO_list[j])
            S[i0:i1, j0:j1] = S_block

    if not symmetric:
        for i in range(n_gtos):
            for j in range(i):  # j < i
                i0 = int(UBS.start_indices[i])
                i1 = int(UBS.end_indices[i])
                j0 = int(UBS.start_indices[j])
                j1 = int(UBS.end_indices[j])

                S_block = S_ab_shell(UBS.GTO_list[i], UBS.GTO_list[j])
                S[i0:i1, j0:j1] = S_block

    else:
        for i in range(n_mat_elem):
            for j in range(i + 1, n_mat_elem):
                S[j, i] = S[i, j]

    return S


def T_UncontractedBasisSet(
    UBS: UncontractedBasisSet, symmetric: bool = True
) -> NDArray[np.float64]:
    n_mat_elem = UBS.n_mat_elem

    T = np.zeros((n_mat_elem, n_mat_elem), dtype=np.float64)
    n_gtos = UBS.n_GTOs

    for i in range(n_gtos):
        for j in range(i, n_gtos):
            i0 = int(UBS.start_indices[i])
            i1 = int(UBS.end_indices[i])
            j0 = int(UBS.start_indices[j])
            j1 = int(UBS.end_indices[j])

            T_block = T_ab_shell(UBS.GTO_list[i], UBS.GTO_list[j])
            T[i0:i1, j0:j1] = T_block

    if not symmetric:
        for i in range(n_gtos):
            for j in range(i):  # j < i
                i0 = int(UBS.start_indices[i])
                i1 = int(UBS.end_indices[i])
                j0 = int(UBS.start_indices[j])
                j1 = int(UBS.end_indices[j])

                T_block = T_ab_shell(UBS.GTO_list[i], UBS.GTO_list[j])
                T[i0:i1, j0:j1] = T_block

    else:
        for i in range(n_mat_elem):
            for j in range(i + 1, n_mat_elem):
                T[j, i] = T[i, j]

    return T


def V_UncontractedBasisSet(
    UBS: UncontractedBasisSet, atom_set: AtomSet, symmetric: bool = True
) -> NDArray[np.float64]:
    n_mat_elem = UBS.n_mat_elem
    n_atoms = atom_set.n_atoms

    V = np.zeros((n_mat_elem, n_mat_elem), dtype=np.float64)
    n_gtos = UBS.n_GTOs

    for i in range(n_gtos):
        for j in range(i, n_gtos):
            i0 = int(UBS.start_indices[i])
            i1 = int(UBS.end_indices[i])
            j0 = int(UBS.start_indices[j])
            j1 = int(UBS.end_indices[j])

            for iatom in range(n_atoms):
                V_block = V_ab_Z_shell(
                    UBS.GTO_list[i],
                    UBS.GTO_list[j],
                    atom_set.q_atoms[iatom],
                    atom_set.r_atoms[iatom],
                )

                V[i0:i1, j0:j1] += V_block

    if not symmetric:
        for i in range(n_gtos):
            for j in range(i):  # j < i
                i0 = int(UBS.start_indices[i])
                i1 = int(UBS.end_indices[i])
                j0 = int(UBS.start_indices[j])
                j1 = int(UBS.end_indices[j])

                for iatom in range(n_atoms):
                    V_block = V_ab_Z_shell(
                        UBS.GTO_list[i],
                        UBS.GTO_list[j],
                        atom_set.q_atoms[iatom],
                        atom_set.r_atoms[iatom],
                    )
                    V[i0:i1, j0:j1] += V_block

    else:
        for i in range(n_mat_elem):
            for j in range(i + 1, n_mat_elem):
                V[j, i] = V[i, j]

    return V


def ERIs_Uncontracted(UBS, kernel: Literal["interest", "oatmeal"] = "interest"):

    if kernel == "interest":
        return _interest_ERIs_Uncontracted(UBS)

    n = UBS.n_mat_elem
    ERI = np.zeros((n, n, n, n), dtype=np.float64)

    n_gtos = UBS.n_GTOs
    for i in range(n_gtos):
        i0, i1 = int(UBS.start_indices[i]), int(UBS.end_indices[i])
        for j in range(n_gtos):
            j0, j1 = int(UBS.start_indices[j]), int(UBS.end_indices[j])
            for k in range(n_gtos):
                k0, k1 = int(UBS.start_indices[k]), int(UBS.end_indices[k])
                for l in range(n_gtos):
                    l0, l1 = int(UBS.start_indices[l]), int(UBS.end_indices[l])

                    block = g_abcd_shell(
                        UBS.GTO_list[i],
                        UBS.GTO_list[j],
                        UBS.GTO_list[k],
                        UBS.GTO_list[l],
                    )

                    ERI[i0:i1, j0:j1, k0:k1, l0:l1] = block

    return ERI


def _interest_ERIs_Uncontracted(UBS, buffer_1=None, symm=None):
    from py_mods.src.integrals.external.interest.interest import interest_full_tensor

    gto_list = UBS.GTO_list

    ERI = interest_full_tensor(gto_list, buffer_1, symm)

    return ERI
