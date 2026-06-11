from py_mods.src.integrals.GTO import (
    GTO,
    create_normalized_GTO,
    _generate_angular_momentum_projections,
    _S_3D_legacy,
    _T_3D_legacy,
    _eri_legacy,
)
from py_mods.src.integrals.internal.coulomb_utils import _V_3D_legacy
from py_mods.src.integrals.GTO import g_abcd_shell
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np


###############################################################################
#                         CGTO Class and Constructor                          #
###############################################################################
@dataclass
class CGTOClass:
    """
    Contracted Gaussian Type Orbital (CGTO) class.

    Attributes
    ----------
    R : NDArray[np.float64]
        Position of the basis.
    exps : NDArray[np.float64]
        Exponents of the primitives.
    N_a : float
        Normalization constant.
    d_i : NDArray[np.float64]
        Contraction coefficients.
    total_l : int
        Total angular momentum.
    l_dim : int
        Number of projections with this angular momentum.
    l_projections : NDArray[np.int32]
        All cartesian projections l.
    primitives : list[GTO]
        List of primitive GTOs.
    """

    R: NDArray[np.float64]  # position of basis
    exps: NDArray[np.float64]  # exponents
    N_a: float  # normalization constant
    d_i: NDArray[np.float64]  # contraction coefficients
    total_l: int
    l_dim: int  # number of projections with this angular momentum
    l_projections: NDArray[np.int32]  # all projections for this angular momentum
    primitives: list[GTO]  # actually needed dut to normalization constants


def create_CGTOClass(
    R: NDArray[np.float64],
    exps: NDArray[np.float64],
    d_i: NDArray[np.float64],
    total_L: int,
    unnormalized: bool = False,
) -> CGTOClass:
    """
    Factory function to create a GTOClass

    Parameters
    ----------
    R : NDArray[np.float64]
        Center of the GTO.
    exps : NDArray[np.float64]
        Exponents of the primitives.
    d_i : NDArray[np.float64]
        Contraction coefficients of the primitives.
    total_L : int
        Total angular momentum of the GTO.

    Returns
    -------
    GTOClass
        Normalized GTOClass object.
    """
    l_projections = _generate_angular_momentum_projections(total_L)
    l_dim = l_projections.shape[0]
    primitives = [create_normalized_GTO(R, exp, total_L) for exp in exps]

    unnormalized_cgto = CGTOClass(
        R=R,
        exps=exps,
        d_i=d_i,
        N_a=1,
        total_l=total_L,
        l_dim=l_dim,
        l_projections=l_projections,
        primitives=primitives,
    )

    if unnormalized:
        N_a = 1.0
    else:
        N_a = calculate_normalization_constant(unnormalized_cgto)

    unnormalized_cgto.N_a = N_a

    return unnormalized_cgto


def calculate_normalization_constant(cont: CGTOClass) -> float:
    """
    Calculate normalization constant for a contracted GTO.

    Parameters
    ----------
    cont : CGTOClass
        Contracted GTO.

    Returns
    -------
    float
        Normalization constant.
    """
    S = S_GTO_proj(cont, 0, cont, 0)
    N = 1.0 / np.sqrt(S)

    return N


###############################################################################
#                             Overlap integrals                               #
###############################################################################


def S_GTO_mat(cont_1: CGTOClass, cont_2: CGTOClass):
    """
    Calculate overlap matrix between two contracted GTOs.

    Parameters
    ----------
    cont_1 : CGTOClass
        First contracted GTO.
    cont_2 : CGTOClass
        Second contracted GTO.

    Returns
    -------
    NDArray[np.float64] of shape (l_dim_1, l_dim_2)
        Overlap integral matrix.
    """
    S_mat = np.zeros((cont_1.l_dim, cont_2.l_dim))

    for p1 in range(cont_1.l_dim):
        for p2 in range(cont_2.l_dim):
            S_mat[p1, p2] = S_GTO_proj(cont_1, p1, cont_2, p2)

    return S_mat * cont_1.N_a * cont_2.N_a


def S_GTO_proj(cont_1: CGTOClass, proj_idx_1: int, cont_2: CGTOClass, proj_idx_2: int):
    """
    Calculate overlap integral between two contracted GTOs
    given their l projections.

    Parameters
    ----------
    cont_1 : CGTOClass
        First contracted GTO.
    proj_idx_1 : int
        Projection index for the first contracted GTO.
    cont_2 : CGTOClass
        Second contracted GTO.
    proj_idx_2 : int
        Projection index for the second contracted GTO.

    Returns
    -------
    float
        Overlap integral value.
    """
    projection_vec_1 = cont_1.l_projections[proj_idx_1]
    projection_vec_2 = cont_2.l_projections[proj_idx_2]

    S: float = 0.0
    for i, gto_1 in enumerate(cont_1.primitives):
        for j, gto_2 in enumerate(cont_2.primitives):
            N_a = gto_1.normalization_constants[proj_idx_1]
            N_b = gto_2.normalization_constants[proj_idx_2]
            primitive_overlap = (
                cont_1.d_i[i]
                * cont_2.d_i[j]
                * _S_3D_legacy(
                    gto_1,
                    projection_vec_1,
                    N_a,
                    gto_2,
                    projection_vec_2,
                    N_b,
                )
            )

            S += primitive_overlap

    return S


###############################################################################
#                         Kinetic energy integrals                            #
###############################################################################


def T_GTO_mat(cont_1: CGTOClass, cont_2: CGTOClass):
    """
    Calculate kinetic energy integral matrix between two contracted GTOs.

    Parameters
    ----------
    cont_1 : CGTOClass
        First contracted GTO.
    cont_2 : CGTOClass
        Second contracted GTO.

    Returns
    -------
    NDArray[np.float64] of shape (l_dim_1, l_dim_2)
        Kinetic energy integral matrix.
    """
    T_mat = np.zeros((cont_1.l_dim, cont_2.l_dim))

    for p1 in range(cont_1.l_dim):
        for p2 in range(cont_2.l_dim):
            T_mat[p1, p2] = T_GTO_proj(cont_1, p1, cont_2, p2)

    return T_mat * cont_1.N_a * cont_2.N_a


def T_GTO_proj(cont_1: CGTOClass, proj_idx_1: int, cont_2: CGTOClass, proj_idx_2: int):
    """
    Calculate kinetic energy integral between two contracted GTOs
    given their l projections.

    Parameters
    ----------
    cont_1 : CGTOClass
        First contracted GTO.
    proj_idx_1 : int
        Projection index for the first contracted GTO.
    cont_2 : CGTOClass
        Second contracted GTO.
    proj_idx_2 : int
        Projection index for the second contracted GTO.

    Returns
    -------
    float
        kinetic energy integral value.
    """
    projection_vec_1 = cont_1.l_projections[proj_idx_1]
    projection_vec_2 = cont_2.l_projections[proj_idx_2]

    T: float = 0.0
    for i, gto_1 in enumerate(cont_1.primitives):
        for j, gto_2 in enumerate(cont_2.primitives):
            N_a = gto_1.normalization_constants[proj_idx_1]
            N_b = gto_2.normalization_constants[proj_idx_2]
            T_primitive = (
                cont_1.d_i[i]
                * cont_2.d_i[j]
                * _T_3D_legacy(
                    gto_1,
                    projection_vec_1,
                    N_a,
                    gto_2,
                    projection_vec_2,
                    N_b,
                )
            )

            T += T_primitive

    return T


###############################################################################
#                        Potential energy integrals                           #
###############################################################################


def V_GTO_mat(
    cont_1: CGTOClass,
    cont_2: CGTOClass,
    atom_pos: NDArray[np.float64],
    atom_charges: NDArray[np.float64],
):
    """
    Calculate the potential energy integral matrix between two contracted GTOs.

    Parameters
    ----------
    cont_1 : CGTOClass
        First contracted GTO.
    cont_2 : CGTOClass
        Second contracted GTO.
    atom_pos : NDArray[np.float64] of shape (n_atoms, 3)
        Positions of the atoms.
    atom_charges : NDArray[np.float64] of shape (n_atoms,)
        Charges of the atoms.

    Returns
    -------
    NDArray[np.float64] of shape (l_dim_1, l_dim_2)
        Potential energy integral matrix.
    """
    S_mat = np.zeros((cont_1.l_dim, cont_2.l_dim))

    for p1 in range(cont_1.l_dim):
        for p2 in range(cont_2.l_dim):
            S_mat[p1, p2] = V_GTO_proj(cont_1, p1, cont_2, p2, atom_pos, atom_charges)

    return S_mat * cont_1.N_a * cont_2.N_a


def V_GTO_proj(
    cont_1: CGTOClass,
    proj_idx_1: int,
    cont_2: CGTOClass,
    proj_idx_2: int,
    atom_pos: NDArray[np.float64],
    atom_charges: NDArray[np.float64],
):
    """
    Calculate potential energy integral between two contracted GTOs
    given their l projections.

    Parameters
    ----------
    cont_1 : CGTOClass
        First contracted GTO.
    proj_idx_1 : int
        Projection index for the first contracted GTO.
    cont_2 : CGTOClass
        Second contracted GTO.
    proj_idx_2 : int
        Projection index for the second contracted GTO.
    atom_pos : NDArray[np.float64] of shape (n_atoms, 3)
        Positions of the atoms.
    atom_charges : NDArray[np.float64] of shape (n_atoms,)
        Charges of the atoms.

    Returns
    -------
    float
        Potential energy integral value.
    """
    if len(atom_pos) != len(atom_charges):
        raise ValueError("atom_pos and atom_charges must have the same length.")

    projection_vec_1 = cont_1.l_projections[proj_idx_1]
    projection_vec_2 = cont_2.l_projections[proj_idx_2]

    V: float = 0.0
    for i, gto_1 in enumerate(cont_1.primitives):
        for j, gto_2 in enumerate(cont_2.primitives):
            for a in range(len(atom_charges)):
                pos = atom_pos[a]
                charge = atom_charges[a]
                N_a = gto_1.normalization_constants[proj_idx_1]
                N_b = gto_2.normalization_constants[proj_idx_2]
                V_primitive = (
                    cont_1.d_i[i]
                    * cont_2.d_i[j]
                    * _V_3D_legacy(
                        gto_1,
                        projection_vec_1,
                        N_a,
                        gto_2,
                        projection_vec_2,
                        N_b,
                        charge,
                        pos,
                    )
                )

                V += V_primitive

    return V


###############################################################################
#                    Electron repulsion energy integrals                      #
###############################################################################


def Eri_GTO_tensor(
    cont_1: CGTOClass, cont_2: CGTOClass, cont_3: CGTOClass, cont_4: CGTOClass
) -> NDArray[np.float64]:
    """
    Calculate ERI tensor between four contracted GTOs.

    Parameters
    ----------
    cont_1 : CGTOClass
        First contracted GTO.
    cont_2 : CGTOClass
        Second contracted GTO.
    cont_3 : CGTOClass
        Third contracted GTO.
    cont_4 : CGTOClass
        Fourth contracted GTO.

    Returns
    -------
    NDArray[np.float64] of shape (l_dim_1, l_dim_2, l_dim_3, l_dim_4)
        ERI tensor.
    """
    eri_mat = np.zeros((cont_1.l_dim, cont_2.l_dim, cont_3.l_dim, cont_4.l_dim))

    for p1 in range(cont_1.l_dim):
        for p2 in range(cont_2.l_dim):
            for p3 in range(cont_3.l_dim):
                for p4 in range(cont_4.l_dim):
                    eri_mat[p1, p2, p3, p4] = eri_GTO_proj(
                        cont_1, p1, cont_2, p2, cont_3, p3, cont_4, p4
                    )

    eri_tensor = eri_mat * cont_1.N_a * cont_2.N_a * cont_3.N_a * cont_4.N_a

    # print(f'Normalization constants are: {cont_1.N_a}, {cont_2.N_a}, {cont_3.N_a}, {cont_4.N_a}')

    return eri_tensor


def Eri_GTO_tensor_intermediates(
    cont_1: CGTOClass, cont_2: CGTOClass, cont_3: CGTOClass, cont_4: CGTOClass
) -> NDArray[np.float64]:

    eri_mat = np.zeros((cont_1.l_dim, cont_2.l_dim, cont_3.l_dim, cont_4.l_dim))

    for i, gto_1 in enumerate(cont_1.primitives):
        for j, gto_2 in enumerate(cont_2.primitives):
            for k, gto_3 in enumerate(cont_3.primitives):
                for l, gto_4 in enumerate(cont_4.primitives):
                    prim_tensor = g_abcd_shell(gto_1, gto_2, gto_3, gto_4)

                    eri_mat += (
                        cont_1.d_i[i] * cont_2.d_i[j] * cont_3.d_i[k] * cont_4.d_i[l]
                    ) * prim_tensor

    return eri_mat * cont_1.N_a * cont_2.N_a * cont_3.N_a * cont_4.N_a


def eri_GTO_proj(
    cont_1: CGTOClass,
    proj_idx_1: int,
    cont_2: CGTOClass,
    proj_idx_2: int,
    cont_3: CGTOClass,
    proj_idx_3: int,
    cont_4: CGTOClass,
    proj_idx_4: int,
) -> float:
    """
    Calculate ERI between four contracted GTOs for given their l projections.

    Parameters
    ----------
    cont_1 : CGTOClass
        First contracted GTO.
    proj_idx_1 : int
        Projection index for the first contracted GTO.
    cont_2 : CGTOClass
        Second contracted GTO.
    proj_idx_2 : int
        Projection index for the second contracted GTO.
    cont_3 : CGTOClass
        Third contracted GTO.
    proj_idx_3 : int
        Projection index for the third contracted GTO.
    cont_4 : CGTOClass
        Fourth contracted GTO.
    proj_idx_4 : int
        Projection index for the fourth contracted GTO.

    Returns
    -------
    float
        ERI value.
    """
    projection_vec_1 = cont_1.l_projections[proj_idx_1]
    projection_vec_2 = cont_2.l_projections[proj_idx_2]
    projection_vec_3 = cont_3.l_projections[proj_idx_3]
    projection_vec_4 = cont_4.l_projections[proj_idx_4]

    eri_val: float = 0.0
    for i, gto_1 in enumerate(cont_1.primitives):
        for j, gto_2 in enumerate(cont_2.primitives):
            for k, gto_3 in enumerate(cont_3.primitives):
                for l, gto_4 in enumerate(cont_4.primitives):
                    N_a = gto_1.normalization_constants[proj_idx_1]
                    N_b = gto_2.normalization_constants[proj_idx_2]
                    N_c = gto_3.normalization_constants[proj_idx_3]
                    N_d = gto_4.normalization_constants[proj_idx_4]

                    abcd_eri = (
                        cont_1.d_i[i]
                        * cont_2.d_i[j]
                        * cont_3.d_i[k]
                        * cont_4.d_i[l]
                        * _eri_legacy(
                            gto_1,
                            projection_vec_1,
                            N_a,
                            gto_2,
                            projection_vec_2,
                            N_b,
                            gto_3,
                            projection_vec_3,
                            N_c,
                            gto_4,
                            projection_vec_4,
                            N_d,
                        )
                    )

                    eri_val += abcd_eri

    return eri_val
