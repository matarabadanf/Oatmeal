from re import I
from typing import Optional, Tuple, Union, List, Literal

import numpy as np
from numpy.typing import NDArray

from py_mods.src.integrals.UncontractedBasisSet import (
    UncontractedBasisSet,
    ERIs_Uncontracted,
)

from py_mods.src.SCF.scf_kernels import calc_p_matrix_comp

import numpy as np
import scipy

from py_mods.src.SCF_4c_dev.types_4c import (
    CS_4c_KU_SCF_Context,
    CS_4c_KU_SCF_Constants,
    CS_4c_KU_SCF_State,
)

from py_mods.src.SCF.linalg import transformation_matrix
from py_mods.src.SCF.CSRHF import guess_density_RHF

from py_mods.src.SCF.utils import initialize_conv_acc

from py_mods.src.SCF_4c_dev.utils import validate_4c_determinant

from py_mods.src.SCF_4c_dev.scf_4c_kernels import scale_4c_integrals


def full_eri_from_Uncontracted_Basis(UBS: UncontractedBasisSet) -> NDArray[np.float64]:
    """
    Compute full ERI tensor from uncontracted basis set.

    Parameters
    ----------
    UBS : UncontractedBasisSet
        The uncontracted basis set.

    Returns
    -------
    eri_tensor : NDArray[np.float64]
        The full ERI tensor.
    """
    eri_tensor = ERIs_Uncontracted(UBS)

    return eri_tensor


def eri_classified(eri: NDArray[np.float64], nL: int) -> NDArray[np.float64]:
    """
    Filter ERI tensor to keep only (LL|LL), (SS|LL), (LL|SS), (SS|SS) terms.

    Parameters
    ----------
    eri : NDArray[np.float64]
        The full electron repulsion integrals tensor.
    nL : int
        Number of large component basis functions.

    Returns
    -------
    eri_classess : NDArray[np.float64]
        The classified ERI tensor.
    """
    eri_classess = np.zeros_like(eri, dtype=np.float64)

    eri_classess[:nL, :nL, :nL, :nL] = eri[:nL, :nL, :nL, :nL]  # LL-LL block
    eri_classess[:nL, :nL, nL:, nL:] = eri[:nL, :nL, nL:, nL:]  # LL-SS block
    eri_classess[nL:, nL:, :nL, :nL] = eri[nL:, nL:, :nL, :nL]  # SS-LL block
    eri_classess[nL:, nL:, nL:, nL:] = eri[nL:, nL:, nL:, nL:]  # SS-SS block

    return eri_classess


def occupation_4c(
    nS: int, nL: int, n_electrons: int, electronic_occ_det: Union[None, NDArray[np.int8]] = None
) -> NDArray[np.int8]:
    """
    Build the occupation vector for 4c calculations.

    Parameters
    ----------
    nS : int
        Number of small component basis functions.
    nL : int
        Number of large component basis functions.
    n_electrons : int
        Number of electrons.
    electronic_occ_det : Union[None, NDArray[np.int8]], optional
        Occupation determinant for electronic states (positive energy solutions). Defaults to None.

    Returns
    -------
    occ : NDArray[np.int8]
        Occupation determinant for electronic and positronic states.
    """
    occ = np.zeros(2 * (nS + nL), dtype=np.int8)

    n_positron_states = 2 * nS

    if electronic_occ_det is None:
        occ[n_positron_states : n_positron_states + n_electrons] = 1
    else:
        assert (
            len(electronic_occ_det) == 2 * nL
        ), "Length of electronic occupation array must be equal to 2*nL"
        assert (
            sum(electronic_occ_det) == n_electrons
        ), "Sum of electronic occupation array must be equal to n_electrons"
        occ[n_positron_states:] = electronic_occ_det

    return occ


def g_matrix_4c(
    P: NDArray[np.complex128], eri: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """
    Construct G matrix (J-K) from the density matrix.

    Parameters
    ----------
    P : NDArray[np.complex128]
        Density matrix.
    eri : NDArray[np.complex128]
        Electron repulsion integrals tensor.

    Returns
    -------
    G_full : NDArray[np.complex128]
        Full G matrix.
    """
    n_bas = P.shape[0]
    n_bas_half = n_bas // 2

    P_aa = P[0:n_bas_half, 0:n_bas_half]
    P_bb = P[n_bas_half:n_bas, n_bas_half:n_bas]
    P_ab = P[0:n_bas_half, n_bas_half:n_bas]
    P_ba = P[n_bas_half:n_bas, 0:n_bas_half]

    P_total = P_aa + P_bb
    J = np.einsum("mnsl, ls -> mn", eri, P_total)

    K_aa = np.einsum("psrq, sr -> pq", eri, P_aa)
    K_bb = np.einsum("psrq, sr -> pq", eri, P_bb)
    K_ab = np.einsum("psrq, sr -> pq", eri, P_ab)
    K_ba = np.einsum("psrq, sr -> pq", eri, P_ba)

    G_aa = J - K_aa
    G_bb = J - K_bb
    G_ab = -K_ab
    G_ba = -K_ba

    G_full = np.zeros((n_bas, n_bas), dtype=np.complex128)
    # And we fill the matrix by blocks
    G_full[0:n_bas_half, 0:n_bas_half] = G_aa
    G_full[n_bas_half:n_bas, n_bas_half:n_bas] = G_bb
    G_full[0:n_bas_half, n_bas_half:n_bas] = G_ab
    G_full[n_bas_half:n_bas, 0:n_bas_half] = G_ba

    return G_full


def scf_iteration(
    F_1: NDArray[np.complex128], X: NDArray[np.complex128], total_occ_det: NDArray[np.int8]
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Perform a single SCF iteration step.

    Parameters
    ----------
    F_1 : NDArray[np.complex128]
        Current Fock matrix.
    X : NDArray[np.complex128]
        Transformation matrix.
    total_occ_det : NDArray[np.int8]
        Occupation determinant.

    Returns
    -------
    e1 : NDArray[np.complex128]
        Eigenvalues (orbital energies).
    w1 : NDArray[np.complex128]
        Eigenvectors in orthogonal basis.
    F_p1 : NDArray[np.complex128]
        Transformed Fock matrix.
    P_1 : NDArray[np.complex128]
        Updated density matrix.
    """
    F_p1 = X.T @ F_1 @ X

    e1, w1 = np.linalg.eigh(F_p1)

    idx = np.argsort(e1)
    e1 = e1[idx]
    w1 = w1[:, idx]

    c_alpha_beta1 = X @ w1
    P_1 = calc_p_matrix_comp(c_alpha_beta1.conj().T, c_alpha_beta1, total_occ_det)

    return e1, w1, F_p1, P_1


def scf_steps(
    n_steps: int,
    H_core: NDArray[np.complex128],
    eri: NDArray[np.complex128],
    X: NDArray[np.complex128],
    total_occ_det: NDArray[np.int8]
) -> List[float]:
    """
    For loop that wraps scf iterations.

    Parameters
    ----------
    n_steps : int
        Number of steps to run.
    H_core : NDArray[np.complex128]
        Core Hamiltonian matrix.
    eri : NDArray[np.complex128]
        Electron repulsion integrals tensor.
    X : NDArray[np.complex128]
        Transformation matrix.
    total_occ_det : NDArray[np.int8]
        Occupation determinant.

    Returns
    -------
    energy_step : List[float]
        Energies at each iteration step.
    """
    energy_step = []
    P_old = np.zeros_like(H_core)

    for i in range(n_steps):
        if i == 0:
            G_new = np.zeros_like(H_core)
        else:
            G_new = g_matrix_4c(P_old, eri)

        if i > 0:
            e_scf = np.linalg.trace(P_old @ H_core + 0.5 * P_old @ G_new)
            energy_step.append(e_scf.real)

        F_new = H_core + G_new

        e_new, w_new, F_p_new, P_2 = scf_iteration(F_new, X, total_occ_det)

        if i == 0:
            e_scf = np.linalg.trace(P_2 @ H_core)
            energy_step.append(e_scf.real)

        P_old = P_2

    return energy_step


# -------------------------------------------------------------
#  CS-4c-KU-SCF Initialization Functions
# -------------------------------------------------------------


def initialize_CS_4c_KU_SCF_extended_context(
    ctx: CS_4c_KU_SCF_Context, ext_ctx: CS_4c_KU_SCF_Constants
) -> None:
    """
    Setup extended context with transformation matrix, validated determinant and scaled integrals.
    Also set up convergence acceleration parameters.

    Parameters
    ----------
    ctx : CS_4c_KU_SCF_Context
        Original context with integrals and parameters.
    ext_ctx : CS_4c_KU_SCF_Constants
        Initialized extended context to compute.

    Returns
    -------
    None
    """

    ext_ctx.dim = len(ctx.S)
    ext_ctx.X = transformation_matrix(ctx.S.astype(np.complex128)).astype(
        np.complex128
    )

    # validate occupation
    ext_ctx.full_det, _ = validate_4c_determinant(
        ctx.nS, ctx.nL, ctx.n_electrons, ctx.occ
    )

    # rescaling the integrals
    T_scaled, V_scaled, W_scaled, ext_ctx.eri_scaled = scale_4c_integrals(
        ctx.T, ctx.V, ctx.W, ctx.eri_classess, ctx.theta
    )

    ext_ctx.H_core = T_scaled + V_scaled + W_scaled
    ext_ctx.core_mask = np.abs(ext_ctx.H_core) > 1e-10

    # eigensolver enforced
    if ctx.theta != 0:
        ext_ctx._eigensolver = "eig"
    else:
        ext_ctx._eigensolver = ctx._eigensolver

    # Convergence acceleration setup
    ext_ctx.acc_iteration_start, ext_ctx.acc_requested = initialize_conv_acc(
        ctx.acc_hist_size, ctx.conv_type, ctx.acc_iteration_start
    )

    return


def initialize_CS_4c_KU_SCF_state_variable(
    ext_ctx: CS_4c_KU_SCF_Constants, state: CS_4c_KU_SCF_State
) -> None:
    """
    Initialize SCF state variables.

    Parameters
    ----------
    ext_ctx : CS_4c_KU_SCF_Constants
        Extended context providing basis dimension info.
    state : CS_4c_KU_SCF_State
        State object to be initialized.

    Returns
    -------
    None
    """
    state.use_conv_acc = False
    state.converged = False
    state.F_guess = []
    state.residuals = []
    state.F_next = np.zeros_like(ext_ctx.H_core)
    state.e_orb = np.zeros(ext_ctx.dim, dtype=np.complex128)
    state.C_prime = np.zeros((ext_ctx.dim, ext_ctx.dim), dtype=np.complex128)
    state.C_munu = np.zeros_like(state.C_prime, dtype=np.complex128)
    state.error = np.complex128(1e10)

    return


def initialize_CS_4c_KU_SCF_P_and_E(
    ctx: CS_4c_KU_SCF_Context,
    state: CS_4c_KU_SCF_State,
) -> None:
    """
    Initialize density matrix and starting energy.

    Parameters
    ----------
    ctx : CS_4c_KU_SCF_Context
        Original context.
    state : CS_4c_KU_SCF_State
        State object to be populated with the initial guess.

    Returns
    -------
    None
    """
    if ctx.theta != 0.0:
        # TODO: this cannot be filled until the routine has been adapted
        pass
        P = guess_density_RHF(ctx.p_guess, len(ctx.S), ctx.initial_orbitals)
        E_prev = np.complex128(0.0)
        # P, unscaled_E = compute_unscaled_density(ctx, ctx.verbose)
        # E_prev = np.complex128(unscaled_E)

    else:
        P = guess_density_RHF(ctx.p_guess, len(ctx.S), ctx.initial_orbitals)
        E_prev = np.complex128(0.0)

    state.P = P
    state.E_prev = E_prev

    return


# TODO: this has to be adaoted once the scf kernel is defined
# def compute_rhf_unscaled_density(
#     ctx: CSRHFContext, verbose: bool
# ) -> Tuple[NDArray[np.complex128], np.complex128]:
#     """
#     Compute unscaled density matrix for theta=0.

#     Parameters
#     ----------
#     ctx : CSRHFContext
#         Original context with integrals and parameters.
#     verbose : bool
#         If True, print status.

#     Returns
#     -------
#     P : NDArray[np.complex128]
#         Unscaled density matrix.
#     E_RHF : np.complex128
#         Unscaled RHF energy.
#     """
#     if verbose:
#         print("Converging unscaled case:")
#     unscaled_ctx = CSRHFContext(
#         S=ctx.S,
#         T=ctx.T,
#         V=ctx.V,
#         eri=ctx.eri,
#         n_electrons=ctx.n_electrons,
#         theta=0.0,
#         occupation=ctx.occupation,
#         max_iter=ctx.max_iter,
#         threshold=ctx.threshold,
#         p_guess=ctx.p_guess,
#         guess_max_iter=ctx.guess_max_iter,
#         initial_orbitals=ctx.initial_orbitals,
#         verbose=ctx.verbose,
#         conv_type=ctx.conv_type,
#         acc_hist_size=ctx.acc_hist_size,
#         acc_iteration_start=10,
#     )

#     unscaled_res = CS_RHF(unscaled_ctx)

#     if verbose:
#         print("Unscaled energy: ", unscaled_res.E_RHF)
#         print("\n\n\nConverging scaled case from unscaled density as reference:")

#     P = unscaled_res.P
#     return P, unscaled_res.E_RHF
