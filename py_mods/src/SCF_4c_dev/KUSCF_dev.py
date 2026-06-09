from re import I
from typing import Optional, Tuple, Union, List, Literal

import numpy as np
from numpy.typing import NDArray

from py_mods.src.integrals.UncontractedBasisSet import (
    UncontractedBasisSet,
    ERIs_Uncontracted,
)

from py_mods.src.SCF.scf_kernels import calc_p_matrix_comp, calc_residual_commutator
from py_mods.src.SCF.CSRHF import (
    is_converged,
    update_rhf_acc_hist_size,
    conv_acc_criteria_met,
    print_table_header,
)
import scipy

from py_mods.src.SCF_4c_dev.types_4c import (
    CS_4c_KU_SCF_Context,
    CS_4c_KU_SCF_Constants,
    CS_4c_KU_SCF_State,
    CS_4c_KU_SCF_Results,
    allocate_CS_4c_KU_SCF_extended_context,
    allocate_CS_4c_KU_SCF_state,
    pack_CS_4c_KU_SCF_results,
)

from py_mods.src.SCF.linalg import transformation_matrix, sign_convention
from py_mods.src.SCF.CSRHF import guess_density_RHF

from py_mods.src.SCF.utils import initialize_conv_acc

from py_mods.src.SCF_4c_dev.utils import (
    validate_4c_determinant,
    validate_CS_4c_KU_SCF_context_input,
)

from py_mods.src.SCF_4c_dev.scf_4c_kernels import (
    scale_4c_integrals,
    calculate_P_next_4c,
)


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
    nS: int,
    nL: int,
    n_electrons: int,
    electronic_occ_det: Union[None, NDArray[np.int8]] = None,
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
    F_1: NDArray[np.complex128], X: NDArray[np.complex128], det: NDArray[np.int8]
) -> Tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
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
    P_1 = calc_p_matrix_comp(c_alpha_beta1.conj().T, c_alpha_beta1, det)

    return e1, w1, F_p1, P_1


def scf_steps(
    n_steps: int,
    H_core: NDArray[np.complex128],
    eri: NDArray[np.complex128],
    X: NDArray[np.complex128],
    det: NDArray[np.int8],
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

        e_new, w_new, F_p_new, P_2 = scf_iteration(F_new, X, det)

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
    ext_ctx.X = transformation_matrix(ctx.S.astype(np.complex128)).astype(np.complex128)

    # validate occupation
    ext_ctx.det, _ = validate_4c_determinant(ctx.nS, ctx.nL, ctx.n_electrons, ctx.occ)

    # rescaling the integrals
    T_scaled, V_scaled, W_scaled, ext_ctx.eri_scaled = scale_4c_integrals(
        ctx.T, ctx.V, ctx.W, ctx.eri_classess, ctx.theta
    )

    ext_ctx.H_core = T_scaled + V_scaled + W_scaled

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


def update_CS_4c_KU_SCF_F_and_r_comp(
    ctx: CS_4c_KU_SCF_Context,
    ext_ctx: CS_4c_KU_SCF_Constants,
    state: CS_4c_KU_SCF_State,
) -> None:
    G = g_matrix_4c(state.P, ext_ctx.eri_scaled)
    state.F = ext_ctx.H_core + G
    state.r = calc_residual_commutator(state.F, state.P, ctx.S.astype(np.complex128))
    state.error = float(np.linalg.norm(state.r.flatten()))


def update_CS_4c_KU_SCF_energy(
    ext_ctx: CS_4c_KU_SCF_Constants,
    state: CS_4c_KU_SCF_State,
) -> None:
    e_scf = np.linalg.trace(state.P @ (ext_ctx.H_core + state.F)) * 0.5
    state.E_SCF = e_scf
    state.E_diff = state.E_SCF - state.E_prev
    state.E_prev = state.E_SCF


def print_cycle_data_4c(convergence_criteria: str, state: CS_4c_KU_SCF_State) -> None:
    print(
        f"{state.iteration:5}     {state.E_SCF:45.16f}     {state.E_diff:45.16f}     {state.error:8.4E}"
    )


def update_CS_4c_KU_SCF_F_matrix(
    ctx: CS_4c_KU_SCF_Context,
    state: CS_4c_KU_SCF_State,
) -> None:
    if not state.use_conv_acc:
        F_next = state.F
    else:
        try:
            F_opt, r_opt = calc_diis_extrapolation_4c(
                state.residuals, state.F_guess, ctx.theta
            )
            F_next = F_opt

            if ctx.conv_type == "CROP":
                state.F_guess[-1] = F_opt
                state.residuals[-1] = r_opt
        except np.linalg.LinAlgError:
            if ctx.verbose:
                print(
                    "!!!!!!!!!!!!!!!! CONVERGENCE ACCELERATION CAUSED A SINGULAR MATRIX. REVERTING TO STANDARD SCF !!!!!!!!!!!!!!!"
                )
            state.use_conv_acc = False
            F_next = state.F

    state.F_next = F_next


def calc_diis_extrapolation_4c(
    residuals: List[NDArray[np.complex128]],
    F_guesses: List[NDArray[np.complex128]],
    theta: float,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    n_guesses = len(residuals)
    eq_sis_dim = n_guesses + 1

    B_matrix = np.zeros((eq_sis_dim, eq_sis_dim), dtype=np.complex128)
    B_matrix[-1, :] = 1
    B_matrix[:, -1] = 1
    B_matrix[-1, -1] = 0

    for i in range(n_guesses):
        for j in range(n_guesses):
            if theta == 0.0:
                # We have to use complex conjugation because the DF Hamiltonian is
                # complex. In the case of the NR case we could just do the scalar
                # product since the hamiltonian is real so we could get away
                # with using np.dot in both cases.
                B_matrix[i, j] = np.vdot(residuals[i].ravel(), residuals[j].ravel())
            else:
                # c-norm metric inner product for complex scaling
                B_matrix[i, j] = np.dot(residuals[i].ravel(), residuals[j].ravel())

    solution = np.zeros(eq_sis_dim, dtype=np.complex128)
    solution[-1] = 1

    try:
        c = np.linalg.solve(B_matrix, solution)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("DIIS matrix singular")

    coeffs = c[:-1]

    F_conv = np.zeros_like(F_guesses[0])
    r_conv = np.zeros_like(residuals[0])

    for k, coef in enumerate(coeffs):
        F_conv += coef * F_guesses[k]
        r_conv += coef * residuals[k]

    return F_conv, r_conv


def update_CS_4c_KU_SCF_density(
    ctx: CS_4c_KU_SCF_Context,
    ext_ctx: CS_4c_KU_SCF_Constants,
    state: CS_4c_KU_SCF_State,
) -> None:
    state.P, state.e_orb, state.C_munu, state.C_prime = calculate_P_next_4c(
        state.F_next, ext_ctx.X, ext_ctx.det, ext_ctx._eigensolver, ctx.theta
    )

    state.C_munu = sign_convention(state.C_munu)
    return


def _kuscf_kernel(ctx: CS_4c_KU_SCF_Context) -> CS_4c_KU_SCF_Results:
    validate_CS_4c_KU_SCF_context_input(ctx)

    ext_ctx = allocate_CS_4c_KU_SCF_extended_context(ctx)
    state = allocate_CS_4c_KU_SCF_state(ctx)

    initialize_CS_4c_KU_SCF_extended_context(ctx, ext_ctx)

    initialize_CS_4c_KU_SCF_P_and_E(ctx, state)
    initialize_CS_4c_KU_SCF_state_variable(ext_ctx, state)

    if ctx.verbose:
        print_table_header()

    for iter_idx in range(ctx.max_iter):
        state.iteration += 1

        update_CS_4c_KU_SCF_F_and_r_comp(ctx, ext_ctx, state)
        update_CS_4c_KU_SCF_energy(ext_ctx, state)

        if ctx.verbose:
            print_cycle_data_4c(ctx._convergence_criteria, state)

        # Reused RHF functions (duck-typed)
        state.converged = is_converged(ctx, state)  # type: ignore
        if state.converged:
            break

        update_rhf_acc_hist_size(ctx, state)  # type: ignore
        state.P_old = state.P.copy()

        update_CS_4c_KU_SCF_F_matrix(ctx, state)

        update_CS_4c_KU_SCF_density(ctx, ext_ctx, state)

        state.use_conv_acc = conv_acc_criteria_met(ctx, ext_ctx, state)  # type: ignore

    update_CS_4c_KU_SCF_density(ctx, ext_ctx, state)
    state.F_next = state.F

    results = pack_CS_4c_KU_SCF_results(ctx, ext_ctx, state)

    return results
