import copy
from typing import Optional, Tuple, Union, List, Literal

import numpy as np
from numpy.typing import NDArray

from py_mods.src.integrals.UncontractedBasisSet import (
    UncontractedBasisSet,
    ERIs_Uncontracted,
)

from py_mods.src.SCF.scf_kernels import calc_p_matrix_comp, calc_residual_commutator
from py_mods.src.SCF.CSRHF import print_table_header
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
from py_mods.src.SCF_4c_dev.scf_4c_kernels import guess_density_4c

from py_mods.src.SCF.utils import initialize_conv_acc

from py_mods.src.SCF_4c_dev.utils import (
    validate_4c_determinant,
    validate_CS_4c_KU_SCF_context_input,
)

from py_mods.src.SCF_4c_dev.scf_4c_kernels import (
    scale_4c_integrals,
    calculate_P_next_4c,
    guess_density_4c,
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
    F_p1 = X.conj().T @ F_1 @ X

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

    n_lindep = np.abs(
        ext_ctx.X.shape[0] - ext_ctx.X.shape[1]
    )  # The difference between row and column size is the number of lindeps

    # validate occupation
    ext_ctx.det, _ = validate_4c_determinant(ctx.nS, ctx.nL, ctx.n_electrons, ctx.occ)

    if n_lindep > 0:
        assert (
            np.sum(ext_ctx.det[-n_lindep:]) == 0
        ), "Linear dependency removal encountered occupied linearly dependent electronic states"
        ext_ctx.det = ext_ctx.det[:-n_lindep]

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
    rhf_state: CS_4c_KU_SCF_State,
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

    P = guess_density_4c(ctx.p_guess, len(ctx.S), ctx.initial_orbitals)
    E_prev = np.complex128(0.0)

    rhf_state.P = P
    rhf_state.E_prev = E_prev

    return


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
    if convergence_criteria == "norm":
        print(f"| {state.iteration:^8} | {state.E_SCF:^45.16f} | {state.E_diff:^45.16f} | {state.error:^22.4E} |")
    elif convergence_criteria == "max":
        print(f"| {state.iteration:^8} | {state.E_SCF:^45.16f} | {state.E_diff:^45.16f} | {state.error:^22.4E} |")


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


def CS_4c_KU_SCF(ctx: CS_4c_KU_SCF_Context) -> CS_4c_KU_SCF_Results:
    """
    Perform a Complex Scaled 4-Component Kramers-Unrestricted Self-Consistent Field (CS-4c-KU-SCF) calculation.

    Takes a context with overlap, kinetic, nuclear attraction, and two-electron
    integrals, optionally applies complex scaling by an angle `theta`, and runs
    the CS-4c-KU-SCF loop using biorthogonal diagonalization. If `theta != 0`, an
    unscaled calculation is performed first to generate a starting guess density.

    Parameters
    ----------
    ctx : CSRHFContext
        Context object containing all parameters for the calculation.

    Returns
    -------
    CSRHFResults
        Results object containing energies, orbitals, and convergence info.
    """
    if ctx.theta == 0:
        return _kuscf_kernel(ctx)

    else:
        scaled_context = copy.deepcopy(ctx)

        # perform unscaled calculation first
        if ctx.verbose:
            print("Converging unscaled case:")
        ctx.theta = 0
        unscaled_rhf = _kuscf_kernel(ctx)

        # use results for scaled calculation
        if ctx.verbose:
            print("Unscaled energy: ", unscaled_rhf.E_SCF)
            print("\n\n\nConverging scaled case from unscaled density as reference:")
        scaled_context.p_guess = "INPORB"
        scaled_context.initial_orbitals = unscaled_rhf.P
        return _kuscf_kernel(scaled_context)


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

        state.converged = is_converged_4c(ctx, state)
        if state.converged:
            break

        update_CS_4c_KU_SCF_acc_hist_size(ctx, state)
        state.P_old = state.P.copy()

        update_CS_4c_KU_SCF_F_matrix(ctx, state)

        update_CS_4c_KU_SCF_density(ctx, ext_ctx, state)

        state.use_conv_acc = conv_acc_criteria_met_4c(ctx, ext_ctx, state)

    update_CS_4c_KU_SCF_density(ctx, ext_ctx, state)
    state.F_next = state.F

    results = pack_CS_4c_KU_SCF_results(ctx, ext_ctx, state)

    return results


def is_converged_4c(
    ctx: CS_4c_KU_SCF_Context,
    state: CS_4c_KU_SCF_State,
) -> bool:
    """
    Check convergence based on residual norms.

    Parameters
    ----------
    ctx : CS_4c_KU_SCF_Context
        Context object containing convergence criteria and threshold.
    state : CS_4c_KU_SCF_State
        Current SCF state.

    Returns
    -------
    converged : bool
        True if converged, else False.
    """
    converged: bool = False

    if ctx._convergence_criteria == "max":
        error_re: float = float(np.max(np.abs(state.r.real)))
        error_im: float = float(np.max(np.abs(state.r.imag)))
        if state.iteration > 1 and np.max([error_re, error_im]) < ctx.threshold:
            converged = True

    elif ctx._convergence_criteria == "norm":
        error = np.linalg.norm(state.r)
        if state.iteration > 1 and error < ctx.threshold:
            converged = True

    if converged and ctx.verbose:
        print("-" * 135)
        print(f"Convergence achieved after {state.iteration} iterations.")

    return converged


def update_CS_4c_KU_SCF_acc_hist_size(
    ctx: CS_4c_KU_SCF_Context, state: CS_4c_KU_SCF_State
) -> None:
    state.F_guess.append(state.F)
    state.residuals.append(state.r)

    if len(state.F_guess) > ctx.acc_hist_size:
        state.F_guess.pop(0)
        state.residuals.pop(0)

    return


def conv_acc_criteria_met_4c(
    ctx: CS_4c_KU_SCF_Context,
    ext_ctx: CS_4c_KU_SCF_Constants,
    state: CS_4c_KU_SCF_State,
) -> bool:
    use_conv_acc = state.use_conv_acc
    if (
        not use_conv_acc
        and state.iteration + 1 >= ctx.acc_iteration_start
        and ext_ctx.acc_requested
    ):
        use_conv_acc = True

        if ctx.verbose:
            msg = f" STARTED {ctx.conv_type} "
            print(f"|{msg:-^133}|")

    return use_conv_acc


def CS_4c_KU_SCF_theta_traj(max_theta, n_points, ctx: CS_4c_KU_SCF_Context):
    """
    Sample energies along theta trajectory for CS-4c-KU-SCF.

    Parameters
    ----------
    max_theta : float
        Max theta (radians).
    n_points : int
        Steps.
    ctx : CS_4c_KU_SCF_Context
        Base context.

    Returns
    -------
    thetas, energies : Tuple[NDArray, NDArray]
        Sampled angles & energies.
    """
    thetas = np.linspace(0, max_theta, n_points)
    energies = []
    for th in thetas:
        ctx.theta = th
        res = CS_4c_KU_SCF(ctx)
        if res.converged:
            energies.append(res.E_SCF)
        else:
            print(f"Traj {th} did not converge.")
        if ctx.verbose and res.converged:
            print(f"Converged point at theta = {th:6.4f} : E = {res.E_SCF:12.8f}")

    return thetas, np.array(energies, dtype=np.complex128)
