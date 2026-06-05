import numpy as np
import copy
from numpy.typing import NDArray
from typing import Literal, Tuple

from py_mods.src.SCF.scf_kernels import (
    calc_g_matrix_comp,
    E_0_comp,
    guess_density_RHF,
    scale_integrals,
    calc_residual_commutator,
    calc_diis_extrapolation,
    calculate_P_next,
)

from py_mods.src.SCF.utils import (
    validate_determinant,
    validate_rhf_context_input,
    initialize_conv_acc,
)

from py_mods.src.SCF.linalg import (
    transformation_matrix,
    sign_convention,
)

from py_mods.src.SCF.CS_SCF_types import (
    CSRHFContext,
    CSRHFResults,
    CSRHFState,
    CSRHFConstants,
    allocate_rhf_extended_context,
    allocate_rhf_state,
    pack_rhf_results,
)


def _csrhf_kernel(ctx: CSRHFContext) -> CSRHFResults:
    """
    Perform Complex Scaled RHF calculation.

    Parameters
    ----------
    ctx : CSRHFContext
        Calculation parameters & integrals.

    Returns
    -------
    CSRHFResults
        Calculation results.
    """
    validate_rhf_context_input(ctx)

    # Allocate extended context and RHF state
    rhf_ext_ctx = allocate_rhf_extended_context(ctx)
    rhf_state = allocate_rhf_state(ctx)

    # Transform matrix & scale & validate determinant & set convergence acceleratio
    initialize_rhf_extended_context(ctx, rhf_ext_ctx)

    # Guess density and initialize E0
    initialize_rhf_P_and_E(ctx, rhf_state)
    initialize_rhf_state_variable(rhf_ext_ctx, rhf_state)

    if ctx.verbose:
        print_table_header()

    for iter_idx in range(ctx.max_iter):
        rhf_state.iteration += 1
        # Calculate next Fock matrix, associated error, and RHF energy
        update_rhf_F_and_r_comp(ctx, rhf_ext_ctx, rhf_state)
        update_rhf_energy(rhf_ext_ctx, rhf_state)

        if ctx.verbose:
            print_cycle_data(ctx._convergence_criteria, rhf_state)

        # Check convergence
        rhf_state.converged = is_converged(ctx, rhf_state)
        if rhf_state.converged:
            break

        # History storage
        update_rhf_acc_hist_size(ctx, rhf_state)
        rhf_state.P_old = rhf_state.P.copy()

        # Update F_next with or without convergence acceleration
        update_rhf_F_matrix(ctx, rhf_state)

        # Compute next Density
        update_rhf_density(ctx, rhf_ext_ctx, rhf_state)

        # Enforce real if theta=0
        if ctx.theta == 0.0:
            rhf_state.P.imag = rhf_state.C_munu.imag = 0

        # Check activation of convergence acceleration
        rhf_state.use_conv_acc = conv_acc_criteria_met(ctx, rhf_ext_ctx, rhf_state)

    # Final update diagonalization
    update_rhf_density(ctx, rhf_ext_ctx, rhf_state)

    rhf_state.F_next = rhf_state.F

    rhf_results = pack_rhf_results(ctx, rhf_ext_ctx, rhf_state)

    return rhf_results


def CS_RHF(ctx: CSRHFContext) -> CSRHFResults:

    if ctx.theta == 0:
        return _csrhf_kernel(ctx)

    else:
        scaled_context = copy.deepcopy(ctx)

        # perform unscaled calculation first
        ctx.theta = 0
        unscaled_rhf = _csrhf_kernel(ctx)

        # use results for scaled calculation
        scaled_context.p_guess = "INPORB"
        scaled_context.initial_orbitals = unscaled_rhf.P
        return _csrhf_kernel(scaled_context)


# -------------------------------------------------------------
#  RHF Initialization Functions
# -------------------------------------------------------------


def initialize_rhf_extended_context(
    ctx: CSRHFContext, rhf_ext_ctx: CSRHFConstants
) -> None:
    """
    Setup extended context with transformation matrix, validated determinant and scaled integrals.  Also set up convergence acceleration parameters.

    Parameters
    ----------
    ctx : CSRHFContext
        Original context with integrals and parameters.
    rhf_ext_ctx : CSRHFConstants
        Initialized extended context to compute.

    Returns
    -------
    None
    """

    rhf_ext_ctx.dim = len(ctx.S)
    rhf_ext_ctx.X = transformation_matrix(ctx.S.astype(np.complex128)).astype(
        np.complex128
    )

    # validate occupation
    rhf_ext_ctx.det, _ = validate_determinant(
        ctx.n_electrons, ctx.occupation, rhf_ext_ctx.dim
    )

    # rescaling the integrals
    T_scaled, V_scaled, rhf_ext_ctx.eri_scaled = scale_integrals(
        ctx.T, ctx.V, ctx.eri, ctx.theta
    )

    rhf_ext_ctx.H_core = T_scaled + V_scaled
    rhf_ext_ctx.core_mask = np.abs(rhf_ext_ctx.H_core) > 1e-10

    # eigensolver enforced
    if ctx.theta != 0:
        rhf_ext_ctx._eigensolver = "eig"
    else:
        rhf_ext_ctx._eigensolver = ctx._eigensolver

    # Convergence acceleration setup
    rhf_ext_ctx.acc_iteration_start, rhf_ext_ctx.acc_requested = initialize_conv_acc(
        ctx.acc_hist_size, ctx.conv_type, ctx.acc_iteration_start
    )

    return


def initialize_rhf_state_variable(
    rhf_ext_ctx: CSRHFConstants, rhf_state: CSRHFState
) -> None:
    rhf_state.use_conv_acc = False
    rhf_state.converged = False
    rhf_state.F_guess = []
    rhf_state.residuals = []
    rhf_state.F_next = np.zeros_like(rhf_ext_ctx.H_core)
    rhf_state.e_orb = np.zeros(rhf_ext_ctx.dim, dtype=np.complex128)
    rhf_state.C_prime = np.zeros(
        (rhf_ext_ctx.dim, rhf_ext_ctx.dim), dtype=np.complex128
    )
    rhf_state.C_munu = np.zeros_like(rhf_state.C_prime, dtype=np.complex128)
    rhf_state.error = np.complex128(1e10)

    return


def initialize_rhf_P_and_E(
    ctx: CSRHFContext,
    rhf_state: CSRHFState,
) -> None:

    if ctx.theta != 0.0:
        P, unscaled_E_RHF = compute_rhf_unscaled_density(ctx, ctx.verbose)
        E_prev = np.complex128(unscaled_E_RHF)

    else:
        P = guess_density_RHF(ctx.p_guess, len(ctx.S), ctx.initial_orbitals)
        E_prev = np.complex128(0.0)

    rhf_state.P = P
    rhf_state.E_prev = E_prev

    return


def compute_rhf_unscaled_density(
    ctx: CSRHFContext, verbose: bool
) -> Tuple[NDArray[np.complex128], np.complex128]:
    """
    Compute unscaled density matrix for theta=0.

    Parameters
    ----------
    ctx : CSRHFContext
        Original context with integrals and parameters.
    verbose : bool
        If True, print status.

    Returns
    -------
    P : NDArray[np.complex128]
        Unscaled density matrix.
    E_RHF : np.complex128
        Unscaled RHF energy.
    """
    if verbose:
        print("Converging unscaled case:")
    unscaled_ctx = CSRHFContext(
        S=ctx.S,
        T=ctx.T,
        V=ctx.V,
        eri=ctx.eri,
        n_electrons=ctx.n_electrons,
        theta=0.0,
        occupation=ctx.occupation,
        max_iter=ctx.max_iter,
        threshold=ctx.threshold,
        p_guess=ctx.p_guess,
        guess_max_iter=ctx.guess_max_iter,
        initial_orbitals=ctx.initial_orbitals,
        verbose=ctx.verbose,
        conv_type=ctx.conv_type,
        acc_hist_size=ctx.acc_hist_size,
        acc_iteration_start=10,
    )

    unscaled_res = CS_RHF(unscaled_ctx)

    if verbose:
        print("Unscaled energy: ", unscaled_res.E_RHF)
        print("\n\n\nConverging scaled case from unscaled density as reference:")

    P = unscaled_res.P
    return P, unscaled_res.E_RHF


# -------------------------------------------------------------
#  RHF Helper Functions
# -------------------------------------------------------------


def is_converged(
    ctx: CSRHFContext,
    rhf_state: CSRHFState,
) -> bool:
    """
    Check convergence based on residual norms.

    Parameters
    ----------
    verbose : bool
        If True, print status.
    threshold : float
        Convergence threshold.
    iter_idx : int
        Current iteration index.
    r : NDArray[np.complex128]
        Residual matrix.

    Returns
    -------
    converged : bool
        True if converged, else False.
    """
    converged: bool = False

    if ctx._convergence_criteria == "max":
        error_re: float = float(np.max(np.abs(rhf_state.r.real)))
        error_im: float = float(np.max(np.abs(rhf_state.r.imag)))
        if rhf_state.iteration > 1 and np.max([error_re, error_im]) < ctx.threshold:
            converged = True

    elif ctx._convergence_criteria == "norm":
        error = np.linalg.norm(rhf_state.r)
        if rhf_state.iteration > 1 and error < ctx.threshold:
            converged = True

    if converged and ctx.verbose:
        print(f"Convergence achieved after {rhf_state.iteration} iterations.")

    return converged


def print_cycle_data(
    _convergence_criteria: Literal["norm", "max"],
    rhf_state: CSRHFState,
) -> None:
    """
    Print SCF iteration data.

    Parameters
    ----------
    iter_idx : int
        Current iteration index.
    r : NDArray[np.complex128]
        Residual matrix.
    E_RHF : np.complex128
        Current RHF energy.
    E_diff : np.complex128
        Energy difference from previous iteration.

    Returns
    -------
    None
    """

    if _convergence_criteria == "norm":
        error = np.linalg.norm(rhf_state.r)
        print(
            f"  {rhf_state.iteration:5}     {rhf_state.E_RHF:24.6E}     {rhf_state.E_diff:24.6E}     {error:8.4E}"
        )
        return

    elif _convergence_criteria == "max":
        error_re, error_im = np.max(np.abs(rhf_state.r.real)), np.max(
            np.abs(rhf_state.r.imag)
        )
        print(
            f"{rhf_state.iteration:5}     {rhf_state.E_RHF:24.6E}     {rhf_state.E_diff:24.6E}     {error_re:8.4E}     {error_im:8.4E}j"
        )
        return


def print_table_header():
    """
    Print SCF iteration table header.

    Returns
    -------
    None
    """
    print("-" * 128)
    print(
        "|   Iter     |                   E_iter                      |                   Delta_e                   |      norm(e_i)      |"
    )
    print("-" * 128)


def conv_acc_criteria_met(
    ctx: CSRHFContext,
    rhf_ext_ctx: CSRHFConstants,
    rhf_state: CSRHFState,
) -> bool:
    use_conv_acc = False
    if (
        rhf_state.iteration - 1 == rhf_ext_ctx.acc_iteration_start
        and rhf_ext_ctx.acc_requested
    ):
        use_conv_acc = True
        if ctx.verbose:
            print("-" * 30, f"   STARTED {ctx.conv_type}  ", "-" * 30)
    return use_conv_acc


# -------------------------------------------------------------
#  RHF Update Functions
# -------------------------------------------------------------


def update_rhf_energy(
    rhf_ext_ctx: CSRHFConstants,
    rhf_state: CSRHFState,
) -> None:
    rhf_state.E_RHF = E_0_comp(rhf_state.P, rhf_ext_ctx.H_core, rhf_state.F)
    rhf_state.E_diff = rhf_state.E_RHF - rhf_state.E_prev
    rhf_state.E_prev = rhf_state.E_RHF


def update_rhf_density(
    ctx: CSRHFContext,
    rhf_ext_ctx: CSRHFConstants,
    rhf_state: CSRHFState,
) -> None:
    """
    Update density matrix and related quantities. For specifics, see calculate_P_next.

    Parameters
    ----------
    n_electrons : int
        Total electron count.
    X : NDArray[np.complex128]
        Transformation matrix.
    det : NDArray[np.int32]
        Occupation determinant.
    core_mask : NDArray[np.bool]
        Mask to mitigate numerical noise.
    F_next : NDArray[np.complex128]
        Next Fock matrix.

    Returns
    -------
    P : NDArray[np.complex128]
        Updated density matrix.
    e_orb : NDArray[np.complex128]
        Updated orbital energies.
    C_munu : NDArray[np.complex128]
        Updated MO coefficients.
    C_prime : NDArray[np.complex128]
        Updated transformed eigenvectors.

    Notes
    -----
    - This function applies a core mask to the density matrix to reduce numerical noise,
    where it is understood that different angular momentum matrix elements must be 0.
    """

    rhf_state.P, rhf_state.e_orb, rhf_state.C_munu, rhf_state.C_prime = (
        calculate_P_next(
            rhf_state.F_next, rhf_ext_ctx.X, rhf_ext_ctx.det, rhf_ext_ctx._eigensolver
        )
    )

    # P, e_orb, C_munu = calculate_P_next_2(
    #     F_next, S, n_electrons, det,
    # )

    rhf_state.P = rhf_state.P * rhf_ext_ctx.core_mask
    rhf_state.C_munu = sign_convention(rhf_state.C_munu)
    return


def update_rhf_acc_hist_size(cxt: CSRHFContext, rhf_state: CSRHFState) -> None:
    rhf_state.F_guess.append(rhf_state.F)
    rhf_state.residuals.append(rhf_state.r)

    if len(rhf_state.F_guess) > cxt.acc_hist_size:
        rhf_state.F_guess.pop(0)
        rhf_state.residuals.pop(0)

    return


def update_rhf_F_matrix(
    ctx: CSRHFContext,
    rhf_state: CSRHFState,
) -> None:
    if not rhf_state.use_conv_acc:
        F_next = rhf_state.F
    else:
        try:
            F_opt, r_opt = calc_diis_extrapolation(
                rhf_state.residuals, rhf_state.F_guess
            )
            F_next = F_opt

            if ctx.conv_type == "CROP":
                rhf_state.F_guess[-1] = F_opt
                rhf_state.residuals[-1] = r_opt
        except np.linalg.LinAlgError:
            if ctx.verbose:
                print(
                    "!!!!!!!!!!!!!!!! CONVERGENCE ACCELERATION CAUSED A SINGULAR MATRIX. REVERTING TO STANDARD SCF !!!!!!!!!!!!!!!"
                )
            rhf_state.use_conv_acc = False
            F_next = rhf_state.F

    rhf_state.F_next = F_next

    return


def update_rhf_F_and_r_comp(
    ctx: CSRHFContext,
    rhf_ext_ctx: CSRHFConstants,
    rhf_state: CSRHFState,
) -> None:
    """
    Calculate Fock matrix & Residual.

    Parameters
    ----------
    P : NDArray[np.complex128]
        Density matrix.
    S : NDArray[np.float64]
        Overlap matrix.
    H_core : NDArray[np.complex128]
        Core Hamiltonian.
    eri : NDArray[np.complex128]
        Integrals.

    Returns
    -------
    Tuple[NDArray, NDArray]
        (F, r).
    """
    rhf_state.F = rhf_ext_ctx.H_core + calc_g_matrix_comp(
        rhf_state.P, rhf_ext_ctx.eri_scaled
    )
    rhf_state.r = calc_residual_commutator(
        rhf_state.F, rhf_state.P, ctx.S.astype(np.complex128)
    )

    return


# -------------------------------------------------------------
#  RHF Trajectory & Plotting Functions
# -------------------------------------------------------------


def rhf_theta_traj(max_theta, n_points, cxt: CSRHFContext):
    """
    Sample energies along theta trajectory.

    Parameters
    ----------
    max_theta : float
        Max theta (radians).
    n_points : int
        Steps.
    cxt : CSRHFContext
        Base context.

    Returns
    -------
    thetas, energies : Tuple[NDArray, NDArray]
        Sampled angles & energies.
    """
    thetas = np.linspace(0, max_theta, n_points)
    energies = []
    for th in thetas:
        cxt.theta = th
        res = CS_RHF(cxt)
        if res.converged:
            energies.append(res.E_RHF)
        else:
            print(f"Traj {th} did not converge.")
        if cxt.verbose and res.converged:
            print(f"Converged point at theta = {th:6.4f} : E = {res.E_RHF:12.8f}")

    return thetas, np.array(energies, dtype=np.complex128)


if __name__ == "__main__":
    pass
