import copy
import numpy as np

from py_mods.src.SCF.scf_kernels import calc_residual_commutator
from py_mods.src.SCF.CSRHF import print_table_header

from py_mods.src.SCF_4c.types_4c import (
    CS_4c_KU_SCF_Context,
    CS_4c_KU_SCF_Constants,
    CS_4c_KU_SCF_State,
    CS_4c_KU_SCF_Results,
    allocate_CS_4c_KU_SCF_extended_context,
    allocate_CS_4c_KU_SCF_state,
    pack_CS_4c_KU_SCF_results,
)

from py_mods.src.SCF.linalg import transformation_matrix, sign_convention
from py_mods.src.SCF_4c.scf_4c_kernels import (
    guess_density_4c,
    g_matrix_4c,
    calc_diis_extrapolation_4c,
)

from py_mods.src.SCF.utils import initialize_conv_acc

from py_mods.src.SCF_4c.utils import (
    validate_4c_determinant,
    validate_CS_4c_KU_SCF_context_input,
)

from py_mods.src.SCF_4c.scf_4c_kernels import (
    scale_4c_integrals,
    calculate_P_next_4c,
    guess_density_4c,
)


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
    ext_ctx.X = transformation_matrix(
        ctx.S.astype(np.complex128), remove_lindep=ctx.remove_lindep
    ).astype(np.complex128)

    n_lindep = np.abs(
        ext_ctx.X.shape[0] - ext_ctx.X.shape[1]
    )  # The difference between row and column size is the number of lindeps

    if n_lindep > 0: 
        H_core = ctx.V + ctx.T + ctx.W
        F_0 = ext_ctx.X.conj().T @ H_core.astype(np.complex128) @ ext_ctx.X
        e_0 = np.linalg.eigvals(F_0)
        n_pos_ener_eigvals = sum([1 for ev in e_0 if ev > -2000])
        n_neg_ener_eigvals = sum([1 for ev in e_0 if ev <= -2000])
        old_nL = ctx.nL
        old_nS = ctx.nS

        if ctx.verbose:
            print(
                f"Linear dependencies removed: LC = {ctx.nL *2- n_pos_ener_eigvals}. SC = {ctx.nS*2 - n_neg_ener_eigvals}"
            )
            print(f"Basis resize: LC = {ctx.nL*2} -> {n_pos_ener_eigvals}, SC = {ctx.nS*2} -> {n_neg_ener_eigvals}. Total size: {ctx.nL*2 + ctx.nS*2} -> {n_pos_ener_eigvals + n_neg_ener_eigvals}.")

        ctx.nL = n_pos_ener_eigvals // 2
        ctx.nS = n_neg_ener_eigvals // 2

        if isinstance(ctx.occ, np.ndarray):
            # if ctx.verbose:
            #     print("Original occupation:\n", ctx.occ, len(ctx.occ))
            
            if len(ctx.occ) == 2 * (old_nL + old_nS):
                old_lc_occ = ctx.occ[2 * old_nS :]
            else:
                old_lc_occ = ctx.occ
                
            lc_occ = old_lc_occ[:n_pos_ener_eigvals]
            new_occ = np.zeros(2 * (ctx.nL + ctx.nS), dtype=np.int32)
            length = min(len(lc_occ), 2 * ctx.nL)
            new_occ[2 * ctx.nS : 2 * ctx.nS + length] = lc_occ[:length]
            ctx.occ = new_occ
            
            # if ctx.verbose:
            #     print("Modified occupation after linear dependency removal:\n", ctx.occ, len(ctx.occ))


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


# -------------------------------------------------------------
#  CS-4c-KU-SCF Helper Functions
# -------------------------------------------------------------


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
        print("-" * 133)
        print(f"Convergence achieved after {state.iteration} iterations.")

    return converged


def print_cycle_data_4c(convergence_criteria: str, state: CS_4c_KU_SCF_State) -> None:
    if convergence_criteria == "norm":
        print(
            f"| {state.iteration:^8} | {state.E_SCF:^45.16f} | {state.E_diff:^45.16f} | {state.error:^22.4E} |"
        )
    elif convergence_criteria == "max":
        print(
            f"| {state.iteration:^8} | {state.E_SCF:^45.16f} | {state.E_diff:^45.16f} | {state.error:^22.4E} |"
        )


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
            print(f"|{msg:-^131}|")

    return use_conv_acc


def update_CS_4c_KU_SCF_energy(
    ext_ctx: CS_4c_KU_SCF_Constants,
    state: CS_4c_KU_SCF_State,
) -> None:
    e_scf = np.linalg.trace(state.P @ (ext_ctx.H_core + state.F)) * 0.5
    state.E_SCF = e_scf
    state.E_diff = state.E_SCF - state.E_prev
    state.E_prev = state.E_SCF


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


def update_CS_4c_KU_SCF_acc_hist_size(
    ctx: CS_4c_KU_SCF_Context, state: CS_4c_KU_SCF_State
) -> None:
    state.F_guess.append(state.F)
    state.residuals.append(state.r)

    if len(state.F_guess) > ctx.acc_hist_size:
        state.F_guess.pop(0)
        state.residuals.pop(0)

    return


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


def update_CS_4c_KU_SCF_F_and_r_comp(
    ctx: CS_4c_KU_SCF_Context,
    ext_ctx: CS_4c_KU_SCF_Constants,
    state: CS_4c_KU_SCF_State,
) -> None:
    G = g_matrix_4c(state.P, ext_ctx.eri_scaled)
    state.F = ext_ctx.H_core + G
    state.r = calc_residual_commutator(state.F, state.P, ctx.S.astype(np.complex128))
    state.error = float(np.linalg.norm(state.r.flatten()))


# -------------------------------------------------------------
#  CS-4c-KU-SCF Trajectory & Plotting Functions
# -------------------------------------------------------------


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
