import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from py_mods.src.SCF.CSRHF import CS_RHF
from py_mods.src.SCF.CS_SCF_types import CSRHFContext
import copy

from py_mods.src.SCF.scf_kernels import (
    E_0_unrestricted_comp,
    guess_density_RHF,
    scale_integrals,
    calc_diis_extrapolation,
    calculate_P_next,
    calculate_unrestricted_F_and_r_comp,
)

from py_mods.src.SCF.utils import (
    validate_unrestricted_determinant,
    validate_uhf_context_input,
    initialize_conv_acc,
)

from py_mods.src.SCF.linalg import (
    transformation_matrix,
    sign_convention,
)

from py_mods.src.SCF.CS_SCF_types import (
    CSUHFContext,
    CSUHFResults,
    CSUHFConstants,
    CSUHFState,
    UHFSpinDiagnostics,
    allocate_uhf_extended_context,
    allocate_uhf_state,
    pack_uhf_results,
)


def CS_UHF(ctx: CSUHFContext) -> CSUHFResults:
    """
    Perform a Complex Scaled RHF calculation.

    Takes overlap, kinetic, nuclear attraction and two-electron integrals,
    applies complex scaling by angle `theta` and runs an UHF loop
    using biorthogonal diagonalization.

    Parameters
    ----------
    context : CS_UHF_Context
        Context object containing all parameters for the calculation.

    Returns
    -------
    CS_UHF_Results
        Results object. For complete description of parameters see definition.

    Notes
    ------
    - The system bust be a closed shell: n_electrons must be even. This is asserted.
    - Integrals must be passed and have the same dimensions. This is asserted.


    - Implementation was done based on "Modern Quantum Chemistry" by Szabo and Ostlund.
    - DIIS implementation was based on [Pulay](https://doi.org/10.1002/jcc.540030413).
    - CROP implementation was based on [Ettenhuber, Jorgensen](https://doi.org/10.1021/ct501114q).

    ^* CROP algorithm does not compute the new trial as t_opt + w_opt, as it breaks convergence here.
    """
    if ctx.theta == 0:
        return _csuhf_kernel(ctx)

    else:
        scaled_context = copy.deepcopy(ctx)

        # perform unscaled calculation first
        ctx.theta = 0
        unscaled_rhf = _csuhf_kernel(ctx)

        # use results for scaled calculation
        scaled_context.p_guess = "INPORB"
        scaled_context.initial_orbitals = [unscaled_rhf.P_alpha, unscaled_rhf.P_beta]
        return _csuhf_kernel(scaled_context)


def _csuhf_kernel(ctx: CSUHFContext) -> CSUHFResults:
    """
    Perform a Complex Scaled RHF calculation.

    Takes overlap, kinetic, nuclear attraction and two-electron integrals,
    applies complex scaling by angle `theta` and runs an UHF loop
    using biorthogonal diagonalization.

    Parameters
    ----------
    context : CS_UHF_Context
        Context object containing all parameters for the calculation.

    Returns
    -------
    CS_UHF_Results
        Results object. For complete description of parameters see definition.

    Notes
    ------
    - The system bust be a closed shell: n_electrons must be even. This is asserted.
    - Integrals must be passed and have the same dimensions. This is asserted.


    - Implementation was done based on "Modern Quantum Chemistry" by Szabo and Ostlund.
    - DIIS implementation was based on [Pulay](https://doi.org/10.1002/jcc.540030413).
    - CROP implementation was based on [Ettenhuber, Jorgensen](https://doi.org/10.1021/ct501114q).

    ^* CROP algorithm does not compute the new trial as t_opt + w_opt, as it breaks convergence here.
    """
    enforce_multiplicity(ctx)
    validate_uhf_context_input(ctx)

    # Allocate extended context and RHF state
    uhf_ext_ctx = allocate_uhf_extended_context(ctx)
    uhf_state = allocate_uhf_state(ctx)

    # Transform matrix & scale & validate determinants & set convergence acceleration
    initialize_uhf_extended_context(ctx, uhf_ext_ctx)

    # Guess density and initialize E0
    initialize_uhf_P_and_E(ctx, uhf_ext_ctx, uhf_state)

    P_guess_alpha = uhf_state.P_alpha.copy()
    P_guess_beta = uhf_state.P_beta.copy()

    if ctx.verbose:
        print("\n\nAlpha occupation: ", uhf_ext_ctx.det[0])
        print("Beta  occupation: ", uhf_ext_ctx.det[1])
        print_table_header()

    # SCF loop
    for iter_idx in range(ctx.max_iter):
        # calculate F_n and r_n from P_n
        uhf_state.iteration += 1
        update_uhf_F_and_r_comp(ctx, uhf_ext_ctx, uhf_state)
        update_uhf_energy(uhf_ext_ctx, uhf_state)

        if ctx.verbose:
            print_cycle_data(uhf_state)

        # Check convergence
        uhf_state.converged = is_converged_uhf(ctx, uhf_state)
        if uhf_state.converged:
            break

        # Save in memory guesses and residuals keeping size of Convergence Algorithm space
        update_uhf_acc_hist_size(ctx, uhf_state)
        uhf_state.P_old_alpha = uhf_state.P_alpha
        uhf_state.P_old_beta = uhf_state.P_beta

        # Choose F for P_{n+1}
        update_uhf_F_matrix(ctx, uhf_state)

        # P_LR, C_munu, e_values, L_munu, C_munu, P_RR, C_prime
        update_uhf_density(uhf_ext_ctx, uhf_state)

        if ctx.theta == 0.0:
            uhf_state.P_alpha.imag = uhf_state.P_beta.imag = 0

        uhf_state.P_total = uhf_state.P_alpha + uhf_state.P_beta
        uhf_state.P_diff = uhf_state.P_alpha - uhf_state.P_beta

        uhf_state.E_prev = uhf_state.E_UHF

        # Check Convergence Algorithm activation
        uhf_state.use_conv_acc = conv_acc_criteria_met(ctx, uhf_ext_ctx, uhf_state)

    update_uhf_density(uhf_ext_ctx, uhf_state)

    S_diagnostics = perform_spin_diagostics(ctx, uhf_state)

    uhf_results = pack_uhf_results(
        ctx, uhf_ext_ctx, uhf_state, P_guess_alpha, P_guess_beta, S_diagnostics
    )

    return uhf_results


# -------------------------------------------------------------
#  RHF Initialization Functions
# -------------------------------------------------------------


def enforce_multiplicity(ctx: CSUHFContext) -> None:
    if ctx.mult == -1:
        ctx.mult = 0 if ctx.n_electrons % 2 == 0 else 1

    return


def initialize_uhf_extended_context(
    ctx: CSUHFContext, uhf_ext_ctx: CSUHFConstants
) -> None:
    """
    Setup extended context with transformation matrix, validated determinants and scaled integrals. Also set up convergence acceleration parameters.

    Parameters
    ----------
    ctx : CSUHFContext
        Original context with integrals and parameters.
    uhf_ext_ctx : CSRHFConstants
        Initialized extended context to compute.

    Returns
    -------
    None
    """
    uhf_ext_ctx.dim = len(ctx.S)
    uhf_ext_ctx.X = np.array(
        transformation_matrix(ctx.S.astype(np.complex128)), dtype=np.complex128
    )

    # validate occupation
    uhf_ext_ctx.det[0], uhf_ext_ctx.det[1], _ = validate_unrestricted_determinant(
        ctx.n_electrons, ctx.occupation, uhf_ext_ctx.dim, ctx.mult
    )

    uhf_ext_ctx.alpha_elec = sum(uhf_ext_ctx.det[0])
    uhf_ext_ctx.beta_elec = sum(uhf_ext_ctx.det[1])

    # rescaling the integrals
    T_scaled, V_scaled, eri_scaled = scale_integrals(ctx.T, ctx.V, ctx.eri, ctx.theta)

    uhf_ext_ctx.H_core = T_scaled + V_scaled
    uhf_ext_ctx.eri_scaled = eri_scaled

    uhf_ext_ctx.core_mask = np.abs(uhf_ext_ctx.H_core) > 1e-10

    # eigensolver enforced
    if ctx.theta != 0:
        uhf_ext_ctx._eigensolver = "eig"
    else:
        uhf_ext_ctx._eigensolver = ctx._eigensolver

    # Convergence acceleration setup
    uhf_ext_ctx.acc_iteration_start, uhf_ext_ctx.acc_requested = initialize_conv_acc(
        ctx.acc_hist_size, ctx.conv_type, ctx.acc_iteration_start
    )

    return


def initialize_uhf_P_and_E(
    ctx: CSUHFContext, ext_uhf_ctx: CSUHFConstants, uhf_state: CSUHFState
) -> None:
    """
    Initialize density matrix and E0.

    Parameters
    ----------
    ctx : CSUHFContext
        Original context with integrals and parameters.
    uhf_state : CSRHFState
        Initialized state to compute.

    Returns
    -------
    None
    """
    guess_density_UHF(ctx, ext_uhf_ctx, uhf_state)

    return


def compute_uhf_unscaled_density(
    ctx: CSUHFContext,
    ext_uhf_ctx: CSUHFConstants,
    uhf_state: CSUHFState,
) -> None:
    if ctx.verbose:
        print("Converging unscaled UHF case:")

    unscaled_ctx = CSUHFContext(
        S=ctx.S.astype(np.complex128),
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

    unscaled_res = CS_UHF(unscaled_ctx)

    if ctx.verbose:
        print("Unscaled energy: ", unscaled_res.E_UHF)
        print("\n\n\nConverging scaled case from unscaled density as reference:")

    uhf_state.P_alpha = unscaled_res.P_alpha
    uhf_state.P_beta = unscaled_res.P_beta
    uhf_state.E_prev = unscaled_res.E_UHF

    return


def guess_density_UHF(
    ctx: CSUHFContext, ext_uhf_ctx: CSUHFConstants, uhf_state: CSUHFState
):
    if ctx.p_guess == "RHF":
        elec_pre = ctx.n_electrons if ctx.n_electrons % 2 == 0 else ctx.n_electrons - 1
        if isinstance(ctx.guess_max_iter, int):
            guess_iter = ctx.guess_max_iter

        else:
            guess_iter = 12
            guess_context = CSRHFContext(
                ctx.S.real,
                ctx.T.real,
                ctx.V.real,
                ctx.eri.real,
                n_electrons=elec_pre,
                theta=0,
                max_iter=guess_iter,
                threshold=1e-14,
                p_guess="core",
                verbose=False,
            )
            guess_scf = CS_RHF(guess_context)

            p_final = guess_scf.P
            uhf_state.E_prev = guess_scf.E_RHF

            uhf_state.P_alpha = np.copy(p_final)
            uhf_state.P_beta = np.copy(p_final)

    elif ctx.p_guess == "INPORB":
        assert (
            ctx.initial_orbitals is not None
        ), "Initial orbitals must be provided when p_guess is 'INPORB'"
        assert (
            len(ctx.initial_orbitals) == 2
        ), "Initial orbitals must be a list of two density matrices [P_alpha, P_beta]"
        assert (
            ctx.initial_orbitals[0].shape
            == ctx.initial_orbitals[1].shape
            == (
                ext_uhf_ctx.dim,
                ext_uhf_ctx.dim,
            )
        ), "Initial alpha density matrix has incorrect dimensions."
        uhf_state.P_alpha = ctx.initial_orbitals[0]
        uhf_state.P_beta = ctx.initial_orbitals[1]

    else:
        uhf_state.P_alpha = uhf_state.P_beta = guess_density_RHF(
            ctx.p_guess, ext_uhf_ctx.dim, None
        )

    if ctx.break_symm is not None:
        if ctx.break_symm == True:
            ctx.break_symm = "arbitrary"

        if ctx.break_symm == "arbitrary":
            # note that breaking symmetry will only make sense when the guess is not zeros
            dim = uhf_state.P_beta.shape[0]
            half = dim // 2

            if np.allclose(uhf_state.P_beta, uhf_state.P_alpha):
                uhf_state.P_beta[:half, :half] += 0.1
                uhf_state.P_alpha[half:, half:] -= 0.1

                uhf_state.P_alpha /= 4
                uhf_state.P_beta /= 4
                
        elif ctx.break_symm == "random":
            raise NotImplementedError("Random symmetry breaking not implemented yet.")

        elif ctx.break_symm == "perturbation":
            raise NotImplementedError(
                "Perturbation symmetry breaking not implemented yet."
            )
        else:
            raise ValueError(
                "Invalid value for break_symm. Accepted values are None, 'arbitrary' (or True), 'random', 'perturbation'."
            )

    return


# -------------------------------------------------------------
#  RHF Helper Functions
# -------------------------------------------------------------


def print_table_header():
    """
    Print SCF iteration table header.

    Returns
    -------
    None
    """
    print("-" * 135)
    print(f"| {'Iter':^8} | {'E_iter':^45} | {'Delta_e':^45} | {'norm(e_i)':^22} |")
    print("-" * 135)


def conv_acc_criteria_met(
    ctx: CSUHFContext,
    uhf_ext_ctx: CSUHFConstants,
    uhf_state: CSUHFState,
) -> bool:
    use_conv_acc = uhf_state.use_conv_acc
    if (
        not use_conv_acc
        and uhf_state.iteration + 1 >= ctx.acc_iteration_start
        and uhf_ext_ctx.acc_requested
    ):  #  and error < conv_thresh and not use_conv:
        use_conv_acc = True

        if ctx.verbose:
            msg = f" STARTED {ctx.conv_type} "
            print(f"|{msg:-^133}|")

    return use_conv_acc


def is_converged_uhf(
    ctx: CSUHFContext,
    uhf_state: CSUHFState,
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
        error_re_alpha = float(np.max(np.abs(uhf_state.r_alpha.real)))
        error_im_alpha = float(np.max(np.abs(uhf_state.r_alpha.imag)))
        error_re_beta = float(np.max(np.abs(uhf_state.r_beta.real)))
        error_im_beta = float(np.max(np.abs(uhf_state.r_beta.imag)))

        error_alpha = max(error_re_alpha, error_im_alpha)
        error_beta = max(error_re_beta, error_im_beta)

        if (
            uhf_state.iteration > 1
            and np.max([error_alpha, error_beta]) < ctx.threshold
        ):
            converged = True

    elif ctx._convergence_criteria == "norm":
        error_alpha = float(np.linalg.norm(uhf_state.r_alpha))
        error_beta = float(np.linalg.norm(uhf_state.r_beta))

        error = max(error_alpha, error_beta)

        if uhf_state.iteration > 1 and error < ctx.threshold:
            converged = True

    if converged and ctx.verbose:
        print(f"Convergence achieved after {uhf_state.iteration} iterations.")

    return converged


def print_cycle_data(uhf_state: CSUHFState) -> None:
    print(f"| {uhf_state.iteration:^8} | {uhf_state.E_UHF:^45.16f} | {uhf_state.E_diff:^45.16f} | {uhf_state.error:^22.4E} |")

    return


# -------------------------------------------------------------
#  UHF Update Functions
# -------------------------------------------------------------


def update_uhf_density(
    uhf_ext_ctx: CSUHFConstants,
    uhf_state: CSUHFState,
) -> None:
    uhf_state.P_alpha, uhf_state.e_orb_alpha, uhf_state.C_munu_alpha, *_ = (
        calculate_P_next(
            uhf_state.F_next_alpha,
            uhf_ext_ctx.X,
            uhf_ext_ctx.det[0],
            uhf_ext_ctx._eigensolver,
        )
    )
    uhf_state.P_beta, uhf_state.e_orb_beta, uhf_state.C_munu_beta, *_ = (
        calculate_P_next(
            uhf_state.F_next_beta,
            uhf_ext_ctx.X,
            uhf_ext_ctx.det[1],
            uhf_ext_ctx._eigensolver,
        )
    )

    uhf_state.C_munu_alpha = sign_convention(uhf_state.C_munu_alpha)
    uhf_state.C_munu_beta = sign_convention(uhf_state.C_munu_beta)

    return


def update_uhf_F_matrix(
    ctx: CSUHFContext,
    uhf_state: CSUHFState,
) -> None:
    if not uhf_state.use_conv_acc:
        uhf_state.F_next_alpha = uhf_state.F_alpha
        uhf_state.F_next_beta = uhf_state.F_beta

    elif uhf_state.use_conv_acc:
        try:
            F_opt_alph, r_opt_alpha = calc_diis_extrapolation(
                uhf_state.residuals_alpha, uhf_state.F_guess_alpha, ctx.theta
            )
            F_opt_beta, r_opt_beta = calc_diis_extrapolation(
                uhf_state.residuals_beta, uhf_state.F_guess_beta, ctx.theta
            )

            # Default is DIIS
            uhf_state.F_next_alpha = F_opt_alph
            uhf_state.F_next_beta = F_opt_beta

            if ctx.conv_type == "CROP":
                uhf_state.F_guess_alpha[-1] = F_opt_alph
                uhf_state.F_guess_beta[-1] = F_opt_beta
                uhf_state.residuals_alpha[-1] = r_opt_alpha
                uhf_state.residuals_beta[-1] = r_opt_beta

                # + r_opt # equation 32 Ettenhuber, r_opt should be here, but it diverges idk why

                uhf_state.F_next_alpha = F_opt_alph  # + r_opt_alpha
                uhf_state.F_next_beta = F_opt_beta  # + r_opt_beta

        except np.linalg.LinAlgError:
            if ctx.verbose:
                print(
                    "!!!!!!!!!!!!!!!! CONVERGENCE ACCELERATION CAUSED A SINGULAR MATRIX. REVERTING TO STANDARD SCF !!!!!!!!!!!!!!!"
                )
            uhf_state.use_conv_acc = False

    return


def update_uhf_acc_hist_size(
    ctx: CSUHFContext,
    uhf_state: CSUHFState,
) -> None:
    uhf_state.F_guess_alpha.append(uhf_state.F_alpha)
    uhf_state.F_guess_beta.append(uhf_state.F_beta)
    uhf_state.residuals_alpha.append(uhf_state.r_alpha)
    uhf_state.residuals_beta.append(uhf_state.r_beta)

    if len(uhf_state.F_guess_alpha) > ctx.acc_hist_size:
        uhf_state.F_guess_alpha.pop(0)
        uhf_state.F_guess_beta.pop(0)
        uhf_state.residuals_alpha.pop(0)
        uhf_state.residuals_beta.pop(0)

    return


def update_uhf_energy(uhf_ext_ctx: CSUHFConstants, uhf_state: CSUHFState) -> None:
    uhf_state.E_UHF = E_0_unrestricted_comp(
        uhf_state.P_alpha,
        uhf_state.P_beta,
        uhf_ext_ctx.H_core,
        uhf_state.F_alpha.reshape(uhf_ext_ctx.H_core.shape),
        uhf_state.F_beta.reshape(uhf_ext_ctx.H_core.shape),
    )

    uhf_state.E_diff = uhf_state.E_UHF - uhf_state.E_prev

    return


def update_uhf_F_and_r_comp(
    ctx: CSUHFContext,
    uhf_ext_ctx: CSUHFConstants,
    uhf_state: CSUHFState,
) -> None:
    uhf_state.F_alpha, uhf_state.r_alpha, uhf_state.F_beta, uhf_state.r_beta = (
        calculate_unrestricted_F_and_r_comp(
            uhf_state.P_alpha,
            uhf_state.P_beta,
            ctx.S.astype(np.complex128),
            uhf_ext_ctx.H_core,
            uhf_ext_ctx.eri_scaled,
        )
    )

    uhf_state.error_alpha = float(np.linalg.norm(uhf_state.r_alpha.flatten()))
    uhf_state.error_beta = float(np.linalg.norm(uhf_state.r_beta.flatten()))

    uhf_state.error = max(uhf_state.error_alpha, uhf_state.error_beta)

    return


# -------------------------------------------------------------
#  UHF Spin diagnostics
# -------------------------------------------------------------


def perform_spin_diagostics(
    ctx: CSUHFContext, uhf_state: CSUHFState
) -> UHFSpinDiagnostics:
    S_diagnostics: UHFSpinDiagnostics = calculate_s2_expectation(
        uhf_state.P_alpha, uhf_state.P_beta, ctx.S.astype(np.complex128), ctx.verbose
    )

    uhf_state.final_alpha_elec = np.trace(
        uhf_state.P_alpha.real @ ctx.S.astype(np.complex128)
    )
    uhf_state.final_beta_elec = np.trace(
        uhf_state.P_beta.real @ ctx.S.astype(np.complex128)
    )

    if (
        abs(uhf_state.final_alpha_elec + uhf_state.final_beta_elec - ctx.n_electrons)
        > 1e-10
    ):
        raise RuntimeError("Number of electrons was not conserved in the calculation")

    return S_diagnostics


def calculate_s2_expectation(P_alpha, P_beta, S, verbose=False):
    """
    Calculate the expectation value of S^2 for a UHF wavefunction.

    Calculated using <S^2> = S_z^2 + S_z + N_beta - Tr(P_alpha @ S @ P_beta @ S)

    Parameters
    ----------
    P_alpha : NDArray, shape (n,n)
    Alpha density matrix
    P_beta : NDArray, shape (n,n)
    Beta density matrix
    S : NDArray, shape (n,n)
    Overlap matrix

    Returns
    -------
    s2 : float
    <S^2> expectation value

    s_z : float
    S_z value
    spin_contamination : float

    Amount of spin contamination (deviation from exact value)
    """

    # Calculate number of electrons
    N_alpha = np.trace(P_alpha.real @ S)
    N_beta = np.trace(P_beta.real @ S)

    # Calculate S_z
    S_z = (N_alpha - N_beta) / 2

    # Calculate <S^2>
    overlap_term = np.trace(P_alpha @ S @ P_beta @ S).real
    s2 = S_z * (S_z + 1) + N_beta - overlap_term

    # Expected value for pure spin state
    s2_exact = S_z * (S_z + 1)
    spin_contamination = s2 - s2_exact

    if verbose:
        print(f"\n---------------  Spin Diagnostics  ---------------")
        print(f"N_alpha = {(N_alpha):6f}")
        print(f"N_beta  = {(N_beta):6f}")
        print(f"S_z = {S_z:.4f}")
        print(f"<S^2> = {s2:.6f}")
        print(f"<S^2>_exact = {s2_exact:.4f}")
        print(f"Spin contamination = {spin_contamination:.6f}")

    return UHFSpinDiagnostics(N_alpha, N_beta, s2, S_z, spin_contamination)


# -------------------------------------------------------------
#  RHF Trajectory & Plotting Functions
# -------------------------------------------------------------


def UHF_theta_traj(max_theta, n_points, context: CSUHFContext):
    """
    Sample CS_RHF energies along a theta trajectory.

    Parameters
    ----------
    max_theta : float
        Maximum theta to sample (radians).
    n_points : int
        Number of points along the trajectory.
    overlap, kin, vnuc, eri : NDArray
        Integral arrays passed to CS_RHF.
    nelec : int
        Number of electrons.
    occupation : int or array-like, optional
        Occupation vector or -1 for default.
    max_iter, threshold, p_guess, verbose : optional
        SCF control parameters forwarded to CS_RHF.
    conv_type : Literal[None, 'DIIS', 'CROP'], optional
        Type of Convergence Algorithm to use. If None, no algorithm is used.
    acc_hist_size : int, optional
        Number of previous Fock matrices and residuals to store for Convergence Algorithm.
    acc_iteration_start : int, optional
        Iteration number to start Convergence Algorithm.

    Returns
    -------
    thetas : NDArray
        Array of sampled theta values.
    energies : list
        List of complex energies for converged points.
    """
    thetas = np.linspace(0, max_theta, n_points)
    energies = []
    for th in thetas:
        context.theta = th
        result = CS_UHF(context)
        if result.converged:
            energies.append(result.E_UHF)
        else:
            print(f"Traj {th} did not converge.")
        if context.verbose and result.converged:
            print(f"Converged point at theta = {th:6.4f} : E = {result.E_UHF:12.8f}")

    return thetas, energies


if __name__ == "__main__":
    pass
