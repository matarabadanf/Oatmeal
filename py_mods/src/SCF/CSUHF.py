import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, Union, List, final
from py_mods.src.SCF.CSRHF import CS_RHF
from py_mods.src.SCF.types import CSRHFContext
from dataclasses import dataclass

from py_mods.src.SCF.scf_kernels import (
    E_0_unrestricted_comp,
    guess_density_RHF,
    scale_integrals,
    calc_diis_extrapolation,
    calculate_P_next,
    calculate_unrestricted_F_and_r_comp,
)

from py_mods.src.SCF.utils import validate_unrestricted_determinant, initialize_conv_acc

from py_mods.src.SCF.linalg import (
    transformation_matrix,
)


@dataclass
class CSUHFContext:
    """
    Context class for CS_UHF calculations.

    Attributes
    ----------
    S : NDArray[np.float64], shape (n, n)
        Overlap matrix.
    T : NDArray[np.float64], shape (n, n)
        Kinetic energy matrix.
    V : NDArray[np.float64], shape (n, n)
        Nuclear attraction matrix.
    eri : NDArray[np.float64], shape (n, n, n, n)
        Electron repulsion integrals.
    n_electrons : int
        Total number of electrons (must be even for closed-shell RHF).
    theta : float
        Complex-scaling angle (radians).
    occupation : int or NDArray[np.int32] or None
        If -1 (or None) build a default RHF occupation vector (2,2,...,0).
        If an ndarray is provided it must sum to n_electrons.
    max_iter : int, optional
        Maximum SCF iterations.
    threshold : float, optional
        Convergence threshold for density matrix difference.
    p_guess : Literal['core', 'ones', 'RHF', 'IMPORB'], optional
        Type of initial guess for density matrix.
    guess_max_iter : int or None, optional
        If p_guess is 'RHF', number of iterations to run the preliminary RHF calculation.
    initial_orbitals : NDArray[np.float64] or None, optional
        If p_guess is 'initial_orbitals', the initial guess orbitals.
    break_symm : bool, optional
        If True, breaks the symmetry of the initial guess density matrix.
    verbose : bool, optional
        If True print iteration progress.
    conv_type : Literal[None, 'DIIS', 'CROP'], optional
        Type of Convergence Algorithm to use. If None, no algorithm is used.
    acc_hist_size : int, optional
        Number of previous Fock matrices and residuals to store for Convergence Algorithm.
    acc_iteration_start : int, optional
        Iteration number to start Convergence Algorithm.

    Notes
    -----
    - Symmety is broken by zeroing the beta density matrix in the occupied space.
    - Breaking symmetry only makes sense when the guess is not zeros.
    """

    # Required
    S: NDArray[np.float64]
    T: NDArray[np.float64]
    V: NDArray[np.float64]
    eri: NDArray[np.float64]
    n_electrons: int

    # Optional
    mult: int = -1
    theta: float = 0.0
    occupation: Union[int, Tuple[NDArray[np.int32], NDArray[np.int32]], None] = None
    max_iter: int = 100
    threshold: float = 1e-12
    p_guess: Literal["core", "ones", "RHF", "IMPORB"] = "core"
    guess_max_iter: Union[int, None] = None
    initial_orbitals: Union[NDArray[np.float64], NDArray[np.complex128], None] = None
    break_symm: bool = False
    verbose: bool = False
    conv_type: Literal[None, "DIIS", "CROP"] = "DIIS"
    acc_hist_size: int = 10
    acc_iteration_start: int = 12

    # Internal
    _eigensolver: Literal["eig", "eigh"] = "eigh"
    _convergence_criteria: Literal["norm", "max"] = "max"


@dataclass
class CSUHFConstants:
    dim: int
    X: NDArray[np.complex128]
    det: NDArray[np.int32]  # of size (Dim, 2)
    alpha_elec: int
    beta_elec: int

    eri_scaled: NDArray[np.complex128]
    H_core: NDArray[np.complex128]
    core_mask: NDArray[np.bool]
    _eigensolver: Literal["eig", "eigh"]
    acc_iteration_start: int = 0
    acc_requested: bool = False


@dataclass
class CSUHFState:
    # control
    iteration: int
    use_conv_acc: bool
    converged: bool

    # Energies
    E_UHF: np.complex128

    e_orb_alpha: NDArray[np.complex128]
    e_orb_beta: NDArray[np.complex128]

    # Coefficients and electrons
    C_munu_alpha: NDArray[np.complex128]
    C_munu_beta: NDArray[np.complex128]
    C_prime_alpha: NDArray[np.complex128]
    C_prime_beta: NDArray[np.complex128]

    final_alpha_elec: int
    final_beta_elec: int

    # memory and error
    F_guess_alpha: List[NDArray[np.complex128]]
    F_guess_beta: List[NDArray[np.complex128]]
    residuals_alpha: List[NDArray[np.complex128]]
    residuals_beta: List[NDArray[np.complex128]]
    P_old_alpha: NDArray[np.complex128]
    P_old_beta: NDArray[np.complex128]

    r_alpha: NDArray[np.complex128]
    r_beta: NDArray[np.complex128]

    error_alpha: complex
    error_beta: complex
    error: complex

    E_diff: np.complex128
    E_prev: np.complex128

    # Fock and densities
    F_alpha: NDArray[np.complex128]
    F_beta: NDArray[np.complex128]
    F_next_alpha: NDArray[np.complex128]
    F_next_beta: NDArray[np.complex128]

    P_alpha: NDArray[np.complex128]
    P_beta: NDArray[np.complex128]

    P_total: NDArray[np.complex128]
    P_diff: NDArray[np.complex128]


@dataclass
class UHFSpinDiagnostics(object):
    N_alpha: int
    N_beta: int
    s2: float
    S_z: float
    spin_contamination: float


@dataclass
class CSUHFResults(object):
    """
    Results class for CS_UHF calculations.

    Attributes
    ----------
    context : CSUHFContext
        Context object used for the calculation.
    converged : bool
        Wether SCF calculation converged.
    E_UHF : float
        Final UHF energy.
    e_alph : NDArray[np.complex128], shape (n,)
        Alpha orbital energies.
    e_beta : NDArray[np.complex128], shape (n,)
        Beta orbital energies.
    X : NDArray[np.complex128], shape (n, n)
        Transformation matrix.
    P_guess_alpha: NDArray[np.complex128], shape (n, n)
        Initial alpha density matrix guess.
    P_guess_beta: NDArray[np.complex128], shape (n, n)
        Initial beta density matrix guess.
    P_alph : NDArray[np.complex128], shape (n, n)
        Alpha density matrix.
    P_beta : NDArray[np.complex128], shape (n, n)
        Beta density matrix.
    P_total : NDArray[np.complex128], shape (n, n)
        Total density matrix.
    P_diff : NDArray[np.complex128], shape (n, n)
        Spin density matrix (P_alpha - P_beta).
    L_alpha : NDArray[np.complex128], shape (n, n)
        Left eigenvector matrix for alpha spin.
    C_alpha : NDArray[np.complex128], shape (n, n)
        Right eigenvector matrix for alpha spin.
    L_beta : NDArray[np.complex128], shape (n, n)
        Left eigenvector matrix for beta spin.
    C_beta : NDArray[np.complex128], shape (n, n)
        Right eigenvector matrix for beta spin.
    """

    context: CSUHFContext
    converged: bool
    E_UHF: np.complex128
    e_alpha: NDArray[np.complex128]
    e_beta: NDArray[np.complex128]
    n_alpha: float
    n_beta: float
    det: Tuple[NDArray[np.int32], NDArray[np.int32]]
    X: NDArray[np.complex128]
    F_final_alph: NDArray[np.complex128]
    F_final_beta: NDArray[np.complex128]
    P_guess_alpha: NDArray[np.complex128]
    P_guess_beta: NDArray[np.complex128]
    P_alpha: NDArray[np.complex128]
    P_beta: NDArray[np.complex128]
    P_total: NDArray[np.complex128]
    P_diff: NDArray[np.complex128]
    C_alpha: NDArray[np.complex128]
    C_beta: NDArray[np.complex128]
    S_diagnostics: UHFSpinDiagnostics
    error: float
    iterations: int
    scaled_eris: NDArray[np.complex128]


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
        uhf_state.P_old_alpha = uhf_state.P_alpha.copy()
        uhf_state.P_old_beta = uhf_state.P_beta.copy()

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

    ResultClass = CSUHFResults(
        context=ctx,
        converged=uhf_state.converged,
        E_UHF=uhf_state.E_UHF,
        e_alpha=uhf_state.e_orb_alpha,
        e_beta=uhf_state.e_orb_beta,
        n_alpha=uhf_state.final_alpha_elec,
        n_beta=uhf_state.final_alpha_elec,
        det=(uhf_ext_ctx.det[0], uhf_ext_ctx.det[1]),
        X=uhf_ext_ctx.X,
        F_final_alph=uhf_state.F_next_alpha,
        F_final_beta=uhf_state.F_next_beta,
        P_guess_alpha=P_guess_alpha,
        P_guess_beta=P_guess_beta,
        P_alpha=uhf_state.P_alpha,
        P_beta=uhf_state.P_beta,
        P_total=uhf_state.P_total,
        P_diff=uhf_state.P_diff,
        C_alpha=uhf_state.C_munu_alpha,
        C_beta=uhf_state.C_munu_beta,
        S_diagnostics=S_diagnostics,
        error=max(uhf_state.error_alpha, uhf_state.error_beta),
        iterations=uhf_state.iteration,
        scaled_eris=uhf_ext_ctx.eri_scaled,
    )

    return ResultClass


def perform_spin_diagostics(
    ctx: CSUHFContext, uhf_state: CSUHFState
) -> UHFSpinDiagnostics:
    S_diagnostics: UHFSpinDiagnostics = calculate_s2_expectation(
        uhf_state.P_alpha, uhf_state.P_beta, ctx.S, ctx.verbose
    )

    uhf_state.final_alpha_elec = np.trace(uhf_state.P_alpha.real @ ctx.S)
    uhf_state.final_beta_elec = np.trace(uhf_state.P_beta.real @ ctx.S)

    if (
        abs(uhf_state.final_alpha_elec + uhf_state.final_beta_elec - ctx.n_electrons)
        > 1e-10
    ):
        raise RuntimeError("Number of electrons was not conserved in the calculation")

    return S_diagnostics


def conv_acc_criteria_met(
    ctx: CSUHFContext,
    uhf_ext_ctx: CSUHFConstants,
    uhf_state: CSUHFState,
) -> bool:
    use_conv_acc = False
    if (
        uhf_state.iteration == ctx.acc_iteration_start and uhf_ext_ctx.acc_requested
    ):  #  and error < conv_thresh and not use_conv:
        use_conv_acc = True
        if ctx.verbose:
            print("-" * 30, f"   STARTED {ctx.conv_type}  ", "-" * 30)

    return use_conv_acc


def update_uhf_density(
    uhf_ext_ctx: CSUHFConstants,
    uhf_state: CSUHFState,
) -> None:
    uhf_state.P_alpha, uhf_state.e_orb_alpha, uhf_state.C_munu_alpha, *_ = (
        calculate_P_next(
            uhf_state.F_next_alpha,
            uhf_ext_ctx.X,
            uhf_ext_ctx.alpha_elec,
            uhf_ext_ctx.det[0],
            mode="UHF",
        )
    )
    uhf_state.P_beta, uhf_state.e_orb_beta, uhf_state.C_munu_beta, *_ = (
        calculate_P_next(
            uhf_state.F_next_beta,
            uhf_ext_ctx.X,
            uhf_ext_ctx.beta_elec,
            uhf_ext_ctx.det[1],
            mode="UHF",
        )
    )

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
                uhf_state.residuals_alpha, uhf_state.F_guess_alpha
            )
            F_opt_beta, r_opt_beta = calc_diis_extrapolation(
                uhf_state.residuals_beta, uhf_state.F_guess_beta
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


def is_converged_uhf(
    ctx: CSRHFContext,
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
        error_re_alpha: float = float(np.max(np.abs(uhf_state.r_alpha.real)))
        error_im_alpha: float = float(np.max(np.abs(uhf_state.r_alpha.imag)))
        error_re_beta: float = float(np.max(np.abs(uhf_state.r_beta.real)))
        error_im_beta: float = float(np.max(np.abs(uhf_state.r_beta.imag)))

        error_alpha = max(error_re_alpha, error_im_alpha)
        error_beta = max(error_re_beta, error_im_beta)

        if (
            uhf_state.iteration > 1
            and np.max([error_alpha, error_beta]) < ctx.threshold
        ):
            converged = True

    elif ctx._convergence_criteria == "norm":
        error_alpha: float = np.linalg.norm(uhf_state.r_alpha)
        error_beta: float = np.linalg.norm(uhf_state.r_beta)

        error: float = max(error_alpha, error_beta)

        if uhf_state.iteration > 1 and error < ctx.threshold:
            converged = True

    if converged and ctx.verbose:
        print(f"Convergence achieved after {uhf_state.iteration} iterations.")

    return converged


def print_cycle_data(uhf_state: CSUHFState) -> None:
    print(
        f"{uhf_state.iteration:5}     {uhf_state.E_UHF:45.16f}     {uhf_state.E_diff:45.16f}     {uhf_state.error:8.4E}"
    )

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
            ctx.S,
            uhf_ext_ctx.H_core,
            uhf_ext_ctx.eri_scaled,
        )
    )

    uhf_state.error_alpha = float(np.linalg.norm(uhf_state.r_alpha.flatten()))
    uhf_state.error_beta = float(np.linalg.norm(uhf_state.r_beta.flatten()))

    uhf_state.error = max(uhf_state.error_alpha, uhf_state.error_beta)

    return


def enforce_multiplicity(ctx: CSUHFContext) -> None:
    if ctx.mult == -1:
        ctx.mult = 0 if ctx.n_electrons % 2 == 0 else 1

    return


def allocate_uhf_extended_context(ctx: CSUHFContext) -> CSUHFConstants:
    dim = len(ctx.S)
    X = np.zeros((dim, dim), dtype=np.complex128)
    det = np.zeros((2, dim), dtype=np.int32)
    eri_scaled = np.zeros((dim, dim, dim, dim), dtype=np.complex128)
    H_core = np.zeros((dim, dim), dtype=np.complex128)
    core_mask = np.zeros((dim, dim), dtype=np.bool)
    _eigensolver = ctx._eigensolver

    return CSUHFConstants(
        dim=dim,
        X=X,
        det=det,
        alpha_elec=0,
        beta_elec=0,
        eri_scaled=eri_scaled,
        H_core=H_core,
        core_mask=core_mask,
        _eigensolver=_eigensolver,
    )


def allocate_uhf_state(ctx: CSUHFContext) -> CSUHFState:
    iteration = 0
    P_alpha = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    P_beta = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    E_prev = np.complex128(0.0)
    use_conv_acc = False
    F_guess_alpha: List[NDArray[np.complex128]] = []
    F_guess_beta: List[NDArray[np.complex128]] = []
    residuals_alpha: List[NDArray[np.complex128]] = []
    residuals_beta: List[NDArray[np.complex128]] = []
    F_next_alpha = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    F_next_beta = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    error_alpha: complex = 1e10
    error_beta: complex = 1e10
    converged: bool = False
    C_munu_alpha = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    C_munu_beta = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    e_orb_alpha = np.zeros(len(ctx.S), dtype=np.complex128)
    e_orb_beta = np.zeros(len(ctx.S), dtype=np.complex128)
    C_prime_alpha = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    C_prime_beta = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    F_alpha = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    F_beta = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    r_alpha = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    r_beta = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    E_UHF = np.complex128(0.0)
    E_diff = np.complex128(0.0)
    P_old_alpha = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    P_old_beta = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    P_total = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    P_diff = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    final_alpha_elec = 0
    final_beta_elec = 0
    error = np.complex128(0)

    return CSUHFState(
        iteration=iteration,
        P_alpha=P_alpha,
        P_beta=P_beta,
        E_prev=E_prev,
        use_conv_acc=use_conv_acc,
        F_guess_alpha=F_guess_alpha,
        F_guess_beta=F_guess_beta,
        residuals_alpha=residuals_alpha,
        residuals_beta=residuals_beta,
        F_next_alpha=F_next_alpha,
        F_next_beta=F_next_beta,
        error_alpha=error_alpha,
        error_beta=error_beta,
        converged=converged,
        C_munu_alpha=C_munu_alpha,
        C_munu_beta=C_munu_beta,
        e_orb_alpha=e_orb_alpha,
        e_orb_beta=e_orb_beta,
        C_prime_alpha=C_prime_alpha,
        C_prime_beta=C_prime_beta,
        F_alpha=F_alpha,
        F_beta=F_beta,
        r_alpha=r_alpha,
        r_beta=r_beta,
        E_UHF=E_UHF,
        E_diff=E_diff,
        P_old_alpha=P_old_alpha,
        P_old_beta=P_old_beta,
        P_total=P_total,
        P_diff=P_diff,
        final_alpha_elec=final_alpha_elec,
        final_beta_elec=final_beta_elec,
        error=error,
    )


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
    uhf_ext_ctx.X = np.array(transformation_matrix(ctx.S), dtype=np.complex128)

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

    if ctx.theta != 0.0:
        compute_uhf_unscaled_density(ctx, ext_uhf_ctx, uhf_state)

    else:
        guess_density_UHF(ctx, ext_uhf_ctx, uhf_state)

    if ctx.break_symm:
        # note that breaking symmetry will only make sense when the guess is not zeros
        uhf_state.P_beta[: ctx.n_electrons, : ctx.n_electrons] = 0

    return


def compute_uhf_unscaled_density(
    ctx: CSUHFContext,
    ext_uhf_ctx: CSUHFConstants,
    uhf_state: CSUHFState,
) -> Tuple[NDArray[np.complex128], float]:
    if ctx.verbose:
        print("Converging unscaled UHF case:")

    unscaled_ctx = CSUHFContext(
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

    unscaled_res = CS_UHF(unscaled_ctx)

    if ctx.verbose:
        print("Unscaled energy: ", unscaled_res.E_UHF)
        print("\n\n\nConverging scaled case from unscaled density as reference:")

    P = unscaled_res.P_alpha
    return P, unscaled_res.E_UHF


def validate_uhf_context_input(ctx: CSUHFContext) -> None:
    if not (len(ctx.T) == len(ctx.V) == len(ctx.S)):
        raise ValueError(
            f"Matrices T, V, S must have the same dimensions. Got N_S={len(ctx.S)}, N_T={len(ctx.T)}, N_V={len(ctx.V)}"
        )

    if ctx.conv_type not in (None, "DIIS", "CROP"):
        raise ValueError("Convergence assist must be either None, 'DIIS', or 'CROP'")

    if ((ctx.n_electrons - ctx.mult) % 2) == 1:
        raise ValueError(
            f"It is not possible to have {ctx.mult} unpaired electrons with {ctx.n_electrons} electrons."
        )


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

    else:
        p_final = guess_density_RHF(ctx.p_guess, ext_uhf_ctx.dim, ctx.initial_orbitals)
        uhf_state.E_prev = np.complex128(0.0)

    uhf_state.P_alpha = p_final
    uhf_state.P_beta = p_final

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
