from matplotlib.pylab import int32
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, Union, List
import matplotlib.pyplot as plt
from dataclasses import dataclass
from py_mods.src.SCF.scf_utils import (
    transformation_matrix,
    calc_g_matrix_comp,
    E_0_comp,
    guess_density_RHF,
    validate_determinant,
    scale_integrals,
    calc_residual_commutator,
    calc_diis_extrapolation,
    calculate_P_next,
    sign_convention,
)


@dataclass
class CS_RHF_ContextClass:
    """
    Context for CS_RHF calculations.

    Attributes
    ----------
    S : NDArray[np.float64]
        Overlap matrix.
    T : NDArray[np.float64]
        Kinetic energy matrix.
    V : NDArray[np.float64]
        Nuclear attraction matrix.
    eri : NDArray[np.float64]
        Electron repulsion integrals.
    n_electrons : int
        Total electron count (must be even).
    theta : float
        Complex-scaling angle (radians).
    occupation : int or NDArray[np.int32] or None
        Occupation vector. If -1/None, build default.
    max_iter : int
        Maximum SCF iterations.
    threshold : float
        Convergence threshold.
    p_guess : {'core', 'ones', 'IMPORB'}
        Initial density guess type.
    guess_max_iter : int or None
        Iterations for preliminary RHF guess (if applicable).
    initial_orbitals : NDArray or None
        Imported orbitals for guess.
    verbose : bool
        If True, print progress.
    conv_type : {None, 'DIIS', 'CROP'}
        Convergence algorithm.
    acc_hist_size : int
        History size for convergence acceleration.
    acc_iteration_start : int
        Iteration to start acceleration.
    """

    # Required
    S: NDArray[np.float64]
    T: NDArray[np.float64]
    V: NDArray[np.float64]
    eri: NDArray[np.float64]
    n_electrons: int

    # Optional
    theta: float = 0.0
    occupation: Union[int, NDArray[np.int32], None] = None
    max_iter: int = 100
    threshold: float = 1e-12
    p_guess: Literal["core", "ones", "IMPORB"] = "core"
    guess_max_iter: Union[int, None] = None
    initial_orbitals: Union[NDArray[np.float64], NDArray[np.complex128], None] = None
    verbose: bool = False
    conv_type: Literal[None, "DIIS", "CROP"] = "DIIS"
    acc_hist_size: int = 10
    acc_iteration_start: int = 12

    # Internal
    _eigensolver: Literal["eig", "eigh"] = "eigh"
    _convergence_criteria: Literal["norm", "max"] = "max"


@dataclass
class CSRHFConstants:
    dim: int
    X: NDArray[np.complex128]
    det: NDArray[np.int32]
    eri_scaled: NDArray[np.complex128]
    H_core: NDArray[np.complex128]
    core_mask: NDArray[np.bool]
    _eigensolver: Literal["eig", "eigh"]
    acc_iteration_start: int = 0
    acc_requested: bool = False


@dataclass
class CSRHFState:
    iteration: int
    P: NDArray[np.complex128]
    E_prev: np.complex128
    use_conv_acc: bool
    F_guess: List[NDArray[np.complex128]]
    residuals: List[NDArray[np.complex128]]
    F_next: NDArray[np.complex128]
    error: complex
    converged: bool
    C_munu: NDArray[np.complex128]
    C_prime: NDArray[np.complex128]
    e_orb: NDArray[np.complex128]
    F: NDArray[np.complex128]
    r: NDArray[np.complex128]
    E_RHF: np.complex128
    E_diff: np.complex128
    P_old: NDArray[np.complex128]


@dataclass
class CS_RHF_ResultsClass:
    """
    Results for CS_RHF calculations.

    Attributes
    ----------
    context : CS_RHF_ContextClass
        Input context.
    converged : bool
        Convergence status.
    E_RHF : complex
        Final RHF energy.
    e_orb : NDArray[np.complex128]
        Orbital energies.
    n_elec : float
        Calculated electron count.
    X : NDArray[np.complex128]
        Transformation matrix.
    F_final : NDArray[np.complex128]
        Final Fock matrix.
    C_prime : NDArray[np.complex128]
        Transformed eigenvectors.
    P_guess : NDArray[np.complex128]
        Initial density guess.
    P : NDArray[np.complex128]
        Final LR density matrix.
    C_munu : NDArray[np.complex128]
        Final MO coefficients.
    error : float
        Final residual norm.
    iterations : int
        Total iterations performed.
    """

    context: CS_RHF_ContextClass
    converged: bool
    E_RHF: np.complex128
    e_orb: NDArray[np.complex128]
    n_elec: int32
    det: NDArray[np.int32]
    H_core: NDArray[np.complex128]
    X: NDArray[np.complex128]
    F_final: NDArray[np.complex128]
    C_prime: NDArray[np.complex128]
    P_guess: NDArray[np.complex128]
    P: NDArray[np.complex128]
    C_munu: NDArray[np.complex128]
    error: float
    iterations: int
    scaled_eris: NDArray[np.complex128]


def CS_RHF(ctx: CS_RHF_ContextClass) -> CS_RHF_ResultsClass:
    """
    Perform Complex Scaled RHF calculation.

    Parameters
    ----------
    ctx : CS_RHF_ContextClass
        Calculation parameters & integrals.

    Returns
    -------
    CS_RHF_ResultsClass
        Calculation results.
    """
    validate_context_input(ctx)

    # Allocate extended context and RHF state
    ext_ctx = allocate_extended_context(ctx)
    rhf_state = allocate_rhf_state(ctx)

    # Transform & Validate inputed matrices & determinant
    setup_extended_context(ctx, ext_ctx)

    # Guess density and initialize E0
    initialize_P_and_E(ctx, rhf_state)
    initialize_state_variables(ext_ctx, rhf_state)

    if ctx.verbose:
        print_table_header()

    for iter_idx in range(ctx.max_iter):
        rhf_state.iteration += 1
        # Calculate next Fock matrix, associated error, and RHF energy
        update_F_and_r_comp(ctx, ext_ctx, rhf_state)
        update_RHF_energy(ext_ctx, rhf_state)

        if ctx.verbose:
            print_cycle_data(ctx._convergence_criteria, rhf_state)

        # Check convergence
        rhf_state.converged = is_converged(ctx, rhf_state)
        if rhf_state.converged:
            break

        # History storage
        update_acc_hist_size(ctx, rhf_state)
        rhf_state.P_old = rhf_state.P.copy()

        # Update F_next with or without convergence acceleration
        update_F_matrix(ctx, rhf_state)

        # Compute next Density
        update_density(ctx, ext_ctx, rhf_state)

        # Enforce real if theta=0
        if ctx.theta == 0.0:
            rhf_state.P.imag = rhf_state.C_munu.imag = 0

        # Check activation of convergence acceleration
        rhf_state.use_conv_acc = conv_acc_criteria_met(ctx, ext_ctx, rhf_state)

    # Final update diagonalization
    update_density(ctx, ext_ctx, rhf_state)

    rhf_state.F_next = rhf_state.F

    return CS_RHF_ResultsClass(
        context=ctx,
        converged=rhf_state.converged,
        E_RHF=rhf_state.E_RHF,
        e_orb=rhf_state.e_orb,
        n_elec=np.int32(ctx.n_electrons),
        det=ext_ctx.det,
        H_core=ext_ctx.H_core,
        X=ext_ctx.X,
        F_final=rhf_state.F_next,
        C_prime=rhf_state.C_prime,
        P_guess=rhf_state.P_old if iter_idx > 0 else rhf_state.P,
        P=rhf_state.P,
        C_munu=rhf_state.C_munu,
        error=rhf_state.r,
        iterations=iter_idx,
        scaled_eris=ext_ctx.eri_scaled,
    )


def update_RHF_energy(
    ext_ctx: CSRHFConstants,
    rhf_state: CSRHFState,
) -> None:
    rhf_state.E_RHF = E_0_comp(rhf_state.P, ext_ctx.H_core, rhf_state.F)
    rhf_state.E_diff = rhf_state.E_RHF - rhf_state.E_prev
    rhf_state.E_prev = rhf_state.E_RHF


def allocate_extended_context(ctx: CS_RHF_ContextClass) -> CSRHFConstants:

    dim = len(ctx.S)
    X = np.zeros((dim, dim), dtype=np.complex128)
    det = np.zeros(dim, dtype=np.int32)
    eri_scaled = np.zeros((dim, dim, dim, dim), dtype=np.complex128)
    H_core = np.zeros((dim, dim), dtype=np.complex128)
    core_mask = np.zeros((dim, dim), dtype=np.bool)
    _eigensolver = ctx._eigensolver

    return CSRHFConstants(
        dim=dim,
        X=X,
        det=det,
        eri_scaled=eri_scaled,
        H_core=H_core,
        core_mask=core_mask,
        _eigensolver=_eigensolver,
    )


def allocate_rhf_state(ctx: CS_RHF_ContextClass) -> CSRHFState:
    iteration = 0
    P = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    E_prev = np.complex128(0.0)
    use_conv_acc = False
    F_guess: List[NDArray[np.complex128]] = []
    residuals: List[NDArray[np.complex128]] = []
    F_next = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    error: complex = 1e10
    converged: bool = False
    C_munu = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    e_orb = np.zeros(len(ctx.S), dtype=np.complex128)
    C_prime = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    F = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    r = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    E_RHF = np.complex128(0.0)
    E_diff = np.complex128(0.0)
    P_old = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)

    return CSRHFState(
        iteration=iteration,
        P=P,
        E_prev=E_prev,
        use_conv_acc=use_conv_acc,
        F_guess=F_guess,
        residuals=residuals,
        F_next=F_next,
        error=error,
        converged=converged,
        C_munu=C_munu,
        e_orb=e_orb,
        C_prime=C_prime,
        F=F,
        r=r,
        E_RHF=E_RHF,
        E_diff=E_diff,
        P_old=P_old,
    )


def validate_context_input(ctx):
    if not len(ctx.T) == len(ctx.V) == len(ctx.S):
        raise ValueError(
            f"Matrices T, V, S must have the same dimensions. Got N_S={len(ctx.S)}, N_T={len(ctx.T)}, N_V={len(ctx.V)}"
        )

    if ctx.n_electrons % 2 != 0:
        raise ValueError("RHF can only be closed-shell systems")

    if ctx.conv_type not in (None, "DIIS", "CROP"):
        raise ValueError("Convergence assist must be either None, 'DIIS', or 'CROP'")

    if ctx._eigensolver not in [
        "eig",
        "eigh",
        "genh",
    ]:
        raise ValueError(
            f"Eigensolver must be either 'eig', 'eigh' or 'genh'. Got {ctx._eigensolver}"
        )


def is_converged(
    ctx: CS_RHF_ContextClass,
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
        error: float = np.linalg.norm(rhf_state.r)
        if rhf_state.iteration > 1 and error < ctx.threshold:
            converged = True

    if converged and ctx.verbose:
        print(f"Convergence achieved after {rhf_state.iteration} iterations.")

    return converged


def initialize_rhf_state(
    ctx: CS_RHF_ContextClass,
    rhf_state: CSRHFState,
) -> None:
    """
    Initialize RHF state with transformed matrices and determinant.

    Parameters
    ----------
    ctx : CS_RHF_ContextClass
        Calculation parameters & integrals.
    ext_ctx : CSRHFConstants
        Extended context with transformed matrices.
    CSRHFState : CSRHFState
        RHF state to initialize.

    Returns
    -------
    None
    """
    dim = len(ctx.S)
    rhf_state.P = np.zeros((dim, dim), dtype=np.complex128)
    rhf_state.E_prev = np.complex128(0.0)
    rhf_state.use_conv_acc = False
    rhf_state.F_guess = []
    rhf_state.residuals = []
    rhf_state.F_next = np.zeros((dim, dim), dtype=np.complex128)
    rhf_state.error = 1e10
    rhf_state.converged = False
    rhf_state.C_munu = np.zeros((dim, dim), dtype=np.complex128)
    rhf_state.e_orb = np.zeros(dim, dtype=np.complex128)
    return


def initialize_P_and_E(
    ctx: CS_RHF_ContextClass,
    rhf_state: CSRHFState,
) -> None:

    if ctx.theta != 0.0:
        P, unscaled_E_RHF = compute_unscaled_density(ctx, ctx.verbose)
        E_prev = np.complex128(unscaled_E_RHF)

    else:
        P = guess_density_RHF(ctx.p_guess, len(ctx.S), ctx.initial_orbitals)
        E_prev = np.complex128(0.0)

    rhf_state.P = P
    rhf_state.E_prev = E_prev

    return


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
            f"  {rhf_state.iter_idx:5}     {rhf_state.E_RHF:24.6E}     {rhf_state.E_diff:24.6E}     {error:8.4E}"
        )
        return

    elif _convergence_criteria == "max":
        error_re, error_im = np.max(np.abs(rhf_state.r.real)), np.max(
            np.abs(rhf_state.r.imag)
        )
        print(
            f"{rhf_state.iter_idx:5}     {rhf_state.E_RHF:24.6E}     {rhf_state.E_diff:24.6E}     {error_re:8.4E}     {error_im:8.4E}j"
        )
        return


def update_density(
    ctx: CS_RHF_ContextClass,
    ext_ctx: CSRHFConstants,
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

    (
        rhf_state.P,
        rhf_state.e_orb,
        rhf_state.C_munu,
        *_,
        rhf_state.C_prime,
    ) = calculate_P_next(rhf_state.F_next, ext_ctx.X, ctx.n_electrons, ext_ctx.det)

    # P, e_orb, C_munu = calculate_P_next_2(
    #     F_next, S, n_electrons, det,
    # )

    rhf_state.P = rhf_state.P * ext_ctx.core_mask
    rhf_state.C_munu = sign_convention(rhf_state.C_munu)
    return


def initialize_state_variables(ext_ctx: CSRHFConstants, rhf_state: CSRHFState) -> None:
    rhf_state.use_conv_acc = False
    rhf_state.converged = False
    rhf_state.F_guess = []
    rhf_state.residuals = []
    rhf_state.F_next = np.zeros_like(ext_ctx.H_core)
    rhf_state.e_orb = np.zeros(ext_ctx.dim, dtype=np.complex128)
    rhf_state.C_prime = np.zeros((ext_ctx.dim, ext_ctx.dim), dtype=np.complex128)
    rhf_state.C_munu = np.zeros_like(rhf_state.C_prime, dtype=np.complex128)
    rhf_state.error = np.complex128(1e10)

    return


def setup_extended_context(ctx: CS_RHF_ContextClass, ext_ctx: CSRHFConstants) -> None:
    """
    Setup extended context with transformed matrices and scaled integrals.

    Parameters
    ----------
    ctx : CS_RHF_ContextClass
        Original context with integrals and parameters.
    ext_ctx : CSRHFConstants
        Initialized extended context to compute.

    Returns
    -------
    None
    """

    ext_ctx.dim = len(ctx.S)
    ext_ctx.X = transformation_matrix(ctx.S.astype(np.complex128)).astype(np.complex128)
    ext_ctx.det, _ = validate_determinant(ctx.n_electrons, ctx.occupation, ext_ctx.dim)

    T_scaled, V_scaled, ext_ctx.eri_scaled = scale_integrals(
        ctx.T, ctx.V, ctx.eri, ctx.theta
    )
    ext_ctx.H_core = T_scaled + V_scaled
    ext_ctx.core_mask = np.abs(ext_ctx.H_core) > 1e-10

    if ctx.theta != 0:
        ext_ctx._eigensolver = "eig"

        # Convergence acceleration setup
    ext_ctx.acc_iteration_start, ext_ctx.acc_requested = setup_conv_acc(
        ctx.acc_hist_size, ctx.conv_type, ctx.acc_iteration_start
    )

    return


def setup_conv_acc(
    acc_hist_size: int,
    conv_type: Literal[None, "DIIS", "CROP"],
    acc_iteration_start: int,
) -> Tuple[int, bool]:
    """
    Setup convergence acceleration parameters.

    Parameters
    ----------
    """
    if conv_type not in [None, "DIIS", "CROP"]:
        print(
            "Convergence assist must be either None, 'DIIS', or 'CROP'. Reverted to no convergence acceleration"
        )
        return int(1e10), False

    acc_requested = conv_type is not None
    acc_iteration_start = (
        min(acc_iteration_start + 1, acc_hist_size)
        if acc_hist_size >= acc_iteration_start
        else max(acc_iteration_start + 1, acc_hist_size)
    )

    return (acc_iteration_start, acc_requested)


def compute_unscaled_density(
    ctx: CS_RHF_ContextClass, verbose: bool
) -> Tuple[NDArray[np.complex128], np.complex128]:
    """
    Compute unscaled density matrix for theta=0.

    Parameters
    ----------
    ctx : CS_RHF_ContextClass
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
    unscaled_ctx = CS_RHF_ContextClass(
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
    ctx: CS_RHF_ContextClass,
    ext_ctx: CSRHFConstants,
    rhf_state: CSRHFState = None,
) -> bool:
    use_conv_acc = False
    if rhf_state.iteration - 1 == ext_ctx.acc_iteration_start and ext_ctx.acc_requested:
        use_conv_acc = True
        if ctx.verbose:
            print("-" * 30, f"   STARTED {ctx.conv_type}  ", "-" * 30)
    return use_conv_acc


def update_acc_hist_size(cxt: CS_RHF_ContextClass, rhf_state: CSRHFState) -> None:
    rhf_state.F_guess.append(rhf_state.F)
    rhf_state.residuals.append(rhf_state.r)

    if len(rhf_state.F_guess) > cxt.acc_hist_size:
        rhf_state.F_guess.pop(0)
        rhf_state.residuals.pop(0)

    return


def update_F_matrix(
    ctx: CS_RHF_ContextClass,
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


def update_F_and_r_comp(
    ctx: CS_RHF_ContextClass,
    ext_ctx: CSRHFConstants,
    rhf_state: CSRHFState,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
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
    rhf_state.F = ext_ctx.H_core + calc_g_matrix_comp(rhf_state.P, ext_ctx.eri_scaled)
    rhf_state.r = calc_residual_commutator(rhf_state.F, rhf_state.P, ctx.S)
    return


def RHF_theta_traj(max_theta, n_points, cxt: CS_RHF_ContextClass):
    """
    Sample energies along theta trajectory.

    Parameters
    ----------
    max_theta : float
        Max theta (radians).
    n_points : int
        Steps.
    cxt : CS_RHF_ContextClass
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


def plot_theta_traj(energies):
    """
    Plot complex energy trajectory.

    Parameters
    ----------
    energies : sequence
        Complex energies.
    """
    reals = [energy.real for energy in energies]
    imags = [energy.imag for energy in energies]
    plt.plot(reals, imags, marker="o")
    plt.xlabel("Re(E)")
    plt.ylabel("Im(E)")
    plt.title("Complex Scaled RHF Energy vs Theta")
    plt.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    plt.ticklabel_format(style="sci")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_theta_orbital_energies(energies, theta, xrange=[0, 0]):
    """
    Scatter plot orbital energies.

    Parameters
    ----------
    energies : sequence
        Orbital energies.
    theta : float
        Current angle.
    xrange : list
        X-axis limits.
    """
    reals = [energy.real for energy in energies]
    imags = [energy.imag for energy in energies]
    if xrange != [0, 0]:
        plt.xlim(xrange)
        reals = [re for re in reals if re < xrange[1]]
        imags = imags[0 : len(reals)]

    plt.scatter(reals, imags, marker="o")
    plt.xlabel("Re(Orbital Energies)")
    plt.ylabel("Im(Orbital Energies)")
    plt.ticklabel_format(style="sci")
    plt.title(f"Complex Scaled RHF Orbital Energies at Theta={theta}")
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    pass
