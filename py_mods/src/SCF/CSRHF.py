from typing import List
from matplotlib.pylab import int32
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, Union
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
    calculate_P_next_2,
    sign_convention,
)
import matplotlib.pyplot as plt
from dataclasses import dataclass


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
    guess_MAX_ITER : int or None
        Iterations for preliminary RHF guess (if applicable).
    INPORB : NDArray or None
        Imported orbitals for guess.
    verbose : bool
        If True, print progress.
    conv_type : {None, 'DIIS', 'CROP'}
        Convergence algorithm.
    conv_MEM : int
        History size for convergence acceleration.
    conv_ITER_START : int
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
    guess_MAX_ITER: Union[int, None] = None
    INPORB: Union[NDArray[np.float64], NDArray[np.complex128], None] = None
    verbose: bool = False
    conv_type: Literal[None, "DIIS", "CROP"] = "DIIS"
    conv_MEM: int = 4
    conv_ITER_START: int = 4


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
    # unpacking
    (
        S,
        T,
        V,
        eri,
        n_electrons,
        theta,
        verbose,
        conv_ITER_START,
        conv_MEM,
        conv_type,
        occupation,
        max_iter,
        threshold,
        p_guess,
        INPORB,
    ) = unpack_ctx_class(ctx)

    assert len(T) == len(V) == len(S), "Matrices T, V, S must have the same dimensions"
    assert n_electrons % 2 == 0, "RHF can only be closed-shell systems"

    # Convergence acceleration setup
    conv_ITER_START, conv_REQUESTED = setup_conv_acc(
        conv_MEM, conv_type, conv_ITER_START
    )

    # Transform & Validate inputed matrices & determinant
    dim, X, det, eri_scaled, H_core, core_mask = setup_mat_and_occ(
        S, T, V, eri, n_electrons, theta, occupation
    )

    # Guess density and initialize E0
    P, E_prev = initialize_P_and_E(ctx, theta, verbose, p_guess, INPORB, dim)

    # Initialize variables
    use_conv_acc, F_guess, residuals, F_next, error, converged, C_munu, e_orb = (
        initialize_scf_variables(dim, H_core)
    )

    if verbose:
        print_table_header()

    for iter_idx in range(max_iter):
        # Calculate next Fock matrix, associated error, and RHF energy
        F, r = calculate_F_and_r_comp(P, S, H_core, eri_scaled)
        E_RHF = E_0_comp(P, H_core, F)

        E_diff = E_RHF - E_prev
        E_prev = E_RHF

        if verbose:
            print_cycle_data(iter_idx, r, E_RHF, E_diff)

        # Check convergence
        converged = check_conv(verbose, threshold, iter_idx, r)
        if converged:
            break

        # History storage
        update_CONV_MEM(conv_MEM, F_guess, residuals, F, r)
        P_old = P.copy()

        # Update F_next with or without convergence acceleration
        F_next = update_F_matrix(
            verbose, conv_type, use_conv_acc, F_guess, residuals, F
        )

        # Compute next Density
        P, e_orb, C_munu, C_prime = update_density(
            n_electrons, X, det, core_mask, F_next
        )

        # Enforce real if theta=0
        if theta == 0.0:
            P, C_munu = clear_imaginary(theta, P, C_munu)

        # Check activation of convergence acceleration
        use_conv_acc = conv_acc_criteria_met(
            verbose, conv_ITER_START, conv_type, conv_REQUESTED, iter_idx
        )

    # Final update diagonalization
    P, e_orb, C_munu, C_prime = update_density(n_electrons, X, det, core_mask, F_next)

    F_next = F

    return CS_RHF_ResultsClass(
        context=ctx,
        converged=converged,
        E_RHF=E_RHF,
        e_orb=e_orb,
        n_elec=int(n_electrons),
        det=det,
        H_core=H_core,
        X=X,
        F_final=F_next,
        C_prime=C_prime,
        P_guess=P_old if iter_idx > 0 else P,
        P=P,
        C_munu=C_munu,
        error=float(error),
        iterations=iter_idx,
    )


def check_conv(verbose, threshold, iter_idx, r):
    converged = False
    error_re, error_im = np.max(np.abs(r.real)), np.max(np.abs(r.imag))
    if iter_idx > 1 and error_re < threshold and error_im < threshold:
        # if iter_idx > 1 and abs(E_diff) < threshold:
        converged = True
        if verbose:
            print(f"Convergence achieved after {iter_idx} iterations.")
    return converged


def initialize_P_and_E(ctx, theta, verbose, p_guess, INPORB, dim):
    if theta != 0.0:
        P, unscaled_E_RHF = compute_unscaled_density(ctx, verbose)
        E_prev = np.complex128(unscaled_E_RHF)
    else:
        P = guess_density_RHF(p_guess, dim, INPORB)
        E_prev: np.complex128 = np.complex128(0.0)
    return P, E_prev


def print_cycle_data(iter_idx, r, E_RHF, E_diff):
    error_re, error_im = np.max(np.abs(r.real)), np.max(np.abs(r.imag))
    print(
        f"{iter_idx:5}     {E_RHF:24.6E}     {E_diff:24.6E}     {error_re:8.4E}     {error_im:8.4E}j"
    )


def update_density(n_electrons, X, det, core_mask, F_next):
    P, e_orb, C_munu, *_, C_prime = calculate_P_next(F_next, X, n_electrons, det)

    # P, e_orb, C_munu = calculate_P_next_2(
    #     F_next, S, n_electrons, det,
    # )

    P = P * core_mask
    C_munu = sign_convention(C_munu)
    return P, e_orb, C_munu, C_prime


def initialize_scf_variables(dim, H_core):
    use_conv_acc: bool = False
    converged: bool = False
    F_guess: List[NDArray[np.complex128]] = []
    residuals: List[NDArray[np.complex128]] = []
    F_next: NDArray[np.complex128] = np.zeros_like(H_core)
    e_orb: NDArray[np.complex128] = np.zeros(dim, dtype=np.complex128)
    C_prime: NDArray[np.complex128] = np.zeros((dim, dim), dtype=np.complex128)
    C_munu: NDArray[np.complex128] = np.zeros_like(C_prime, dtype=np.complex128)
    error: complex = np.inf
    converged: bool = False

    return use_conv_acc, F_guess, residuals, F_next, error, converged, C_munu, e_orb


def unpack_ctx_class(ctx):
    S = ctx.S.astype(np.complex128)
    T = ctx.T.astype(np.complex128)
    V = ctx.V.astype(np.complex128)
    eri = ctx.eri.astype(np.complex128)
    n_electrons = ctx.n_electrons
    theta = ctx.theta
    verbose = ctx.verbose
    conv_ITER_START = ctx.conv_ITER_START
    conv_MEM = ctx.conv_MEM
    conv_type = ctx.conv_type
    occupation = ctx.occupation
    n_occ = n_electrons // 2
    max_iter = ctx.max_iter
    threshold = ctx.threshold
    p_guess = ctx.p_guess
    guess_MAX_ITER = ctx.guess_MAX_ITER
    INPORB = ctx.INPORB
    return (
        S,
        T,
        V,
        eri,
        n_electrons,
        theta,
        verbose,
        conv_ITER_START,
        conv_MEM,
        conv_type,
        occupation,
        max_iter,
        threshold,
        p_guess,
        INPORB,
    )


def setup_mat_and_occ(S, T, V, eri, n_electrons, theta, occupation):
    dim = len(S)
    X = transformation_matrix(S).astype(np.complex128)
    det, _ = validate_determinant(n_electrons, occupation, dim)

    T_scaled, V_scaled, eri_scaled = scale_integrals(T, V, eri, theta)
    H_core = T_scaled + V_scaled
    core_mask = np.abs(H_core) > 1e-10
    return dim, X, det, eri_scaled, H_core, core_mask


def setup_conv_acc(conv_MEM, conv_type, conv_ITER_START):
    conv_REQUESTED = conv_type is not None
    conv_ITER_START = (
        min(conv_ITER_START + 1, conv_MEM)
        if conv_MEM >= conv_ITER_START
        else max(conv_ITER_START + 1, conv_MEM)
    )

    return conv_ITER_START, conv_REQUESTED


def compute_unscaled_density(ctx, verbose: bool):
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
        guess_MAX_ITER=ctx.guess_MAX_ITER,
        INPORB=ctx.INPORB,
        verbose=ctx.verbose,
        conv_type=ctx.conv_type,
        conv_MEM=ctx.conv_MEM,
        conv_ITER_START=10,
    )

    unscaled_res = CS_RHF(unscaled_ctx)

    if verbose:
        print("Unscaled energy: ", unscaled_res.E_RHF)
        print("\n\n\nConverging scaled case from unscaled density as reference:")

    P = unscaled_res.P
    return P, unscaled_res.E_RHF


def print_table_header():
    print("-" * 128)
    print(
        "|   Iter     |                   E_iter                      |                   Delta_e                   |      norm(e_i)      |"
    )
    print("-" * 128)


def clear_imaginary(theta, P, C_munu):
    if theta == 0.0:
        P = P.real.astype(np.complex128)
        C_munu = C_munu.real.astype(np.complex128)
    return P, C_munu


def conv_acc_criteria_met(
    verbose, conv_ITER_START, conv_type, conv_REQUESTED, iter_idx
):
    use_conv_acc = False
    if iter_idx == conv_ITER_START and conv_REQUESTED:
        use_conv_acc = True
        if verbose:
            print("-" * 30, f"   STARTED {conv_type}  ", "-" * 30)
    return use_conv_acc


def update_CONV_MEM(conv_MEM, F_guess, residuals, F, r):
    F_guess.append(F)
    residuals.append(r)

    if len(F_guess) > conv_MEM:
        F_guess.pop(0)
        residuals.pop(0)


def update_F_matrix(verbose, conv_type, use_conv_acc, F_guess, residuals, F):
    if not use_conv_acc:
        F_next = F
    else:
        try:
            F_opt, r_opt = calc_diis_extrapolation(residuals, F_guess)
            F_next = F_opt

            if conv_type == "CROP":
                F_guess[-1] = F_opt
                residuals[-1] = r_opt
        except np.linalg.LinAlgError:
            if verbose:
                print(
                    "!!!!!!!!!!!!!!!! CONVERGENCE ACCELERATION CAUSED A SINGULAR MATRIX. REVERTING TO STANDARD SCF !!!!!!!!!!!!!!!"
                )
            use_conv_acc = False
            F_next = F
    return F_next


def calculate_F_and_r_comp(
    P: NDArray[np.complex128],
    S: NDArray[np.complex128],
    H_core: NDArray[np.complex128],
    eri: NDArray[np.complex128],
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
    F = H_core + calc_g_matrix_comp(P, eri)
    r = calc_residual_commutator(F, P, S)
    return F, r


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
