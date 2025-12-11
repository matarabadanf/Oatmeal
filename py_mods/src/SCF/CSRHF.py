from matplotlib.pylab import int32
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, Union
from py_mods.src.SCF.scf_utils import (
    diagonalize_biorthogonal,
    transformation_matrix,
    calc_g_matrix_comp,
    E_0_comp,
    guess_density_RHF,
    validate_determinant,
    scale_integrals,
    calc_residual_commutator,
    calc_diis_extrapolation,
    calculate_P_next,
    canonicalize,
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
    conv_MEM: int = 8
    conv_ITER_START: int = 12


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
    R_munu : NDArray[np.complex128]
        Right MO coefficients.
    L_munu : NDArray[np.complex128]
        Left MO coefficients.
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
    X: NDArray[np.complex128]
    F_final: NDArray[np.complex128]
    C_prime: NDArray[np.complex128]
    P_guess: NDArray[np.complex128]
    P: NDArray[np.complex128]
    R_munu: NDArray[np.complex128]
    L_munu: NDArray[np.complex128]
    error: float
    iterations: int


def CS_RHF(ctx: CS_RHF_ContextClass) -> CS_RHF_ResultsClass:
    """
    Perform Complex Scaled RHF calculation.

    Parameters
    ----------
    ctx : CS_RHF_ContextClass
        Calculation parameters and integrals.

    Returns
    -------
    CS_RHF_ResultsClass
        Calculation results.
    """
    # unpacking
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

    assert len(T) == len(V) == len(S), "Matrices T, V, S must have the same dimensions"
    assert n_electrons % 2 == 0, "RHF can only be closed-shell systems"

    # setup
    conv_REQUESTED = True if conv_type is not None else False
    conv_ITER_START = (
        min(conv_ITER_START + 1, conv_MEM)
        if conv_MEM >= conv_ITER_START
        else max(conv_ITER_START + 1, conv_MEM)
    )

    # Transform & Validate
    dim = len(S)
    X = transformation_matrix(S).astype(np.complex128)
    det, _ = validate_determinant(n_electrons, occupation, dim)

    # Scaling
    T_scaled, V_scaled, eri_scaled = scale_integrals(T, V, eri, theta)
    H_core = T_scaled + V_scaled

    # Guess
    P = guess_density_RHF(p_guess, dim, INPORB)

    # State variables
    E_prev = 0.0 + 0.0j
    use_conv = False
    converged = False
    F_guess = []
    residuals = []

    # Placeholders for results
    F_next = np.zeros_like(H_core)
    e_orb = np.zeros(dim, dtype=np.complex128)
    C_prime = np.zeros((dim, dim), dtype=np.complex128)
    L_munu = np.zeros_like(C_prime)
    R_munu = np.zeros_like(C_prime)

    iter_idx = 0

    if verbose:
        print("-" * 128)
        print(
            "|   Iter     |                   E_iter                      |                   Delta_e                   |      norm(e_i)      |"
        )
        print("-" * 128)

    for iter_idx in range(max_iter):
        F, r = calculate_F_and_r_comp(P, S, H_core, eri_scaled)

        error = np.linalg.norm(r.ravel())
        E_RHF = E_0_comp(P, H_core, F)
        E_diff = E_RHF - E_prev

        if verbose:
            print(
                f"{iter_idx:5}     {E_RHF:45.16f}     {E_diff:45.16f}     {error:8.4E}"
            )

        # Check convergence
        if iter_idx > 1 and error < threshold:
            converged = True
            if verbose:
                print(f"Convergence achieved after {iter_idx} iterations.")

            # Final diagonalization
            P, e_orb, R_munu, *_, C_prime = calculate_P_next(
                F_next, X, n_electrons, det
            )

            F_next = F
            break

        # History storage
        F_guess.append(F)
        residuals.append(r)

        if len(F_guess) > conv_MEM:
            F_guess.pop(0)
            residuals.pop(0)

        # Update F_next
        if not use_conv:
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
                use_conv = False
                F_next = F

        # Compute next Density
        P_old = P.copy()
        P, e_orb, R_munu, *_, C_prime = calculate_P_next(F_next, X, n_electrons, det)

        # Stability Patch: Enforce real if theta=0
        if theta == 0.0:
            P = P.real.astype(np.complex128)
            L_munu = L_munu.real.astype(np.complex128)
            R_munu = R_munu.real.astype(np.complex128)

        E_prev = E_RHF

        # Activate DIIS
        if iter_idx == conv_ITER_START and conv_REQUESTED:
            use_conv = True
            if verbose:
                print("-" * 30, f"   STARTED {conv_type}  ", "-" * 30)


    R_munu = sign_convention(R_munu)

    # R_munu, e_orb = canonicalize(R_munu, F_next , n_occ)


    return CS_RHF_ResultsClass(
        context=ctx,
        converged=converged,
        E_RHF=E_RHF,
        e_orb=e_orb,
        n_elec=int(n_electrons),
        det=det,
        X=X,
        F_final=F_next,
        C_prime=C_prime,
        P_guess=P_old if iter_idx > 0 else P,
        P=P,
        R_munu=R_munu,
        L_munu=R_munu.T,
        error=float(error),
        iterations=iter_idx,
    )


def calculate_F_and_r_comp(
    P: NDArray[np.complex128],
    S: NDArray[np.complex128],
    H_core: NDArray[np.complex128],
    eri: NDArray[np.complex128],
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Calculate Fock matrix and Residual.

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
        Sampled angles and energies.
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
