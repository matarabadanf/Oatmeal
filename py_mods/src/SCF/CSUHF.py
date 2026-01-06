import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, Union
from py_mods.src.SCF.CSRHF import CS_RHF, CS_RHF_ContextClass
from py_mods.src.SCF.scf_utils import (
    transformation_matrix,
    E_0_unrestricted_comp,
    guess_density_RHF,
    validate_unrestricted_determinant,
    scale_integrals,
    calc_diis_extrapolation,
    calculate_P_next,
    calculate_unrestricted_F_and_r_comp,
)
from dataclasses import dataclass


@dataclass
class CS_UHF_ContextClass:
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
    guess_MAX_ITER : int or None, optional
        If p_guess is 'RHF', number of iterations to run the preliminary RHF calculation.
    INPORB : NDArray[np.float64] or None, optional
        If p_guess is 'INPORB', the initial guess orbitals.
    break_symm : bool, optional
        If True, breaks the symmetry of the initial guess density matrix.
    verbose : bool, optional
        If True print iteration progress.
    conv_type : Literal[None, 'DIIS', 'CROP'], optional
        Type of Convergence Algorithm to use. If None, no algorithm is used.
    conv_MEM : int, optional
        Number of previous Fock matrices and residuals to store for Convergence Algorithm.
    conv_ITER_START : int, optional
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
    mult: Union[None, int] = None
    theta: float = 0.0
    occupation: Union[int, Tuple[NDArray[np.int32], NDArray[np.int32]], None] = None
    max_iter: int = 100
    threshold: float = 1e-12
    p_guess: Literal["core", "ones", "RHF", "IMPORB"] = "core"
    guess_MAX_ITER: Union[int, None] = None
    INPORB: Union[NDArray[np.float64], NDArray[np.complex128], None] = None
    break_symm: bool = False
    verbose: bool = False
    conv_type: Literal[None, "DIIS", "CROP"] = "DIIS"
    conv_MEM: int = 8
    conv_ITER_START: int = 12


@dataclass
class _UHF_SpinDiagnosticsClass(object):
    N_alpha: int
    N_beta: int
    s2: float
    S_z: float
    spin_contamination: float


@dataclass
class CS_UHF_ResultsClass(object):
    """
    Results class for CS_UHF calculations.

    Attributes
    ----------
    context : CS_UHF_ContextClass
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
    R_alpha : NDArray[np.complex128], shape (n, n)
        Right eigenvector matrix for alpha spin.
    L_beta : NDArray[np.complex128], shape (n, n)
        Left eigenvector matrix for beta spin.
    R_beta : NDArray[np.complex128], shape (n, n)
        Right eigenvector matrix for beta spin.
    """

    context: CS_UHF_ContextClass
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
    R_alpha: NDArray[np.complex128]
    R_beta: NDArray[np.complex128]
    S_diagnostics: _UHF_SpinDiagnosticsClass
    error: float
    iterations: int


def CS_UHF(context: CS_UHF_ContextClass) -> CS_UHF_ResultsClass:
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
    # unpacking
    S = context.S.astype(np.complex128)
    T = context.T.astype(np.complex128)
    V = context.V.astype(np.complex128)
    eri = context.eri.astype(np.complex128)
    n_electrons = context.n_electrons
    theta = context.theta
    mult = context.mult
    verbose = context.verbose
    conv_ITER_START = context.conv_ITER_START
    conv_MEM = context.conv_MEM
    conv_type = context.conv_type
    occupation = context.occupation
    max_iter = context.max_iter
    threshold = context.threshold
    p_guess = context.p_guess
    guess_MAX_ITER = context.guess_MAX_ITER
    INPORB = context.INPORB
    break_symm = context.break_symm

    assert len(T) == len(V) == len(S), "Matrices T, V, S must have the same dimensions"
    # assert n_electrons % 2 != 0, "For closed-shell calculations use RHF routine."
    assert conv_type in [
        None,
        "DIIS",
        "CROP",
    ], "Convergence assist must be either None, DIIS, or CROP"

    # setup
    conv_REQUESTED = True if conv_type is not None else False
    conv_ITER_START = (
        min(conv_ITER_START + 1, conv_MEM)
        if conv_MEM >= conv_ITER_START
        else max(conv_ITER_START + 1, conv_MEM)
    )

    if mult is None:
        mult = int(0) if n_electrons % 2 == 0 else int(1)
    assert (
        n_electrons - mult
    ) % 2 != 1, f"It is not possible to have {mult} unpaired electrons with { n_electrons} electrons."

    # Otain transformation matrix and validate occupation determinant
    dim = len(S)
    X = np.array(transformation_matrix(S), dtype=np.complex128)
    det_alpha, det_beta, _ = validate_unrestricted_determinant(
        n_electrons, occupation, dim, mult
    )
    alpha_elec = sum(det_alpha)
    beta_elec = sum(det_beta)

    if verbose:
        print("\n\nAlpha occupation: ", det_alpha)
        print("Beta  occupation: ", det_beta)

    # rescaling the integrals
    T_scaled, V_scaled, eri_scaled = scale_integrals(T, V, eri, theta)
    H_core = T_scaled + V_scaled

    # Guess initial density matrix
    P_alph = guess_density_UHF(
        p_guess, n_electrons, dim, S, T, V, eri, X, guess_MAX_ITER, INPORB
    )

    # P_alph *= S # this leads to a closer guess to the PySCF one
    P_beta = np.copy(P_alph)

    if (
        break_symm
    ):  # note that breaking symmetry will only make sense when the guess is not zeros
        P_beta[:n_electrons, :n_electrons] = 0

    P_guess_alpha = np.copy(P_alph)
    P_guess_beta = np.copy(P_beta)

    # initialize variables and lists
    E_prev = 0.0 + 0.0j
    use_conv = False
    converged = False
    F_guess_alph = []
    F_guess_beta = []
    residuals_alph = []
    residuals_beta = []

    mem_iter = max_iter
    conv_thresh = 1e-4

    if verbose:
        print("-" * 128)
        print(
            "|   Iter   |               E_iter                  |                       Delta_e                   |        norm(e_i)        |"
        )
        print("-" * 128)

    # SCF loop
    for iteration in range(max_iter):
        # calculate F_n and r_n from P_n
        F_alph, r_alph, F_beta, r_beta = calculate_unrestricted_F_and_r_comp(
            P_alph, P_beta, S, H_core, eri_scaled
        )

        error_alph = float(np.linalg.norm(r_alph.flatten()))
        error_beta = float(np.linalg.norm(r_beta.flatten()))

        error = max(error_alph, error_beta)

        E_UHF = E_0_unrestricted_comp(
            P_alph,
            P_beta,
            H_core,
            F_alph.reshape(H_core.shape),
            F_beta.reshape(H_core.shape),
        )

        E_diff = E_UHF - E_prev

        if verbose:
            print(
                f"{iteration:5}     {E_UHF:45.16f}     {E_diff:45.16f}     {error:8.4E}"
            )

        # Check convergence
        if iteration > 5 and error < threshold:
            converged = True
            if verbose:
                print(
                    f"Convergence achieved after {iteration} iterations.\n\n:: Final SCF energy = {E_UHF:5}\n\nFinal SCF energy in parseable format\n%% {E_UHF.real:.14E} {E_UHF.imag:.14E} {theta:.6f}"
                )
            P_alph, e_alph, R_alph, *_ = calculate_P_next(
                F_next_alph, X, alpha_elec, det_alpha, mode="UHF"
            )
            P_beta, e_beta, R_beta, *_ = calculate_P_next(
                F_next_beta, X, beta_elec, det_beta, mode="UHF"
            )
            break

        # Save in memory guesses and residuals keeping size of Convergence Algorithm space

        F_guess_alph.append(F_alph)
        F_guess_beta.append(F_beta)
        residuals_alph.append(r_alph)
        residuals_beta.append(r_beta)

        if len(F_guess_alph) > conv_MEM:
            F_guess_alph.pop(0)
            F_guess_beta.pop(0)
            residuals_alph.pop(0)
            residuals_beta.pop(0)

        # Choose F for P_{n+1}
        if not use_conv:
            F_next_alph: NDArray[np.complex128] = F_alph
            F_next_beta: NDArray[np.complex128] = F_beta

        elif use_conv:
            try:
                F_opt_alph, r_opt_alpha = calc_diis_extrapolation(
                    residuals_alph, F_guess_alph
                )
                F_opt_beta, r_opt_beta = calc_diis_extrapolation(
                    residuals_beta, F_guess_beta
                )

                # Default is DIIS
                F_next_alph = F_opt_alph
                F_next_beta = F_opt_beta

                if conv_type == "CROP":
                    F_guess_alph[-1] = F_opt_alph
                    F_guess_beta[-1] = F_opt_beta
                    residuals_alph[-1] = r_opt_alpha
                    residuals_beta[-1] = r_opt_beta

                    # + r_opt # equation 32 Ettenhuber, r_opt should be here, but it diverges idk why

                    F_next_alph = F_opt_alph  # + r_opt_alpha
                    F_next_beta = F_opt_beta  # + r_opt_beta
            except np.linalg.LinAlgError:
                if verbose:
                    print(
                        "!!!!!!!!!!!!!!!! CONVERGENCE ACCELERATION CAUSED A SINGULAR MATRIX. REVERTING TO STANDARD SCF !!!!!!!!!!!!!!!"
                    )
                use_conv = False

        # P_LR, C_munu, e_values, L_munu, R_munu, P_RR, C_prime
        P_alph, e_alph, R_alph, *_ = calculate_P_next(
            F_next_alph, X, alpha_elec, det_alpha, mode="UHF"
        )
        P_beta, e_beta, R_beta, *_ = calculate_P_next(
            F_next_beta, X, beta_elec, det_beta, mode="UHF"
        )

        if theta == 0.0:
            P_alph = P_alph.real.astype(np.complex128)
            R_alph = R_alph.real.astype(np.complex128)

        P_total = P_alph + P_beta
        P_diff = P_alph - P_beta

        E_prev = E_UHF

        # Check Convergence Algorithm activation
        if (
            iteration == conv_ITER_START and conv_REQUESTED
        ):  #  and error < conv_thresh and not use_conv:
            use_conv = True
            if verbose:
                print("-" * 30, f"   STARTED {conv_type}  ", "-" * 30)

    S_diagnostics = calculate_s2_expectation(P_alph, P_beta, S, verbose)

    n_alpha = np.trace(P_alph.real @ S)
    n_beta = np.trace(P_beta.real @ S)

    assert (
        abs(n_alpha + n_beta - n_electrons) < 1e-10
    ), "Number of electrons was not conserved in the calculation"

    # R_alph, _ = canonicalize(R_alph, F_next_alph.reshape(X.shape))
    # R_beta, _ = canonicalize(R_beta, F_next_beta.reshape(X.shape))

    ResultClass = CS_UHF_ResultsClass(
        context=context,
        converged=converged,
        E_UHF=E_UHF,
        e_alpha=e_alph,
        e_beta=e_beta,
        n_alpha=n_alpha,
        n_beta=n_beta,
        det=(det_alpha, det_beta),
        X=X,
        F_final_alph=F_next_alph,
        F_final_beta=F_next_beta,
        P_guess_alpha=P_guess_alpha,
        P_guess_beta=P_guess_beta,
        P_alpha=P_alph,
        P_beta=P_beta,
        P_total=P_total,
        P_diff=P_diff,
        R_alpha=R_alph,
        R_beta=R_beta,
        S_diagnostics=S_diagnostics,
        error=error,
        iterations=iteration,
    )

    return ResultClass


def guess_density_UHF(
    p_guess,
    n_electrons,
    dim,
    S=None,
    T=None,
    V=None,
    eri=None,
    X=None,
    guess_MAX_ITER=10,
    INPORB=None,
):
    if p_guess == "RHF":
        elec_pre = n_electrons if n_electrons % 2 == 0 else n_electrons - 1
        if isinstance(guess_MAX_ITER, int):
            guess_iter = guess_MAX_ITER

        else:
            guess_iter = 12
            guess_context = CS_RHF_ContextClass(
                S.real,
                T.real,
                V.real,
                eri.real,
                n_electrons=elec_pre,
                theta=0,
                max_iter=guess_iter,
                threshold=1e-14,
                p_guess="core",
                verbose=False,
            )
            guess_scf = CS_RHF(guess_context)
            return guess_scf.P
    else:
        return guess_density_RHF(p_guess, dim, INPORB)


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

    return _UHF_SpinDiagnosticsClass(N_alpha, N_beta, s2, S_z, spin_contamination)


def UHF_theta_traj(max_theta, n_points, context: CS_UHF_ContextClass):
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
    conv_MEM : int, optional
        Number of previous Fock matrices and residuals to store for Convergence Algorithm.
    conv_ITER_START : int, optional
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
