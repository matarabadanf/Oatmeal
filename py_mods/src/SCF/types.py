from dataclasses import dataclass
from typing import List, Union, Literal, Tuple
import numpy as np
from numpy.typing import NDArray

# -------------------------------------------------------------
#  RHF types and allocators
# -------------------------------------------------------------


@dataclass
class CSRHFContext:
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
    _convergence_criteria: Literal["norm", "max"] = "norm"


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
class CSRHFResults:
    """
    Results for CS_RHF calculations.

    Attributes
    ----------
    context : CSRHFContext
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

    context: CSRHFContext
    converged: bool
    E_RHF: np.complex128
    e_orb: NDArray[np.complex128]
    n_elec: np.int32
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


def allocate_rhf_extended_context(ctx: CSRHFContext) -> CSRHFConstants:

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


def allocate_rhf_state(ctx: CSRHFContext) -> CSRHFState:
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


def pack_rhf_results(
    ctx: CSRHFContext,
    rhf_ext_ctx: CSRHFConstants,
    rhf_state: CSRHFState,
) -> CSRHFResults:
    return CSRHFResults(
        context=ctx,
        converged=rhf_state.converged,
        E_RHF=rhf_state.E_RHF,
        e_orb=rhf_state.e_orb,
        n_elec=np.int32(ctx.n_electrons),
        det=rhf_ext_ctx.det,
        H_core=rhf_ext_ctx.H_core,
        X=rhf_ext_ctx.X,
        F_final=rhf_state.F_next,
        C_prime=rhf_state.C_prime,
        P_guess=rhf_state.P_old if rhf_state.iteration > 0 else rhf_state.P,
        P=rhf_state.P,
        C_munu=rhf_state.C_munu,
        error=rhf_state.r,
        iterations=rhf_state.iteration,
        scaled_eris=rhf_ext_ctx.eri_scaled,
    )


# -------------------------------------------------------------
#  UHF types and allocators
# -------------------------------------------------------------


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
    _convergence_criteria: Literal["norm", "max"] = "norm"


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


def pack_uhf_results(
    ctx: CSUHFContext,
    uhf_ext_ctx: CSUHFConstants,
    uhf_state: CSUHFState,
    P_guess_alpha: NDArray[np.complex128],
    P_guess_beta: NDArray[np.complex128],
    S_diagnostics: UHFSpinDiagnostics,
) -> CSUHFResults:
    return CSUHFResults(
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


if __name__ == "__main__":
    pass
