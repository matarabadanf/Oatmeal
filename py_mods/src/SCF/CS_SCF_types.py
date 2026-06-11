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
    Context for (CS)RHF calculation.

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
        Total electron count (must be an even number).
    theta : float, optional
        Complex-scaling angle in radians. Default is 0.0.
    occupation : int, NDArray[np.int32] or None, optional
        Occupation vector. If an integer, it specifies the number of doubly occupied
        orbitals. If an array, it explicitly defines the occupation numbers.
        If None, a default occupation is built based on `n_electrons`. Default is None.
    max_iter : int, optional
        Maximum number of SCF iterations. Default is 100.
    threshold : float, optional
        Convergence threshold. Default is 1e-12.
    p_guess : {'core', 'ones', 'INPORB'}, optional
        Initial density guess type. Default is 'core'.
    guess_max_iter : int or None, optional
        Maximum iterations for a preliminary RHF guess (if applicable). Default is None.
    initial_orbitals : NDArray[np.float64], NDArray[np.complex128] or None, optional
        Imported orbitals for the initial guess. Required if `p_guess` is 'INPORB'.
        Can be real or complex. Default is None.
    verbose : bool, optional
        If True, print progress information to stdout. Default is False.
    conv_type : {None, 'DIIS', 'CROP'}, optional
        Convergence acceleration algorithm. Default is 'DIIS'.
    acc_hist_size : int, optional
        History size for the convergence acceleration subspace. Default is 10.
    acc_iteration_start : int, optional
        Iteration number to start applying convergence acceleration. Default is 12.
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
    p_guess: Literal["core", "ones", "INPORB"] = "core"
    guess_max_iter: Union[int, None] = None
    initial_orbitals: Union[NDArray[np.float64], NDArray[np.complex128], None] = None
    verbose: bool = False
    conv_type: Literal[None, "DIIS", "CROP"] = "DIIS"
    acc_hist_size: int = 5
    acc_iteration_start: int = 6

    # Internal
    _eigensolver: Literal["eig", "eigh"] = "eigh"
    _convergence_criteria: Literal["norm", "max"] = "norm"


@dataclass
class CSRHFConstants:
    """
    Constants and scaled/calculated matrices for (CS)RHF.

    Attributes
    ----------
    dim : int
        Dimension of the basis set (number of MO basis functions).
    X : NDArray[np.complex128]
        Orthogonalization matrix (S^{-1/2} for canonical orthogonalization).
    det : NDArray[np.int32]
        Occupation vector.
    eri_scaled : NDArray[np.complex128]
        Complex-scaled eris in the atomic orbital basis.
    H_core : NDArray[np.complex128]
        Complex-scaled core Hamiltonian matrix (T + V).
    core_mask : NDArray[np.bool]
        Boolean mask of Hcore interactions.
    _eigensolver : {'eig', 'eigh'}
        Solver to use for diagonalization.
    acc_iteration_start : int, optional
        Iteration number at which to begin convergence acceleration. Default is 10.
    acc_requested : bool, optional
        Boolean indicating if convergence acceleration (DIIS/CROP) is requested. Default is False.
    """

    dim: int
    X: NDArray[np.complex128]
    det: NDArray[np.int32]
    eri_scaled: NDArray[np.complex128]
    H_core: NDArray[np.complex128]
    core_mask: NDArray[np.bool]
    _eigensolver: Literal["eig", "eigh"]
    acc_iteration_start: int = 10
    acc_requested: bool = False


@dataclass
class CSRHFState:
    """
    State struct for (CS)RHF procedure.

    Attributes
    ----------
    iteration : int
        Current SCF iteration number.
    P : NDArray[np.complex128]
        Current density matrix.
    E_prev : np.complex128
        Energy from previous iteration.
    use_conv_acc : bool
        Bool indicating if convergence acceleration is used.
    F_guess : List[NDArray[np.complex128]]
        DIIS/CROP history of F matrices.
    residuals : List[NDArray[np.complex128]]
        DIIS/CROP history of error vectors.
    F_next : NDArray[np.complex128]
        Extrapolated F matrix.
    error : complex
        Current error.
    converged : bool
        True if converged with threshold.
    C_munu : NDArray[np.complex128]
        MO coefficients in the atomic orbital basis.
    C_prime : NDArray[np.complex128]
        MO coefficients in the orthogonalized basis.
    e_orb : NDArray[np.complex128]
        Current orbital energies.
    F : NDArray[np.complex128]
        Current Fock matrix.
    r : NDArray[np.complex128]
        Current residual/error vector.
    E_RHF : np.complex128
        Current total RHF energy.
    E_diff : np.complex128
        Change in energy from the previous iteration.
    P_old : NDArray[np.complex128]
        Density matrix from the previous iteration.
    """

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
    Struct containing the results of a (CS)RHF calculation.

    Attributes
    ----------
    context : CSRHFContext
        Input context.
    converged : bool
        True if the SCF converged.
    E_RHF : np.complex128
        Complex-Scaled RHF energy.
    e_orb : NDArray[np.complex128]
        Orbital energies.
    n_elec : np.int32
        Final number of electrons.
    det : NDArray[np.int32]
        Occupation vector.
    H_core : NDArray[np.complex128]
        (CS)H_core.
    X : NDArray[np.complex128]
        Orthogonalization transformation matrix.
    F_final : NDArray[np.complex128]
        The Fock matrix of the final iteration.
    C_prime : NDArray[np.complex128]
        Molecular orbital coefficients in the orthogonalized basis.
    P_guess : NDArray[np.complex128]
        Initial density matrix.
    P : NDArray[np.complex128]
        Final density matrix.
    C_munu : NDArray[np.complex128]
        Final MO coefficients.
    error : float or complex
        Final error.
    iterations : int
        Number of iterations.
    scaled_eris : NDArray[np.complex128]
        Scaled eris.
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
    error: Union[float, complex]
    iterations: int
    scaled_eris: NDArray[np.complex128]
    homo_index: int


def allocate_rhf_extended_context(ctx: CSRHFContext) -> CSRHFConstants:
    """
    Allocate extended context for (CS)RHF calculations.

    Parameters
    ----------
    ctx : CSRHFContext
        Input context with system parameters.

    Returns
    -------
    CSRHFConstants
        Allocated extended context with preallocated matrices.
    """
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
    """
    Allocate state for CSRHF calculations.

    Parameters
    ----------
    ctx : CSRHFContext
        Input context with system parameters.

    Returns
    -------
    CSRHFState
        Allocated state with preallocated matrices and initial values.
    """
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
    """
    Pack results from (CS)RHF calculation into a results dataclass.

    Parameters
    ----------
    ctx : CSRHFContext
        Input context used for the calculation.
    rhf_ext_ctx : CSRHFConstants
        Extended context with preallocated matrices.
    rhf_state : CSRHFState
        Final state after SCF iterations.

    Returns
    -------
    CSRHFResults
        Packed results from the CSRHF calculation.
    """
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
        error=rhf_state.error,
        iterations=rhf_state.iteration,
        scaled_eris=rhf_ext_ctx.eri_scaled,
        homo_index=int(np.int32(ctx.n_electrons) / 2),
    )


# -------------------------------------------------------------
#  UHF types and allocators
# -------------------------------------------------------------


@dataclass
class CSUHFContext:
    """
    Context for (CS)UHF calculations.

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
        Total electron count.
    mult : int, optional
        Spin multiplicity (2S + 1). Default is -1 (calculate from n_electrons).
    theta : float, optional
        Complex-scaling angle in radians. Default is 0.0.
    occupation : Union[int, Tuple[NDArray[np.int32], NDArray[np.int32]], None], optional
        Occupation vector. If int/None, built defaults. If Tuple, explicit (alpha, beta) vectors.
    max_iter : int, optional
        Maximum SCF iterations. Default is 100.
    threshold : float, optional
        Convergence threshold. Default is 1e-12.
    p_guess : Literal['core', 'ones', 'RHF', 'INPORB'], optional
        Initial density guess type. Default is 'core'.
    guess_max_iter : int or None, optional
        Max iterations for preliminary RHF guess.
    initial_orbitals : List[NDArray[np.complex128]] or None, optional
        List of [alpha, beta] orbitals for 'INPORB' guess.
    break_symm : Literal[None, True, 'arbitrary', 'random', 'perturbation'], optional
        Method to break initial guess symmetry.
    verbose : bool, optional
        If True, print progress.
    conv_type : Literal[None, 'DIIS', 'CROP'], optional
        Convergence algorithm. Default is 'DIIS'.
    acc_hist_size : int, optional
        History size for convergence acceleration. Default is 10.
    acc_iteration_start : int, optional
        Iteration to start acceleration. Default is 12.
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
    p_guess: Literal["core", "ones", "RHF", "INPORB"] = "core"
    guess_max_iter: Union[int, None] = None
    initial_orbitals: Union[List[NDArray[np.complex128]], None] = None
    break_symm: Literal[None, True, "arbitrary", "random", "perturbation"] = (
        None  # type of symmetry breaking. If True is given, 'arbitrary' will be used
    )
    verbose: bool = False
    conv_type: Literal[None, "DIIS", "CROP"] = "DIIS"
    acc_hist_size: int = 10
    acc_iteration_start: int = 12

    # Internal
    _eigensolver: Literal["eig", "eigh"] = "eigh"
    _convergence_criteria: Literal["norm", "max"] = "norm"


@dataclass
class CSUHFConstants:
    """
    Constants and scaled/calculated matrices for (CS)UHF.

    Attributes
    ----------
    dim : int
        Dimension of the basis set.
    X : NDArray[np.complex128]
        Orthogonalization matrix (S^{-1/2}).
    det : NDArray[np.int32]
        Occupation vector for (alpha, beta).
    alpha_elec : int
        Number of alpha electrons.
    beta_elec : int
        Number of beta electrons.
    eri_scaled : NDArray[np.complex128]
        Complex-scaled eris.
    H_core : NDArray[np.complex128]
        Complex-scaled core Hamiltonian (T + V).
    core_mask : NDArray[np.bool]
        Boolean mask of Hcore interactions.
    _eigensolver : {'eig', 'eigh'}
        Solver for diagonalization.
    acc_iteration_start : int, optional
        Iteration to start acceleration. Default is 0.
    acc_requested : bool, optional
        If True, convergence acceleration is requested. Default is False.
    """

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
    """
    State struct for CSUHF procedure.

    Attributes
    ----------
    iteration : int
        Current SCF iteration number.
    use_conv_acc : bool
        If convergence acceleration is active.
    converged : bool
        True if converged with threshold.
    E_UHF : np.complex128
        Current total UHF energy.
    e_orb_alpha : NDArray[np.complex128]
        Alpha orbital energies.
    e_orb_beta : NDArray[np.complex128]
        Beta orbital energies.
    C_munu_alpha : NDArray[np.complex128]
        Alpha MO coefficients (AO basis).
    C_munu_beta : NDArray[np.complex128]
        Beta MO coefficients (AO basis).
    C_prime_alpha : NDArray[np.complex128]
        Alpha MO coefficients (Orthogonal basis).
    C_prime_beta : NDArray[np.complex128]
        Beta MO coefficients (Orthogonal basis).
    final_alpha_elec : int
        Calculated alpha electron count.
    final_beta_elec : int
        Calculated beta electron count.
    F_guess_alpha : List[NDArray[np.complex128]]
        Alpha Fock history (DIIS/CROP).
    F_guess_beta : List[NDArray[np.complex128]]
        Beta Fock history (DIIS/CROP).
    residuals_alpha : List[NDArray[np.complex128]]
        Alpha error vector history.
    residuals_beta : List[NDArray[np.complex128]]
        Beta error vector history.
    P_old_alpha : NDArray[np.complex128]
        Alpha density from previous iteration.
    P_old_beta : NDArray[np.complex128]
        Beta density from previous iteration.
    r_alpha : NDArray[np.complex128]
        Alpha residual vector.
    r_beta : NDArray[np.complex128]
        Beta residual vector.
    error_alpha : complex
        Alpha component error.
    error_beta : complex
        Beta component error.
    error : complex
        Total error metric.
    E_diff : np.complex128
        Change in energy from previous iteration.
    E_prev : np.complex128
        Energy from previous iteration.
    F_alpha : NDArray[np.complex128]
        Current Alpha Fock matrix.
    F_beta : NDArray[np.complex128]
        Current Beta Fock matrix.
    F_next_alpha : NDArray[np.complex128]
        Extrapolated Alpha Fock matrix.
    F_next_beta : NDArray[np.complex128]
        Extrapolated Beta Fock matrix.
    P_alpha : NDArray[np.complex128]
        Current Alpha density matrix.
    P_beta : NDArray[np.complex128]
        Current Beta density matrix.
    P_total : NDArray[np.complex128]
        Total density (P_alpha + P_beta).
    P_diff : NDArray[np.complex128]
        Spin density (P_alpha - P_beta).
    """

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
    """
    Struct for spin analysis.

    Attributes
    ----------
    N_alpha : int
        Number of alpha electrons.
    N_beta : int
        Number of beta electrons.
    s2 : float
        Expectation value <S^2>.
    S_z : float
        Expectation value <S_z>.
    spin_contamination : float
        Difference between calculated and exact <S^2>.
    """

    N_alpha: int
    N_beta: int
    s2: float
    S_z: float
    spin_contamination: float


@dataclass
class CSUHFResults(object):
    """
    Results for CS_UHF calculations.

    Attributes
    ----------
    context : CSUHFContext
        Input context.
    converged : bool
        Convergence status.
    E_UHF : np.complex128
        Final UHF energy.
    e_alpha : NDArray[np.complex128]
        Alpha orbital energies.
    e_beta : NDArray[np.complex128]
        Beta orbital energies.
    n_alpha : float
        Alpha electron count.
    n_beta : float
        Beta electron count.
    det : Tuple[NDArray[np.int32], NDArray[np.int32]]
        Occupation vectors (alpha, beta).
    X : NDArray[np.complex128]
        Transformation matrix.
    F_final_alph : NDArray[np.complex128]
        Final Alpha Fock matrix.
    F_final_beta : NDArray[np.complex128]
        Final Beta Fock matrix.
    P_guess_alpha : NDArray[np.complex128]
        Initial alpha density guess.
    P_guess_beta : NDArray[np.complex128]
        Initial beta density guess.
    P_alpha : NDArray[np.complex128]
        Final alpha density matrix.
    P_beta : NDArray[np.complex128]
        Final beta density matrix.
    P_total : NDArray[np.complex128]
        Total density matrix.
    P_diff : NDArray[np.complex128]
        Spin density matrix.
    C_alpha : NDArray[np.complex128]
        Alpha MO coefficients.
    C_beta : NDArray[np.complex128]
        Beta MO coefficients.
    S_diagnostics : UHFSpinDiagnostics
        Spin diagnostic struct.
    error : float
        Final error.
    iterations : int
        Total iterations performed.
    scaled_eris : NDArray[np.complex128]
        Scaled eris.
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
    homo_index: int


def allocate_uhf_extended_context(ctx: CSUHFContext) -> CSUHFConstants:
    """
    Allocate extended context for CSUHF calculations.

    Parameters
    ----------
    ctx : CSUHFContext
        Input context with system parameters.

    Returns
    -------
    CSUHFConstants
        Allocated extended context with preallocated matrices.
    """
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
    """
    Allocate state for CSUHF calculations.

    Parameters
    ----------
    ctx : CSUHFContext
        Input context with system parameters.

    Returns
    -------
    CSUHFState
        Allocated state with preallocated matrices and initial values.
    """
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
    """
    Pack results from CSUHF calculation into a results dataclass.

    Parameters
    ----------
    ctx : CSUHFContext
        Input context used for the calculation.
    uhf_ext_ctx : CSUHFConstants
        Extended context with preallocated matrices.
    uhf_state : CSUHFState
        Final state after SCF iterations.
    P_guess_alpha : NDArray[np.complex128]
        Initial alpha density guess.
    P_guess_beta : NDArray[np.complex128]
        Initial beta density guess.
    S_diagnostics : UHFSpinDiagnostics
        Spin diagnostic struct.

    Returns
    -------
    CSUHFResults
        Packed results from the CSUHF calculation.
    """
    return CSUHFResults(
        context=ctx,
        converged=uhf_state.converged,
        E_UHF=uhf_state.E_UHF,
        e_alpha=uhf_state.e_orb_alpha,
        e_beta=uhf_state.e_orb_beta,
        n_alpha=uhf_state.final_alpha_elec,
        n_beta=uhf_state.final_beta_elec,
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
        homo_index=int(
            max(uhf_state.final_alpha_elec.real, uhf_state.final_beta_elec.real)
        ),
    )


if __name__ == "__main__":
    pass
