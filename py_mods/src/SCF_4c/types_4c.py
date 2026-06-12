from typing import Union, List, Literal
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# -------------------------------------------------------------
#  CS-4c-KU-SCF types and allocators
# -------------------------------------------------------------


@dataclass
class CS_4c_KU_SCF_Context:
    """
    Context for complex-scaled four-component Kramers-unrestricted self-consistent field (CS-4c-KU-SCF) calculations.

    Attributes
    ----------
    nL : int
        Number of AO large component basis.
    nS : int
        Number of AO small component basis.
    S : NDArray[np.float64]
        Overlap matrix.
    T : NDArray[np.complex128]
        Kinetic energy matrix.
    V : NDArray[np.float64]
        Nuclear attraction matrix.
    W : NDArray[np.float64]
        V_sC - 2mc^2 matrix.
    eri_classess : NDArray[np.float64]
        Electron repulsion integrals zeroed in non used combinations such as (LS|SS).
    n_electrons : int
        Total electron count (must be an even number).
    theta : float, optional
        Complex-scaling angle in radians. Default is 0.0.
    occ : int, NDArray[np.int32] or None, optional
        Occupation vector.
        If same size as nL, it is expanded to include small components.
        If same size as 2*(nL+nS), it is used directly.
        If an array, it explicitly defines the occupation numbers.
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
    nL: int
    nS: int
    S: NDArray[np.float64]
    T: NDArray[np.complex128]
    V: NDArray[np.float64]
    W: NDArray[np.float64]
    eri_classess: NDArray[np.float64]
    n_electrons: int

    # Optional
    theta: float = 0.0
    occ: Union[int, NDArray[np.int32], None] = None
    max_iter: int = 100
    threshold: float = 1e-12
    p_guess: Literal["core", "ones", "INPORB"] = "core"
    guess_max_iter: Union[int, None] = None
    initial_orbitals: Union[NDArray[np.float64], NDArray[np.complex128], None] = None
    verbose: bool = True
    conv_type: Literal[None, "DIIS", "CROP"] = "DIIS"
    acc_hist_size: int = 5
    acc_iteration_start: int = 6
    remove_lindep: bool = True
    lindep_thres: float = 1e-6

    # Internal
    _eigensolver: Literal["eig", "eigh"] = "eigh"
    _convergence_criteria: Literal["norm", "max"] = "norm"


@dataclass
class CS_4c_KU_SCF_Constants:
    """
    Constants and scaled/calculated matrices for CS-4c-KU-SCF calculations.

    Attributes
    ----------
    dim : int
        Total number of basis functions 2*(nL + nS).
    X : NDArray[np.complex128]
        Orthogonalization matrix (S^{-1/2} for canonical orthogonalization).
    det : NDArray[np.int8]
        Occupation vector including small and large components.
    eri_scaled : NDArray[np.complex128]
        Complex-scaled class eris in the atomic orbital basis.
    H_core : NDArray[np.complex128]
        Complex-scaled core Hamiltonian matrix (T + V).
    core_mask : NDArray[np.bool_]
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
    core_mask: NDArray[np.bool_]
    _eigensolver: Literal["eig", "eigh"]
    acc_iteration_start: int = 10
    acc_requested: bool = False


@dataclass
class CS_4c_KU_SCF_State:
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
    e_electronic_orb : NDArray[np.complex128]
        Current electronic solution orbital energies. (size is 2*nL)
    F : NDArray[np.complex128]
        Current Fock matrix.
    r : NDArray[np.complex128]
        Current residual/error vector.
    E_SCF : np.complex128
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
    e_electronic_orb: NDArray[np.complex128]
    F: NDArray[np.complex128]
    r: NDArray[np.complex128]
    E_SCF: np.complex128
    E_diff: np.complex128
    P_old: NDArray[np.complex128]


@dataclass
class CS_4c_KU_SCF_Results:
    """
    Struct containing the results of a (CS)RHF calculation.

    Attributes
    ----------
    context : CSRHFContext
        Input context.
    converged : bool
        True if the SCF converged.
    E_SCF : np.complex128
        Complex-Scaled RHF energy.
    e_orb : NDArray[np.complex128]
        Orbital energies.
    e_electronic_orb : NDArray[np.complex128]
        Electronic solution orbital energies (size is 2*nL).
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

    context: CS_4c_KU_SCF_Context
    converged: bool
    E_SCF: np.complex128
    e_orb: NDArray[np.complex128]
    e_electronic_orb: NDArray[np.complex128]
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


def allocate_CS_4c_KU_SCF_extended_context(
    ctx: CS_4c_KU_SCF_Context,
) -> CS_4c_KU_SCF_Constants:
    """
    Allocate extended context for complex-scaled four-component Kramers-unrestricted self-consistent field (CS-4c-KU-SCF) calculations.

    Parameters
    ----------
    ctx : CS_4c_KU_SCF_Context
        Input context with system parameters.

    Returns
    -------
    CS_4c_KU_SCF_Constants
        Allocated extended context with preallocated matrices.
    """
    nL = ctx.nL
    nS = ctx.nS
    spatial_dim = nL + nS
    full_dim = 2 * spatial_dim
    X = np.zeros((full_dim, full_dim), dtype=np.complex128)
    det = np.zeros(full_dim, dtype=np.int32)
    eri_scaled = np.zeros(
        (spatial_dim, spatial_dim, spatial_dim, spatial_dim), dtype=np.complex128
    )
    H_core = np.zeros((full_dim, full_dim), dtype=np.complex128)
    core_mask = np.zeros((full_dim, full_dim), dtype=np.bool_)
    _eigensolver = ctx._eigensolver

    return CS_4c_KU_SCF_Constants(
        dim=full_dim,
        X=X,
        det=det,
        eri_scaled=eri_scaled,
        H_core=H_core,
        core_mask=core_mask,
        _eigensolver=_eigensolver,
    )


def allocate_CS_4c_KU_SCF_state(ctx: CS_4c_KU_SCF_Context) -> CS_4c_KU_SCF_State:
    """
    Allocate state for CSRHF calculations.

    Parameters
    ----------
    ctx : CS_4c_KU_SCF_Context
        Input context with system parameters.

    Returns
    -------
    CS_4c_KU_SCF_State
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
    e_electronic_orb = np.zeros(ctx.nL * 2, dtype=np.complex128)
    C_prime = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    F = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    r = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)
    E_SCF = np.complex128(0.0)
    E_diff = np.complex128(0.0)
    P_old = np.zeros((len(ctx.S), len(ctx.S)), dtype=np.complex128)

    return CS_4c_KU_SCF_State(
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
        e_electronic_orb=e_electronic_orb,
        C_prime=C_prime,
        F=F,
        r=r,
        E_SCF=E_SCF,
        E_diff=E_diff,
        P_old=P_old,
    )


def pack_CS_4c_KU_SCF_results(
    ctx: CS_4c_KU_SCF_Context,
    ext_ctx: CS_4c_KU_SCF_Constants,
    state: CS_4c_KU_SCF_State,
) -> CS_4c_KU_SCF_Results:
    """
    Pack results from (CS)RHF calculation into a results dataclass.

    Parameters
    ----------
    ctx : CS_4c_KU_SCF_Context
        Input context used for the calculation.
    ext_ctx : CS_4c_KU_SCF_Constants
        Extended context with preallocated matrices.
    state : CS_4c_KU_SCF_State
        Final state after SCF iterations.

    Returns
    -------
    CS_4c_KU_SCF_Results
        Packed results from the CS-4c-KU-SCF calculation.
    """
    return CS_4c_KU_SCF_Results(
        context=ctx,
        converged=state.converged,
        E_SCF=state.E_SCF,
        e_orb=state.e_orb,
        e_electronic_orb=state.e_orb[-2*ctx.nL:],
        n_elec=np.int32(ctx.n_electrons),
        det=ext_ctx.det,
        H_core=ext_ctx.H_core,
        X=ext_ctx.X,
        F_final=state.F_next,
        C_prime=state.C_prime,
        P_guess=state.P_old if state.iteration > 0 else state.P,
        P=state.P,
        C_munu=state.C_munu,
        error=state.error,
        iterations=state.iteration,
        scaled_eris=ext_ctx.eri_scaled,
        homo_index=int(np.int32(ctx.n_electrons) / 2),
    )
