from dataclasses import dataclass
from typing import List, Union, Literal
import numpy as np
from numpy.typing import NDArray


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


if __name__ == "__main__":
    pass
