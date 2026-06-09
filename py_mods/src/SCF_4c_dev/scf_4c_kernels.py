from tkinter import W
from typing import Tuple, Union
import numpy as np
from numpy.typing import NDArray


def scale_4c_integrals(
    T: Union[NDArray[np.complex128], NDArray[np.float64]],
    V: Union[NDArray[np.complex128], NDArray[np.float64]],
    W: Union[NDArray[np.complex128], NDArray[np.float64]],
    eri_classess: Union[NDArray[np.complex128], NDArray[np.float64]],
    theta: float,
) -> Tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """
    Scale 4c integrals by a complex factor exp(-i*theta).

    Parameters
    ----------
    T : NDArray[np.float64]
        Kinetic energy matrix.
    V : NDArray[np.float64]
        Nuclear attraction matrix.
    eri_classess : NDArray[np.float64]
        ERIs.
    theta : float
        Complex scaling angle.

    Returns
    -------
    T_s, V_s, eri_s : Tuple[NDArray[np.complex128], ...]
        Scaled integrals.
    """
    exp_t1 = np.exp(-1j * theta)
    # Ensure output is complex128 even if input is float
    return (
        (T * exp_t1).astype(np.complex128),
        (V * exp_t1).astype(np.complex128),
        (W * exp_t1).astype(np.complex128),
        (eri_classess * exp_t1).astype(np.complex128),
    )

from typing import Literal

def calculate_P_next_4c(
    F_next: NDArray[np.complex128],
    X: NDArray[np.complex128],
    det: NDArray[np.int8],
    solver: Literal["eig", "eigh"],
    theta: float
) -> Tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    from py_mods.src.SCF.scf_kernels import calc_p_matrix_comp

    F_prime = X.T @ F_next @ X
    
    if solver == "eigh":
        e_values, C_prime = np.linalg.eigh(F_prime)
        idx = np.argsort(e_values.real)
        e_values = e_values[idx]
        C_prime = C_prime[:, idx]
    else:
        e_values, C_prime = np.linalg.eig(F_prime)
        idx = np.argsort(e_values.real)
        e_values = e_values[idx]
        C_prime = C_prime[:, idx]
        
    C_munu = X @ C_prime
    
    if theta == 0.0:
        L_munu = C_munu.conj().T
    else:
        L_munu = C_munu.T
        
    P_munu = calc_p_matrix_comp(L_munu, C_munu, det)

    return P_munu, e_values, C_munu, C_prime
