from typing import Tuple, Union, Literal
import numpy as np
from numpy.typing import NDArray
from py_mods.src.SCF.scf_kernels import calc_p_matrix_comp
import scipy


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
        W.astype(np.complex128),
        (eri_classess * exp_t1).astype(np.complex128),
    )


def calculate_P_next_4c(
    F_next: NDArray[np.complex128],
    X: NDArray[np.complex128],
    det: NDArray[np.int8],
    solver: Literal["eig", "eigh"],
    theta: float,
) -> Tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:

    F_prime = X.conj().T @ F_next @ X

    if solver == "eigh" and theta == 0.0:
        e_values, C_prime = np.linalg.eigh(F_prime)
        idx = np.argsort(e_values.real)
        e_values = e_values[idx]
        C_prime = C_prime[:, idx]

        C_munu = X @ C_prime
        L_munu = C_munu.conj().T
    else:
        e_values, vl, C_prime = scipy.linalg.eig(F_prime, left=True, right=True)
        idx = np.argsort(e_values.real)
        e_values = e_values[idx]
        C_prime = C_prime[:, idx]
        vl = vl[:, idx]

        # scipy returns left eigenvectors as columns of vl where vl.conj().T @ A = w * vl.conj().T
        L_prime = vl.conj().T

        # normalize to satisfy biorthogonality L_prime @ C_prime = I
        overlap = np.sum(L_prime * C_prime.T, axis=1)
        L_prime = L_prime / overlap[:, None]

        # Return left and right eigenvectors to AO basis: L_AO = L_prime @ X.dagg, C_AO = X @ C_prime
        C_munu = X @ C_prime
        L_munu = L_prime @ X.conj().T

    P_munu = calc_p_matrix_comp(L_munu, C_munu, det)

    return P_munu, e_values, C_munu, C_prime


def guess_density_4c(
    p_guess: Literal["core", "ones", "INPORB"],
    dim: int,
    INPORB: Union[NDArray[np.complex128], NDArray[np.float64], None],
) -> NDArray[np.complex128]:
    """
    Generate initial guess density (complex) for 4c calculations.

    Parameters
    ----------
    p_guess : {'core', 'ones', 'INPORB'}
        Guess method.
    dim : int
        Basis dimension.
    INPORB : {NDArray[np.complex128], NDArray[np.float64], None}
        Imported guess orbitals.

    Returns
    -------
    P_guess : NDArray[np.complex128]
        Guess density matrix.
    """
    if p_guess == "INPORB":
        assert INPORB is not None, "Empty INPORB for guess"

        assert isinstance(INPORB, np.ndarray) and INPORB.shape == (
            dim,
            dim,
        ), f"Wrong type ({type(INPORB)}) or dimensions ({INPORB.shape}) of import guess orbitals. Dimension expected is {(dim, dim)}"

        return np.copy(INPORB.astype(np.complex128))

    elif p_guess == "core":
        return np.zeros((dim, dim), dtype=np.complex128)

    elif p_guess == "ones":
        return np.ones((dim, dim), dtype=np.complex128)

    else:
        raise ValueError(
            f"Invalid method. Choose 'core', 'ones' or 'INPORB' (inputed {p_guess})."
        )
