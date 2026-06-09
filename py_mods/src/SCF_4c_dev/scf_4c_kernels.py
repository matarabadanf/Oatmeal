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
