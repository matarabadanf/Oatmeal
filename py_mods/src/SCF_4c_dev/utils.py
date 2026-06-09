from py_mods.src.SCF_4c_dev.types_4c import CS_4c_KU_SCF_Context
import numpy as np

from typing import Tuple, Union

from numpy.typing import NDArray


def validate_CS_4c_KU_SCF_context_input(ctx: CS_4c_KU_SCF_Context):
    """
    Validate the input context for CS-4c-KU-SCF calculations.

    Raises
    ------
    ValueError
        If any of the validation checks fail, a ValueError is raised with an appropriate message.

    Notes
    -----
    This function checks the following:
    - The dimension of the overlap matrix S (and T and V) must be 2*(nL+nS). 2 * since we have both alpha and beta components with the same AO basis.
    - The dimensions of S, T, and V must be the same.
    - The dimension of T must be 2*dimension of S.
    - The convergence assist type must be either None, 'DIIS', or 'CROP'.
    - The eigensolver must be either 'eig', 'eigh', or 'genh'.
    """
    if not 2 * (ctx.nL + ctx.nS) == len(ctx.S):
        raise ValueError(
            f"Dimension of S (and T, V, W) must be 2*(nL+nS). Got {len(ctx.S)}, expected {2 * (ctx.nL + ctx.nS)}"
        )

    if not len(ctx.S) == len(ctx.W) == len(ctx.V) == len(ctx.T):
        raise ValueError(
            f"Dimension of S, V, W, and T must be the same. Got {len(ctx.S)}, {len(ctx.V)}, {len(ctx.W)},  {len(ctx.T)}"
        )

    if ctx.conv_type not in (None, "DIIS", "CROP"):
        raise ValueError("Convergence assist must be either None, 'DIIS', or 'CROP'")

    if ctx._eigensolver not in [
        "eig",
        "eigh",
        "genh",
    ]:
        raise ValueError(
            f"Eigensolver must be either 'eig', 'eigh' or 'genh'. Got {ctx._eigensolver}"
        )


def validate_4c_determinant(
    nS: int,
    nL: int,
    n_electrons: int,
    full_det: Union[int, NDArray[np.int32], None],
) -> Tuple[NDArray[np.int8], bool]:
    """
    Validate or construct occupation determinant.

    Parameters
    ----------
    n_electrons : int
        Total electron count.
    full_det : int, NDArray[np.int32], or None
        If -1/None, build aufbau ordered occupation in the positive energy solutions.
    expected_dim : int
        Expected vector length.

    Returns
    -------
    determinant : NDArray[np.int32]
        Validated occupation vector.
    natural_occupation : bool
        True if constructed as natural (RHF) occupation.
    """
    natural_occupation = True

    if full_det is None:
        full_det = -1

    if isinstance(full_det, int):
        if full_det == -1:
            det_arr = np.zeros(2 * (nL + nS), dtype=np.int8)
            n_occ = n_electrons
            det_arr[2 * nS : 2 * nS + n_occ] = 1
            return det_arr, natural_occupation
        else:
            raise TypeError("determinant must be -1, None or a numpy array")

    if not isinstance(full_det, np.ndarray):
        raise TypeError("determinant must be a numpy.ndarray when not -1/None")

    natural_occupation = False
    det_arr = full_det.astype(np.int8)

    if int(np.sum(det_arr)) != n_electrons:
        raise ValueError(
            f"determinant sum ({int(np.sum(det_arr))}) != n_electrons ({n_electrons})"
        )

    if int(np.sum(det_arr[: 2 * nS])) != 0 and len(det_arr) == 2 * (nL + nS):
        raise ValueError(
            f"Positron states must be unoccupied. Got sum {int(np.sum(det_arr[:2*nS]))} in the first 2*nS."
        )

    expected_dim = 2 * (nL + nS)

    if len(det_arr) != expected_dim:
        new_occ = np.zeros(expected_dim, dtype=np.int8)
        length = min(len(det_arr), expected_dim)
        new_occ[2 * nS : 2 * nS + length] = det_arr[:length]
        det_arr = new_occ

    return det_arr, natural_occupation
