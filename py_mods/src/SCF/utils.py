import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, Literal


def validate_determinant(
    n_electrons: int,
    determinant: Union[int, NDArray[np.int32], None],
    expected_dim: int,
) -> Tuple[NDArray[np.int32], bool]:
    """
    Validate or construct occupation determinant.

    Parameters
    ----------
    n_electrons : int
        Total electron count.
    determinant : int, NDArray[np.int32], or None
        If -1/None, build default RHF vector (2,2,...,0).
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

    if determinant is None:
        determinant = -1

    if isinstance(determinant, int):
        if determinant == -1:
            det_arr = np.zeros(expected_dim, dtype=np.int32)
            n_occ = n_electrons // 2
            det_arr[:n_occ] = 2
            return det_arr, natural_occupation
        else:
            raise TypeError("determinant must be -1, None or a numpy array")

    if not isinstance(determinant, np.ndarray):
        raise TypeError("determinant must be a numpy.ndarray when not -1/None")

    natural_occupation = False
    det_arr = determinant.astype(np.int32)

    if int(np.sum(det_arr)) != n_electrons:
        raise ValueError(
            f"determinant sum ({int(np.sum(det_arr))}) != n_electrons ({n_electrons})"
        )

    if len(det_arr) != expected_dim:
        new_occ = np.zeros(expected_dim, dtype=np.int32)
        length = min(len(det_arr), expected_dim)
        new_occ[:length] = det_arr[:length]
        det_arr = new_occ

    return det_arr, natural_occupation


def validate_unrestricted_determinant(
    n_electrons: int,
    determinants: Union[int, Tuple[NDArray[np.int32], NDArray[np.int32]], None],
    expected_dim: int,
    multiplicity: int,
) -> Tuple[NDArray[np.int32], NDArray[np.int32], bool]:
    """
    Validate or construct unrestricted occupation determinants.

    Parameters
    ----------
    n_electrons : int
        Total electron count.
    determinants : int, Tuple[NDArray, NDArray], or None
        If -1/None, build default vector. Tuple is (alpha, beta).
    expected_dim : int
        Expected vector length.
    multiplicity : int
        Number of unpaired electrons.

    Returns
    -------
    alpha_det : NDArray[np.int32]
        Alpha occupation vector.
    beta_det : NDArray[np.int32]
        Beta occupation vector.
    natural_occupation : bool
        True if constructed as natural (UHF) occupation.
    """
    natural_occupation = True

    if determinants is None:
        determinants = -1

    if isinstance(determinants, int):
        if determinants == -1:
            alpha_det = np.zeros(expected_dim, dtype=np.int32)
            beta_det = np.zeros(expected_dim, dtype=np.int32)

            n_doubly_occ = (n_electrons - multiplicity) // 2
            alpha_det[:n_doubly_occ] = 1
            beta_det[:n_doubly_occ] = 1

            # Remaining electrons are unpaired alpha
            n_unpaired = n_electrons - (n_doubly_occ * 2)
            alpha_det[n_doubly_occ : n_doubly_occ + n_unpaired] = 1

            valid, _ = check_unpaired(alpha_det, beta_det, multiplicity)
            assert valid, f"Multiplicity mismatch for auto-generated determinant."

            return alpha_det, beta_det, natural_occupation
        else:
            raise TypeError("determinant must be -1, None or a tuple of arrays")

    if not isinstance(determinants, (list, tuple)):
        raise TypeError("determinant must be a list or tuple of arrays")

    natural_occupation = False

    alpha_in = np.array(determinants[0], dtype=np.int32)
    beta_in = np.array(determinants[1], dtype=np.int32)

    alpha_det = np.zeros(expected_dim, dtype=np.int32)
    beta_det = np.zeros(expected_dim, dtype=np.int32)

    alpha_det[: len(alpha_in)] = alpha_in
    beta_det[: len(beta_in)] = beta_in

    total_elec = np.sum(alpha_det) + np.sum(beta_det)
    assert (
        total_elec == n_electrons
    ), f"Occupation sum ({total_elec}) != n_electrons ({n_electrons})"

    valid, _ = check_unpaired(alpha_det, beta_det, multiplicity)
    assert valid, f"Multiplicity mismatch."

    return alpha_det, beta_det, natural_occupation


def check_unpaired(
    alpha_det: NDArray[np.int32], beta_det: NDArray[np.int32], multiplicity: int
) -> Tuple[bool, int]:
    """
    Verify unpaired electrons match multiplicity.

    Parameters
    ----------
    alpha_det : NDArray[np.int32]
        Alpha occupation vector.
    beta_det : NDArray[np.int32]
        Beta occupation vector.
    multiplicity : int
        Expected multiplicity.

    Returns
    -------
    is_valid : bool
        True if valid.
    calculated_mult : int
        Calculated multiplicity.
    """
    diff = alpha_det - beta_det
    calc_mult = int(np.sum(np.abs(diff)))

    return (calc_mult == multiplicity), calc_mult


def validate_rhf_context_input(ctx):
    if not len(ctx.T) == len(ctx.V) == len(ctx.S):
        raise ValueError(
            f"Matrices T, V, S must have the same dimensions. Got N_S={len(ctx.S)}, N_T={len(ctx.T)}, N_V={len(ctx.V)}"
        )

    if ctx.n_electrons % 2 != 0:
        raise ValueError("RHF can only be closed-shell systems")

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


def initialize_conv_acc(
    acc_hist_size: int,
    conv_type: Literal[None, "DIIS", "CROP"],
    acc_iteration_start: int,
) -> Tuple[int, bool]:
    """
    Setup convergence acceleration parameters.

    Parameters
    ----------
    """
    if conv_type not in [None, "DIIS", "CROP"]:
        print(
            "Convergence assist must be either None, 'DIIS', or 'CROP'. Reverted to no convergence acceleration"
        )
        return int(1e10), False

    acc_requested = conv_type is not None
    acc_iteration_start = (
        min(acc_iteration_start + 1, acc_hist_size)
        if acc_hist_size >= acc_iteration_start
        else max(acc_iteration_start + 1, acc_hist_size)
    )

    return (acc_iteration_start, acc_requested)


if __name__ == "__main__":
    pass
