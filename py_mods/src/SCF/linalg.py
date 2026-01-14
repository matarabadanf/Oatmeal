import numpy as np
from numpy.typing import NDArray
from typing import Literal, Union, Tuple, Dict
from scipy.linalg import block_diag


def transformation_matrix(
    S_munu: NDArray[np.complex128],
    method: Literal["canonical", "symmetric"] = "symmetric",
    verbose: bool = False,
) -> NDArray[np.float64]:
    """
    Calculate basis transformation matrix X.

    Parameters
    ----------
    S_munu : NDArray[np.float64]
        Overlap matrix.
    method : {'canonical', 'symmetric'}
        Orthogonalization method.
    verbose : bool
        If True, print transformed matrix.

    Returns
    -------
    X : NDArray[np.float64]
        Transformation matrix.
    """
    assert method in [
        "canonical",
        "symmetric",
    ], "method must be 'canonical' or 'symmetric'"

    # diagonalize U.T @ S @ U = s
    s, U = np.linalg.eigh(S_munu)
    s_root = np.diag(1.0 / np.sqrt(s))

    if method == "symmetric":
        X = U @ s_root @ U.T
    elif method == "canonical":
        X = U @ s_root

    transformed = X.T @ S_munu @ X

    if verbose:
        print(transformed)

    # Use identity matrix of correct size for check
    assert np.allclose(
        transformed, np.eye(len(S_munu)), atol=1e-7
    ), "Transformation failed"

    return X


def diagonalize_biorthogonal(
    F_prime: NDArray[np.complex128],
) -> Tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """
    Diagonalize matrix using c-norm solution. Yields orthonormalized basis.

    Parameters
    ----------
    F_prime : NDArray[np.complex128]
        Transformed Fock matrix.

    Returns
    -------
    e_values : NDArray[np.complex128]
        Eigenvalues.
    C_prime : NDArray[np.complex128]
        Right eigenvectors.
    L_prime : NDArray[np.complex128]
        Left eigenvectors.
    R_prime : NDArray[np.complex128]
        Right eigenvectors (copy).
    LFR : NDArray[np.complex128]
        Diagonal matrix (L @ F @ R).
    """

    R_prime, _, e_values, C_prime = _diagonalize_gram(F_prime, None)

    # e_values, R_prime = np.linalg.eigh(F_prime)

    C_prime = R_prime
    L_prime = R_prime.T

    LR = L_prime @ R_prime

    LFR = L_prime @ F_prime @ R_prime

    diag_LFR = np.diag(np.diagonal(LFR))

    # assert np.allclose(LFR, diag_LFR, atol=1E-8), "Matrix product L' @ F' @ R' is not diagonal"
    # assert np.allclose(LR, np.eye(len(LR)), atol=1E-8)

    return e_values, C_prime, L_prime, R_prime, LFR


def _diagonalize_gram(
    F_prime: Union[NDArray[np.complex128], NDArray[np.float64]],
    solver: Literal["eig", "eigh"],
) -> Tuple:

    use_eigh = (
        np.allclose(F_prime.T, F_prime)
        and np.linalg.norm(F_prime.imag) < 1e-14
        or solver == "eigh"
    )

    if use_eigh:
        e_values, C_prime = np.linalg.eigh(F_prime)

    else:
        e_values, C_prime = np.linalg.eig(F_prime)

    # Sort
    idx = e_values.argsort()
    e_values = e_values[idx]
    C_prime = C_prime[:, idx]

    if not use_eigh:
        C_norm = orthonormalize_solutions2(e_values, C_prime)
    else:
        C_norm = C_prime

    C_norm_p = C_norm

    R_prime = np.copy(C_norm_p)
    L_prime = np.copy(C_norm_p.T)

    return R_prime, L_prime, e_values, C_prime


def count_degen2(e_orb: NDArray[np.complex128]) -> Dict[complex, int]:
    """
    Count eigenvalue degeneracies.

    Parameters
    ----------
    e_orb : NDArray[np.complex128]
        Orbital energies.

    Returns
    -------
    Dict[complex, int]
        Degeneracy map.
    """
    counts: Dict[complex, int] = {}
    for item in e_orb:
        val = np.round(item, 10)
        counts[val] = counts.get(val, 0) + 1

    keys = np.array(list(counts.keys()))
    degs = np.array(list(counts.values()))

    return keys, degs


def c_projector(
    u: NDArray[np.complex128], v: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """
    Calculate c-projection of v onto u.

    Parameters
    ----------
    u : NDArray[np.complex128]
        Target vector.
    v : NDArray[np.complex128]
        Source vector.

    Returns
    -------
    NDArray[np.complex128]
        Projected vector.
    """
    num = np.dot(v, u)  # c-product (no conjugation)
    den = np.dot(u, u)
    if abs(den) < 1e-14:
        return np.zeros_like(v)
    return (num / den) * u


def c_norm(u: NDArray[np.complex128]) -> np.complex128:
    """
    Calculate c-norm (sqrt(u.u)).

    Parameters
    ----------
    u : NDArray[np.complex128]
        Input vector.

    Returns
    -------
    np.complex128
        Calculated norm.
    """
    return np.sqrt(np.dot(u, u))


def gram_schmidt(v: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Apply Gram-Schmidt orthonormalization.

    Parameters
    ----------
    v : NDArray[np.complex128]
        Input vectors as columns.

    Returns
    -------
    NDArray[np.complex128]
        Orthonormalized vectors as columns.
    """
    # initialize the space
    dim, vectors = v.shape
    e = np.zeros((dim, vectors), dtype=np.complex128)
    u = np.zeros((dim, vectors), dtype=np.complex128)

    # obtain first basis vector
    u[:, 0] = v[:, 0].copy()
    e[:, 0] = v[:, 0] / c_norm(v[:, 0])

    # for every remaining vector, subtract the projection of the previous basis vectors
    for i in range(1, vectors):
        v_i = v[:, i]
        proj = np.zeros(dim, dtype=np.complex128)
        for j in range(i):
            proj += c_projector(u[:, j], v_i)
        u[:, i] = v_i - proj

        # and normalize to get the next basis vector
        e[:, i] = u[:, i] / c_norm(u[:, i])

    return e


def modified_gram_schmidt(v: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Apply Modified Gram-Schmidt orthonormalization.

    Parameters
    ----------
    v : NDArray[np.complex128]
        Input vectors as columns.

    Returns
    -------
    NDArray[np.complex128]
        Orthonormalized vectors.
    """

    # exctract number of dimensions and of vectors (array in column form has a shape ())
    dim, vectors = v.shape
    e = np.zeros((dim, vectors), dtype=np.complex128)

    # copy the original vectors to modify them at each step
    v_copy = v.astype(np.complex128)

    for _ in range(2):
        for i in range(vectors):
            # normalize the i-th basis vector
            threshold = 1e-14
            v_copy[:, i].imag[np.abs(v_copy[:, i].imag) < threshold] = 0.0

            # normalize
            e[:, i] = v_copy[:, i] / c_norm(v_copy[:, i])

            # remove component in this direction of all the remaining vectors
            for j in range(i + 1, vectors):
                proj = c_projector(e[:, i], v_copy[:, j])
                v_copy[:, j] -= proj

    return e


def orthonormalize_solutions2(eval, evec):
    ener, n_deg = count_degen2(eval)
    # print("Energy snd degeneracy in Fock eigenvalues:")

    # for e, d in zip(ener, n_deg):
    # print(f"{e:.6f}   {d}")

    copyy = np.copy(evec)

    distinct_evals = len(n_deg)
    for i in range(distinct_evals):
        # print(f"Processing degenerate set {ener[i]} with degeneracy {n_deg[i]}")
        if n_deg[i] != 1:
            # print(n_deg[0:i-1])
            idx_str = sum(n_deg[0:i])
            # print('Starting on column ', idx_str)
            idx_end = idx_str + n_deg[i]
            # print(evec[:, idx_str:idx_end].real)
            v = modified_gram_schmidt(
                evec[:, idx_str:idx_end]
            )  # WIWOWIWOWIWO the issue was here looks like
            copyy[:, idx_str:idx_end] = v

        # else:
        # print('Nothing to do\n')

    # reorthonormalize all
    copyy = modified_gram_schmidt(copyy)

    # set to zero imaginary part below threshold

    norms = np.array([c_norm(col) for col in copyy.T])

    return copyy / norms


def canonicalize(
    C_munu: NDArray[np.complex128],
    F: NDArray[np.complex128],
    n_occ: int,
) -> NDArray[np.complex128]:
    F_mo = np.einsum("mu, uv, vn -> mn", C_munu.conj().T, F, C_munu)

    F_vv = F_mo[n_occ:, n_occ:]

    eps_virt, U_virt = np.linalg.eigh(F_vv)

    U_full = block_diag(np.eye(n_occ), U_virt)

    C_new = C_munu @ U_full
    eps_occ = np.diag(F_mo)[:n_occ]
    epsilon_new = np.concatenate([eps_occ, eps_virt])

    return C_new, epsilon_new


def sign_convention(matrix):
    for i, col in enumerate(matrix.T):
        if np.abs(max(col.real)) < np.abs(min(col.real)):
            matrix[:, i] *= -1
    return matrix


if __name__ == "__main__":
    pass
