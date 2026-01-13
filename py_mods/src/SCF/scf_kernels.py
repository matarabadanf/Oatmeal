import numpy as np
from numpy.typing import NDArray
from typing import Literal, Optional, Union, Tuple, Sequence
import scipy
from py_mods.src.SCF.linalg import diagonalize_biorthogonal


# --- Real SCF Helper Functions ---
def calc_p_matrix(
    C_matrix: NDArray[np.float64], n_electrons: int
) -> NDArray[np.float64]:
    """
    Calculate RHF density matrix (real).

    Parameters
    ----------
    C_matrix : NDArray[np.float64]
        MO coefficient matrix.
    n_electrons : int
        Electron count.

    Returns
    -------
    P : NDArray[np.float64]
        Density matrix.
    """
    n_occ = n_electrons // 2
    C_occ = C_matrix[:, :n_occ]
    return 2.0 * (C_occ @ C_occ.T)


def calc_g_matrix(
    P_matrix: NDArray[np.float64], eri: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate real G matrix.

    Parameters
    ----------
    P_matrix : NDArray[np.float64]
        Density matrix.
    eri : NDArray[np.float64]
        ERIs.

    Returns
    -------
    G : NDArray[np.float64]
        G matrix.
    """
    J = np.einsum("mnls,ls->mn", eri, P_matrix)
    K = np.einsum("mlns,ls->mn", eri, P_matrix)
    return J - 0.5 * K


def E_0(
    P: NDArray[np.float64], H_core: NDArray[np.float64], F: NDArray[np.float64]
) -> float:
    """
    Calculate real Hartree-Fock energy.

    Parameters
    ----------
    P : NDArray[np.float64]
        Density matrix.
    H_core : NDArray[np.float64]
        Core Hamiltonian.
    F : NDArray[np.float64]
        Fock matrix.

    Returns
    -------
    float
        Electronic energy.
    """
    return 0.5 * np.sum(P * (H_core + F))


def V_NN(
    positions: NDArray[np.float64],
    charges: Union[NDArray[np.int32], Sequence[int]],
    units: Literal["Bohr", "Angstrom"] = "Bohr",
) -> float:
    """
    Calculate nuclear repulsion energy.

    Parameters
    ----------
    positions : NDArray[np.float64]
        Atom coordinates.
    charges : Union[NDArray, Sequence]
        Nuclear charges.
    units : {'Bohr', 'Angstrom'}
        Input units.

    Returns
    -------
    float
        Nuclear repulsion energy.
    """
    energy = np.float64(0.0)
    n_atoms = len(positions)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist > 1e-12:
                energy += (charges[i] * charges[j]) / dist

    if units == "Angstrom":
        energy *= 0.529177249

    return float(energy)


# --- Complex SCF Helper Functions ---


def scale_integrals(
    T: NDArray[np.complex128],
    V: NDArray[np.complex128],
    eri: NDArray[np.complex128],
    theta: float,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Scale integrals by complex-scaling angle theta.

    Parameters
    ----------
    T : NDArray[np.float64]
        Kinetic energy matrix.
    V : NDArray[np.float64]
        Nuclear attraction matrix.
    eri : NDArray[np.float64]
        ERIs.
    theta : float
        Complex scaling angle.

    Returns
    -------
    T_s, V_s, eri_s : Tuple[NDArray[np.complex128], ...]
        Scaled integrals.
    """
    exp_t2 = np.exp(-2j * theta)
    exp_t1 = np.exp(-1j * theta)
    # Ensure output is complex128 even if input is float
    return (
        (T * exp_t2).astype(np.complex128),
        (V * exp_t1).astype(np.complex128),
        (eri * exp_t1).astype(np.complex128),
    )


def guess_density_RHF(
    p_guess: Literal["core", "ones", "IMPORB"],
    dim: int,
    INPORB: Union[NDArray[np.complex128], NDArray[np.float64], None],
) -> NDArray[np.complex128]:
    """
    Generate initial guess density (complex).

    Parameters
    ----------
    dim : int
        Basis dimension.
    method : {'core', 'ones', 'IMPORB'}
        Guess method.
    INPORB : {NDArray[np.complex128], NDArray[np.float64], None}
        Imported guess orbitals.

    Returns
    -------
    P_guess : NDArray[np.complex128]
        Guess density matrix.
    """
    if p_guess == "INPORB":
        assert INPORB is not None, "Empty INPORB alpha for guess"

        assert (
            isinstance(INPORB, np.array) and INPORB.shape == X.shape
        ), f"Wrong type ({type(INPORB)}) or dimensions ({INPORB.shape}) of import guess orbitals, expexted {type(X)} and {X.shape}"

        return np.copy(INPORB.astype(np.complex128))

    elif p_guess == "core":
        return np.zeros((dim, dim), dtype=np.complex128)

    elif p_guess == "ones":
        return np.ones((dim, dim), dtype=np.complex128)

    else:
        raise ValueError("Invalid method. Choose 'core', 'ones' or 'IMPORB'.")


def calc_p_matrix_comp(
    l_matrix: NDArray[np.complex128],
    r_matrix: NDArray[np.complex128],
    n_electrons: int,
    determinant: Optional[NDArray[np.int32]] = None,
    natural_occupation: bool = True,
    mode: Literal["RHF", "UHF"] = "RHF",
) -> NDArray[np.complex128]:
    """
    Calculate complex density matrix.

    P = 2 * sum(r_a * l_a)

    Parameters
    ----------
    l_matrix : NDArray[np.complex128]
        Left MO coefficients.
    r_matrix : NDArray[np.complex128]
        Right MO coefficients.
    n_electrons : int
        Electron count.
    determinant : Optional[NDArray]
        Occupation determinant.
    natural_occupation : bool
        If True, use natural occupation.

    Returns
    -------
    NDArray[np.complex128]
        Density matrix.
    """
    assert (
        n_electrons == sum(determinant) if determinant is not None else True
    ), "n_electrons must match determinant sum"

    if determinant is None:
        raise ValueError("Must provide determinant if natural_occupation is False")
    det_arr = determinant

    if mode == "UHF":
        mask = (det_arr == 1).astype(np.complex128)
        P = np.einsum("ma, a, an -> mn", r_matrix, mask, l_matrix)
    else:
        mask = (det_arr == 2).astype(np.complex128)
        P = 2.0 * np.einsum("ma, a, an->mn", r_matrix, mask, l_matrix)

    return P


def calc_g_matrix_comp(
    P_matrix: NDArray[np.complex128], eri: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """
    Calculate complex G matrix.

    Parameters
    ----------
    P_matrix : NDArray[np.complex128]
        Density matrix.
    eri : NDArray[np.complex128]
        ERIs.

    Returns
    -------
    NDArray[np.complex128]
        G matrix.
    """
    J = np.einsum("mnls,ls->mn", eri, P_matrix)
    K = np.einsum("mlns,ls->mn", eri, P_matrix)
    return J - 0.5 * K


def E_0_comp(
    P: NDArray[np.complex128], H_core: NDArray[np.complex128], F: NDArray[np.complex128]
) -> np.complex128:
    """
    Calculate complex Hartree-Fock energy.

    Parameters
    ----------
    P : NDArray[np.complex128]
        Density matrix.
    H_core : NDArray[np.complex128]
        Core Hamiltonian.
    F : NDArray[np.complex128]
        Fock matrix.

    Returns
    -------
    np.complex128
        Electronic energy.
    """
    return np.sum(P * (H_core + F)) * 0.5


def E_0_unrestricted_comp(
    P_alpha: NDArray[np.complex128],
    P_beta: NDArray[np.complex128],
    H_core: NDArray[np.complex128],
    F_alpha: NDArray[np.complex128],
    F_beta: NDArray[np.complex128],
) -> np.complex128:
    """
    Calculate complex UHF energy.

    Parameters
    ----------
    P_alpha : NDArray[np.complex128]
        Alpha density.
    P_beta : NDArray[np.complex128]
        Beta density.
    H_core : NDArray[np.complex128]
        Core Hamiltonian.
    F_alpha : NDArray[np.complex128]
        Alpha Fock matrix.
    F_beta : NDArray[np.complex128]
        Beta Fock matrix.

    Returns
    -------
    np.complex128
        Total electronic energy.
    """
    E_alpha = np.sum(P_alpha * (H_core + F_alpha))
    E_beta = np.sum(P_beta * (H_core + F_beta))
    return 0.5 * (E_alpha + E_beta)


# --- Convergence Utilities ---


def calc_residual_commutator(
    F: NDArray[np.complex128],
    P: NDArray[np.complex128],
    S: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """
    Calculate residual commutator [F, P]S.

    Parameters
    ----------
    F : NDArray[np.complex128]
        Fock matrix.
    P : NDArray[np.complex128]
        Density matrix.
    S : NDArray[np.float64]
        Overlap matrix.

    Returns
    -------
    NDArray[np.complex128]
        Residual matrix (S P F - F P S).
    """
    return S @ P @ F - F @ P @ S


def calc_diis_extrapolation(
    residuals: Sequence[NDArray[np.complex128]],
    F_guesses: Sequence[NDArray[np.complex128]],
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Calculate DIIS extrapolation.

    Parameters
    ----------
    residuals : Sequence[NDArray]
        History of residuals.
    F_guesses : Sequence[NDArray]
        History of Fock matrices.

    Returns
    -------
    Tuple[NDArray, NDArray]
        Extrapolated (F, r).
    """
    n_guesses = len(residuals)
    eq_sis_dim = n_guesses + 1

    B_matrix = np.zeros((eq_sis_dim, eq_sis_dim), dtype=np.complex128)
    B_matrix[-1, :] = 1
    B_matrix[:, -1] = 1
    B_matrix[-1, -1] = 0

    for i in range(n_guesses):
        for j in range(n_guesses):
            # dot product of flattened arrays
            B_matrix[i, j] = np.dot(residuals[i].ravel(), residuals[j].ravel())

    solution = np.zeros(eq_sis_dim, dtype=np.complex128)
    solution[-1] = 1

    try:
        c = np.linalg.solve(B_matrix, solution)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("DIIS matrix singular")

    # Reconstruct
    coeffs = c[:-1]

    F_conv = np.zeros_like(F_guesses[0])
    r_conv = np.zeros_like(residuals[0])

    for k, coef in enumerate(coeffs):
        F_conv += coef * F_guesses[k]
        r_conv += coef * residuals[k]

    return F_conv, r_conv


def calculate_P_next(
    F: NDArray[np.complex128],
    X: NDArray[np.complex128],
    n_electrons: int,
    det: NDArray[np.int32],
    mode: Literal["RHF", "UHF"] = "RHF",
) -> Tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """
    Calculate the next density matrix P_{n+1} given Fock matrix F_n and transformation matrix X.

    Parameters
    ----------
    F_0 : NDArray[np.float64] of dimension (n, n)
        Fock matrix at iteration n.
    X : NDArray[np.float64] of dimension (n, n)
        Transformation matrix.
    n_electrons : int
        Number of electrons.

    Returns
    -------
    Tuple containing:
        - P_1 (NDArray[np.float64][n, n]): Next density matrix
        - C_munu (NDArray[np.float64][n, n]): Molecular orbital coefficients.
        - e_values (NDArray[np.float64][n, n]): Orbital energies.

    Notes
    ------
    Diagonalization algorithm used is np.linalg.eigh due to the matrix being symmetric.
    """
    F = F.reshape(X.shape)

    F_prime = X @ F @ X.T

    mat_norm = np.linalg.norm(F_prime)

    # print(mat_norm)

    F_prime /= mat_norm  # divide by the norm to avoid numerical instability

    e_values, C_prime, L_prime, R_prime, LFR = diagonalize_biorthogonal(F_prime)

    e_values *= mat_norm

    diag_LFR = np.diag(np.diagonal(LFR))

    # try:
    assert np.allclose(
        LFR, diag_LFR, atol=1e-6
    ), "Matrix product L' @ F' @ R' is not diagonal"

    # except AssertionError:
    #     plot_map(LFR-diag_LFR)

    # Obtain untransformed MO coefficients
    C_munu = X @ C_prime
    L_munu = L_prime @ X
    R_munu = X @ R_prime

    # Build new density matrix
    P_LR = calc_p_matrix_comp(L_munu, R_munu, n_electrons, det, mode=mode)
    P_RR = np.copy(P_LR)

    return P_LR, e_values, C_munu, R_munu, L_munu, P_RR, C_prime


def calculate_P_next_2(
    F: NDArray[np.complex128],
    S: NDArray[np.complex128],
    n_electrons: int,
    det: NDArray[np.int32],
    mode: Literal["RHF", "UHF"] = "RHF",
) -> Tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """
    Calculate the next density matrix P_{n+1} given Fock matrix F_n and transformation matrix X.

    Parameters
    ----------
    F_0 : NDArray[np.float64] of dimension (n, n)
        Fock matrix at iteration n.
    X : NDArray[np.float64] of dimension (n, n)
        Transformation matrix.
    n_electrons : int
        Number of electrons.

    Returns
    -------
    Tuple containing:
        - P_1 (NDArray[np.float64][n, n]): Next density matrix
        - C_munu (NDArray[np.float64][n, n]): Molecular orbital coefficients.
        - e_values (NDArray[np.float64][n, n]): Orbital energies.

    Notes
    ------
    Diagonalization algorithm used is np.linalg.eigh due to the matrix being symmetric.
    """
    e_values, C_munu = scipy.linalg.eigh(F, S)

    # Build new density matrix
    P_LR = calc_p_matrix_comp(C_munu.T, C_munu, n_electrons, det, mode=mode)

    return P_LR, e_values, C_munu


# --- UHF helper functions ---
def calculate_unrestricted_F_and_r_comp(
    P_alpha: NDArray[np.complex128],
    P_beta,
    S: NDArray[np.complex128],
    H_core: NDArray[np.complex128],
    eri: NDArray[np.complex128],
) -> Tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """
    Calculate Fock matrix F and residual r from P.

    Parameters
    ----------
    P : NDArray[np.float64] of dimension (n, n)
        Density matrix.
    S : NDArray[np.float64] of dimension (n, n)
        Overlap matrix.
    H_core : NDArray[np.float64] of dimension (n, n)
        Core Hamiltonian matrix.
    eri : NDArray[np.float64] of dimension (n, n, n, n)
        Electron repulsion integrals.

    Returns
    -------
    Tuple containing:
        - F (NDArray[np.float64][n, n]): Fock matrix.
        - r (NDArray[np.float64][n, n]): Residual matrix.
    """
    G_alpha, G_beta = calc_g_matrix_spin_comp(P_alpha, P_beta, eri)
    F_alpha = H_core + G_alpha
    F_beta = H_core + G_beta
    r_alpha = calc_residual_commutator(F_alpha, P_alpha, S)
    r_beta = calc_residual_commutator(F_beta, P_beta, S)

    return F_alpha.flatten(), r_alpha.flatten(), F_beta.flatten(), r_beta.flatten()


def calc_g_matrix_spin_comp(
    P_alpha, P_beta, eri
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:

    P_total = P_alpha + P_beta
    # J from total density
    J = np.einsum("mnsl, ls -> mn", eri, P_total)
    # K from same-spin density
    K_alpha = np.einsum("mlns, ls -> mn", eri, P_alpha)
    K_beta = np.einsum("mlns, ls -> mn", eri, P_beta)

    G_alpha = J - K_alpha
    G_beta = J - K_beta

    return G_alpha, G_beta


if __name__ == "__main__":
    pass
