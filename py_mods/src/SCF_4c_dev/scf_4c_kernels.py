from typing import Tuple, Union, Literal, List
from py_mods.src.integrals.UncontractedBasisSet import UncontractedBasisSet, ERIs_Uncontracted
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
    det: NDArray[np.int32],
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
        e_values, vl, C_prime = scipy.linalg.eig(F_prime, left=True, right=True)  # type: ignore
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


def full_eri_from_Uncontracted_Basis(UBS: UncontractedBasisSet) -> NDArray[np.float64]:
    """
    Compute full ERI tensor from uncontracted basis set.

    Parameters
    ----------
    UBS : UncontractedBasisSet
        The uncontracted basis set.

    Returns
    -------
    eri_tensor : NDArray[np.float64]
        The full ERI tensor.
    """
    eri_tensor = ERIs_Uncontracted(UBS)

    return eri_tensor



def eri_classified(eri: NDArray[np.float64], nL: int) -> NDArray[np.float64]:
    """
    Filter ERI tensor to keep only (LL|LL), (SS|LL), (LL|SS), (SS|SS) terms.

    Parameters
    ----------
    eri : NDArray[np.float64]
        The full electron repulsion integrals tensor.
    nL : int
        Number of large component basis functions.

    Returns
    -------
    eri_classess : NDArray[np.float64]
        The classified ERI tensor.
    """
    eri_classess = np.zeros_like(eri, dtype=np.float64)

    eri_classess[:nL, :nL, :nL, :nL] = eri[:nL, :nL, :nL, :nL]  # LL-LL block
    eri_classess[:nL, :nL, nL:, nL:] = eri[:nL, :nL, nL:, nL:]  # LL-SS block
    eri_classess[nL:, nL:, :nL, :nL] = eri[nL:, nL:, :nL, :nL]  # SS-LL block
    eri_classess[nL:, nL:, nL:, nL:] = eri[nL:, nL:, nL:, nL:]  # SS-SS block

    return eri_classess



def occupation_4c(
    nS: int,
    nL: int,
    n_electrons: int,
    electronic_occ_det: Union[None, NDArray[np.int32]] = None,
) -> NDArray[np.int32]:
    """
    Build the occupation vector for 4c calculations.

    Parameters
    ----------
    nS : int
        Number of small component basis functions.
    nL : int
        Number of large component basis functions.
    n_electrons : int
        Number of electrons.
    electronic_occ_det : Union[None, NDArray[np.int32]], optional
        Occupation determinant for electronic states (positive energy solutions). Defaults to None.

    Returns
    -------
    occ : NDArray[np.int32]
        Occupation determinant for electronic and positronic states.
    """
    occ = np.zeros(2 * (nS + nL), dtype=np.int32)

    n_positron_states = 2 * nS

    if electronic_occ_det is None:
        occ[n_positron_states : n_positron_states + n_electrons] = 1
    else:
        assert (
            len(electronic_occ_det) == 2 * nL
        ), "Length of electronic occupation array must be equal to 2*nL"
        assert (
            sum(electronic_occ_det) == n_electrons
        ), "Sum of electronic occupation array must be equal to n_electrons"
        occ[n_positron_states:] = electronic_occ_det

    return occ



def g_matrix_4c(
    P: NDArray[np.complex128], eri: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """
    Construct G matrix (J-K) from the density matrix.

    Parameters
    ----------
    P : NDArray[np.complex128]
        Density matrix.
    eri : NDArray[np.complex128]
        Electron repulsion integrals tensor.

    Returns
    -------
    G_full : NDArray[np.complex128]
        Full G matrix.
    """
    n_bas = P.shape[0]
    n_bas_half = n_bas // 2

    P_aa = P[0:n_bas_half, 0:n_bas_half]
    P_bb = P[n_bas_half:n_bas, n_bas_half:n_bas]
    P_ab = P[0:n_bas_half, n_bas_half:n_bas]
    P_ba = P[n_bas_half:n_bas, 0:n_bas_half]

    P_total = P_aa + P_bb
    J = np.einsum("mnsl, ls -> mn", eri, P_total)

    K_aa = np.einsum("psrq, sr -> pq", eri, P_aa)
    K_bb = np.einsum("psrq, sr -> pq", eri, P_bb)
    K_ab = np.einsum("psrq, sr -> pq", eri, P_ab)
    K_ba = np.einsum("psrq, sr -> pq", eri, P_ba)

    G_aa = J - K_aa
    G_bb = J - K_bb
    G_ab = -K_ab
    G_ba = -K_ba

    G_full = np.zeros((n_bas, n_bas), dtype=np.complex128)
    # And we fill the matrix by blocks
    G_full[0:n_bas_half, 0:n_bas_half] = G_aa
    G_full[n_bas_half:n_bas, n_bas_half:n_bas] = G_bb
    G_full[0:n_bas_half, n_bas_half:n_bas] = G_ab
    G_full[n_bas_half:n_bas, 0:n_bas_half] = G_ba

    return G_full



def scf_iteration(
    F_1: NDArray[np.complex128], X: NDArray[np.complex128], det: NDArray[np.int32]
) -> Tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """
    Perform a single SCF iteration step.

    Parameters
    ----------
    F_1 : NDArray[np.complex128]
        Current Fock matrix.
    X : NDArray[np.complex128]
        Transformation matrix.
    total_occ_det : NDArray[np.int32]
        Occupation determinant.

    Returns
    -------
    e1 : NDArray[np.complex128]
        Eigenvalues (orbital energies).
    w1 : NDArray[np.complex128]
        Eigenvectors in orthogonal basis.
    F_p1 : NDArray[np.complex128]
        Transformed Fock matrix.
    P_1 : NDArray[np.complex128]
        Updated density matrix.
    """
    F_p1 = X.conj().T @ F_1 @ X

    e1, w1 = np.linalg.eigh(F_p1)

    idx = np.argsort(e1)
    e1 = e1[idx]
    w1 = w1[:, idx]

    c_alpha_beta1 = X @ w1
    P_1 = calc_p_matrix_comp(c_alpha_beta1.conj().T, c_alpha_beta1, det)

    return e1, w1, F_p1, P_1



def scf_steps(
    n_steps: int,
    H_core: NDArray[np.complex128],
    eri: NDArray[np.complex128],
    X: NDArray[np.complex128],
    det: NDArray[np.int32],
) -> List[float]:
    """
    For loop that wraps scf iterations.

    Parameters
    ----------
    n_steps : int
        Number of steps to run.
    H_core : NDArray[np.complex128]
        Core Hamiltonian matrix.
    eri : NDArray[np.complex128]
        Electron repulsion integrals tensor.
    X : NDArray[np.complex128]
        Transformation matrix.
    total_occ_det : NDArray[np.int32]
        Occupation determinant.

    Returns
    -------
    energy_step : List[float]
        Energies at each iteration step.
    """
    energy_step = []
    P_old = np.zeros_like(H_core)

    for i in range(n_steps):
        if i == 0:
            G_new = np.zeros_like(H_core)
        else:
            G_new = g_matrix_4c(P_old, eri)

        if i > 0:
            e_scf = np.linalg.trace(P_old @ H_core + 0.5 * P_old @ G_new)
            energy_step.append(e_scf.real)

        F_new = H_core + G_new

        e_new, w_new, F_p_new, P_2 = scf_iteration(F_new, X, det)

        if i == 0:
            e_scf = np.linalg.trace(P_2 @ H_core)
            energy_step.append(e_scf.real)

        P_old = P_2

    return energy_step


# -------------------------------------------------------------
#  CS-4c-KU-SCF Initialization Functions
# -------------------------------------------------------------



def calc_diis_extrapolation_4c(
    residuals: List[NDArray[np.complex128]],
    F_guesses: List[NDArray[np.complex128]],
    theta: float,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    n_guesses = len(residuals)
    eq_sis_dim = n_guesses + 1

    B_matrix = np.zeros((eq_sis_dim, eq_sis_dim), dtype=np.complex128)
    B_matrix[-1, :] = 1
    B_matrix[:, -1] = 1
    B_matrix[-1, -1] = 0

    for i in range(n_guesses):
        for j in range(n_guesses):
            if theta == 0.0:
                # We have to use complex conjugation because the DF Hamiltonian is
                # complex. In the case of the NR case we could just do the scalar
                # product since the hamiltonian is real so we could get away
                # with using np.dot in both cases.
                B_matrix[i, j] = np.vdot(residuals[i].ravel(), residuals[j].ravel())
            else:
                # c-norm metric inner product for complex scaling
                B_matrix[i, j] = np.dot(residuals[i].ravel(), residuals[j].ravel())

    solution = np.zeros(eq_sis_dim, dtype=np.complex128)
    solution[-1] = 1

    try:
        c = np.linalg.solve(B_matrix, solution)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("DIIS matrix singular")

    coeffs = c[:-1]

    F_conv = np.zeros_like(F_guesses[0])
    r_conv = np.zeros_like(residuals[0])

    for k, coef in enumerate(coeffs):
        F_conv += coef * F_guesses[k]
        r_conv += coef * residuals[k]

    return F_conv, r_conv




# -------------------------------------------------------------
#  CS-4c-KU-SCF Update Functions
# -------------------------------------------------------------

