import numpy as np
from numpy.typing import NDArray
from typing import Literal, Optional, Union, Tuple, Sequence, Dict

from py_mods.src.SCF.plot_utilities import plot_map

# --- Linalg Utilities ---

def transformation_matrix(
    S_munu: NDArray[np.complex128], 
    method: Literal['canonical', 'symmetric'] = 'symmetric', 
    verbose: bool = False
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
    assert method in ['canonical', 'symmetric'], "method must be 'canonical' or 'symmetric'"
    
    # diagonalize U.T @ S @ U = s
    s, U = np.linalg.eigh(S_munu)
    s_root = np.diag(1.0 / np.sqrt(s))

    if method == 'symmetric':
        X = U @ s_root @ U.T
    elif method == 'canonical':
        X = U @ s_root
    
    transformed = X.T @ S_munu @ X
    
    if verbose:
        print(transformed)

    # Use identity matrix of correct size for check
    assert np.allclose(transformed, np.eye(len(S_munu)), atol=1e-7), "Transformation failed"

    return X


# --- Real SCF Helper Functions ---

def validate_determinant(
    n_electrons: int,
    determinant: Union[int, NDArray[np.int32], None],
    expected_dim: int
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
            raise TypeError('determinant must be -1, None or a numpy array')

    if not isinstance(determinant, np.ndarray):
        raise TypeError('determinant must be a numpy.ndarray when not -1/None')

    natural_occupation = False
    det_arr = determinant.astype(np.int32)

    if int(np.sum(det_arr)) != n_electrons:
        raise ValueError(f'determinant sum ({int(np.sum(det_arr))}) != n_electrons ({n_electrons})')

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
            beta_det  = np.zeros(expected_dim, dtype=np.int32)     

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
            raise TypeError('determinant must be -1, None or a tuple of arrays')

    if not isinstance(determinants, (list, tuple)):
        raise TypeError('determinant must be a list or tuple of arrays')

    natural_occupation = False
    
    alpha_in = np.array(determinants[0], dtype=np.int32)
    beta_in = np.array(determinants[1], dtype=np.int32)

    alpha_det = np.zeros(expected_dim, dtype=np.int32)
    beta_det = np.zeros(expected_dim, dtype=np.int32)

    alpha_det[:len(alpha_in)] = alpha_in
    beta_det[:len(beta_in)] = beta_in

    total_elec = np.sum(alpha_det) + np.sum(beta_det)
    assert total_elec == n_electrons, f"Occupation sum ({total_elec}) != n_electrons ({n_electrons})"
    
    valid, _ = check_unpaired(alpha_det, beta_det, multiplicity)
    assert valid, f"Multiplicity mismatch."
    
    return alpha_det, beta_det, natural_occupation

def check_unpaired(
    alpha_det: NDArray[np.int32],
    beta_det: NDArray[np.int32],
    multiplicity: int
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

def calc_p_matrix(
    C_matrix: NDArray[np.float64], 
    n_electrons: int
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
    P_matrix: NDArray[np.float64], 
    eri: NDArray[np.float64]
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
    J = np.einsum('mnls,ls->mn', eri, P_matrix)
    K = np.einsum('mlns,ls->mn', eri, P_matrix)
    return J - 0.5 * K

def E_0(
    P: NDArray[np.float64], 
    H_core: NDArray[np.float64], 
    F: NDArray[np.float64]
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
    units: Literal['Bohr', 'Angstrom'] = 'Bohr'
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
    energy = np.float64(0.)
    n_atoms = len(positions)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist > 1e-12:
                energy += (charges[i] * charges[j]) / dist
    
    if units == 'Angstrom':
        energy *= 0.529177249
    
    return float(energy)


# --- Complex SCF Helper Functions ---

def scale_integrals(
    T: NDArray[np.complex128], 
    V: NDArray[np.complex128], 
    eri: NDArray[np.complex128], 
    theta: float
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
    return (T * exp_t2).astype(np.complex128), \
           (V * exp_t1).astype(np.complex128), \
           (eri * exp_t1).astype(np.complex128)

def guess_density_RHF(p_guess: Literal['core', 'ones', 'IMPORB'], dim, INPORB=None) -> NDArray[np.complex128]:
    """
    Generate initial guess density (complex).

    Parameters
    ----------
    dim : int
        Basis dimension.
    method : {'core', 'ones'}
        Guess method.

    Returns
    -------
    P_guess : NDArray[np.complex128]
        Guess density matrix.
    """
    if p_guess == 'INPORB':
        assert INPORB is not None, 'Empty INPORB alpha for guess'

        assert isinstance(INPORB, np.array) and INPORB.shape == X.shape, f'Wrong type ({type(INPORB)}) or dimensions ({INPORB.shape}) of import guess orbitals, expexted {type(X)} and {X.shape}'
        return np.copy(INPORB) 

    elif p_guess == 'core':
        return np.zeros((dim, dim), dtype=np.complex128)
    
    elif p_guess == 'ones':
        return np.ones((dim, dim), dtype=np.complex128)
    
    else:
        raise ValueError("Invalid method. Choose 'core', 'ones' or 'IMPORB'.")

def diagonalize_biorthogonal(F_prime: NDArray[np.complex128]) -> Tuple[
    NDArray[np.complex128], 
    NDArray[np.complex128], 
    NDArray[np.complex128], 
    NDArray[np.complex128], 
    NDArray[np.complex128]
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

    R_prime, L_prime, e_values, C_prime = _diagonalize_gram(F_prime)

    LR = L_prime @ R_prime
    
    LFR = L_prime @ F_prime @ R_prime

    diag_LFR = np.diag(np.diagonal(LFR))

    # assert np.allclose(LFR, diag_LFR, atol=1E-8), "Matrix product L' @ F' @ R' is not diagonal"
    # assert np.allclose(LR, np.eye(len(LR)), atol=1E-8)

    return e_values, C_prime, L_prime, R_prime, LFR

def _diagonalize_gram(F_prime):
    e_values, C_prime = np.linalg.eig(F_prime)

    # Sort by real part of eigenvalues (standard for SCF stability)
    idx = e_values.argsort()
    e_values = e_values[idx]
    C_prime = C_prime[:, idx]

    degeneracies = count_degen(e_values)
    C_norm = orthonormalize_solutions(e_values, C_prime, degeneracies)

    R_prime = np.copy(C_norm)
    L_prime = np.copy(C_norm.T) # Biorthogonal: L = C^{-1}, here approximated via ortho logic

    return R_prime, L_prime, e_values, C_prime

def count_degen(e_orb: NDArray[np.complex128]) -> Dict[complex, int]:
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
        val = np.round(item, 5)
        counts[val] = counts.get(val, 0) + 1
    return counts

def c_projector(u: NDArray[np.complex128], v: NDArray[np.complex128]) -> NDArray[np.complex128]:
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
    num = np.dot(v, u) # c-product (no conjugation)
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
        Input vectors (rows).

    Returns
    -------
    NDArray[np.complex128]
        Orthonormalized vectors.
    """
    size, dim = v.shape
    e = np.zeros((size, dim), dtype=np.complex128)
    u = np.zeros((size, dim), dtype=np.complex128)

    u[0] = v[0].copy()
    e[0] = v[0] / c_norm(v[0])

    for i in range(1, size):
        v_i = v[i]
        proj = np.zeros(dim, dtype=np.complex128)
        for j in range(i):
            proj += c_projector(u[j], v_i) 
        
        u[i] = v_i - proj
        e[i] = u[i] / c_norm(u[i])
    
    return e

def orthonormalize_solutions(
    e_orb: NDArray[np.complex128], 
    C: NDArray[np.complex128], 
    deg_dict: Dict[complex, int]
) -> NDArray[np.complex128]:
    """
    Orthonormalize degenerate solutions.

    Parameters
    ----------
    e_orb : NDArray[np.complex128]
        Orbital energies.
    C : NDArray[np.complex128]
        Eigenvectors.
    deg_dict : Dict[complex, int]
        Degeneracy map.

    Returns
    -------
    NDArray[np.complex128]
        Normalized eigenvectors.
    """
    degeneracies = deg_dict.copy()
    C_orth = C.copy()
    
    n_orb = len(e_orb)
    i = 0
    while i < n_orb:
        val = np.round(e_orb[i], 10)
        deg = degeneracies.get(val, 0)
        
        if deg > 1:
            # Extract subspace
            v = C[:, i : i + deg].T
            # Orthogonalize
            e = gram_schmidt(v)
            # Place back
            C_orth[:, i : i + deg] = e.T
            i += deg
        else:
            i += 1

    # Normalize all columns
    norms = np.array([c_norm(col) for col in C_orth.T])
    return C_orth / norms

def calc_p_matrix_comp(
    l_matrix: NDArray[np.complex128], 
    r_matrix: NDArray[np.complex128], 
    n_electrons: int,
    determinant: Optional[NDArray[np.int32]] = None,
    natural_occupation: bool = True,
    mode: Literal['RHF', 'UHF'] = 'RHF'
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
    assert n_electrons == sum(determinant) if determinant is not None else True, "n_electrons must match determinant sum"

    if determinant is None:
            raise ValueError("Must provide determinant if natural_occupation is False")
    det_arr = determinant

    if mode == 'UHF':
        mask = (det_arr == 1).astype(np.complex128)
        P = np.einsum('ma, a, an -> mn', r_matrix, mask, l_matrix)
    else:
        mask = (det_arr == 2).astype(np.complex128)
        P = 2.0 * np.einsum('ma, a, an->mn', r_matrix, mask, l_matrix )

    return P

def calc_g_matrix_comp(
    P_matrix: NDArray[np.complex128], 
    eri: NDArray[np.complex128]
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
    J = np.einsum('mnls,ls->mn', eri, P_matrix)
    K = np.einsum('mlns,ls->mn', eri, P_matrix)
    return J - 0.5 * K

def E_0_comp(
    P: NDArray[np.complex128],
    H_core: NDArray[np.complex128],
    F: NDArray[np.complex128]
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
    F_beta: NDArray[np.complex128]
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
    E_beta  = np.sum(P_beta  * (H_core + F_beta))
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
    F_guesses: Sequence[NDArray[np.complex128]]
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
            B_matrix[i,j] = np.dot(residuals[i].ravel(), residuals[j].ravel())
    
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

def calculate_P_next(F: NDArray[np.complex128], X: NDArray[np.complex128], n_electrons: int, det: NDArray[np.int32], mode: Literal['RHF', 'UHF'] = 'RHF') -> Tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
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

    F_prime /= mat_norm #divide by the norm to avoid numerical instability

    e_values, C_prime, L_prime, R_prime, LFR = diagonalize_biorthogonal(F_prime)

    e_values *= mat_norm

    # try:
    #     assert np.allclose(LFR, diag_LFR, atol=1E-6), "Matrix product L' @ F' @ R' is not diagonal"
    
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


# --- UHF helper functions ---
def calculate_unrestricted_F_and_r_comp(P_alpha: NDArray[np.complex128], P_beta, S: NDArray[np.complex128], H_core: NDArray[np.complex128], eri: NDArray[np.complex128]) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
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

def calc_g_matrix_spin_comp(P_alpha, P_beta, eri) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:

    P_total = P_alpha + P_beta
    # J from total density
    J = np.einsum('mnsl, ls -> mn', eri, P_total)
    # K from same-spin density
    K_alpha = np.einsum('mlns, ls -> mn', eri, P_alpha)
    K_beta  = np.einsum('mlns, ls -> mn', eri, P_beta)

    G_alpha = J - K_alpha
    G_beta  = J - K_beta

    return G_alpha, G_beta
