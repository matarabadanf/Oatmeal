import numpy as np
from numpy.typing import NDArray
from typing import Literal, Optional, Union, Tuple, Sequence

# --- Linalg Utilities ---

def kroeneker_delta(i: int, j: int) -> int:
    """Kronecker delta function.

    Parameters
    ----------
    i : int
        First index.
    j : int
        Second index.

    Returns
    -------
    int
        1 if i equals j, otherwise 0.
    """
    return 1 if i == j else 0

def equiv_matrix(
    prev: NDArray[np.float64], 
    curr: NDArray[np.float64], 
    threshold=0.0000001
) -> bool:
    """
    Check equality between two arrays. Uses the max difference as metric. 

    Parameters
    ------
    prev : NDArray[np.float64], shape (n, n)
        Previous array to compare.
    curr : NDArray[np.float64], shape (n, n)
        Current array to compare.
    threshold : float
        Convergence threshold.

    Returns
    ------
    converged : bool
        True if the maximum difference between arrays is less than threshold.
    """
    return np.max(np.abs(curr - prev)) < threshold


def is_diagonal(matrix: NDArray, atol: float = 1e-8) -> bool:
    """Return True if matrix is diagonal (within numerical tolerance).

    Parameters
    ----------
    matrix : NDArray, shape (n, n)
        Matrix to check.

    Returns
    -------
    is_diag : bool
        True if off-diagonal elements are (near) zero.
    """
    dim = len(matrix)
    ty = type(matrix[0,0])

    reference = np.zeros(dim, dtype=ty)

    for i in range(dim):
        matrix[i,i] = 0

    if equiv_matrix(matrix, reference):
        return True
    
    return False


def transformation_matrix(
    S_munu: NDArray[np.float64], 
    method: Literal['canonical', 'symmetric'] ='symmetric', 
    verbose: bool = False
) -> NDArray[np.float64]:
    """
    Calculate the basis transformation matrix X.

    Uses symmetric or canonical orthogonalization.
    - symmetric: X = S^{-1/2} = U @ s^{-1/2} @ U^T
    - canonical: X = U @ s^{-1/2}

    Parameters
    ----------
    S_munu : NDArray[np.float64], shape (n, n)
        Overlap matrix.
    method : {'canonical', 'symmetric'}
        Orthogonalization method.
    verbose : bool
        If True prints the transformed matrix.

    Returns
    -------
    X : NDArray[np.float64], shape (n, n)
        Transformation matrix.
    """
    assert method in ['canonical', 'symmetric'], "method must be either 'canonical' or 'symmetric'"
    dim = len(S_munu)

    # diagonalize U.T @ S @ U = s
    s, U = np.linalg.eigh(S_munu)

    s_root = np.diag(1 / np.sqrt(s))

    if method == 'symmetric':
        X = U @ s_root @ U.T
    elif method == 'canonical':
        X = U @ s_root
    
    # transformation matrix test
    transformed = X.T @ S_munu @ X

    np.savetxt(f'trans.dat', transformed)
    
    if verbose:
        print(transformed)

    assert equiv_matrix(transformed, np.identity(len(S_munu))), "transformation matrix calculation failed"

    return X


# --- Real SCF Helper Functions ---

def validate_determinant(
    n_electrons: int,
    determinant: Union[int, NDArray[np.int32], None],
    expected_dim: int
) -> Tuple[NDArray[np.int32], bool]:
    """
    Validate or construct an occupation-number determinant.

    Parameters
    ----------
    n_electrons : int
        Total number of electrons.
    determinant : int or NDArray[np.int32] or None
        If -1 (or None) build a default RHF occupation vector (2,2,...,0).
        If an ndarray is provided it must sum to n_electrons.
    expected_dim : int
        Expected length of the determinant vector; arrays shorter are padded with zeros.

    Returns
    -------
    determinant : NDArray[np.int32]
        Validated (and possibly padded) occupation vector with dtype int32.
    natural_occupation : bool
        True if determinant was constructed as the natural (RHF) occupation.
    """
    natural_occupation = True

    if determinant is None:
        determinant = -1

    if isinstance(determinant, int):
        if determinant == -1:
            determinant = np.zeros([expected_dim], dtype=np.int32)
            for i in range(int(n_electrons / 2)):
                determinant[i] = 2
        else:
            raise TypeError('determinant must be -1, None or a numpy array of occupations')

    if not isinstance(determinant, np.ndarray):
        raise TypeError('determinant must be a numpy.ndarray when not -1/None')

    natural_occupation = False
    if determinant.dtype.kind not in ('i', 'u'):
        determinant = determinant.astype(np.int32)

    if int(np.sum(determinant)) != n_electrons:
        raise ValueError(f'determinant sum ({int(np.sum(determinant))}) != n_electrons ({n_electrons})')

    if len(determinant) != expected_dim:
        new_occ = np.zeros(expected_dim, dtype=np.int32)
        for i, oc in enumerate(determinant):
            if i >= expected_dim:
                break
            new_occ[i] = int(oc)
        determinant = new_occ

    return determinant.astype(np.int32), natural_occupation

def calc_p_matrix(
    C_matrix: NDArray[np.float64], 
    n_electrons: int
) -> NDArray[np.float64]:
    """
    Calculate the (RHF) density matrix from MO coefficients.

    P_{mu, nu} = 2 * sum_{a}^{n_occ} C_{mu, a} * C_{nu, a}^*

    Parameters
    ----------
    C_matrix : NDArray[np.float64], shape (n, n)
        Molecular orbital coefficient matrix.
    n_electrons : int
        Number of electrons (must be even for RHF).

    Returns
    -------
    P : NDArray[np.float64], shape (n, n)
        Density matrix.
    """
    dim = len(C_matrix)
    n_occ = n_electrons // 2 
    P = 2 * np.einsum('mu,nu->mn', C_matrix[:, :n_occ], np.conj(C_matrix[:, :n_occ]))

    return P


def calc_g_matrix(
    P_matrix: NDArray[np.float64], 
    eri: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate G matrix using: 

    G_{mu, nu} = sum_{la, si} P_{la, si} * ( <mu nu|la si> - 0.5 * <mu la|nu si> )

    Parameters
    ----------
    P_matrix : NDArray[np.float64] of dimension (n, n)
        Density matrix.
    eri : NDArray[np.float64] of dimension (n, n, n, n)
        Electron repulsion integrals.

    Returns
    -------
    g_mat : NDArray[np.float64] of dimension (n, n)
        G matrix.
    """
    return np.einsum('mnls,ls->mn', eri, P_matrix) - 0.5 * np.einsum('mlns,ls->mn', eri, P_matrix)

def E_0(P: NDArray[np.float64], H_core: NDArray[np.float64], F: NDArray[np.float64]) -> float:
    """
    Calculate Hartree-Fock electronic energy:

    E_0 = 0.5 * sum_{mu,nu} P_{mu,nu} * (H_core_{mu,nu} + F_{mu,nu})

    Parameters
    ----------
    P : NDArray[np.float64], shape (n, n)
        Density matrix.
    H_core : NDArray[np.float64], shape (n, n)
        Core Hamiltonian (kinetic + nuclear attraction).
    F : NDArray[np.float64], shape (n, n)
        Fock matrix.

    Returns
    -------
    energy : float
        Electronic Hartree-Fock energy.
    """
    return 0.5 * np.sum(P * (H_core + F))

def V_NN(
    positions: NDArray[np.float64], 
    charges: Union[NDArray[np.int32], Sequence[int]], 
    units: Literal['Bohr', 'Angstrom'] = 'Bohr'
) -> float:
    """
    Nuclear repulsion energy.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_atoms, 3)
        Atomic coordinates.
    charges : array-like of ints, shape (n_atoms,)
        Nuclear charges.
    units : {'Bohr', 'Angstrom'}
        Units of positions input. If 'Angstrom', positions are converted to Bohr.

    Returns
    -------
    V_NN : float
        Nuclear repulsion energy in Hartree.
    """
    if units == 'Angstrom':
        positions /= 0.529177249

    V_NN = 0.0
    n_atoms = len(positions)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            R_ij = np.linalg.norm(positions[i] - positions[j])
            V_NN += (charges[i] * charges[j]) / R_ij
    
    return V_NN


# --- Complex SCF Helper Functions ---
def scale_integrals(T: NDArray[np.float64], V: NDArray[np.float64], eri: NDArray[np.float64], theta: float) -> NDArray[np.complex128]:
    """
    Scale integrals according to the complex-scaling angle theta.

    Parameters
    ----------
    T : NDArray[np.float64], shape (n, n)
        Kinetic energy matrix.
    V : NDArray[np.float64], shape (n, n)
        Nuclear attraction matrix.
    eri : NDArray[np.float64], shape (n, n, n, n)
        Electron repulsion integrals.
    theta : float
        Complex-scaling angle (radians).

    Returns
    -------
    T_scaled, V_scaled, eri_scaled : tuple of NDArray[np.complex128]
        Scaled kinetic, potential and two-electron integrals.
    """
    exp_t2 = np.exp(-2j * theta)
    exp_t1 = np.exp(-1j * theta)
    return T * exp_t2, V * exp_t1, eri * exp_t1

def guess_density(dim: int, method: Literal['core', 'ones']) -> NDArray[np.complex128]:
    """Generate an initial guess density matrix.

    Parameters
    ----------
    dim : int
        Dimension of the basis (number of AOs).
    method : {'core', 'ones'}
        Method for generating the guess density.

    Returns
    -------
    P_guess : NDArray[np.complex128], shape (dim, dim)
        Initial guess density matrix.
    """
    if method == 'core':
        P_guess = np.zeros((dim, dim), dtype=np.complex128)
    elif method == 'ones':
        P_guess = np.ones((dim, dim), dtype=np.complex128)
    else:
        raise ValueError("Invalid method for guess density. Choose 'core' or 'ones'.")

    return P_guess

def diagonalize_biorthogonal(F_prime: NDArray[np.complex128]):
    """
    Diagonalize a (generally non-Hermitian) transformed Fock matrix F'.

    Uses numpy.linalg.eig to obtain right eigenvectors; constructs left
    eigenvectors as the inverse of the right eigenvector matrix.

    Parameters
    ----------
    F_prime : NDArray[np.complex128], shape (n, n)
        Transformed Fock matrix to diagonalize.

    Returns
    -------
    e_values : NDArray[np.complex128], shape (n,)
        Eigenvalues (orbital energies), sorted ascending.
    C_prime : NDArray[np.complex128], shape (n, n)
        Right eigenvector matrix whose columns are eigenvectors.
    L_prime : NDArray[np.complex128], shape (n, n)
        Left eigenvector matrix (inverse of C_prime).
    R_prime : NDArray[np.complex128], shape (n, n)
        Copy of C_prime (right eigenvectors).
    LFR : NDArray[np.complex128], shape (n, n)
        Product L_prime @ F_prime @ R_prime, should be diagonal.
    """
    e_values, C_prime = np.linalg.eig(F_prime)

    idx = e_values.argsort()
    e_values = e_values[idx]
    C_prime = C_prime[:, idx]
    R_prime = np.copy(C_prime)

    L_prime = np.linalg.inv(C_prime)

    LFR = L_prime @ F_prime @ R_prime

    return e_values, C_prime, L_prime, R_prime, LFR

def calc_p_matrix_comp(
    l_matrix: NDArray[np.complex128], 
    r_matrix: NDArray[np.complex128], 
    n_electrons: int,
    determinant: Optional[Union[NDArray[np.int32], Sequence[int]]] = None,
    natural_occupation: bool = False,
) -> NDArray[np.complex128]:
    """
    Calculate density matrix from biorthonormal left/right MO coefficient matrices.

    P_{mu,nu} = 2 * sum_{a in occupied} r_{mu,a} * l_{nu,a}

    Parameters
    ----------
    l_matrix, r_matrix : NDArray[np.complex128], shape (n, n)
        Left and right molecular orbital coefficient matrices (biorthonormal).
    n_electrons : int
        Total number of electrons (must be even for RHF-like occupancy).
    determinant : ndarray or sequence of ints, optional
        Occupation vector (e.g. [2,2,0,...]) selecting which orbitals are occupied.
        If None and natural_occupation is True a default RHF occupation is used.
    natural_occupation : bool
        If True ignores determinant and uses the lowest n_electrons/2 orbitals.

    Returns
    -------
    P : NDArray[np.complex128], shape (n, n)
        Complex density matrix.
    """
    assert n_electrons % 2 == 0, 'This only works for RHF for now'
    
    P = np.zeros((len(r_matrix), len(r_matrix)), dtype=np.complex128)

    if natural_occupation:
        n_occ = n_electrons // 2 
        P += 2 * np.einsum('mu,nu->mn', r_matrix[:, :n_occ], l_matrix[:, :n_occ])
    else:
        for mu in range(len(r_matrix)):
            for nu in range(len(r_matrix)):
                for a in range(len(r_matrix)):
                    P[mu, nu] += 2 * r_matrix[mu, a] * l_matrix[nu, a] * kroeneker_delta(2, determinant[a])

    return P

def calc_g_matrix_comp(
    P_matrix: NDArray[np.complex128], 
    eri: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """
    Calculate G matrix using: 

    G_{mu, nu} = sum_{la, si} P_{la, si} * ( <mu nu|la si> - 0.5 * <mu la|nu si> )

    Parameters
    ----------
    P_matrix : NDArray[np.float64] of dimension (n, n)
        Density matrix.
    eri : NDArray[np.float64] of dimension (n, n, n, n)
        Electron repulsion integrals.

    Returns
    -------
    g_mat : NDArray[np.float64] of dimension (n, n)
        G matrix.
    """
    return np.einsum('mnls,ls->mn', eri, P_matrix) - 0.5 * np.einsum('mlns,ls->mn', eri, P_matrix)

def E_0_comp(
    P: NDArray[np.complex128],
    H_core: NDArray[np.complex128],
    F: NDArray[np.complex128]
 ) -> np.complex128:
    """
    Complex-valued Hartree-Fock energy:

    E_0 = 0.5 * sum_{mu,nu} P_{mu,nu} * (H_core_{mu,nu} + F_{mu,nu})

    Parameters
    ----------
    P : NDArray[np.complex128], shape (n, n)
        Density matrix (complex).
    H_core : NDArray[np.complex128], shape (n, n)
        Core Hamiltonian (complex).
    F : NDArray[np.complex128], shape (n, n)
        Fock matrix (complex).

    Returns
    -------
    energy : np.complex128
        Complex electronic energy.
    """
    return np.sum(P * (H_core + F)) * 0.5

