import numpy as np
from numpy.typing import NDArray
from typing import Literal

def transformation_matrix(S_munu: NDArray[np.float64], method: Literal['symmetric'] ='symmetric') -> NDArray[np.float64]:
    """
    Calculate The normalization transformation matrix X.

    Uses symmetric orthogonalization. This is obtaining the matrix S^{-1/2}
    by obtaining the diagonal form of S, s. 
    
    U^{dagger} @ S @ U = s
    
    The diagonal matrix s^{-1/2} is easily computed and to obtain S^{-1/2} we
    use the transformation:
    
    X = S^{-1/2} = U @ s^{-1/2} @ U^{dagger} 

    Parameters
    ------
    S_munu : np.ndarray of dimension (n, n)
        Overlap matrix. 
    
    Returns
    ------
    X : np.ndarray of dimension (n, n)
        Transformation matrix X.

    Notes 
    ------
    The operation S^{-1/2} @ S @ S^{-1/2} = Identity must always hold. This is asserted.
    """
    assert method in ['canonical', 'symmetric'], "method must be either 'canonical' or 'symmetric'"
    dim = len(S_munu)

    # diagonalize U.T @ S @ U = s
    s, U = np.linalg.eigh(S_munu)

    s_root = np.zeros([dim, dim])

    # calculate s^{-0.5}
    for index, eigenvalue in enumerate(s):
        s_root[index,index] = 1/np.sqrt(eigenvalue)

    if method == 'symmetric':
        X = U @ s_root @ U.T
    elif method == 'canonical':
        X = U @ s_root
    
    # transformation matrix test
    transformed = X.T @ S_munu @ X
    assert equiv_matrix(transformed, np.identity(len(S_munu))), "transformation matrix calculation failed"

    return X

def equiv_matrix(prev: NDArray[np.float64], curr: NDArray[np.float64], threshold=0.0000001) -> bool:
    """
    Check equality between two arrays. Uses the max difference as metric. 

    Parameters
    ------
    prev : NDArray[np.float64] of dimension (n, n)
        Previous array to compare.
    curr : NDArray[np.float64] of dimension (n, n)
        Current array to compare.
    threshold : float
        Convergence threshold.

    Returns
    ------
    converged : bool
        True if the maximum difference between arrays is less than threshold.
    """
    diff = np.abs(curr - prev)
    max_diff = np.max(diff)

    return max_diff < threshold

def calc_g_matrix(P_matrix: NDArray[np.float64], eri: NDArray[np.float64]) -> NDArray[np.float64]:
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

    
    Notes
    ------
    The system bust be a closed shell: n_electrons must be even. This is asserted.

    Integrals must be passed and have the same dimensions. This is asserted.

    Diagonalization algorithm used is np.linalg.eigh due to the matrix being symmetric.
    
    The algorithm steps are:
        - Obtain transformation matrix X from S.
        - Guess initial density matrix P.
        - Build core Hamiltonian H_core = T + V.
        - SCF loop:
            - Build G matrix from P and eri.
            - Build Fock matrix F = H_core + G.
            - Obtain transformed Fock matrix F' = X.T @ F @ X.
            - Diagonalize F' to obtain orbital energies and transformed MO coefficients.
            - Obtain untransformed MO coefficients C = X @ C'.
            - Build new density matrix P from C.
            - Calculate RHF energy E_RHF.
            - Check convergence.
    """
    dim = len(P_matrix)
    g_mat = np.zeros([dim, dim])

    for mu in range(0, dim):
        for nu in range(0, dim):
            for si in range(0,dim):
                for la in range(0, dim):
                    g_mat[mu, nu] += P_matrix[la, si] * (eri[mu, nu, la, si] - 0.5 * eri[mu, la, nu, si])

    
    return g_mat

def calc_p_matrix(C_matrix: NDArray[np.float64], n_electrons: int) -> NDArray[np.float64]:
    """
    Calculate density matrix from MO coefficients using: 

    P_{mu, nu} = 2 * sum_{a}^{n_occ} C_{mu, a} * C_{nu, a}^*

    Parameters
    ----------
    C_matrix : NDArray[np.float64] of dimension (n, n)
        Overlap matrix.
    n_electrons : int
        Number of electrons.

    Returns
    -------
    P : NDArray[np.float64] of dimension (n, n)
        Density matrix.
    
    Notes
    -------
    n_occ is divided by 2 due to this being used for the RHF case.
    """
    dim = len(C_matrix)
    P = np.zeros([dim, dim])

    n_occ = int(n_electrons / 2) 

    # print(n_occ)

    for mu in range(0, dim):
        for nu in range(0, dim):
            for a in range(0, n_occ):
                P[mu, nu] += 2 * C_matrix[mu, a] * np.conj(C_matrix[nu, a])
    
    return P

def E_0(P: NDArray[np.float64], H_core: NDArray[np.float64], F: NDArray[np.float64]) -> float:
    """
    Calculate Hartree-Fock energy using: 

    E_0 = 0.5 * sum_{mu, nu} P_{mu, nu} * (H^core_{mu, nu} + F_{mu, nu})

    Parameters
    ----------
    P : NDArray[np.float64] of dimension (n, n)
        Overlap matrix.
    H_core : NDArray[np.float64] of dimension (n, n)
        Kinetic energy matrix.
    F : NDArray[np.float64] of dimension (n, n)
        Nuclear attraction matrix.

    Returns
    -------
    energy: float
        Hartree-Fock energy. 
    """
    energy = 0.0
    dim = len(P)

    for mu in range(0, dim):
        for nu in range(0, dim):
            energy += 0.5 * P[mu, nu] * (H_core[mu, nu] + F[mu, nu])
    
    return energy

def V_NN(positions: NDArray[np.float64], charges: NDArray, units: Literal['Bohr', 'Angstrom'] = 'Bohr') -> float:
    """
    Calculate nuclear repulsion energy.

    Parameters
    ----------
    positions : NDArray[np.float64] of dimension (n_atoms, 3)
        Atomic positions.
    charges : NDArray[np.int] of dimension (n_atoms,)
        Atomic charges.

    Returns
    -------
    V_NN : float
        Nuclear repulsion energy.
    """
    if units == 'Angstrom':
        positions /= 0.529177249
        print(positions)

    V_NN = 0.0
    n_atoms = len(positions)

    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            R_ij = np.linalg.norm(positions[i] - positions[j])
            V_NN += (charges[i] * charges[j]) / R_ij
    
    return V_NN