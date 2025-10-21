import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple

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

def RHF(
    S: NDArray[np.float64],
    T: NDArray[np.float64], 
    V: NDArray[np.float64], 
    eri: NDArray[np.float64], 
    n_electrons: int, 
    max_iter: int = 100, 
    threshold: float = 1E-12, 
    p_guess: Literal['core', 'ones'] = 'core', 
    verbose: bool = False
) -> Tuple[bool, float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform a RHF calculation.

    Takes S, T, V and eri matrix elements and computes the RHF procedure. 

    Implementation was done based on "Modern Quantum Chemistry" 
    by Szabo and Ostlund.

    Parameters
    ----------
    S : NDArray[np.float64] of dimension (n, n)
        Overlap matrix.
    T : NDArray[np.float64] of dimension (n, n)
        Kinetic energy matrix.
    V : NDArray[np.float64] of dimension (n, n)
        Nuclear attraction matrix.
    eri : NDArray[np.float64] of dimension (n, n, n, n)
        Electron repulsion integrals.
    n_electrons : int
        Number of electrons.
    max_iter : int, optional
        Maximum number of SCF iterations.
    threshold : float, optional
        Convergence threshold for max density matrix diff.
    p_guess : Literal['core'], optional
        Initial density matrix guess.
    verbose : bool, optional
        If True, prints iterations.

    Returns
    -------
    Tuple containing:
        - converged (bool): Convergence status.
        - E_RHF (float): Final RHF energy.
        - e_values (NDArray[np.float64][n, n]): Orbital energies.
        - C_munu (NDArray[np.float64][n, n]): Molecular orbital coefficients.
        - P (NDArray[np.float64][n, n]): Final density matrix.
    
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
    assert len(T) == len(V) == len(S), "Matrices T, V, S must have the same dimensions"
    assert n_electrons % 2 == 0, "RHF can only be closed-shell systems"

    # Otain transformation matrix 
    dim = len(S)
    X = transformation_matrix(S)

    # Guess initial density matrix
    if p_guess == 'core':
        P = np.zeros([dim, dim])
    elif p_guess == 'ones':
        P = np.ones([dim, dim])
    
    P_old = np.zeros([dim, dim])
    P_new = np.copy(P)

    # Build core Hamiltonian
    H_core = T + V 

    E_iter = 0 
    Delta_E = 0 

    if verbose:
        print('-'*70)
        print('|   Iter   |           E_iter           |            Delta_e         |')
        print('-'*70)

    # SCF loop
    for iter in range(max_iter):
        if iter != 0 and equiv_matrix(P_new, P_old, threshold=threshold):
            converged = True
            if verbose:
                print(f'{iter:5}     {E_iter:25.16f}     {Delta_E:25.16f}')
                print(f'Convergence achieved after {iter} iterations. Final SCF energy = {E_iter}')

            break
        
        # Obtain G matrix from P and eris. Build Fock matrix
        G = calc_g_matrix(P_new, eri)
        F = G + H_core

        # Obtain transformed Fock matrix 
        F_prime = X @ F @ X.T

        # Diagonalize transformed Fock matrix to obtain energies and transformed MO coefficients
        e_values, C_prime = np.linalg.eigh(F_prime) # here is eigh because we are in the non-scaled case

        # Obtain untransformed MO coefficients
        C_munu = X @ C_prime

        # Build new density matrix
        P_old = np.copy(P_new)
        P_new = calc_p_matrix(C_munu, n_electrons=n_electrons)

        # Calculate HF energy
        E_old = E_iter
        E_iter = E_0(P_new, H_core, F)
        Delta_E = E_iter - E_old

        if verbose:
            print(f'{iter:5}     {E_iter:25.16f}     {Delta_E:25.16f}')

    E_RHF = E_iter
    return converged, E_RHF, e_values, C_munu, P



if __name__ == "__main__":
    """
    S_sto3g_H2 = np.loadtxt('./data/s_H2.dat')
    T_sto3g_H2 = np.loadtxt('./data/kin_H2.dat')
    V_sto3g_H2 = np.loadtxt('./data/vnuc_H2.dat')
    eri_sto3g_H2 = np.load('./data/eri_H2.npy')

    idn = np.identity(2)

    # test 1: successful transformation matrix
    X = transformation_matrix(S_sto3g_H2)
    transformed = X @ S_sto3g_H2 @ X      # sould be the identity    
    assert equiv_matrix(transformed, idn), "transformation matrix calculation failed"

    # test 2: SCF convergence for H2 in STO-3G
    e_values, C_munu, P = scf(S_sto3g_H2, T_sto3g_H2, V_sto3g_H2, eri_sto3g_H2, n_electrons=2, max_iter=100, threshold=1E-14, p_guess='core')

    print(e_values)
    """


    S_sto3g_li    = np.loadtxt('../tests/data/Li_plus_S_6-31g.dat')
    T_sto3g_li    = np.loadtxt('../tests/data/Li_plus_kin_6-31g.dat')
    V_sto3g_li    = np.loadtxt('../tests/data/Li_plus_vnuc_6-31g.dat')
    eri_sto3g_li  = np.load('../tests/data/Li_plus_eri_6-31g.npy')
    E_hf_sto3g_li = np.load('../tests/data/Li_plus_e_hf_6-31g.npy')

    idn = np.identity(len(S_sto3g_li))

    # test 1: successful transformation matrix
    X = transformation_matrix(S_sto3g_li)
    transformed = X.T @ S_sto3g_li @ X      # sould be the identity    
    assert equiv_matrix(transformed, idn), "transformation matrix calculation failed"

    # test 2: SCF convergence for li in 6-31g
    converged, E_hf, E_e_values, C_munu, P = RHF(S_sto3g_li, T_sto3g_li, V_sto3g_li, eri_sto3g_li, n_electrons=2, max_iter=100, threshold=1E-14, p_guess='core', verbose=True)
    assert converged, "Calculation did not converge"
    assert abs(E_hf - E_hf_sto3g_li) < 1E-8, f"SCF energy does not match reference value {E_hf} != {E_hf_sto3g_li}"


    