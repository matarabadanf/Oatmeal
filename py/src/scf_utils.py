import numpy as np
from numpy.typing import NDArray

def transformation_matrix(S_munu: np.ndarray) -> np.ndarray:
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
    S_munu : np.ndarray of square dimensions.
    
    Returns
    ------
    X : np.ndarray of same shape as S_munu
        Transformation matrix X.

    Notes 
    ------
    The operation S^{-1/2} @ S @ S^{-1/2} = Identity must always hold.
    """
    dim = len(S_munu)

    # diagonalize U.T @ S @ U = s
    s, U = np.linalg.eigh(S_munu)

    s_root = np.zeros([dim, dim])

    for index, eigenvalue in enumerate(s):
        s_root[index,index] = 1/np.sqrt(eigenvalue)

    X = U @ s_root @ U.T

    transformed = X @ S_munu @ X    

    assert equiv_matrix(transformed, np.identity(len(S_munu))), "transformation matrix calculation failed"
    print(X)

    return X

def equiv_matrix(prev: NDArray[np.float64], curr: NDArray[np.float64], threshold=0.0000001) -> bool:
    """
    Check equality between two arrays. Uses the max difference as metric. 

    Parameters
    ------
    prev : NDArray[np.float64]
        Previous array to compare.
    curr : NDArray[np.float64]
        Current array to compare.
    threshold : float
        Convergence threshold.

    Returns
    ------
    converged : bool
        True if the maximum absolute difference between prev and curr is less than threshold.
    """
    diff = np.abs(curr - prev)
    max_diff = np.max(diff)

    return max_diff < threshold

def calc_g_matrix(P_matrix: np.ndarray, eri: np.ndarray) -> np.ndarray:
    dim = len(P_matrix)
    g_mat = np.zeros([dim, dim])

    for mu in range(0, dim):
        for nu in range(0, dim):
            for si in range(0,dim):
                for la in range(0, dim):
                    g_mat[mu, nu] += P_matrix[la, si] * (eri[mu, nu, la, si] - 0.5 * eri[mu, la, nu, si])
                    # g_mat[mu, nu] += P_matrix[la, si] * (eri[mu, si, si, la] - 0.5 * eri[mu, la, si, nu])
    
    return g_mat

def calc_p_matrix(C_matrix: np.ndarray, n_electrons: int) -> np.ndarray:
    
    dim = len(C_matrix)
    P = np.zeros([dim, dim])

    n_occ = int(n_electrons / 2) 

    # print(n_occ)

    for mu in range(0, dim):
        for nu in range(0, dim):
            for a in range(0, n_occ):
                P[mu, nu] += 2 * C_matrix[mu, a] * np.conj(C_matrix[nu, a])
    
    return P

def E_0(P, H_core, F):
    energy = 0.0
    dim = len(P)

    for mu in range(0, dim):
        for nu in range(0, dim):
            energy += 0.5 * P[mu, nu] * (H_core[mu, nu] + F[mu, nu])
    
    return energy

def scf(S, T, V, eri, n_electrons:int, max_iter=100, threshold=1E-6, p_guess='core'):
    assert len(T) == len(V) == len(S), "Matrices T, V, S must have the same dimensions"
    assert n_electrons % 2 == 0, "RHF can only be closed-shell systems"

    dim = len(S)
    X = transformation_matrix(S)

    if p_guess == 'core':
        P = np.zeros([dim, dim])
    
    P_old = np.zeros([dim, dim])
    P_new = np.copy(P)

    H_core = T + V 

    for iter in range(max_iter):
        if iter != 0 and equiv_matrix(P_new, P_old, threshold=threshold):
            converged = True
            print(f'Convergence achieved after {iter} iterations. Final SCF energy = {E_iter}')
            break

        G = calc_g_matrix(P_new, eri)
        F = G + H_core
        F_prime = X @ F @ X.T

        e_values, C_prime = np.linalg.eigh(F_prime) # here is eigh because we are in the non-scaled case

        # idx = e_values.argsort()
        # e_values = e_values[idx]
        # C_prime = C_prime[:, idx]

        print(f'\n\nIteration {iter}')
        print(e_values)
        print(f'\nEigenvectors:')
        print(C_prime)

        C_munu = X @ C_prime

        # print(iter)
        print(f'\nTransformed Eigenvectors:')
        print(C_munu)

        P_old = np.copy(P_new)
        P_new = calc_p_matrix(C_munu, n_electrons=n_electrons)

        print(f'\nNew Density Matrix:')
        print(P_new)

        E_iter = E_0(P_new, H_core, F)
        print(f"Eigenvalues at iter {iter}: {e_values}")
        print(f"Energy: {E_iter}")

    
    return e_values, C_munu, P



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


    S_sto3g_li = np.loadtxt('./data/s_li.dat')
    T_sto3g_li = np.loadtxt('./data/kin_li.dat')
    V_sto3g_li = np.loadtxt('./data/vnuc_li.dat')
    eri_sto3g_li = np.load('./data/eri_li.npy')

    idn = np.identity(5)

    # test 1: successful transformation matrix
    X = transformation_matrix(S_sto3g_li)
    transformed = X @ S_sto3g_li @ X      # sould be the identity    
    assert equiv_matrix(transformed, idn), "transformation matrix calculation failed"

    # test 2: SCF convergence for li in STO-3G
    e_values, C_munu, P = scf(S_sto3g_li, T_sto3g_li, V_sto3g_li, eri_sto3g_li, n_electrons=2, max_iter=2, threshold=1E-14, p_guess='core')

    # expected [-4.40795283 -0.93897155 -0.87757914 -0.87757914 -0.87757914]
    # expected 
    """
    [[ 1.00548232 -0.2252598   0.          0.          0.        ]
    [-0.02384599  1.03013011  0.          0.          0.        ]
    [-0.         -0.          1.          0.          0.        ]
    [-0.          0.          0.          1.          0.        ]
    [-0.          0.          0.          0.          1.        ]]
    """
    