import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple
from py_mods.src.SCF.scf_utils import transformation_matrix, equiv_matrix, calc_g_matrix, calc_p_matrix, E_0

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
    # print(X)

    # Guess initial density matrix
    if p_guess == 'core':
        P = np.zeros([dim, dim])
    elif p_guess == 'ones':
        P = np.ones([dim, dim])
    
    P_old = np.zeros([dim, dim])
    P_new = np.copy(P)

    # Build core Hamiltonian
    H_core = T + V 

    E_iter = 0.
    Delta_E = 0.
    converged = False

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
                
            E_RHF = E_iter
            return converged, E_RHF, e_values, C_munu, P_new
        
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
    return converged, E_RHF, e_values, C_munu, P_new

if __name__ == "__main__":
    pass 