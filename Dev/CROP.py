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
    verbose: bool = False,
    CROP_MEM: int = 5,
    CROP_ITER_START: int = 5,
    CROP_REQUESTED: bool = True,
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
    CROP_MEM : int, optional
        Number of previous Fock matrices and residuals to store for CROP.
    CROP_ITER_START : int, optional
        Iteration number to start CROP.
    CROP_REQUESTED : bool, optional
        If True, enables CROP after CROP_ITER_START iterations.

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
    

    # Build core Hamiltonian
    H_core = T + V 

    # initialize variables and lists
    E_prev = 0.
    use_CROP = False 
    converged = False
    F_guess = []
    residuals = []

    if verbose:
        print('-'*83)
        print('|   Iter   |           E_iter           |           Delta_e        |  Sum(Error)  |')
        print('-'*83)

    # SCF loop
    for iter in range(0, max_iter):

        # calculate F_n and r_n from P_n
        F, r = calculate_F_and_r(P, S, H_core, eri)
        error = r.flatten() @ r.flatten()
        E_RHF = E_0(P, H_core, F.reshape(H_core.shape))
        E_diff = E_RHF - E_prev

        if verbose:
            print(f'{iter:5}     {E_RHF:25.16f}     {E_diff:25.16f}     {error:8.4E}')

        # Check convergence
        if iter > 1 and error < threshold:
            converged = True
            break

        # Save in memory guesses and residuals keeping size of CROP space
        F_guess.append(F)
        residuals.append(r)

        if len(F_guess) > CROP_MEM:
            F_guess.pop(0)
            residuals.pop(0)

        # Choose F for P_{n+1}
        if not use_CROP:
            F_next = F 
        
        elif use_CROP:
            F_opt, r_opt = CROP_guess(residuals, F_guess)
            F_guess[-1] = F_opt
            residuals[-1] = r_opt  

            F_next = F_opt # + r_opt # equation 32 Ettenhuber, r_opt should be here, but it diverges idk why

        # Calculate P_{n+1}
        P, C_munu, orbital_energies = calculate_P_next(F_next.reshape(X.shape), X, n_electrons)

        E_prev = E_RHF
    
        # Check CROP activation
        if iter == CROP_ITER_START and CROP_REQUESTED:
            use_CROP = True 
            print('-'*30,  '   STARTED CROP   ', '-' *30)
    
    return converged, E_RHF, orbital_energies, C_munu, P

def calculate_P_next(F_0: NDArray[np.float64], X: NDArray[np.float64], n_electrons: int) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
    F_prime = X @ F_0 @ X.T

    e_values, C_prime = np.linalg.eigh(F_prime)

    C_munu = X @ C_prime

    P_1 = calc_p_matrix(C_munu, n_electrons=n_electrons)

    return P_1, C_munu, e_values

def calculate_F_and_r(P: NDArray[np.float64], S: NDArray[np.float64], H_core: NDArray[np.float64], eri: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    F = H_core + calc_g_matrix(P, eri)
    r = residual(F, P, S)

    return F.flatten(), r.flatten()

def residual(F: NDArray[np.float64], P: NDArray[np.float64], S: NDArray[np.float64]) -> NDArray[np.float64]:
    """ 
    Calculate the residual matrix r = S P F - F P S

    Parameters
    ----------
    F : NDArray[np.float64] of dimension (n, n)
        Fock matrix.
    P : NDArray[np.float64] of dimension (n, n)
        Density matrix.
    S : NDArray[np.float64] of dimension (n, n)
        Overlap matrix.
    
    Returns
    -------
    NDArray[np.float64] of dimension (n, n)
    """
    return S @ P @ F - F @ P @ S

def CROP_guess(residuals: NDArray[np.float64], F_guesses: NDArray[np.float64]) -> NDArray[np.float64]:
    """ 
    Calculate the CROP extrapolated Fock matrix.

    Parameters
    ----------
    residuals : List of NDArray[np.float64] of dimension (n, n)
        List of residual matrices.
    F_guesses : List of NDArray[np.float64] of dimension (n, n)
        List of Fock matrices.
    
    Returns
    -------
    NDArray[np.float64] of dimension (n, n)
    """
    n_guesses = len(residuals)
    eq_sis_dim = n_guesses + 1
    
    # build the system of equations
    B_matrix = np.zeros([eq_sis_dim, eq_sis_dim])
    B_matrix[-1,:] = B_matrix[:,-1] = 1
    B_matrix[-1,-1] = 0

    for i in range(n_guesses):
        for j in range(n_guesses):
            B_matrix[i,j] = residuals[i] @ residuals[j]

    solution = np.zeros(eq_sis_dim)
    solution[-1] = 1

    # solve the system of equations
    c = np.linalg.solve(B_matrix, solution)

    F_CROP = sum([c[i] * F_guesses[i] for i in range(len(c)-1)])
    r_crop = sum([c[i] * residuals[i] for i in range(len(c)-1)])

    return F_CROP, r_crop

CROP_RHF = RHF 

if __name__ == "__main__":
    pass 