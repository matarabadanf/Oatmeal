import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, Union
from py_mods.src.SCF.scf_utils import transformation_matrix, calc_g_matrix_comp, calc_p_matrix_comp, E_0_comp, guess_density, validate_determinant, scale_integrals, is_diagonal, diagonalize_biorthogonal, equiv_matrix
import matplotlib.pyplot as plt

def CS_RHF(
    S: NDArray[np.float64],
    T: NDArray[np.float64], 
    V: NDArray[np.float64], 
    eri: NDArray[np.float64], 
    n_electrons: int, 
    theta: float,
    occupation: Union[int, NDArray[np.int32], None] = None,
    max_iter: int = 100, 
    threshold: float = 1E-12, 
    p_guess: Literal['core', 'ones'] = 'core', 
    verbose: bool = False,
    conv_type: Literal[None, 'DIIS', 'CROP'] = 'DIIS',
    conv_MEM: int = 5,
    conv_ITER_START: int = 5,
) -> Tuple[bool, float, NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Perform a Complex Scaled RHF calculation.

    Takes overlap, kinetic, nuclear attraction and two-electron integrals,
    applies complex scaling by angle `theta` and runs an RHF-like SCF loop
    using biorthogonal diagonalization.

    Parameters
    ----------
    S : NDArray[np.float64], shape (n, n)
        Overlap matrix.
    T : NDArray[np.float64], shape (n, n)
        Kinetic energy matrix.
    V : NDArray[np.float64], shape (n, n)
        Nuclear attraction matrix.
    eri : NDArray[np.float64], shape (n, n, n, n)
        Electron repulsion integrals.
    n_electrons : int
        Total number of electrons (must be even for closed-shell RHF).
    theta : float
        Complex-scaling angle (radians).
    occupation : int or NDArray[np.int32] or None
        If -1 (or None) build a default RHF occupation vector (2,2,...,0).
        If an ndarray is provided it must sum to n_electrons.
    max_iter : int, optional
        Maximum SCF iterations.
    threshold : float, optional
        Convergence threshold for density matrix difference.
    p_guess : {'core', 'ones'}, optional
        Initial density guess.
    verbose : bool, optional
        If True print iteration progress.
    conv_type : Literal[None, 'DIIS', 'CROP'], optional
        Type of Convergence Algorithm to use. If None, no algorithm is used.
    conv_MEM : int, optional
        Number of previous Fock matrices and residuals to store for Convergence Algorithm.
    conv_ITER_START : int, optional
        Iteration number to start Convergence Algorithm.

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
    - The system bust be a closed shell: n_electrons must be even. This is asserted.
    - Integrals must be passed and have the same dimensions. This is asserted.


    - Implementation was done based on "Modern Quantum Chemistry" by Szabo and Ostlund.
    - DIIS implementation was based on [Pulay](https://doi.org/10.1002/jcc.540030413).
    - CROP implementation was based on [Ettenhuber, Jorgensen](https://doi.org/10.1021/ct501114q).

    ^* CROP algorithm does not compute the new trial as t_opt + w_opt, as it breaks convergence here.
    """
    assert len(T) == len(V) == len(S), "Matrices T, V, S must have the same dimensions"
    assert n_electrons % 2 == 0, "RHF can only be closed-shell systems"
    assert conv_type in [None, 'DIIS', 'CROP'], 'Convergence assist must be either None, DIIS, or CROP'

    conv_REQUESTED = True if conv_type is not None else False
    
    conv_ITER_START = min(conv_ITER_START+1, conv_MEM)

    # Otain transformation matrix and validate occupation determinant
    dim = len(S)
    X = transformation_matrix(S) + 0j
    det, natural_occupation = validate_determinant(n_electrons, occupation, dim)

    # rescaling the integrals
    T_scaled, V_scaled, eri_scaled = scale_integrals(T, V, eri, theta)
    H_core = T_scaled + V_scaled

    # Guess initial density matrix
    P = guess_density(dim, p_guess)

    # initialize variables and lists
    E_prev = 0.+0.j
    use_conv = False 
    converged = False
    F_guess = []
    residuals = []

    if verbose:
        print('-'*128)
        print('|   Iter   |               E_iter                  |                       Delta_e                   |        norm(e_i)        |')
        print('-'*128)

    # SCF loop
    for iter in range(max_iter):
        # calculate F_n and r_n from P_n
        F, r = calculate_F_and_r_comp(P, S, H_core, eri_scaled)
        error = np.linalg.norm(r.flatten())
        E_RHF = E_0_comp(P, H_core, F.reshape(H_core.shape))
        E_diff = E_RHF - E_prev

        if verbose:
            print(f'{iter:5}     {E_RHF:45.16f}     {E_diff:45.16f}     {error:8.4E}')
        # Check convergence
        if iter > 1 and error < threshold:
            converged = True
            if verbose:
                print(f'Convergence achieved after {iter} iterations. Final SCF energy = {E_RHF:5}')
            break

        # Save in memory guesses and residuals keeping size of Convergence Algorithm space
        F_guess.append(F)
        residuals.append(r)

        if len(F_guess) > conv_MEM:
            F_guess.pop(0)
            residuals.pop(0)
        
        # Choose F for P_{n+1}
        if not use_conv:
            F_next = F 
        
        elif use_conv:
            F_opt, r_opt = conv_guess(residuals, F_guess)

            F_next = F_opt # Default is DIIS

            if conv_type == 'CROP':
                F_guess[-1] = F_opt
                residuals[-1] = r_opt  
                F_next = F_opt # + r_opt # equation 32 Ettenhuber, r_opt should be here, but it diverges idk why

        P_old = np.copy(P)
        
        P, C_munu, orbital_energies = calculate_P_next(F_next.reshape(X.shape), X, n_electrons, det, natural_occupation)


        E_prev = E_RHF 

        # Check Convergence Algorithm activation
        if iter == conv_ITER_START and conv_REQUESTED:
            use_conv = True 
            if verbose:
                print('-'*30,  f'   STARTED {conv_type}  ', '-' *30)

    return converged, E_RHF, orbital_energies, C_munu, P

def calculate_P_next(F_0: NDArray[np.float64], X: NDArray[np.float64], n_electrons: int, det, natural_occ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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

    e_values, C_prime, L_prime, R_prime, LFR = diagonalize_biorthogonal(F_prime)

    assert is_diagonal(LFR), "Matrix product L' @ F' @ R' is not diagonal" 

    # Obtain untransformed MO coefficients
    C_munu = X @ C_prime
    L_munu = L_prime @ X
    R_munu = X @ R_prime

    # Build new density matrix
    P_1 = calc_p_matrix_comp(L_munu.T, R_munu, n_electrons, determinant=det, natural_occupation=natural_occ) 

    return P_1, C_munu, e_values

def calculate_F_and_r_comp(P: NDArray[np.float64], S: NDArray[np.float64], H_core: NDArray[np.float64], eri: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    F = H_core + calc_g_matrix_comp(P, eri)
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
    r = S @ P @ F - F @ P @ S
    # print(np.dot(r.flatten() , r.flatten()))
    return r 

def conv_guess(residuals: NDArray[np.float64], F_guesses: NDArray[np.float64]) -> NDArray[np.float64]:
    """ 
    Calculate the Convergence Algorithm extrapolated Fock matrix.

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
    B_matrix = np.zeros([eq_sis_dim, eq_sis_dim], dtype=complex)
    B_matrix[-1,:] = B_matrix[:,-1] = 1
    B_matrix[-1,-1] = 0

    for i in range(n_guesses):
        for j in range(n_guesses):
            B_matrix[i,j] = residuals[i] @ residuals[j]
    
    solution = np.zeros(eq_sis_dim)
    solution[-1] = 1

    # solve the system of equations
    c = np.linalg.solve(B_matrix, solution)

    F_conv = sum([c[i] * F_guesses[i] for i in range(len(c)-1)])
    r_conv = sum([c[i] * residuals[i] for i in range(len(c)-1)])

    return F_conv, r_conv

def theta_traj(max_theta, n_points, overlap, kin, vnuc, eri, nelec, occupation=-1, max_iter=100, threshold=1E-12, p_guess='core', verbose=False, 
    conv_type: Literal[None, 'DIIS', 'CROP'] = 'DIIS',
    conv_MEM: int = 5,
    conv_ITER_START: int = 5,
    ):
    """
    Sample CS_RHF energies along a theta trajectory.

    Parameters
    ----------
    max_theta : float
        Maximum theta to sample (radians).
    n_points : int
        Number of points along the trajectory.
    overlap, kin, vnuc, eri : NDArray
        Integral arrays passed to CS_RHF.
    nelec : int
        Number of electrons.
    occupation : int or array-like, optional
        Occupation vector or -1 for default.
    max_iter, threshold, p_guess, verbose : optional
        SCF control parameters forwarded to CS_RHF.
    conv_type : Literal[None, 'DIIS', 'CROP'], optional
        Type of Convergence Algorithm to use. If None, no algorithm is used.
    conv_MEM : int, optional
        Number of previous Fock matrices and residuals to store for Convergence Algorithm.
    conv_ITER_START : int, optional
        Iteration number to start Convergence Algorithm.

    Returns
    -------
    thetas : NDArray
        Array of sampled theta values.
    energies : list
        List of complex energies for converged points.
    """
    thetas = np.linspace(0, max_theta, n_points)
    energies = []
    for th in thetas:
        converged, E_elec, E_e_values, C_munu, P = CS_RHF(overlap, kin, vnuc, eri, nelec, th, occupation=occupation, max_iter=max_iter, threshold=threshold, p_guess=p_guess, verbose=verbose, conv_type=conv_type, conv_MEM=conv_MEM, conv_ITER_START=conv_ITER_START)
        if converged:
            energies.append(E_elec)
        else:
            print(f'Traj {th} did not converge.')
        if verbose and converged:
            print(f'Converged point at theta = {th:6.4f} : E = {E_elec:12.8f}') 

    return thetas, energies

def plot_theta_traj(energies):
    """
    Plot the complex energy trajectory (Im vs Re) for a list of energies.

    Parameters
    ----------
    energies : sequence of complex
        Energies to plot.

    Returns
    -------
    None
    """
    reals = [energy.real for energy in energies]
    imags = [energy.imag for energy in energies]
    plt.plot(reals, imags, marker='o')
    plt.xlabel('Re(E)')
    plt.ylabel('Im(E)')
    plt.title('Complex Scaled RHF Energy vs Theta')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    # plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    # plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.ticklabel_format(style='sci')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_theta_orbital_energies(energies, theta, xrange=[0,0]):
    """
    Scatter plot of orbital energies (Im vs Re).

    Parameters
    ----------
    energies : sequence of complex
        Orbital energies.
    theta : float
        Theta value used (for title).
    xrange : sequence, optional
        x-axis limits for filtering/zooming. Default disables filtering.

    Returns
    -------
    None
    """
    reals = [energy.real for energy in energies]
    imags = [energy.imag for energy in energies]
    if xrange != [0,0]:
        plt.xlim(xrange)
        reals = [re for re in reals if re < xrange[1]]
        imags = imags[0:len(reals)]

    plt.scatter(reals, imags, marker='o')
    plt.xlabel('Re(Orbital Energies)')
    plt.ylabel('Im(Orbital Energies)')
    plt.ticklabel_format(style='sci')
    plt.title(f'Complex Scaled RHF Orbital Energies at Theta={theta}')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    plt.grid(True, alpha=0.3)
    plt.show()



if __name__ == "__main__":
    pass 