import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, Union
from py_mods.src.SCF.scf_utils import transformation_matrix, calc_g_matrix_comp, calc_p_matrix_comp, E_0_comp, guess_density, validate_determinant, scale_integrals, diagonalize_biorthogonal
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class CS_RHF_ContextClass:
    """
    Context class for CS_UHF calculations.

    Attributes
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
    p_guess : Literal['core', 'ones', 'RHF', 'IMPORB'], optional
        Type of initial guess for density matrix.
    guess_MAX_ITER : int or None, optional
        If p_guess is 'RHF', number of iterations to run the preliminary RHF calculation.
    INPORB : NDArray[np.float64] or None, optional
        If p_guess is 'INPORB', the initial guess orbitals.
    break_symm : bool, optional
        If True, breaks the symmetry of the initial guess density matrix.
    verbose : bool, optional
        If True print iteration progress.
    conv_type : Literal[None, 'DIIS', 'CROP'], optional
        Type of Convergence Algorithm to use. If None, no algorithm is used.
    conv_MEM : int, optional
        Number of previous Fock matrices and residuals to store for Convergence Algorithm.
    conv_ITER_START : int, optional
        Iteration number to start Convergence Algorithm.

    Notes
    -----
    - Symmety is broken by zeroing the beta density matrix in the occupied space. 
    - Breaking symmetry only makes sense when the guess is not zeros.
    """
    # Required 
    S: NDArray[np.float64]
    T: NDArray[np.float64]
    V: NDArray[np.float64]
    eri: NDArray[np.float64]
    n_electrons: int

    # Optional 
    theta: float = 0.
    occupation: Union[int, NDArray[np.int32], None] = None
    max_iter: int = 100
    threshold: float = 1E-12
    p_guess: Literal['core', 'ones', 'IMPORB'] = 'core'
    guess_MAX_ITER: Union[int, None] = None
    INPORB: Union[NDArray[np.float64], NDArray[np.complex128], None] = None
    verbose: bool = False
    conv_type: Literal[None, 'DIIS', 'CROP'] = 'DIIS'
    conv_MEM: int = 8
    conv_ITER_START: int = 12

@dataclass
class _RHF_LR_DiagnosticsClass(object):
    E_RHF_LR: np.complex128
    E_RHF_RR: np.complex128
    mean_LR: np.complex128
    max_LR: np.complex128
    P_diff: np.float64
    LR_diff: np.float64
    P_herm: np.float64
    LR_herm: np.float64

@dataclass
class CS_RHF_ResultsClass(object):
    """
    Results class for CS_RHF calculations.

    Attributes
    ----------
    context : CS_RHF_ContextClass
        Context object used for the calculation.
    converged : bool
        Whether the SCF calculation converged.
    E_RHF : complex
        Final RHF energy.
    e_orb : NDArray[np.complex128], shape (n,)
        Orbital energies.
    n_elec : int
        Number of electrons.
    X : NDArray[np.complex128], shape (n, n)
        Transformation matrix.
    P_guess : NDArray[np.complex128], shape (n, n)
        Initial density matrix guess.
    P_LR : NDArray[np.complex128], shape (n, n)
        Final LR density matrix.
    R_munu : NDArray[np.complex128], shape (n, n)
        Right molecular orbital coefficients.
    L_munu : NDArray[np.complex128], shape (n, n)
        Left molecular orbital coefficients.
    LR_diagnostics : _RHF_LR_DiagnosticsClass
        Diagnostics comparing LR and RR densities/energies.
    error : float
        Final residual norm.
    iterations : int
        Number of SCF iterations performed.
    """
    context: CS_RHF_ContextClass
    converged: bool
    E_RHF: float
    e_orb: NDArray[np.complex128]
    n_elec: float
    X: NDArray[np.complex128]
    F_final: NDArray[np.complex128]
    C_prime: NDArray[np.complex128]
    P_guess: NDArray[np.complex128]
    P_LR: NDArray[np.complex128]
    R_munu: NDArray[np.complex128]
    L_munu: NDArray[np.complex128]
    LR_diagnostics: _RHF_LR_DiagnosticsClass
    error: float
    iterations: int


def CS_RHF(ctx: CS_RHF_ContextClass) -> CS_RHF_ResultsClass:
    """
    Perform a Complex Scaled RHF calculation.

    Takes overlap, kinetic, nuclear attraction and two-electron integrals,
    applies complex scaling by angle `theta` and runs an RHF-like SCF loop
    using biorthogonal diagonalization.

    Parameters
    ----------
    context : CS_RHF_ContextClass
        Context object containing parameters and integrals.

    Returns
    -------
    CS_RHF_ResultsClass
        Results object containing convergence status, final energy, orbital energies,
        density matrices, molecular orbital coefficients, and diagnostics.

    Notes
    ------
    - The system bust be a closed shell: n_electrons must be even. This is asserted.
    - Integrals must be passed and have the same dimensions. This is asserted.


    - Implementation was done based on "Modern Quantum Chemistry" by Szabo and Ostlund.
    - DIIS implementation was based on [Pulay](https://doi.org/10.1002/jcc.540030413).
    - CROP implementation was based on [Ettenhuber, Jorgensen](https://doi.org/10.1021/ct501114q).

    ^* CROP algorithm does not compute the new trial as t_opt + w_opt, as it breaks convergence here.
    """
    # unpacking. Easiest to maintain compatibility. 
    S = ctx.S
    T = ctx.T
    V = ctx.V
    eri = ctx.eri
    n_electrons = ctx.n_electrons
    theta = ctx.theta
    occupation = ctx.occupation
    max_iter = ctx.max_iter
    threshold = ctx.threshold
    p_guess = ctx.p_guess
    verbose = ctx.verbose
    conv_type = ctx.conv_type
    conv_MEM = ctx.conv_MEM
    conv_ITER_START = ctx.conv_ITER_START

    assert len(T) == len(V) == len(S), "Matrices T, V, S must have the same dimensions"
    assert n_electrons % 2 == 0, "RHF can only be closed-shell systems"
    assert conv_type in [None, 'DIIS', 'CROP'], 'Convergence assist must be either None, DIIS, or CROP'

    # setup
    conv_REQUESTED = True if  conv_type is not None else False
    conv_ITER_START = min(conv_ITER_START+1,  conv_MEM) if  conv_MEM >= conv_ITER_START else  max(conv_ITER_START+1,  conv_MEM)

    # Otain transformation matrix and validate occupation determinant
    dim = len(S)
    X = transformation_matrix(S) + 0j
    det, natural_occupation = validate_determinant(n_electrons, occupation, dim)

    # rescaling the integrals
    T_scaled, V_scaled, eri_scaled = scale_integrals(T, V, eri, theta)
    H_core = T_scaled + V_scaled

    # Guess initial density matrix
    P_LR = guess_density(dim, p_guess)

    # initialize variables and lists
    E_prev = 0.+0.j
    use_conv = False 
    converged = False
    F_guess = []
    residuals = []

    if verbose:
        print('-'*128)
        print('|   Iter     |                   E_iter                      |                   Delta_e                   |      norm(e_i)      |')
        print('-'*128)

    # SCF loop
    for iter in range(max_iter):
        # calculate F_n and r_n from P_n
        F, r = calculate_F_and_r_comp(P_LR, S, H_core, eri_scaled)
        # print(f'Normal condition: {np.allclose(F.conj().T @ F, F @ F.conj().T)}')
        error = np.linalg.norm(r.flatten())
        E_RHF = E_0_comp(P_LR, H_core, F.reshape(H_core.shape))
        E_diff = E_RHF - E_prev

        if verbose:
            print(f'{iter:5}     {E_RHF:45.16f}     {E_diff:45.16f}     {error:8.4E}')
        # Check convergence
        if iter > 1 and error < threshold:
            converged = True
            if verbose:
                print(f'Convergence achieved after {iter} iterations.\n\n:: Final SCF energy = {E_RHF:5}\n\nFinal SCF energy in parseable format\n%% {E_RHF.real:.14E} {E_RHF.imag:.14E} {theta:.6f}')
            
            P_LR, C_munu, orbital_energies, L_munu, R_munu, P_RR, C_prime = calculate_P_next(
                F.reshape(X.shape), 
                X, 
                n_electrons, 
                det, 
                theta, 
                natural_occupation
            )
            F_next = F 
            
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
            try:
                F_opt, r_opt = conv_guess(residuals, F_guess)

                F_next = F_opt # Default is DIIS

                if conv_type == 'CROP':
                    F_guess[-1] = F_opt
                    residuals[-1] = r_opt  
                    F_next = F_opt # + r_opt # equation 32 Ettenhuber, r_opt should be here, but it diverges idk why

            except np.linalg.LinAlgError:
                if verbose:
                    print('!!!!!!!!!!!!!!!! CONVERGENCE ACCELERATION CAUSED A SINGULAR MATRIX. REVERTING TO STANDARD SCF !!!!!!!!!!!!!!!')
                use_conv = False 

        P_old = np.copy(P_LR)
        
        P_LR, C_munu, orbital_energies, L_munu, R_munu, P_RR, C_prime = calculate_P_next(F_next.reshape(X.shape), X, n_electrons, det, theta, natural_occupation)

        if theta == 0.:
            P_LR.imag = 0 
            L_munu.imag = 0 
            R_munu.imag = 0 
            P_RR.imag = 0 

        E_prev = E_RHF 

        # Check Convergence Algorithm activation
        if iter == conv_ITER_START and conv_REQUESTED:
            use_conv = True 
            if verbose:
                print('-'*30,  f'   STARTED {conv_type}  ', '-' *30)
    
    LR_diagnostics = lr_diagonstics(P_LR, P_RR, L_munu, R_munu, X, H_core, F, n_electrons, verbose)

    ResultClass = CS_RHF_ResultsClass(
        context=ctx,
        converged=converged,
        E_RHF=E_RHF,
        e_orb=orbital_energies,
        n_elec=n_electrons,
        X=X,
        F_final = F_next,
        C_prime=C_prime,
        P_guess=P_old,
        P_LR=P_LR,
        R_munu=R_munu,
        L_munu=L_munu,
        LR_diagnostics=LR_diagnostics,
        error=error,
        iterations=iter
    )

    return ResultClass

def calculate_P_next(F_0: NDArray[np.float64], X: NDArray[np.float64], n_electrons: int, det, theta, natural_occ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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

    normal = True if np.allclose(F_prime @ F_prime.conj().T, F_prime.conj().T @ F_prime) and theta == 0. else False

    unscaled = True if theta == 0. else False

    e_values, C_prime, L_prime, R_prime, LFR = diagonalize_biorthogonal(F_prime)

    # if normal:
    #     plot_map(LFR.real)
    #     assert is_diagonal(LFR, 1E-5), "LFR is not diagonal in schur. Check."
    # else:
    # assert is_diagonal(LFR), "Matrix product L' @ F' @ R' is not diagonal" 
    # print(f'C_munu @ C_munu = {np.conj(C_prime.T) @ C_prime}')

    # Obtain untransformed MO coefficients
    C_munu = X @ C_prime
    L_munu = L_prime @ X
    R_munu = X @ R_prime

    # Build new density matrix
    P_LR = calc_p_matrix_comp(L_munu.T, R_munu, n_electrons, determinant=det, natural_occupation=natural_occ)
    P_RR = calc_p_matrix_comp(np.conj(R_munu), R_munu, n_electrons, determinant=det, natural_occupation=natural_occ)
    
    return P_LR, C_munu, e_values, L_munu, R_munu, P_RR, C_prime

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

def lr_diagonstics(P_LR: NDArray[np.complex128], P_RR: NDArray[np.complex128], L_munu: NDArray[np.complex128], R_munu: NDArray[np.complex128], X, H_core: NDArray[np.complex128], F: NDArray[np.complex128], n_elec, verbose:bool = False)->_RHF_LR_DiagnosticsClass:
    E_RHF_LR = E_0_comp(P_LR, H_core, F.reshape(H_core.shape))
    E_RHF_RR = E_0_comp(P_LR.T, H_core, F.reshape(H_core.shape))

    P_diff = P_LR - P_RR.T
    P_diff = P_LR - P_LR.T

    P_frobenius_norm = np.sqrt(np.trace(P_diff @ np.conjugate(P_diff))).real
    P_frobenius_norm /= n_elec

    invx = np.linalg.inv(X)

    LR_diff = (L_munu @ invx).T - invx @ R_munu

    LR_frobenius_norm = np.sqrt(np.trace(LR_diff @ np.conjugate(LR_diff))).real
    LR_frobenius_norm /= n_elec


    if verbose:
        print('\n')
        print('-'*30,  f'   DIAGNOSTICS   ', '-' *30)
        print(f'LR energy: {E_RHF_LR:.8f}')
        print(f'RR energy: {E_RHF_RR:.8f}')
        print(f'LR-RR E_diff: {E_RHF_LR-E_RHF_RR:.8f}')
        print(f'\nMean P_LR-P_RR difference: {np.mean(P_LR.flatten()-P_RR.flatten()):.8E}')
        print(f'Max  P_LR-P_RR difference: {np.max(P_LR.flatten()-P_RR.flatten()):.8E}')
        print(f'Mean C_LR-C_RR difference: {np.mean(LR_diff):.8E}')
        print(f'Max  C_LR-C_RR difference: {np.max(LR_diff):.8E}')
        print(f'\nP_LR hermiticity diagnostic (P_LR / P_RR): {P_frobenius_norm}')
        print(f'\nC_LR hermiticity diagnostic (L_mn / R_mn): {LR_frobenius_norm}')


    LR_diagnostics = _RHF_LR_DiagnosticsClass(
        E_RHF_LR=E_RHF_LR,
        E_RHF_RR=E_RHF_RR,
        mean_LR=np.mean(P_LR.flatten()-P_RR.flatten()),
        max_LR=np.max(P_LR.flatten()-P_RR.flatten()),
        P_diff=P_diff,
        LR_diff=LR_diff,
        P_herm=P_frobenius_norm,
        LR_herm=LR_frobenius_norm,
    )

    return LR_diagnostics

def RHF_theta_traj(max_theta, n_points, cxt: CS_RHF_ContextClass):
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
        cxt.theta = th
        res = CS_RHF(cxt)
        if res.converged:
            energies.append(res.E_RHF)
        else:
            print(f'Traj {th} did not converge.')
        if cxt.verbose and res.converged:
            print(f'Converged point at theta = {th:6.4f} : E = {res.E_RHF:12.8f}') 

    return thetas, np.array(energies, dtype=np.complex128)

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