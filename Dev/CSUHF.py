from re import A
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, Union
from py_mods.src.SCF.scf_utils import transformation_matrix, guess_density, validate_unrestricted_determinant, scale_integrals, is_diagonal
from Dev.CSRHF import CS_RHF
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class CS_UHF_ContextClass:
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
    mult: Union[None, int] = None
    theta: float = 0.
    occupation: Union[int, NDArray[np.int32], None] = None
    max_iter: int = 100
    threshold: float = 1E-12
    p_guess: Literal['core', 'ones', 'RHF', 'IMPORB'] = 'core'
    guess_MAX_ITER: Union[int, None] = None
    INPORB: Union[NDArray[np.float64], NDArray[np.complex128], None] = None
    break_symm: bool = False
    verbose: bool = False
    conv_type: Literal[None, 'DIIS', 'CROP'] = 'DIIS'
    conv_MEM: int = 8
    conv_ITER_START: int = 12

@dataclass
class _UHF_LR_DiagnosticsClass(object):
    E_RHF_LR: np.complex128
    E_RHF_RR: np.complex128
    mean_LR_alpha: np.complex128
    max_LR_alpha: np.complex128
    mean_LR_beta: np.complex128
    max_LR_beta: np.complex128

@dataclass
class _UHF_SpinDiagnosticsClass(object):
    N_alpha: int
    N_beta: int
    s2: float
    S_z: float
    spin_contamination: float


@dataclass
class CS_UHF_ResultsClass(object):
    """ 
    Results class for CS_UHF calculations.

    Attributes
    ----------
    context : CS_UHF_ContextClass
        Context object used for the calculation.
    converged : bool
        Wether SCF calculation converged.
    E_UHF : float
        Final UHF energy.
    e_alph : NDArray[np.complex128], shape (n,)
        Alpha orbital energies.
    e_beta : NDArray[np.complex128], shape (n,)
        Beta orbital energies.
    X : NDArray[np.complex128], shape (n, n)
        Transformation matrix.
    P_guess_alpha: NDArray[np.complex128], shape (n, n)
        Initial alpha density matrix guess.
    P_guess_beta: NDArray[np.complex128], shape (n, n)
        Initial beta density matrix guess.
    P_LR_alph : NDArray[np.complex128], shape (n, n)
        Alpha density matrix.
    P_LR_beta : NDArray[np.complex128], shape (n, n)
        Beta density matrix.
    P_total : NDArray[np.complex128], shape (n, n)
        Total density matrix.
    P_diff : NDArray[np.complex128], shape (n, n)
        Spin density matrix (P_alpha - P_beta).
    L_alpha : NDArray[np.complex128], shape (n, n)
        Left eigenvector matrix for alpha spin.
    R_alpha : NDArray[np.complex128], shape (n, n)
        Right eigenvector matrix for alpha spin.
    L_beta : NDArray[np.complex128], shape (n, n)
        Left eigenvector matrix for beta spin.
    R_beta : NDArray[np.complex128], shape (n, n)
        Right eigenvector matrix for beta spin.
    """
    context: CS_UHF_ContextClass
    converged: bool
    E_UHF: float
    e_alpha: NDArray[np.complex128]
    e_beta: NDArray[np.complex128]
    n_alpha: float
    n_beta: float
    X: NDArray[np.complex128]
    P_guess_alpha: NDArray[np.complex128]
    P_guess_beta: NDArray[np.complex128]
    P_LR_alpha: NDArray[np.complex128]
    P_LR_beta: NDArray[np.complex128]
    P_total: NDArray[np.complex128]
    P_diff: NDArray[np.complex128]
    L_alpha: NDArray[np.complex128]
    R_alpha: NDArray[np.complex128]
    L_beta: NDArray[np.complex128]
    R_beta: NDArray[np.complex128]
    LR_diagnostics: _UHF_LR_DiagnosticsClass
    S_diagnostics: _UHF_SpinDiagnosticsClass
    error: float
    iterations: int

def CS_UHF(context: CS_UHF_ContextClass) -> CS_UHF_ResultsClass:
    """
    Perform a Complex Scaled RHF calculation.

    Takes overlap, kinetic, nuclear attraction and two-electron integrals,
    applies complex scaling by angle `theta` and runs an UHF loop
    using biorthogonal diagonalization.

    Parameters
    ----------
    context : CS_UHF_Context
        Context object containing all parameters for the calculation.

    Returns
    -------
    CS_UHF_Results
        Results object. For complete description of parameters see definition. 

    Notes
    ------
    - The system bust be a closed shell: n_electrons must be even. This is asserted.
    - Integrals must be passed and have the same dimensions. This is asserted.


    - Implementation was done based on "Modern Quantum Chemistry" by Szabo and Ostlund.
    - DIIS implementation was based on [Pulay](https://doi.org/10.1002/jcc.540030413).
    - CROP implementation was based on [Ettenhuber, Jorgensen](https://doi.org/10.1021/ct501114q).

    ^* CROP algorithm does not compute the new trial as t_opt + w_opt, as it breaks convergence here.
    """
    assert len(context.T) == len(context.V) == len(context.S), "Matrices T, V, S must have the same dimensions"
    # assert n_electrons % 2 != 0, "For closed-shell calculations use RHF routine."
    assert context.conv_type in [None, 'DIIS', 'CROP'], 'Convergence assist must be either None, DIIS, or CROP'

    # setup
    conv_REQUESTED = True if  context.conv_type is not None else False
    conv_ITER_START = min(context.conv_ITER_START+1,  context.conv_MEM) if  context.conv_MEM >= context.conv_ITER_START else  max(context.conv_ITER_START+1,  context.conv_MEM)
    
    if context.mult is None:
        context.mult = int(0) if  context.n_electrons % 2 == 0 else int(1) 
    assert (context.n_electrons - context.mult) % 2 != 1, f"It is not possible to have {context.mult} unpaired electrons with { context.n_electrons} electrons."
    
    # obtain the occupations
    dim = len(context.S)
    det_alpha, det_beta, _ = validate_unrestricted_determinant( context.n_electrons, context.occupation, dim, context.mult)

    alpha_elec = sum(det_alpha)
    context.n_alpha_initial = alpha_elec
    context.alpha_occ = det_alpha
    beta_elec = sum(det_beta)
    context.n_beta_initial = beta_elec
    context.beta_occ = det_beta

    if context.verbose:
        print('\n\nAlpha occupation: ', det_alpha)
        print('Beta  occupation: ', det_beta)

    # Otain transformation matrix and validate occupation determinant
    dim = len(context.S)
    X = transformation_matrix(context.S) + 0j

    # rescaling the integrals
    T_scaled, V_scaled, eri_scaled = scale_integrals(context.T, context.V, context.eri, context.theta)
    H_core = T_scaled + V_scaled

    # Guess initial density matrix
    if context.p_guess == 'RHF':
        elec_pre = context.n_electrons if context.n_electrons % 2 == 0 else context.n_electrons-1 
        if isinstance(context.guess_MAX_ITER, int):
            guess_iter = context.guess_MAX_ITER
        else: 
            guess_iter = 8
        _, _, _, _, P_LR_alph, *_ = CS_RHF(context.S, context.T, context.V, context.eri, n_electrons=elec_pre, theta=0, max_iter=guess_iter, threshold=1E-14, p_guess='core', verbose=False)


    elif context.p_guess == 'INPORB':
        assert context.INPORB is not None, 'Empty INPORB alpha for guess'

        assert isinstance(context.INPORB, np.array) and context.INPORB.shape == X.shape, f'Wrong type ({type(context.INPORB)}) or dimensions ({context.INPORB.shape}) of import guess orbitals, expexted {type(X)} and {X.shape}'
        P_LR_alph = np.copy(context.INPORB) 

    else: 
        P_LR_alph = guess_density(dim, context.p_guess)
    
    # P_LR_alph *= context.S # this leads to a closer guess to the PySCF one
    P_LR_beta = np.copy(P_LR_alph)
    
    if context.break_symm: #note that breaking symmetry will only make sense when the guess is not zeros
        P_LR_beta[:context.n_electrons, :context.n_electrons] = 0

    P_guess_alpha = np.copy(P_LR_alph)
    P_guess_beta = np.copy(P_LR_beta)

    # initialize variables and lists
    E_prev = 0.+0.j
    use_conv = False 
    converged = False
    F_guess_alph = []
    F_guess_beta = []
    residuals_alph = []
    residuals_beta = []

    mem_iter = context.max_iter
    conv_thresh = 1E-4

    if context.verbose:
        print('-'*128)
        print('|   Iter   |               E_iter                  |                       Delta_e                   |        norm(e_i)        |')
        print('-'*128)

    # SCF loop
    for iteration in range(context.max_iter):
        # calculate F_n and r_n from P_n
        F_alph, r_alph, F_beta, r_beta = calculate_F_and_r_comp(P_LR_alph, P_LR_beta, context.S, H_core, eri_scaled)

        error_alph = np.linalg.norm(r_alph.flatten())
        error_beta = np.linalg.norm(r_beta.flatten())

        error = max(error_alph, error_beta)

        E_UHF = E_0_unrestricted_comp(P_LR_alph, P_LR_beta, H_core, F_alph.reshape(H_core.shape), F_beta.reshape(H_core.shape))

        E_diff = E_UHF - E_prev

        if context.verbose:
            print(f'{iteration:5}     {E_UHF:45.16f}     {E_diff:45.16f}     {error:8.4E}')

        # Check convergence
        if iteration > 5 and error < context.threshold:
            converged = True
            if context.verbose:
                print(f'Convergence achieved after {iteration} iterations.\n\n:: Final SCF energy = {E_UHF:5}\n\nFinal SCF energy in parseable format\n%% {E_UHF.real:.14E} {E_UHF.imag:.14E} {context.theta:.6f}')
            break

        # Save in memory guesses and residuals keeping size of Convergence Algorithm space

        F_guess_alph.append(F_alph)
        F_guess_beta.append(F_beta)
        residuals_alph.append(r_alph)
        residuals_beta.append(r_beta)

        if len(F_guess_alph) > context.conv_MEM:
            F_guess_alph.pop(0)
            F_guess_beta.pop(0)
            residuals_alph.pop(0)
            residuals_beta.pop(0)
        
        # Choose F for P_{n+1}
        if not use_conv:
            F_next_alph = F_alph
            F_next_beta = F_beta 
        
        elif use_conv:
            try:
                F_opt_alph, r_opt_alpha = conv_guess(residuals_alph, F_guess_alph)
                F_opt_beta, r_opt_beta  = conv_guess(residuals_beta, F_guess_beta)

                # Default is DIIS
                F_next_alph = F_opt_alph
                F_next_beta = F_opt_beta             

                if context.conv_type == 'CROP':
                    F_guess_alph[-1] = F_opt_alph 
                    F_guess_beta[-1] = F_opt_beta 
                    residuals_alph[-1] = r_opt_alpha
                    residuals_beta[-1] = r_opt_beta 

                    # + r_opt # equation 32 Ettenhuber, r_opt should be here, but it diverges idk why

                    F_next_alph = F_opt_alph # + r_opt_alpha
                    F_next_beta = F_opt_beta # + r_opt_beta
            except np.linalg.LinAlgError:
                print('!!!!!!!!!!!!!!!! CONVERGENCE ACCELERATION CAUSED A SINGULAR MATRIX. REVERTING TO STANDARD SCF !!!!!!!!!!!!!!!')
                use_conv = False 

        
        P_LR_alph, R_alph, L_alph, e_alph, P_RR_alph = calculate_P_next(F_next_alph, X, alpha_elec, det_alpha)
        P_LR_beta, R_beta, L_beta, e_beta, P_RR_beta = calculate_P_next(F_next_beta, X,  beta_elec, det_beta)

        P_total = P_LR_alph + P_LR_beta
        P_diff = P_LR_alph - P_LR_beta

        E_prev = E_UHF

        # Check Convergence Algorithm activation
        if iteration == conv_ITER_START and conv_REQUESTED: #  and error < conv_thresh and not use_conv:
            use_conv = True 
            if context.verbose:
                print('-'*30,  f'   STARTED {context.conv_type}  ', '-' *30)


    LR_diagnostics = UHF_LR_diagnostic(P_LR_alph, P_LR_beta, P_RR_alph, P_RR_beta, H_core, F_alph, F_beta, context.verbose)
    S_diagnostics = calculate_s2_expectation(P_LR_alph, P_LR_beta, context.S, context.verbose)

    n_alpha = np.trace(P_LR_alph.real @ context.S)
    n_beta  = np.trace(P_LR_beta.real @ context.S)

    assert abs(n_alpha + n_beta - context.n_electrons) < 1E-10, 'Number of electrons was not conserved in the calculation'

    ResultClass = CS_UHF_ResultsClass(context, converged, E_UHF, e_alph, e_beta, n_alpha, n_beta, X, P_guess_alpha, P_guess_beta, P_LR_alph, P_LR_beta, P_total, P_diff, L_alph, R_alph, L_beta, R_beta, LR_diagnostics, S_diagnostics, error, iteration)

    return ResultClass

def calculate_P_next(F_alpha: NDArray[np.float64], X: NDArray[np.float64], n_electrons: int, det) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
    F_alpha = F_alpha.reshape(X.shape)

    F_prime = X @ F_alpha @ X.T

    e_values, C_prime, L_prime, R_prime, LFR = diagonalize_biorthogonal(F_prime)

    assert is_diagonal(LFR), "Matrix product L' @ F' @ R' is not diagonal"

    # Obtain untransformed MO coefficients
    C_munu = X @ C_prime
    L_munu = L_prime @ X
    R_munu = X @ R_prime

    # Build new density matrix
    mask = (det == 1).astype(float)
    P_LR = np.einsum('ma, a, an -> mn', R_munu, mask, L_munu)
    P_RR = np.einsum('ma, a, an -> mn', R_munu, mask, R_munu)


    return P_LR, R_munu, L_munu, e_values, P_RR

def calculate_F_and_r_comp(P_alpha: NDArray[np.float64], P_beta, S: NDArray[np.float64], H_core: NDArray[np.float64], eri: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    r_alpha = residual(F_alpha, P_alpha, S)
    r_beta = residual(F_beta, P_beta, S)

    return F_alpha.flatten(), r_alpha.flatten(), F_beta.flatten(), r_beta.flatten()

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



def E_0_unrestricted_comp(P_alpha, P_beta, H_core, F_alpha, F_beta):

    E_elec = 0.5 * (
        np.einsum('mn,mn->', P_alpha, H_core + F_alpha)
    + np.einsum('mn,mn->', P_beta,  H_core  + F_beta)
    )
    return E_elec


def calc_g_matrix_spin_comp(P_alpha, P_beta, eri) -> NDArray[np.complex128]:

    P_total = P_alpha + P_beta
    # J from total density
    J = np.einsum('mnsl, ls -> mn', eri, P_total)
    # K from same-spin density
    K_alpha = np.einsum('mlns, ls -> mn', eri, P_alpha)
    K_beta  = np.einsum('mlns, ls -> mn', eri, P_beta)

    G_alpha = J - K_alpha
    G_beta  = J - K_beta

    return G_alpha, G_beta

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

    assert is_diagonal(LFR), "LFR is not diagonal. Check."

    return e_values, C_prime, L_prime, R_prime, LFR

def UHF_LR_diagnostic(P_LR_alph, P_LR_beta, P_RR_alph, P_RR_beta, H_core, F_alph, F_beta, verbose=False):
    E_RHF_LR = E_0_unrestricted_comp(P_LR_alph, P_LR_beta, H_core, F_alph.reshape(H_core.shape), F_beta.reshape(H_core.shape))
    E_RHF_RR = E_0_unrestricted_comp(P_RR_alph, P_RR_beta, H_core, F_alph.reshape(H_core.shape), F_beta.reshape(H_core.shape))
    mean_LR_alpha = np.mean(P_LR_alph.flatten()-P_RR_alph.flatten())
    max_LR_alpha = np.max(P_LR_alph.flatten()-P_RR_alph.flatten())
    mean_LR_beta = np.mean(P_LR_beta.flatten()-P_RR_beta.flatten())
    max_LR_beta = np.max(P_LR_beta.flatten()-P_RR_beta.flatten())

    if verbose:
        print('\n')
        print('-'*30,  f'   LR DIAGNOSTICS   ', '-' *30)
        print(f'LR energy (Normal LR for both alpha and beta): {E_RHF_LR:.8f}')
        print(f'RR energy (Using  RR for both alpha and beta): {E_RHF_RR:.8f}')
        print(f'LR-RR E_diff: {E_RHF_LR-E_RHF_RR:.8f}')
        print(f'\nMean Alpha P_LR-P_RR difference: {mean_LR_alpha:16.8E}')
        print(f'Max  Alpha P_LR-P_RR difference: {max_LR_alpha:16.8E}')
        print(f'\nMean Beta  P_LR-P_RR difference: {mean_LR_beta:16.8E}')
        print(f'Max  Beta  P_LR-P_RR difference: {max_LR_beta:16.8E}')
    
    return _UHF_LR_DiagnosticsClass(E_RHF_LR, E_RHF_RR, mean_LR_alpha, max_LR_alpha, mean_LR_beta, max_LR_beta)


def calculate_s2_expectation(P_alpha, P_beta, S, verbose=False):
    """
    Calculate the expectation value of S^2 for a UHF wavefunction.

    Calculated using <S^2> = S_z^2 + S_z + N_beta - Tr(P_alpha @ S @ P_beta @ S)

    Parameters
    ----------
    P_alpha : NDArray, shape (n,n)
    Alpha density matrix
    P_beta : NDArray, shape (n,n)
    Beta density matrix
    S : NDArray, shape (n,n)
    Overlap matrix

    Returns
    -------
    s2 : float
    <S^2> expectation value

    s_z : float
    S_z value
    spin_contamination : float

    Amount of spin contamination (deviation from exact value)
    """

    # Calculate number of electrons
    N_alpha = np.trace(P_alpha.real @ S)
    N_beta  = np.trace(P_beta.real @ S)

    # Calculate S_z
    S_z = (N_alpha - N_beta) / 2

    # Calculate <S^2>
    overlap_term = np.trace(P_alpha @ S @ P_beta @ S).real
    s2 = S_z * (S_z + 1) + N_beta - overlap_term

    # Expected value for pure spin state
    s2_exact = S_z * (S_z + 1)
    spin_contamination = s2 - s2_exact

    if verbose:
        print(f"\n---------------  Spin Diagnostics  ---------------")
        print(f"N_alpha = {(N_alpha):6f}")
        print(f"N_beta  = {(N_beta):6f}")
        print(f"S_z = {S_z:.4f}")
        print(f"<S^2> = {s2:.6f}")
        print(f"<S^2>_exact = {s2_exact:.4f}")
        print(f"Spin contamination = {spin_contamination:.6f}")

    return _UHF_SpinDiagnosticsClass(N_alpha, N_beta, s2, S_z, spin_contamination)


def UHF_theta_traj(max_theta, n_points, context: CS_UHF_ContextClass):
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
        context.theta = th
        result = CS_UHF(context)
        if result.converged:
            energies.append(result.E_UHF)
        else:
            print(f'Traj {th} did not converge.')
        if context.verbose and result.converged:
            print(f'Converged point at theta = {th:6.4f} : E = {result.E_UHF:12.8f}') 

    return thetas, energies

if __name__ == "__main__":
    pass