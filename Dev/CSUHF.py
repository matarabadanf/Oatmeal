from ast import iter_child_nodes
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, Union
from py_mods.src.SCF.scf_utils import transformation_matrix, guess_density, validate_unrestricted_determinant, scale_integrals, is_diagonal
import matplotlib.pyplot as plt

def CS_UHF(
    S: NDArray[np.float64],
    T: NDArray[np.float64], 
    V: NDArray[np.float64], 
    eri: NDArray[np.float64], 
    n_electrons: int, 
    mult: int = 1,
    theta: float = 0.,
    occupation: Union[int, NDArray[np.int32], None] = None,
    max_iter: int = 100, 
    threshold: float = 1E-12, 
    p_guess: Literal['core', 'ones'] = 'core', 
    verbose: bool = False,
    conv_type: Literal[None, 'DIIS', 'CROP'] = 'DIIS',
    conv_MEM: int = 8,
    conv_ITER_START: int = 12,
    diagnostics: bool = False,
) -> Tuple[bool, float, NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
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
    diagnostics : bool, optional
        If True, print L-R eigenvector diagnostics at the end of the calculation.

    Returns
    -------
    Tuple containing:
        - converged (bool): Convergence status.
        - E_RHF (float): Final RHF energy.
        - e_values (NDArray[np.float64][n, n]): Orbital energies.
        - C_munu (NDArray[np.float64][n, n]): R Molecular orbital coefficients.
        - P_LR (NDArray[np.float64][n, n]): Final density matrix.
        - L_munu (NDArray[np.float64][n, n]): L Molecular orbital coefficients.
        - R_munu (NDArray[np.float64][n, n]): R Molecular orbital coefficients.
        - P_RR (NDArray[np.float64][n, n]): Final RR density matrix.

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
    # assert n_electrons % 2 != 0, "For closed-shell calculations use RHF routine."
    assert (n_electrons - mult) % 2 != 1, f"It is not possible to have {mult} unpaired electrons with {n_electrons} electrons."
    assert conv_type in [None, 'DIIS', 'CROP'], 'Convergence assist must be either None, DIIS, or CROP'

    # setup
    conv_REQUESTED = True if conv_type is not None else False
    conv_ITER_START = min(conv_ITER_START+1, conv_MEM) if conv_MEM >= conv_ITER_START else  max(conv_ITER_START+1, conv_MEM)

    # obtain the occupations
    dim = len(S)
    det_alpha, det_beta, natural_occ = validate_unrestricted_determinant(n_electrons, occupation, dim, mult)

    alpha_elec = sum(det_alpha)
    beta_elec = sum(det_beta)
    print('Alpha occupation: ', det_alpha)
    print('Beta  occupation: ', det_beta)

    # Otain transformation matrix and validate occupation determinant
    dim = len(S)
    X = transformation_matrix(S) + 0j

    # rescaling the integrals
    T_scaled, V_scaled, eri_scaled = scale_integrals(T, V, eri, theta)
    H_core = T_scaled + V_scaled
    # H_core = T + V 

    # Guess initial density matrix
    P_LR_alph = guess_density(dim, p_guess)
    P_LR_beta = np.copy(P_LR_alph)

    P_LR_alph, _, _, _ = calculate_P_next(H_core.flatten(), X, alpha_elec, det_alpha, diagnostics)
    P_LR_beta, _, _, _ = calculate_P_next(H_core.flatten(), X,  beta_elec, det_beta, diagnostics)

    P_LR_alph_0 = P_LR_alph

    # initialize variables and lists
    E_prev = 0.+0.j
    use_conv = False 
    converged = False
    F_guess_alph = []
    F_guess_beta = []
    residuals_alph = []
    residuals_beta = []

    mem_iter = max_iter
    conv_thresh = 1E-4

    if verbose:
        print('-'*128)
        print('|   Iter   |               E_iter                  |                       Delta_e                   |        norm(e_i)        |')
        print('-'*128)

    # SCF loop
    for iter in range(max_iter):
        # calculate F_n and r_n from P_n
        F_alph, r_alph, F_beta, r_beta = calculate_F_and_r_comp(P_LR_alph, P_LR_beta, S, H_core, eri_scaled)

        error_alph = np.linalg.norm(r_alph.flatten())
        error_beta = np.linalg.norm(r_beta.flatten())

        error = max(error_alph, error_beta)

        E_UHF = E_0_unrestricted_comp(P_LR_alph, P_LR_beta, H_core, F_alph.reshape(H_core.shape), F_beta.reshape(H_core.shape))

        E_diff = E_UHF - E_prev

        if verbose:
            print(f'{iter:5}     {E_UHF:45.16f}     {E_diff:45.16f}     {error:8.4E}')

        # Check convergence
        if iter > 1 and error < threshold:
            converged = True
            if verbose:
                print(f'Convergence achieved after {iter} iterations. Final SCF energy = {E_UHF:5}')
            break

        # Save in memory guesses and residuals keeping size of Convergence Algorithm space

        F_guess_alph.append(F_alph )
        F_guess_beta.append(F_beta )
        residuals_alph.append(r_alph )
        residuals_beta.append(r_beta )

        if len(F_guess_alph) > conv_MEM:
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

                if conv_type == 'CROP':
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

        
        P_LR_alph, R_alph, L_alph, e_alph = calculate_P_next(F_next_alph, X, alpha_elec, det_alpha, diagnostics)
        P_LR_beta, R_beta, L_beta, e_beta = calculate_P_next(F_next_beta, X,  beta_elec, det_beta,  diagnostics)
        
        P_total = P_LR_alph + P_LR_beta
        P_minus = P_LR_alph - P_LR_beta

        
        E_prev = E_UHF

        # Check Convergence Algorithm activation
        if iter == conv_ITER_START and conv_REQUESTED: #  and error < conv_thresh and not use_conv:
            use_conv = True 
            if verbose:
                print('-'*30,  f'   STARTED {conv_type}  ', '-' *30)

    # if diagnostics:
    #     E_RHF_LR = E_0_comp(P_LR, H_core, F.reshape(H_core.shape))
    #     E_RHF_RR = E_0_comp(P_RR, H_core, F.reshape(H_core.shape))
    #     print('\n')
    #     print('-'*30,  f'   DIAGNOSTICS   ', '-' *30)
    #     print(f'LR energy: {E_RHF_LR:.8f}')
    #     print(f'RR energy: {E_RHF_RR:.8f}')
    #     print(f'LR-RR E_diff: {E_RHF_LR-E_RHF_RR:.8f}')
    #     print(f'\nMean P_LR-P_RR difference: {np.mean(P_LR.flatten()-P_RR.flatten()):.8E}')
    #     print(f'Max  P_LR-P_RR difference: {np.max(P_LR.flatten()-P_RR.flatten()):.8E}')
    P_orthogonal = X @ P_total @ X.T

    return converged, E_UHF, e_alph, e_beta, P_LR_alph, P_LR_beta, P_LR_alph_0, P_orthogonal# , orbital_energies, C_munu, L_munu.T, R_munu, P_RR

def calculate_P_next(F_alpha: NDArray[np.float64], X: NDArray[np.float64], n_electrons: int, det, diagnostics=False) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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

    # sort_idx = np.argsort(e_values.real)
    # e_values = e_values[sort_idx]
    # C_prime = C_prime[:, sort_idx]
    # L_prime = L_prime[:, sort_idx]
    # R_prime = R_prime[:, sort_idx]
    # LFR = LFR[sort_idx][:, sort_idx] if LFR is not None else LFR

    assert is_diagonal(LFR), "Matrix product L' @ F' @ R' is not diagonal"

    # Obtain untransformed MO coefficients
    C_munu = X @ C_prime
    L_munu = L_prime @ X
    R_munu = X @ R_prime

    # Build new density matrix

    mask = (det == 1).astype(float)
    P_spin = np.einsum('ma, a, an -> mn', R_munu, mask, L_munu)

    return P_spin, R_munu, L_munu, e_values

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




def UHF_theta_traj(max_theta, n_points, overlap, kin, vnuc, eri, nelec, occupation=-1, max_iter=100, threshold=1E-12, p_guess='core', verbose=False, 
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
        converged, E_elec, *_ = CS_UHF(overlap, kin, vnuc, eri, nelec, th, occupation=occupation, max_iter=max_iter, threshold=threshold, p_guess=p_guess, verbose=verbose, conv_type=conv_type, conv_MEM=conv_MEM, conv_ITER_START=conv_ITER_START)
        if converged:
            energies.append(E_elec)
        else:
            print(f'Traj {th} did not converge.')
        if verbose and converged:
            print(f'Converged point at theta = {th:6.4f} : E = {E_elec:12.8f}') 

    return thetas, energies
if __name__ == "__main__":
    pass