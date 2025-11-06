import numpy as np
from py_mods.src.SCF.scf_utils import (
    validate_determinant,
    transformation_matrix,
    calc_g_matrix_comp,
    calc_p_matrix_comp,
    E_0_comp,
    is_diagonal,
    guess_density,
    diagonalize_biorthogonal,
    scale_integrals
)

from numpy.typing import NDArray
from typing import Literal, Union, Tuple
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
    verbose: bool = False
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

    Returns
    -------
    converged : bool
        True if SCF converged within max_iter.
    E_RHF : complex
        Final complex electronic energy (Hartree).
    e_values : NDArray[np.complex128], shape (n,)
        Orbital energies (possibly complex).
    C_munu : NDArray[np.complex128], shape (n, n)
        Molecular orbital coefficients (untransformed).
    P_new : NDArray[np.complex128], shape (n, n)
        Final density matrix.
    """
    assert len(T) == len(V) == len(S), "Matrices T, V, S must have the same dimensions"
    assert n_electrons % 2 == 0, "RHF can only be closed-shell systems"

    # Otain transformation matrix 
    dim = len(S)

    det, natural_occupation = validate_determinant(n_electrons, occupation, dim)

    X = transformation_matrix(S) + 0j
    # print(type(X[0][0]))

    # rescaling the integrals
    T_scaled, V_scaled, eri_scaled = scale_integrals(T, V, eri, theta)
    H_core = T_scaled + V_scaled

    # Guess initial density matrix
    P_new = guess_density(dim, p_guess)
    P_old = np.copy(P_new)

    E_iter = 0+0j
    Delta_E = 0+0j
    converged = False
    Error = 13

    if verbose:
        print('-'*128)
        print('|   Iter   |               E_iter                  |                       Delta_e                   |        norm(e_i)        |')
        print('-'*128)

    # SCF loop
    for iter in range(max_iter):
        if iter != 0 and Error < threshold:
            converged = True
            if verbose:
                print(f'Convergence achieved after {iter-1} iterations. Final SCF energy = {E_iter}')

            break
        
        # Obtain G matrix from P and eris. Build Fock matrix
        G = calc_g_matrix_comp(P_new, eri_scaled)
        F = G + H_core

        if iter > 0:
            Error_vec = (S @ P_new @ F - F @ P_new @ S).flatten()
            Error = np.sqrt(Error_vec @ Error_vec)

        # Obtain transformed Fock matrix 
        F_prime = X @ F @ X.T

        # Diagonalize transformed Fock matrix to obtain energies and transformed MO coefficients
        e_values, C_prime, L_prime, R_prime, LFR = diagonalize_biorthogonal(F_prime)

        assert is_diagonal(LFR), "Matrix product L' @ F' @ R' is not diagonal" 

        # Obtain untransformed MO coefficients
        C_munu = X @ C_prime
        L_munu = L_prime @ X
        R_munu = X @ R_prime

        # Build new density matrix
        P_old = np.copy(P_new)
        P_new = calc_p_matrix_comp(L_munu.T, R_munu, n_electrons, determinant=det, natural_occupation=natural_occupation) 

        # Calculate HF energy
        E_old = E_iter
        E_iter = E_0_comp(P_new, H_core, F)
        Delta_E = E_iter - E_old

        if verbose:
            print(f'{iter:5}     {E_iter:25.16f}     {Delta_E:45.16f}     {Error:8.4E}')

    E_RHF = E_iter

    return converged, E_RHF, e_values, C_munu, P_new


def theta_traj(max_theta, n_points, overlap, kin, vnuc, eri, nelec, occupation=-1, max_iter=100, threshold=1E-12, p_guess='core', verbose=False):
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
        converged, E_elec, E_e_values, C_munu, P = CS_RHF(overlap, kin, vnuc, eri, nelec, th, occupation=occupation, max_iter=max_iter, threshold=threshold, p_guess=p_guess, verbose=verbose)
        if converged:
            energies.append(E_elec)
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
