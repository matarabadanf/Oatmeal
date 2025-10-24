import numpy as np 
from py_mods.src.RHF import RHF
from py_mods.src.scf_utils import * 
from pyscf import gto, scf
from pathlib import Path
from numpy.typing import NDArray
from typing import Literal, Tuple
import matplotlib.pyplot as plt 
import scipy

data_path = Path(__file__).parent / "data"

# print(data_path)

mol_He= gto.M(atom = 'He 0 0 0', spin=0, charge=0, basis='aug-cc-pvtz')
mol_He.basis = {'He': gto.basis.parse(
    '''He    S
      2.340000E+02           0.000000E+00           2.587000E-03           0.000000E+00
      3.516000E+01           0.000000E+00           1.953300E-02           0.000000E+00
      7.989000E+00           0.000000E+00           9.099800E-02           0.000000E+00
      2.212000E+00           0.000000E+00           2.720500E-01           0.000000E+00
      6.669000E-01           1.000000E+00           4.780650E-01           0.000000E+00
      2.089000E-01           0.000000E+00           3.077370E-01           1.000000E+00
He    S
      0.0513800              1.0000000
He    P
      3.044000E+00           1.000000E+00           0.000000E+00
      7.580000E-01           0.000000E+00           1.000000E+00
He    P
      0.1993000              1.0000000
He    D
      1.965000E+00           1.0000000
He    D
      0.4592000              1.0000000
    '''
)}

# mol_He.build()

# print(mol_He.basis)


kin = mol_He.intor('int1e_kin')
vnuc = mol_He.intor('int1e_nuc')
overlap = mol_He.intor('int1e_ovlp')
eri = mol_He.intor('int2e')


# np.savetxt(f'{data_path}/He_kin_ccpvdz.dat', kin)
# np.savetxt(f'{data_path}/He_vnuc_ccpvdz.dat', vnuc)
# np.savetxt(f'{data_path}/He_S_ccpvdz.dat', overlap)
# np.save(f'{data_path}/He_eri_ccpvdz.npy', eri) # cannot He savetxt, has to He np binary

rhf_He = scf.RHF(mol_He)
e_He = rhf_He.kernel()

# np.save('../He_e_hf_ccpvdz', e_He)

print(e_He)


def CS_RHF(
    S: NDArray[np.float64],
    T: NDArray[np.float64], 
    V: NDArray[np.float64], 
    eri: NDArray[np.float64], 
    n_electrons: int, 
    theta: float,
    max_iter: int = 100, 
    threshold: float = 1E-12, 
    p_guess: Literal['core', 'ones'] = 'core', 
    verbose: bool = False
) -> Tuple[bool, float, NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Perform a RHF calculation.

    Takes S, T, V and eri matrix elements and computes the RHF procedure. 

    Introduces complex scaling by an angle theta. 

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
    X = transformation_matrix(S) + 0j
    # print(type(X[0][0]))

    # rescaling the integrals
    T_scaled = T * np.exp(-(0+2j) * theta)
    V_scaled = V * np.exp(-(0+1j) * theta)
    eri_scaled = eri * np.exp(-(0+1j) * theta)

    # print(np.max(np.abs(T-T_scaled)))
    # print(np.max(np.abs(V-V_scaled)))
    # print(np.max(np.abs(eri-eri_scaled)))

    # print(X)

    # Guess initial density matrix
    if p_guess == 'core':
        P = np.zeros([dim, dim], dtype=np.complex128)
    elif p_guess == 'ones':
        P = np.ones([dim, dim], dtype=np.complex128)
    
    P_old = np.zeros([dim, dim], dtype=np.complex128)
    P_new = np.copy(P)

    # Build core Hamiltonian
    H_core = T_scaled + V_scaled

    E_iter = 0+0j
    Delta_E = 0+0j
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

            break
        
        # Obtain G matrix from P and eris. Build Fock matrix
        G = calc_g_matrix_comp(P_new, eri_scaled)
        F = G + H_core

        # Obtain transformed Fock matrix 
        F_prime = X @ F @ X.T

        # print(F_prime)

        # Diagonalize transformed Fock matrix to obtain energies and transformed MO coefficients
        e_values, C_prime = np.linalg.eig(F_prime) # here is eig because we are in the scaled case

        # to explore tomorrow. We need to look at biorthogonal solutions. 
        # eigvals, C_R_prime, C_L_prime = scipy.linalg.eig(F, S, left=True, right=True)

        idx = e_values.argsort()
        e_values = e_values[idx]
        C_prime = C_prime[:,idx]

        # print(e_values)

        R_prime = np.copy(C_prime)
        L_prime = np.linalg.inv(C_prime)

        LFR = L_prime @ F_prime @ R_prime

        if iter == -1:
            print('C_prime at first iteration:\n', C_prime)
            print('\nL_prime at first iteration:\n', L_prime)
            print('\nR_prime at first iteration:\n', R_prime)
            print('\nF_prime at first iteration:\n', F_prime)
            print("\nL'F'R' at first iteration:\n", LFR)
            print('\nEigenvalues at first iteration:\n', e_values)

        assert is_diagonal(LFR), "Matrix product L' @ F' @ R' is not diagonal" 

        # Obtain untransformed MO coefficients
        C_munu = X @ C_prime
        L_munu = L_prime @ X
        R_munu = X @ R_prime

        
        if iter == -1:
            print('\n\n\nC_munu at first iteration:\n', C_munu)
            print('\nL_munu at first iteration:\n', L_munu)
            print('\nR_munu at first iteration:\n', R_munu)

        # Build new density matrix
        P_old = np.copy(P_new)
        P_new = calc_p_matrix_comp(L_munu.T, R_munu, n_electrons, iter) # TODO: why do i have to transpose here??

        # Calculate HF energy
        E_old = E_iter
        E_iter = E_0_comp(P_new, H_core, F)
        Delta_E = E_iter - E_old

        if verbose:
            print(f'{iter:5}     {E_iter:25.16f}     {Delta_E:25.16f}')

    # print('Type of P_old matrix is:', type(P_new[0][0]))
    # print('Type of P_new matrix is:', type(P_new[0][0]))
    # print('Type of E_iter is:', type(E_iter))
    # print('Type of e_values is:', type(e_values[0]))
    # print('Type of C_munu is:', type(C_munu[0][0]))
    # print('Type of converged is:', type(converged))
    # print('Type of X is:', type(X[0][0]))
    # print('Type of T_scaled is:', type(T_scaled[0][0]))
    # print('Type of V_scaled is:', type(V_scaled[0][0]))
    # print('Type of eri_scaled is:', type(eri_scaled[0][0][0][0]))
    # print('Type of H_core is:', type(H_core[0][0]))
    # print('Type of F_prime is:', type(F_prime[0][0]))

    E_RHF = E_iter

    return converged, E_RHF, e_values, C_munu, P_new

nelec = 2
theta = .5

converged, E_elec_comp, E_e_values, C_munu, P = CS_RHF(overlap, kin, vnuc, eri, nelec, theta, max_iter=100, threshold=1E-12, p_guess='core', verbose=False)


converged, E_elec, E_e_values, C_munu, P = RHF(overlap, kin, vnuc, eri, nelec, max_iter=100, threshold=1E-12, p_guess='core', verbose=False)
print('\n\n\n', E_elec_comp)
print(E_elec)
