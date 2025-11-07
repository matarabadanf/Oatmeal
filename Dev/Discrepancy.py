import  Dev.CS_CROP 
import py_mods.src.SCF.CSRHF
from py_mods.src.SCF.RHF import plot_map
from pyscf import gto, scf
import numpy as np 

dist = 1.4 * 0.529177249

mol_He= gto.M(atom = 'He 0 0 0', spin=0, charge=0,) # basis='aug-cc-pVqZ')

big_basis='''
He    S
      1.145000E+03           0.000000E+00           0.000000E+00           3.590000E-04           0.000000E+00           0.000000E+00
      1.717000E+02           0.000000E+00           0.000000E+00           2.771000E-03           0.000000E+00           0.000000E+00
      3.907000E+01           0.000000E+00           0.000000E+00           1.425100E-02           0.000000E+00           0.000000E+00
      1.104000E+01           0.000000E+00           0.000000E+00           5.556600E-02           0.000000E+00           0.000000E+00
      3.566000E+00           1.000000E+00           0.000000E+00           1.620910E-01           0.000000E+00           0.000000E+00
      1.240000E+00           0.000000E+00           1.000000E+00           3.321970E-01           0.000000E+00           0.000000E+00
      4.473000E-01           0.000000E+00           0.000000E+00           4.196150E-01           1.000000E+00           0.000000E+00
      1.640000E-01           0.000000E+00           0.000000E+00           1.861280E-01           0.000000E+00           1.000000E+00
He    S
      0.0466400              1.0000000
He    P
      1.015300E+01           1.000000E+00           0.000000E+00           0.000000E+00           0.000000E+00
      3.627000E+00           0.000000E+00           1.000000E+00           0.000000E+00           0.000000E+00
      1.296000E+00           0.000000E+00           0.000000E+00           1.000000E+00           0.000000E+00
      4.630000E-01           0.000000E+00           0.000000E+00           0.000000E+00           1.000000E+00
He    P
      0.1400000              1.0000000
He    D
      7.666000E+00           1.000000E+00           0.000000E+00           0.000000E+00
      2.647000E+00           0.000000E+00           1.000000E+00           0.000000E+00
      9.140000E-01           0.000000E+00           0.000000E+00           1.000000E+00
He    D
      0.2892000              1.0000000
He    F
      5.411000E+00           1.000000E+00           0.000000E+00
      1.707000E+00           0.000000E+00           1.000000E+00
He    F
      0.5345000              1.0000000
He    G
      3.430000E+00           1.0000000
He    G
      0.7899000              1.0000000
'''

mol_He.basis = {'He': gto.basis.parse(big_basis)}
mol_He.build()

T = mol_He.intor('int1e_kin')
V = mol_He.intor('int1e_nuc')
S = mol_He.intor('int1e_ovlp')
eri = mol_He.intor('int2e')

rhf_He = scf.RHF(mol_He)
# rhf_He.init_guess = 'hcore'
# rhf_He.max_cycle = 0

e_He = rhf_He.kernel()
e_elec = rhf_He.energy_elec()

occupation_determinant = np.array([0,2,0])

# test : SCF convergence for H2 in STO-3G
converged, E_RHF, orbital_energies, C_munu, P_1 =           Dev.CS_CROP.CS_RHF(S, T, V, eri, n_electrons=2, theta=.05, max_iter=100, threshold=1E-12,  occupation=occupation_determinant, p_guess='core', verbose=True, conv_type='CROP')
converged, E_RHF, orbital_energies, C_munu, P_2 = py_mods.src.SCF.CSRHF.CS_RHF(S, T, V, eri, n_electrons=2, theta=.05, max_iter=100, threshold=1E-12,  occupation=occupation_determinant, p_guess='core', verbose=True)

plot_map(P_1.imag- P_2.imag)
plot_map(P_1.real- P_2.real)

traj_energies = Dev.CS_CROP.theta_traj(0.08, 9, S, T, V, eri, 2, max_iter=1000, threshold=1E-13, p_guess='core', verbose=False)
traj_ener = np.array(traj_energies[1], dtype=complex)

print(traj_ener)

print(f'Error with pyscf {E_RHF -  rhf_He.energy_elec()[0]}')

