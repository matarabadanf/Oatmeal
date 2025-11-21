from pyscf import gto, scf
import numpy as np 
from Dev.CSRHF_dev import CS_RHF_ContextClass, CS_RHF

mol_He= gto.M(atom = 'C 0 0 0', spin=0, charge=0, basis='aug-cc-pvqz') 
mol_He.build()

kin = mol_He.intor('int1e_kin')
vnuc = mol_He.intor('int1e_nuc')
overlap = mol_He.intor('int1e_ovlp')
eri = mol_He.intor('int2e')

# prepare UHF calculation
nelec = 6
gs_det = np.array([2,2,2])
ex_det = np.array([2,2,0,0,0,0,2])
H2_context = CS_RHF_ContextClass(overlap, kin, vnuc, eri, n_electrons=nelec, verbose=False, conv_ITER_START=30)

# unscaled calculations
print('\n\n\n Case 1s2 He aug-cc-pvqz, theta = 0')
H2_context.occupation = gs_det
one_s2_theta0 = CS_RHF(H2_context)
print('\n\n\n Case 2s2 He aug-cc-pvqz, theta = 0')
H2_context.occupation = ex_det
two_s2_theta0 = CS_RHF(H2_context)

# scaled calculations
theta = 0.02
H2_context.theta = theta
print(f'\n\n\n Case 1s2 He aug-cc-pvqz, theta = {theta}')
H2_context.occupation = gs_det
one_s2_theta0 = CS_RHF(H2_context)
print(f'\n\n\n Case 2s2 He aug-cc-pvqz, theta = {theta}')
H2_context.occupation = ex_det
two_s2_theta0 = CS_RHF(H2_context)