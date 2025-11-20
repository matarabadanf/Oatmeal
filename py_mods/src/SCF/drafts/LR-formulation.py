from pyscf import gto, scf
import numpy as np 
from Dev.CSUHF_dev import CS_UHF_ContextClass, CS_UHF

mol_He= gto.M(atom = 'He 0 0 0', spin=0, charge=0, basis='aug-cc-pvqz') # basis='aug-cc-pVqZ')

# mol_He.basis = {'He': gto.basis.parse(He_tempered_str)}
mol_He.build()

kin = mol_He.intor('int1e_kin')
vnuc = mol_He.intor('int1e_nuc')
overlap = mol_He.intor('int1e_ovlp')
eri = mol_He.intor('int2e')

rhf_He = scf.RHF(mol_He)
e_He = rhf_He.kernel()
orb = rhf_He.mo_coeff

# prepare UHF calculation
nelec = 2
gs_dets = [np.array([1,0]), np.array([1,0])]
ex_dets = [np.array([0,1]), np.array([0,1])]
H2_context = CS_UHF_ContextClass(overlap, kin, vnuc, eri, n_electrons=nelec, verbose=True)

print('\n\n\n Case 1s2 He aug-cc-pvqz, theta = 0')
H2_context.occupation = gs_dets
one_s2_theta0 = CS_UHF(H2_context)
print('\n\n\n Case 2s2 He aug-cc-pvqz, theta = 0')
H2_context.occupation = ex_dets
two_s2_theta0 = CS_UHF(H2_context)


theta = 0.1
H2_context.theta = theta
print(f'\n\n\n Case 1s2 He aug-cc-pvqz, theta = {theta}')
H2_context.occupation = gs_dets
one_s2_theta0 = CS_UHF(H2_context)
print(f'\n\n\n Case 2s2 He aug-cc-pvqz, theta = {theta}')
H2_context.occupation = ex_dets
two_s2_theta0 = CS_UHF(H2_context)