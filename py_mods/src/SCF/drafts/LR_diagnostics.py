from pyscf import gto, scf
import numpy as np 
from py_mods.src.SCF.CSRHF import CS_RHF

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
# get the two-electron integrals as a numpy array
ref_eri = mol_He.ao2mo(orb)


nelec = 2
theta = 0.0

# even_tempered_demonstration(7.668876968794860E-002, 1.9581497063588078, 29)
print('\n\n\n Case 1s2 He aug-cc-pvqz, theta = 0')
converged, E_elec_comp, E_e_values, R_munu_prime, P, L_munu, R_munu, P_RR = CS_RHF(overlap, kin, vnuc, eri, nelec, theta, max_iter=500, threshold=1E-8, p_guess='core', verbose=True, diagnostics=True)
print('\n\n\n Case 2s2 He aug-cc-pvqz, theta = 0')
converged, E_elec_comp, E_e_values, R_munu_prime, P, L_munu, R_munu, P_RR = CS_RHF(overlap, kin, vnuc, eri, nelec, theta, occupation=np.array([0,2,0]),threshold=1E-8,  verbose=True, diagnostics=True)


theta = 0.1
print('\n\n\n Case 1s2 He aug-cc-pvqz, theta = 0.1')
converged, E_elec_comp, E_e_values, R_munu_prime, P, L_munu, R_munu, P_RR = CS_RHF(overlap, kin, vnuc, eri, nelec, theta, max_iter=500, threshold=1E-8, p_guess='core', verbose=True, diagnostics=True)
print('\n\n\n Case 2s2 He aug-cc-pvqz, theta = 0.1')
converged, E_elec_comp, E_e_values, R_munu_prime, P, L_munu, R_munu, P_RR = CS_RHF(overlap, kin, vnuc, eri, nelec, theta, occupation=np.array([0,2,0]),threshold=1E-8,  verbose=True, diagnostics=True)


# print(f'Mean difference between left and right solutions: {np.mean(R_munu-L_munu):.4E}')
# print(f'Max difference between left and right solutions : {np.max(R_munu-L_munu):.4E}')

# # mu to p 
# tmp1 = np.einsum("m p, m n l s -> p n l s", L_munu, eri)         

# # nu to q 
# tmp2 = np.einsum("n q, p n l s -> p q l s", L_munu, tmp1)       

# # lambda to r 
# tmp3 = np.einsum("l r, p q l s -> p q r s", R_munu, tmp2)      

# # sigma to s
# mo_eri = np.einsum("s t, p q r s -> p q r t", R_munu, tmp3)   

# # print(f'\nDifference between MO by pyscf and by regular contraction: {np.max(ref_eri-mo_eri)}')
# # print(f'Difference between MO by pyscf and by regular contraction: {np.mean(ref_eri-mo_eri)}')

# print("\n (vv|oo) =?= (oo|vv):")
# print(f'{mo_eri[0,0,1,1]:.4E}')
# print(f'{mo_eri[1,1,0,0]:.4E}')

# print("\n (vv|oo) == (oo|vv):")
# print(mo_eri[0,3,0,3])
# print(mo_eri[3,0,0,3])

# print("\n (vv|oo) == (oo|vv):")
# print(mo_eri[3,0,3,0])
# print(mo_eri[0,3,0,3])

if __name__ == '__main__':
    pass