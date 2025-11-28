from pyscf import gto, scf, mp, ao2mo
import numpy as np
from pathlib import Path
# from py_mods.src.SCF.CSUHF import CS_UHF_ContextClass, CS_UHF
from Dev.CSRHF_dev import CS_RHF_ContextClass, CS_RHF
from py_mods.src.SCF.plot_utilities import plot_map
from Dev.CSMP2_dev import CS_MP2

data_path = Path(__file__).parent.parent

# pyscf data
mol = gto.M(atom = 'He 0 0 0', spin=0, charge=0, basis='6-31g')

kin = mol.intor('int1e_kin')
vnuc = mol.intor('int1e_nuc')
overlap = mol.intor('int1e_ovlp')
eri = mol.intor('int2e')

mf = scf.RHF(mol) 

e_He = mf.kernel()
e_elec = mf.energy_elec()

mymp = mp.RMP2(mf).run() # this is UMP2
print('MP2 total energy = ', mymp.e_tot)

#print(mymp.t2)

eris_mo = ao2mo.kernel(mol, mf.mo_coeff, aosym='1')

print(type(eris_mo))

# implementation and calculation
Li_context = CS_RHF_ContextClass(overlap, kin, vnuc, eri, n_electrons=4, theta=0.1)
# Li_context.verbose = True

Li_UHF_results = CS_RHF(Li_context)
print(f'\n\n\nSCF energy: {Li_UHF_results.E_RHF}')
# print(type(Li_UHF_results))

# Li_UHF_results.R_munu = mf.mo_coeff

# print( Li_UHF_results.e_orb)

# plot_map((mf.mo_coeff - Li_UHF_results.R_munu.real), title='C_pyscf-C_calc')
# plot_map((mf.mo_coeff), title='C_pyscf

plot_map((Li_UHF_results.L_munu-Li_UHF_results.R_munu.T), title='C_calc')

mp_resutls = CS_MP2(Li_UHF_results, eris_mo)

plot_map(mymp.t2[0,0,:,:])

print(f'\n\nMP2 calc: {mp_resutls.E_MP2}, E_corr = {mp_resutls.E_corr}')
print(f'MP2 pyscf: {mymp.e_tot}, E_corr = {mymp.e_corr}')
print(f'Differences: {mp_resutls.E_MP2 - mymp.e_tot}, E_corr = {mp_resutls.E_corr - mymp.e_corr}')