from pyscf import gto, scf, mp, ao2mo
import numpy as np
from pathlib import Path
# from py_mods.src.SCF.CSUHF import CS_UHF_ContextClass, CS_UHF
from py_mods.src.SCF.CSRHF import CS_RHF, CS_RHF_ContextClass
from py_mods.src.SCF.plot_utilities import plot_map
from Dev.CSMP2_dev import CS_MP2

data_path = Path(__file__).parent.parent

# pyscf data
mol = gto.M(atom = 'He 0 0 0', spin=0, charge=0, basis='aug-cc-pvqz')

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

# plot_map((mf.mo_coeff), title='C calc PYSCF')

print(type(eris_mo))

# implementation and calculation
Li_context = CS_RHF_ContextClass(overlap, kin, vnuc, eri, n_electrons=2, theta=0.0, verbose=False)
# Li_context.verbose = True

RHF_res = CS_RHF(Li_context)
print(f'\n\n\nSCF energy: {RHF_res.E_RHF}')
# print(type(RHF_res))



# print( RHF_res.e_orb)

# plot_map((mf.mo_coeff - RHF_res.R_munu.real), title='C_pyscf-C_calc')
# plot_map((mf.mo_coeff), title='C_pyscf

# plot_map((RHF_res.R_munu), title='C calc IMPL')

# plot_map((RHF_res.C_prime @ RHF_res.C_prime), title='Diff')

mp_resutls = CS_MP2(RHF_res)

# plot_map(mymp.t2[0,0,:,:], title='PYSCF T2')

print(f'\n\nMP2 calc: {mp_resutls.E_MP2}, E_corr = {mp_resutls.E_corr}')
print(f'MP2 pyscf: {mymp.e_tot}, E_corr = {mymp.e_corr}')
print(f'Differences: {mp_resutls.E_MP2 - mymp.e_tot}, E_corr = {mp_resutls.E_corr - mymp.e_corr}\n')

# to see later, the exact use of slices in this https://pycrawfordprogproj.readthedocs.io/en/latest/Project_04/Project_04.html
