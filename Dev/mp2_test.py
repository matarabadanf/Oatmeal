from pyscf import gto, scf, mp
import numpy as np
from pathlib import Path
# from py_mods.src.SCF.CSUHF import CS_UHF_ContextClass, CS_UHF
from Dev.CSRHF_dev import CS_RHF_ContextClass, CS_RHF
from py_mods.src.SCF.plot_utilities import plot_map
from Dev.CSMP2_dev import CS_MP2

data_path = Path(__file__).parent.parent

# pyscf data
mol_He = gto.M(atom = 'Be 0 0 0', spin=0, charge=0, basis='cc-pvdz')

kin = mol_He.intor('int1e_kin')
vnuc = mol_He.intor('int1e_nuc')
overlap = mol_He.intor('int1e_ovlp')
eri = mol_He.intor('int2e')

mf = scf.RHF(mol_He)

e_He = mf.kernel()
e_elec = mf.energy_elec()

mymp = mp.RMP2(mf).run() # this is UMP2
print('MP2 total energy = ', mymp.e_tot)


# implementation and calculation
Li_context = CS_RHF_ContextClass(overlap, kin, vnuc, eri, n_electrons=4, theta=0.0)
# Li_context.verbose = True

Li_UHF_results = CS_RHF(Li_context)
print(f'\n\n\nSCF energy: {Li_UHF_results.E_RHF}')
# print(type(Li_UHF_results))

mp_resutls = CS_MP2(Li_UHF_results)


print(f'\n\nMP2 calc: {mp_resutls.E_MP2}, E_corr = {mp_resutls.E_corr}')