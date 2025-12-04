from pyscf import gto, scf, mp, ao2mo
import numpy as np
from pathlib import Path
# from py_mods.src.SCF.CSUHF import CS_UHF_ContextClass, CS_UHF
from py_mods.src.SCF.CSRHF import CS_RHF, CS_RHF_ContextClass
from py_mods.src.SCF.plot_utilities import plot_map
from Dev.CSMP2_dev import CS_MP2
# from Dev.naive_MP2 import CS_MP2
from py_mods.src.SCF.external import RHF_context_from_pyscf
import matplotlib.pyplot as plt 

# pyscf data
pyscf_args = {
    'atom': 'Li 0 0 0',
    'spin': 0,
    'charge': -1,
    'basis': 'aug-cc-pvqz',
}

mol = gto.M(**pyscf_args)

mf = scf.RHF(mol) 

e_He = mf.kernel()
e_elec = mf.energy_elec()

mymp = mp.RMP2(mf).run() # this is UMP2


# implementation and calculation
RHF_cxt = RHF_context_from_pyscf(**pyscf_args)
RHF_res = CS_RHF(RHF_cxt)

print(f'\nSCF energy: {RHF_res.E_RHF.real} (converged: {RHF_res.converged})')
print(f'SCF pyscf: {e_He}')
print(f'Difference: {RHF_res.E_RHF.real - e_He} \n')

mp_results = CS_MP2(RHF_res)

print(f'\n\nMP2 calc: {mp_results.E_MP2}, E_corr = {mp_results.E_corr}')
print(f'MP2 pyscf: {mymp.e_tot}, E_corr = {mymp.e_corr}')
print(f'Differences: {mp_results.E_MP2 - mymp.e_tot}, E_corr = {mp_results.E_corr - mymp.e_corr}\n')

