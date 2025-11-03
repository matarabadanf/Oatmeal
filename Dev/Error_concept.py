from py_mods.src.SCF.RHF import RHF, plot_map
from py_mods.src.SCF.CSUHF import CS_RHF
from py_mods.src.SCF.scf_utils import V_NN

import numpy as np 
from pyscf import gto, scf

dist = 1.4 * 0.529177249

mol_H2 = gto.M(atom = f'He 0 0 0', spin=0, basis='aug-cc-PVqZ')

T_sto3g_H2 = mol_H2.intor('int1e_kin')
V_sto3g_H2 = mol_H2.intor('int1e_nuc')
S_sto3g_H2 = mol_H2.intor('int1e_ovlp')
eri_sto3g_H2 = mol_H2.intor('int2e')

rhf_H2 = scf.RHF(mol_H2)

pyscf_e_H2 = rhf_H2.kernel()
e_elec = rhf_H2.energy_elec()

print(f"H energy calculated by pyscf = {pyscf_e_H2}")

# test : SCF convergence for H2 in STO-3G
converged, E_elec, E_e_values, C_munu, P = CS_RHF(S_sto3g_H2, T_sto3g_H2, V_sto3g_H2, eri_sto3g_H2, n_electrons=2, theta=0.1, occupation=np.array([0,2,0]), max_iter=1000, threshold=1E-12, p_guess='core', verbose=True)