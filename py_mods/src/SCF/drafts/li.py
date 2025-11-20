from pyscf import gto, scf
import numpy as np
from pathlib import Path
from py_mods.src.SCF.CSUHF import CS_UHF_ContextClass, CS_UHF
from py_mods.src.SCF.RHF import plot_map

data_path = Path(__file__).parent.parent

# pyscf data
mol_He = gto.M(atom = 'Li 0 0 0', spin=1, charge=0, basis='aug-cc-pvqz')

kin = mol_He.intor('int1e_kin')
vnuc = mol_He.intor('int1e_nuc')
overlap = mol_He.intor('int1e_ovlp')
eri = mol_He.intor('int2e')

rhf_He = scf.UHF(mol_He)

e_He = rhf_He.kernel()
e_elec = rhf_He.energy_elec()


# implementation and calculation
Li_context = CS_UHF_ContextClass(overlap, kin, vnuc, eri, n_electrons=3)
Li_context.verbose = True
Li_context.diagnostics = True

Li_UHF_results = CS_UHF(Li_context)

plot_map(Li_UHF_results.P_diff.real)

if __name__ == '__main__':
    pass
 
