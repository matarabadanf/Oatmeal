from Dev.DIIS import DIIS_RHF
from py_mods.src.SCF.scf_utils import V_NN

import numpy as np 
from pyscf import gto, scf

dist = 1.4 * 0.529177249

mol = gto.M(atom = f'H 0 0 0; F 1.1 0 0', spin=0, basis='aug-cc-PVqZ')

T = mol.intor('int1e_kin')
V = mol.intor('int1e_nuc')
S = mol.intor('int1e_ovlp')
eri = mol.intor('int2e')

rhf = scf.RHF(mol)

pyscf_e = rhf.kernel()
e_elec = rhf.energy_elec()

print(f"H energy calculated by pyscf = {pyscf_e}")

# test : SCF convergence for H2 in STO-3G
converged, E_RHF, orbital_energies, C_munu, P = DIIS_RHF(S, T, V, eri, n_electrons=10, max_iter=100, threshold=1E-20, p_guess='core', verbose=True, DIIS_REQUESTED=True)

print(f'Error with pyscf {E_RHF -  rhf.energy_elec()[0]}')

