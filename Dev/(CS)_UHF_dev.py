from Dev.CSUHF import CS_UHF
from pyscf import gto, scf, ao2mo
import numpy as np 

spin = 1

mol_He= gto.M(atom = 'Li 0 0 0', spin=spin, charge=0, basis='aug-cc-pvtz') #  basis='6-311g') # basis='aug-cc-pVqZ')

# mol_He.basis = {'He': gto.basis.parse(He_tempered_str)}
mol_He.build()

kin = mol_He.intor('int1e_kin')
vnuc = mol_He.intor('int1e_nuc')
overlap = mol_He.intor('int1e_ovlp')
eri = mol_He.intor('int2e')

rhf_He = scf.UHF(mol_He)
e_He = rhf_He.kernel()

nelec = 3
theta = 0

converged, E_elec_comp,  e_values = CS_UHF(overlap, kin, vnuc, eri, nelec, mult=spin, verbose=True, threshold=1E-10, max_iter=600, conv_type=None)


