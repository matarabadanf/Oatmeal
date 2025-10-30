from py_mods.src.SCF.basis_utils import even_temp_uncontr_str
from pyscf import gto, scf
import numpy as np

He_tempered_str = even_temp_uncontr_str('He', 'S', 7.668876968794860E-002, 1.9581497063588078, 29)

mol_He= gto.M(atom = 'He 0 0 0', spin=0, charge=0,) # basis='aug-cc-pVqZ')

mol_He.basis = {'He': gto.basis.parse(He_tempered_str)}
mol_He.build()

kin = mol_He.intor('int1e_kin')
vnuc = mol_He.intor('int1e_nuc')
overlap = mol_He.intor('int1e_ovlp')
eri = mol_He.intor('int2e')

# print(overlap)
# print(kin)
# print(vnuc)
# print(eri)

np.savetxt('../He_kin_29s.dat', kin)
np.savetxt('../He_vnuc_29s.dat', vnuc)
np.savetxt('../He_S_29s.dat', overlap)
np.save('../He_eri_29s.npy', eri) # cannot He savetxt, has to He np binary

rhf_He = scf.RHF(mol_He)
# rhf_He.init_guess = 'hcore'
# rhf_He.max_cycle = 0

e_He = rhf_He.kernel()
e_elec = rhf_He.energy_elec()

print(e_He, e_elec, rhf_He.cycles)
# print(rhf_He.mo_coeff)

np.save('../He_e_hf_29s', e_He)