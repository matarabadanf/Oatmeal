from pyscf import gto, scf
import numpy as np
from pathlib import Path

data_path = Path(__file__).parent.parent

mol_He = gto.M(atom = 'N 0 0 0', spin=1, charge=0, basis='cc-pvqz')

kin = mol_He.intor('int1e_kin')
vnuc = mol_He.intor('int1e_nuc')
overlap = mol_He.intor('int1e_ovlp')
eri = mol_He.intor('int2e')

# print(overlap)
# print(kin)
# print(vnuc)
# print(eri)

np.savetxt(f'{data_path}/N_kin_cc-pvqz.dat', kin)
np.savetxt(f'{data_path}/N_vnuc_cc-pvqz.dat', vnuc)
np.savetxt(f'{data_path}/N_S_cc-pvqz.dat', overlap)
np.save(f'{data_path}/N_eri_cc-pvqz.npy', eri) # cannot He savetxt, has to He np binary

rhf_He = scf.UHF(mol_He)
# rhf_He.init_guess = 'hcore'
# rhf_He.max_cycle = 0

e_He = rhf_He.kernel()
e_elec = rhf_He.energy_elec()

print(e_He, e_elec, rhf_He.cycles)
# print(rhf_He.mo_coeff)

np.save(f'{data_path}/N_e_hf_cc-pvqz', e_He)