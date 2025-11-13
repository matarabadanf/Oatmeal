from pyscf import gto, scf
import numpy as np
from pathlib import Path

data_path = Path(__file__).parent.parent



mol_HF = gto.M(atom = f'H 0 0 0; F 1.1 0 0', spin=0, basis='aug-cc-PVtZ')

kin = mol_HF.intor('int1e_kin')
vnuc = mol_HF.intor('int1e_nuc')
overlap = mol_HF.intor('int1e_ovlp')
eri = mol_HF.intor('int2e')

# print(overlap)
# print(kin)
# print(vnuc)
# print(eri)

np.savetxt(f'{data_path}/HF_kin_augccpvqz.dat', kin)
np.savetxt(f'{data_path}/HF_vnuc_augccpvqz.dat', vnuc)
np.savetxt(f'{data_path}/HF_S_augccpvqz.dat', overlap)
np.save(f'{data_path}/HF_eri_augccpvqz.npy', eri) # cannot be savetxt, has to be np binary

rhf_HF = scf.RHF(mol_HF)
# rhf_HF.init_guess = 'hcore'
# rhf_HF.max_cycle = 0

e_HF = rhf_HF.kernel()
e_elec = rhf_HF.energy_elec()

print(e_HF, e_elec, rhf_HF.cycles)
# print(rhf_HF.mo_coeff)
np.save(f'{data_path}/HF_e_hf_augccpvqz', e_elec[0])


