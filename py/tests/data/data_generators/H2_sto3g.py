from pyscf import gto, scf
import numpy as np

mol_H2 = gto.M(atom = 'H 0 0 0; H 0 0 0.740848', spin=0)

kin = mol_H2.intor('int1e_kin')
vnuc = mol_H2.intor('int1e_nuc')
overlap = mol_H2.intor('int1e_ovlp')
eri = mol_H2.intor('int2e')

# print(overlap)
# print(kin)
# print(vnuc)
# print(eri)

np.savetxt('../H2_kin_sto3g.dat', kin)
np.savetxt('../H2_vnuc_sto3g.dat', vnuc)
np.savetxt('../H2_S_sto3g.dat', overlap)
np.save('../H2_eri_sto3g.npy', eri) # cannot be savetxt, has to be np binary

rhf_H2 = scf.RHF(mol_H2)

e_H2 = rhf_H2.kernel()
e_elec = rhf_H2.energy_elec()

# print(e_H2, e_elec)
np.save('../H2_e_hf_sto3g', e_H2)