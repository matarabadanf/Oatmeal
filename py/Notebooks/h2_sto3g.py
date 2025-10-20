from pyscf import gto, scf
import numpy as np

mol_H2 = gto.M(atom = 'H 0 0 0; H 0 0 0.740848', spin=0)

kin = mol_H2.intor('int1e_kin')
vnuc = mol_H2.intor('int1e_nuc')
overlap = mol_H2.intor('int1e_ovlp')
eri = mol_H2.intor('int2e')

print(overlap)
# print(kin)
# print(vnuc)
# print(eri)

np.savetxt('./data/kin_H2.dat', kin)
np.savetxt('./data/vnuc_H2.dat', vnuc)
np.savetxt('./data/S_H2.dat', overlap)
np.save('./data/eri_H2.npy', eri) # cannot be savetxt, has to be np binary

rhf_H2 = scf.RHF(mol_H2)

e_H2 = rhf_H2.kernel()
e_elec = rhf_H2.energy_elec()

print(e_H2, e_elec)