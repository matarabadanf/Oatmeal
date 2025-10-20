from pyscf import gto, scf
import numpy as np

mol_Li = gto.M(atom = 'Li 0 0 0', spin=1, charge=2)

kin = mol_Li.intor('int1e_kin')
vnuc = mol_Li.intor('int1e_nuc')
overlap = mol_Li.intor('int1e_ovlp')
eri = mol_Li.intor('int2e')

# print(overlap)
# print(kin)
# print(vnuc)
# print(eri)

np.savetxt('./data/kin_li.dat', kin)
np.savetxt('./data/vnuc_li.dat', vnuc)
np.savetxt('./data/S_li.dat', overlap)
np.save('./data/eri_li.npy', eri) # cannot be savetxt, has to be np binary

rhf_li = scf.RHF(mol_Li)
rhf_li.init_guess = 'hcore'
rhf_li.max_cycle = 0

e_li = rhf_li.kernel()
e_elec = rhf_li.energy_elec()

print(e_li, e_elec, rhf_li.cycles)
print(rhf_li.mo_coeff)