from pyscf import gto, scf
import numpy as np

mol_Li = gto.M(atom = 'Li 0 0 0', spin=0, charge=+1, basis='6-31g')

kin = mol_Li.intor('int1e_kin')
vnuc = mol_Li.intor('int1e_nuc')
overlap = mol_Li.intor('int1e_ovlp')
eri = mol_Li.intor('int2e')

# print(overlap)
# print(kin)
# print(vnuc)
# print(eri)

np.savetxt('../Li_plus_kin_6-31g.dat', kin)
np.savetxt('../Li_plus_vnuc_6-31g.dat', vnuc)
np.savetxt('../Li_plus_S_6-31g.dat', overlap)
np.save('../Li_plus_eri_6-31g.npy', eri) # cannot be savetxt, has to be np binary

rhf_li = scf.RHF(mol_Li)
# rhf_li.init_guess = 'hcore'
# rhf_li.max_cycle = 0

e_li = rhf_li.kernel()
e_elec = rhf_li.energy_elec()

print(e_li, e_elec, rhf_li.cycles)
# print(rhf_li.mo_coeff)

np.save('../Li_plus_e_hf_6-31g', e_li)