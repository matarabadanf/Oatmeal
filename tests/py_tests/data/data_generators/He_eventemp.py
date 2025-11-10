from pyscf import gto, scf
import numpy as np

mol_Be = gto.M(atom = 'Be 0 0 0', spin=0, charge=0, basis='ccpvdz')

kin = mol_Be.intor('int1e_kin')
vnuc = mol_Be.intor('int1e_nuc')
overlap = mol_Be.intor('int1e_ovlp')
eri = mol_Be.intor('int2e')

# print(overlap)
# print(kin)
# print(vnuc)
# print(eri)

np.savetxt('../Be_kin_ccpvdz.dat', kin)
np.savetxt('../Be_vnuc_ccpvdz.dat', vnuc)
np.savetxt('../Be_S_ccpvdz.dat', overlap)
np.save('../Be_eri_ccpvdz.npy', eri) # cannot be savetxt, has to be np binary

rhf_Be = scf.RHF(mol_Be)
# rhf_Be.init_guess = 'hcore'
# rhf_Be.max_cycle = 0

e_Be = rhf_Be.kernel()
e_elec = rhf_Be.energy_elec()

print(e_Be, e_elec, rhf_Be.cycles)
# print(rhf_Be.mo_coeff)

np.save('../Be_e_hf_ccpvdz', e_Be)