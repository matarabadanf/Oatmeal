from pyscf import gto, scf
import numpy as np

mol_He = gto.M(atom = 'He 0 0 0', spin=0, charge=0, basis='ccpvdz')

kin = mol_He.intor('int1e_kin')
vnuc = mol_He.intor('int1e_nuc')
overlap = mol_He.intor('int1e_ovlp')
eri = mol_He.intor('int2e')

# print(overlap)
# print(kin)
# print(vnuc)
# print(eri)

np.savetxt('../He_kin_ccpvdz.dat', kin)
np.savetxt('../He_vnuc_ccpvdz.dat', vnuc)
np.savetxt('../He_S_ccpvdz.dat', overlap)
np.save('../He_eri_ccpvdz.npy', eri) # cannot He savetxt, has to He np binary

rhf_He = scf.RHF(mol_He)
# rhf_He.init_guess = 'hcore'
# rhf_He.max_cycle = 0

e_He = rhf_He.kernel()
e_elec = rhf_He.energy_elec()

print(e_He, e_elec, rhf_He.cycles)
# print(rhf_He.mo_coeff)

np.save('../He_e_hf_ccpvdz', e_He)