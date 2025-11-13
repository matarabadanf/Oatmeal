from pyscf import gto, scf
import numpy as np
from pathlib import Path

data_path = Path(__file__).parent.parent

mol_He= gto.M(atom = 'He 0 0 0', spin=0, charge=0,) # basis='aug-cc-pVqZ')

big_basis='''
He    S
      1.145000E+03           0.000000E+00           0.000000E+00           3.590000E-04           0.000000E+00           0.000000E+00
      1.717000E+02           0.000000E+00           0.000000E+00           2.771000E-03           0.000000E+00           0.000000E+00
      3.907000E+01           0.000000E+00           0.000000E+00           1.425100E-02           0.000000E+00           0.000000E+00
      1.104000E+01           0.000000E+00           0.000000E+00           5.556600E-02           0.000000E+00           0.000000E+00
      3.566000E+00           1.000000E+00           0.000000E+00           1.620910E-01           0.000000E+00           0.000000E+00
      1.240000E+00           0.000000E+00           1.000000E+00           3.321970E-01           0.000000E+00           0.000000E+00
      4.473000E-01           0.000000E+00           0.000000E+00           4.196150E-01           1.000000E+00           0.000000E+00
      1.640000E-01           0.000000E+00           0.000000E+00           1.861280E-01           0.000000E+00           1.000000E+00
He    S
      0.0466400              1.0000000
He    P
      1.015300E+01           1.000000E+00           0.000000E+00           0.000000E+00           0.000000E+00
      3.627000E+00           0.000000E+00           1.000000E+00           0.000000E+00           0.000000E+00
      1.296000E+00           0.000000E+00           0.000000E+00           1.000000E+00           0.000000E+00
      4.630000E-01           0.000000E+00           0.000000E+00           0.000000E+00           1.000000E+00
He    P
      0.1400000              1.0000000
He    D
      7.666000E+00           1.000000E+00           0.000000E+00           0.000000E+00
      2.647000E+00           0.000000E+00           1.000000E+00           0.000000E+00
      9.140000E-01           0.000000E+00           0.000000E+00           1.000000E+00
He    D
      0.2892000              1.0000000
He    F
      5.411000E+00           1.000000E+00           0.000000E+00
      1.707000E+00           0.000000E+00           1.000000E+00
He    F
      0.5345000              1.0000000
He    G
      3.430000E+00           1.0000000
He    G
      0.7899000              1.0000000
'''

mol_He.basis = {'He': gto.basis.parse(big_basis)}
mol_He.build()

kin = mol_He.intor('int1e_kin')
vnuc = mol_He.intor('int1e_nuc')
overlap = mol_He.intor('int1e_ovlp')
eri = mol_He.intor('int2e')

# print(overlap)
# print(kin)
# print(vnuc)
# print(eri)

np.savetxt(f'{data_path}/He_kin_aug-cc-pv(5+d)z.dat', kin)
np.savetxt(f'{data_path}/He_vnuc_aug-cc-pv(5+d)z.dat', vnuc)
np.savetxt(f'{data_path}/He_S_aug-cc-pv(5+d)z.dat', overlap)
np.save(f'{data_path}/He_eri_aug-cc-pv(5+d)z.npy', eri) # cannot He savetxt, has to He np binary

rhf_He = scf.RHF(mol_He)
# rhf_He.init_guess = 'hcore'
# rhf_He.max_cycle = 0

e_He = rhf_He.kernel()
e_elec = rhf_He.energy_elec()

print(e_He, e_elec, rhf_He.cycles)
# print(rhf_He.mo_coeff)

np.save(f'{data_path}/He_e_hf_aug-cc-pv(5+d)z', e_He)