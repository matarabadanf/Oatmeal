from pyscf import gto, scf, mp
from py_mods.src.SCF.CSUHF import CS_UHF
from py_mods.src.SCF.plot_utilities import plot_map
from Dev.CSMP2_dev import CS_MP2
from py_mods.src.SCF.external import UHF_context_from_pyscf
import matplotlib.pyplot as plt
import numpy as np
from pyscf.tools import molden

large_basis = """
He    S
      5.285000E+02           0.000000E+00           9.400000E-04           0.000000E+00           0.000000E+00
      7.931000E+01           0.000000E+00           7.214000E-03           0.000000E+00           0.000000E+00
      1.805000E+01           0.000000E+00           3.597500E-02           0.000000E+00           0.000000E+00
      5.085000E+00           0.000000E+00           1.277820E-01           0.000000E+00           0.000000E+00
      1.609000E+00           1.000000E+00           3.084700E-01           0.000000E+00           0.000000E+00
      5.363000E-01           0.000000E+00           4.530520E-01           1.000000E+00           0.000000E+00
      1.833000E-01           0.000000E+00           2.388840E-01           0.000000E+00           1.000000E+00
He    S
      0.0481900              1.0000000
He    P
      5.994000E+00           1.000000E+00           0.000000E+00           0.000000E+00
      1.745000E+00           0.000000E+00           1.000000E+00           0.000000E+00
      5.600000E-01           0.000000E+00           0.000000E+00           1.000000E+00
He    P
      0.1626000              1.0000000
He    D
      4.299000E+00           1.000000E+00           0.000000E+00
      1.223000E+00           0.000000E+00           1.000000E+00
He    D
      0.3510000              1.0000000
He    F
      2.680000E+00           1.0000000
He    F
      0.6906000              1.0000000
END
"""

# pyscf data
pyscf_args = {
    "atom": "Ne 0 0 0; Ne 0 0 1",
    "spin": 0,
    "charge": 0,
    "basis": "cc-pvtz",
}
override = True  # override calculated MO coefficients by PySCF's

mol = gto.M(**pyscf_args)
# mol.basis = {"He": gto.basis.parse(large_basis)}
# mol.build()

mf = scf.UHF(mol)

e_He = mf.kernel()
e_elec = mf.energy_elec()
py_e_orb = mf.mo_energy
py_mo_coeff = mf.mo_coeff

# print(py_e_orb)

mymp = mp.UMP2(mf).run()  # this is UMP2

# implementation and calculation
UHF_cxt = UHF_context_from_pyscf(**pyscf_args)
UHF_res = CS_UHF(UHF_cxt)
# print(UHF_res.e_alpha)
# print(UHF_res.e_beta)

print(f"\nSCF energy: {UHF_res.E_UHF} (converged: {UHF_res.converged})")
print(f"SCF pyscf: {e_He}")
print(f"Difference: {UHF_res.E_UHF.real - e_He} \n")

# print('Orbital energies UHF implementation:')
# e_orb =(UHF_res.e_alpha.real, UHF_res.e_beta.real)
# print(e_orb)

# print(mf.mo_coeff[0].shape)
# print(UHF_res.R_alpha.shape)

# Once again, enforcingg MO coefficients should yield the same results as PySCF with our implementation
# print(f"Maximum difference between Alpha and Beta MO coefficients: {np.max(mf.mo_coeff[0] - mf.mo_coeff[1])}")
if override:
    UHF_res.R_alpha = mf.mo_coeff[0]
    UHF_res.R_beta = mf.mo_coeff[1]
    UHF_res.e_alpha = mf.mo_energy[0]
    UHF_res.e_beta = mf.mo_energy[1]

# not enforcing MOs and Energies, error of MP2 energy of  0.003220297697885094 for He/aug-cc-pvqz
#     Enforcing MOS and energies, error of MP2 energy of  1E-17                for He/aug-cc-pvqz
# not enforcing MOs and Energies, error of MP2 energy of -0.006341869083391349 for Be/aug-cc-pvqz
#     Enforcing MOS and energies, error of MP2 energy of  7E-17                for Be/aug-cc-pvqz
# not enforcing MOs and Energies, error of MP2 energy of  6.72990602135215e-09 for Ne/aug-cc-pvqz
#     Enforcing MOS and energies, error of MP2 energy of  4E-16                for Ne/aug-cc-pvqz

molden.from_scf(mf, 'hf_result.molden')


mp_results = CS_MP2(UHF_res)

print(f"\n\nMP2 calc: {mp_results.E_MP2}, E_corr = {mp_results.E_corr}")
print(f"MP2 pyscf: {mymp.e_tot}, E_corr = {mymp.e_corr}")
print(
    f"Differences: {mp_results.E_MP2 - mymp.e_tot}, E_corr = {mp_results.E_corr - mymp.e_corr}\n"
)
