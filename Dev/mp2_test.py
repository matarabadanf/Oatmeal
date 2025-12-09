from re import I
from pyscf import gto, scf, mp, ao2mo
import numpy as np
from pathlib import Path

# from py_mods.src.SCF.CSUHF import CS_UHF_ContextClass, CS_UHF
from py_mods.src.SCF.CSRHF import CS_RHF, CS_RHF_ContextClass
from py_mods.src.SCF.plot_utilities import plot_map
from Dev.CSMP2_dev import CS_MP2

# from Dev.naive_MP2 import CS_MP2
from py_mods.src.SCF.external import RHF_context_from_pyscf
import matplotlib.pyplot as plt

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
    "atom": "Ne 0 0 0",
    "spin": 0,
    "charge": 0,
    "basis": "aug-cc-pvqz",
}

mol = gto.M(**pyscf_args)
# mol.basis = {'He': gto.basis.parse(large_basis)}
# mol.build()

mf = scf.RHF(mol)

e_He = mf.kernel()
e_elec = mf.energy_elec()
py_e_orb = mf.mo_energy
py_mo_coeff = mf.mo_coeff

# print(py_e_orb)

# plot_map(mf.mo_coeff)

mymp = mp.RMP2(mf).run()  # this is UMP2

# implementation and calculation
RHF_cxt = RHF_context_from_pyscf(**pyscf_args)
RHF_cxt.theta = 0.00
# RHF_cxt.occupation = np.array([2,0])
RHF_res = CS_RHF(RHF_cxt)

# print(RHF_res.e_orb)

# print(RHF_res.e_orb - py_e_orb)

# plot_map(RHF_res.R_munu.real)

print(f"\nSCF energy: {RHF_res.E_RHF} (converged: {RHF_res.converged})")
print(f"SCF pyscf: {e_He}")
print(f"Difference: {RHF_res.E_RHF.real - e_He} \n")

# Brute forcing this:
RHF_res.R_munu = mf.mo_coeff
RHF_res.e_orb = mf.mo_energy

mp_results = CS_MP2(RHF_res)

# The energy result difference is 1.3E-17 with forced MO coefficients and enregies
# The energy result difference is 0.00145 with self-obtained MO coefficients and orbitals

print(f"\n\nMP2 calc: {mp_results.E_MP2}, E_corr = {mp_results.E_corr}")
print(f"MP2 pyscf: {mymp.e_tot}, E_corr = {mymp.e_corr}")
print(
    f"Differences: {mp_results.E_MP2 - mymp.e_tot}, E_corr = {mp_results.E_corr - mymp.e_corr}\n"
)
