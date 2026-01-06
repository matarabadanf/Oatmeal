from pyscf import gto, scf, mp
from py_mods.src.SCF.CSUHF import CS_UHF
from py_mods.src.MP2.CSMP2 import CS_MP2
from py_mods.src.SCF.external import UHF_context_from_pyscf

# pyscf data
pyscf_args = {
    "atom": "Ne 0 0 0; Ne 0 0 100",
    "spin": 0,
    "charge": 0,
    "basis": "cc-pvtz",
}
override = False  # override calculated MO coefficients by PySCF's

mol = gto.M(**pyscf_args)
mol.verbose = 0

# Pyscf calculation
mf = scf.UHF(mol)
mf.conv_tol = 1e-14
mf.conv_tol_grad = 1e-14
mf.max_cycle = 200
e_He = mf.kernel()
e_elec = mf.energy_elec()[0]
py_e_orb = mf.mo_energy
py_mo_coeff = mf.mo_coeff
mymp = mp.UMP2(mf).run()  # this is UMP2

# Implementation calculation
UHF_cxt = UHF_context_from_pyscf(**pyscf_args)
UHF_cxt.break_symm = True
UHF_res = CS_UHF(UHF_cxt)
mp_results = CS_MP2(UHF_res)

# results

print("------------- UHF -------------")
print(f"\nSCF energy: {UHF_res.E_UHF} (converged: {UHF_res.converged})")
print(f"SCF pyscf: {e_elec}")
print(f"Difference: {UHF_res.E_UHF.real - e_elec} \n")

print("------------- UMP2 -------------")
print(f"\nMP2 calc: {mp_results.E_MP2}, E_corr = {mp_results.E_corr}")
print(f"MP2 pyscf: {e_elec - mymp.e_corr}, E_corr = {mymp.e_corr}")
print(
    f"Differences: {mp_results.E_MP2 - e_elec - mymp.e_corr}, E_corr = {mp_results.E_corr - mymp.e_corr}\n"
)
