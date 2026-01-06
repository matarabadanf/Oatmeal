from pyscf import gto, scf, mp
import numpy as np
from py_mods.src.MP2.CSMP2 import CS_MP2
from py_mods.src.SCF.CSRHF import CS_RHF
from py_mods.src.SCF.CSUHF import CS_UHF
from py_mods.src.SCF.external import RHF_context_from_pyscf, UHF_context_from_pyscf

def test_RMP2():
    basis = "aug-cc-pvdz"
    pyscf_args = {
        "atom": "He 0 0 0",
        "spin": 0,
        "charge": 0,
        "basis": f"{basis}",
    }

    atoms = [
        "He 0 0 0",
        "Mg 0 0 0",
        "Ne 0 0 0",
        "Ar 0 0 0",
        "Kr 0 0 0",
    ]

    abs_errors = []
    rel_errors = []

    for atom in atoms:
        pyscf_args["atom"] = atom
        mol = gto.M(**pyscf_args)
        mol.verbose = 0 

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        mf.conv_tol_grad = 1e-14
        mf.kernel()
        mymp = mp.RMP2(mf).run() 

        # implementation and calculation
        RHF_cxt = RHF_context_from_pyscf(**pyscf_args)

        RHF_res = CS_RHF(RHF_cxt)
        mp_results = CS_MP2(RHF_res)

        abs_errors.append(mymp.e_tot - mp_results.E_MP2)
        rel_errors.append(np.abs((mp_results.E_MP2 - mymp.e_tot) * 100 / mymp.e_tot))

        assert np.max(abs_errors) < 1E-12

def test_UMP2():
    pyscf_args = {
        "atom": "Ne 0 0 0; Ne 0 0 1",
        "spin": 0,
        "charge": 0,
        "basis": "cc-pvtz",
    }

    mol = gto.M(**pyscf_args)
    mol.verbose = 0 

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-14
    mf.max_cycle = 200
    e_He = mf.kernel()
    e_elec = mf.energy_elec()[0]
    e_nuc = mol.energy_nuc()

    UHF_cxt = UHF_context_from_pyscf(**pyscf_args)
    UHF_cxt.break_symm = True
    UHF_cxt.p_guess = 'RHF'

    UHF_res = CS_UHF(UHF_cxt)

    assert UHF_res.E_UHF.real - e_elec < 1E-14, 'Difference in convergence of SCF with reference'

    mymp = mp.UMP2(mf).run()
    mp_results = CS_MP2(UHF_res)

    assert mp_results.E_corr - mymp.e_corr < 1E-12, 'Difference in correlation MP2 energy with reference'

if __name__ == "__main__":
    pass