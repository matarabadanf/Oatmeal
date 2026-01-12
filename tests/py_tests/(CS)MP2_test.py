from pyscf import gto, scf, mp
import numpy as np
from py_mods.src.MP2.CSMP2 import CS_MP2
from py_mods.src.SCF.CSRHF import CS_RHF
from py_mods.src.SCF.CSUHF import CS_UHF
from py_mods.src.SCF.external import RHF_context_from_pyscf, UHF_context_from_pyscf
from py_mods.src.SCF.basis_utils import even_temp_uncontr_str


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

        assert np.max(abs_errors) < 1e-12


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
    UHF_cxt.p_guess = "RHF"

    UHF_res = CS_UHF(UHF_cxt)

    assert (
        UHF_res.E_UHF.real - e_elec < 1e-14
    ), "Difference in convergence of SCF with reference"

    mymp = mp.UMP2(mf).run()
    mp_results = CS_MP2(UHF_res)

    assert (
        mp_results.E_corr - mymp.e_corr < 1e-12
    ), "Difference in correlation MP2 energy with reference"


def test_MP2_theta_traj_Qchem():

    # Setup
    He_tempered_str = even_temp_uncontr_str(
        "He", "S", 7.668876968794860e-002, 1.9581497063588078, 21
    )  # because this is the reference data

    pyscf_args_qz = {
        "atom": "He 0 0 0",
        "spin": 0,
        "charge": 0,
        "basis": "aug-cc-pvqz",
    }

    He_1s2_qz_cxt = RHF_context_from_pyscf(**pyscf_args_qz)
    He_2s2_qz_cxt = RHF_context_from_pyscf(**pyscf_args_qz)
    He_1s2_even_cxt = RHF_context_from_pyscf(
        **pyscf_args_qz, parseable_basis=He_tempered_str
    )
    He_2s2_even_cxt = RHF_context_from_pyscf(
        **pyscf_args_qz, parseable_basis=He_tempered_str
    )

    He_2s2_qz_cxt.occupation = np.array([0, 2, 0])
    He_2s2_even_cxt.occupation = np.array([0, 2, 0])

    # TEST: 1s2/aug-cc-pvqz theta = 0.04
    qz_1s2_ref_theta = 0.04
    qz_1s2_ref_ener = -2.89725754 + 8.624e-05j

    He_1s2_qz_cxt = RHF_context_from_pyscf(**pyscf_args_qz)
    He_1s2_qz_cxt.theta = qz_1s2_ref_theta

    He_1s2_qz_res = CS_RHF(He_1s2_qz_cxt)
    He_1s2_qz_mp2 = CS_MP2(He_1s2_qz_res)

    # print("He 1s2 aug-cc-pvqz MP2 Energy:", He_1s2_qz_mp2.E_MP2 - qz_1s2_ref_ener)
    assert abs(He_1s2_qz_mp2.E_MP2 - qz_1s2_ref_ener) < 1e-8

    # TEST: 1s2/even-tempered theta = 0.05
    even_1s2_ref_theta = 0.05
    even_1s2_ref_ener = -2.87517703 + 1e-08j

    He_1s2_even_cxt.theta = even_1s2_ref_theta
    He_1s2_even_res = CS_RHF(He_1s2_even_cxt)
    He_1s2_even_mp2 = CS_MP2(He_1s2_even_res)
    # print("He 1s2 even-tempered MP2 Energy:", He_1s2_even_mp2.E_MP2 - even_1s2_ref_ener)
    assert abs(He_1s2_even_mp2.E_MP2 - even_1s2_ref_ener) < 1e-8

    # TEST: 2s2/aug-cc-pvqz theta = 0.05
    qz_2s2_ref_theta = 0.05
    qz_2s2_ref_ener = -0.73805956 - 0.00201875j

    He_2s2_qz_cxt.theta = qz_2s2_ref_theta
    He_2s2_qz_res = CS_RHF(He_2s2_qz_cxt)
    He_2s2_qz_mp2 = CS_MP2(He_2s2_qz_res)
    # print("He 2s2 aug-cc-pvqz MP2 Energy:", He_2s2_qz_mp2.E_MP2 - qz_2s2_ref_ener)
    assert abs(He_2s2_qz_mp2.E_MP2 - qz_2s2_ref_ener) < 1e-8

    # TEST: 2s2/even-tempered theta = 0.23
    even_2s2_ref_theta = 0.23
    even_2s2_ref_ener = -0.7259088 - 0.01099962j

    He_2s2_even_cxt.theta = even_2s2_ref_theta
    He_2s2_even_res = CS_RHF(He_2s2_even_cxt)
    He_2s2_even_mp2 = CS_MP2(He_2s2_even_res)
    # print("He 2s2 even-tempered MP2 Energy:", He_2s2_even_mp2.E_MP2 - even_2s2_ref_ener)
    assert abs(He_2s2_even_mp2.E_MP2 - even_2s2_ref_ener) < 1e-8


if __name__ == "__main__":
    pass
