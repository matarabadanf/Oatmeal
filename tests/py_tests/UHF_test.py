import numpy as np
from py_mods.src.SCF.CSUHF import CS_UHF, CSUHFContext
from pathlib import Path

data_path = Path(__file__).parent / "data"


def test_B_huge() -> None:
    S_aug_cc_pvqz_B = np.loadtxt(f"{data_path}/B_S_aug-cc-pvqz.dat")
    T_aug_cc_pvqz_B = np.loadtxt(f"{data_path}/B_kin_aug-cc-pvqz.dat")
    V_aug_cc_pvqz_B = np.loadtxt(f"{data_path}/B_vnuc_aug-cc-pvqz.dat")
    eri_aug_cc_pvqz_B = np.load(f"{data_path}/B_eri_aug-cc-pvqz.npy")
    E_hf_aug_cc_pvqz_B = np.load(f"{data_path}/B_e_hf_aug-cc-pvqz.npy")

    # test: SCF convergence for B in aug-cc-pvqz

    context = CSUHFContext(
        S=S_aug_cc_pvqz_B,
        T=T_aug_cc_pvqz_B,
        V=V_aug_cc_pvqz_B,
        eri=eri_aug_cc_pvqz_B,
        n_electrons=5,
        max_iter=500,
        threshold=1e-7,
        p_guess="RHF",
        verbose=True,
        conv_type="DIIS",
    )

    CS_UHF_results = CS_UHF(context)
    assert CS_UHF_results.converged, "Calculation did not converge"
    assert (
        abs(CS_UHF_results.E_UHF.real - E_hf_aug_cc_pvqz_B) < 1e-8
    ), f"SCF energy does not match reference value {CS_UHF_results.E_UHF.real} != {E_hf_aug_cc_pvqz_B}"


def test_N() -> None:
    S_aug_cc_pvqz_N = np.loadtxt(f"{data_path}/N_S_cc-pvqz.dat")
    T_aug_cc_pvqz_N = np.loadtxt(f"{data_path}/N_kin_cc-pvqz.dat")
    V_aug_cc_pvqz_N = np.loadtxt(f"{data_path}/N_vnuc_cc-pvqz.dat")
    eri_aug_cc_pvqz_N = np.load(f"{data_path}/N_eri_cc-pvqz.npy")
    E_hf_aug_cc_pvqz_N = np.load(f"{data_path}/N_e_hf_cc-pvqz.npy")

    # test: SCF convergence for B in cc-pvqz

    context = CSUHFContext(
        S=S_aug_cc_pvqz_N,
        T=T_aug_cc_pvqz_N,
        V=V_aug_cc_pvqz_N,
        eri=eri_aug_cc_pvqz_N,
        n_electrons=7,
        max_iter=500,
        threshold=1e-8,
        p_guess="RHF",
        verbose=True,
        conv_type="DIIS",
    )

    CS_UHF_results = CS_UHF(context)
    assert CS_UHF_results.converged, "Calculation did not converge"
    assert (
        abs(CS_UHF_results.E_UHF.real - E_hf_aug_cc_pvqz_N) < 1e-8
    ), f"SCF energy does not match reference value {CS_UHF_results.E_UHF.real} != {E_hf_aug_cc_pvqz_N}"


def test_Cl() -> None:
    S_aug_cc_pvqz_Cl = np.loadtxt(f"{data_path}/Cl_S_cc-pvqz.dat")
    T_aug_cc_pvqz_Cl = np.loadtxt(f"{data_path}/Cl_kin_cc-pvqz.dat")
    V_aug_cc_pvqz_Cl = np.loadtxt(f"{data_path}/Cl_vnuc_cc-pvqz.dat")
    eri_aug_cc_pvqz_Cl = np.load(f"{data_path}/Cl_eri_cc-pvqz.npy")
    E_hf_aug_cc_pvqz_Cl = np.load(f"{data_path}/Cl_e_hf_cc-pvqz.npy")

    # test: SCF convergence for B in cc-pvqz

    context = CSUHFContext(
        S=S_aug_cc_pvqz_Cl,
        T=T_aug_cc_pvqz_Cl,
        V=V_aug_cc_pvqz_Cl,
        eri=eri_aug_cc_pvqz_Cl,
        n_electrons=17,
        max_iter=500,
        threshold=1e-8,
        p_guess="RHF",
        verbose=True,
        conv_type="DIIS",
    )

    CS_UHF_results = CS_UHF(context)
    assert CS_UHF_results.converged, "Calculation did not converge"
    assert (
        abs(CS_UHF_results.E_UHF.real - E_hf_aug_cc_pvqz_Cl) < 1e-8
    ), f"SCF energy does not match reference value {CS_UHF_results.E_UHF.real} != {E_hf_aug_cc_pvqz_Cl}"


def test_H2_dissociation():
    # change here to see different curves.
    from pyscf import gto

    n_points = 20
    element_1 = "H"
    element_2 = "H"
    basis = "aug-cc-pvdz"
    n_elec = 2

    reference_E = np.array(
        [
            -2.37960889,
            -1.43448906,
            -1.22282276,
            -1.15386862,
            -1.11765807,
            -1.09515991,
            -1.07981773,
            -1.06868512,
            -1.06023855,
            -1.05361053,
            -1.04827084,
            -1.04387712,
            -1.04019844,
            -1.03707339,
            -1.03438574,
            -1.03204966,
            -1.03000041,
            -1.02818821,
            -1.02657417,
            -1.02512749,
        ]
    )

    distances = np.linspace(0.3, 20, n_points)
    Imp_RHF_eners = np.zeros_like(distances)
    for i, dist in enumerate(distances):
        mol = gto.M(
            atom=f"  {element_1} 0 0 0; {element_2} {dist} 0 0",
            spin=0,
            charge=0,
            basis=basis,
        )

        kin = mol.intor("int1e_kin")
        vnuc = mol.intor("int1e_nuc")
        overlap = mol.intor("int1e_ovlp")
        eri = mol.intor("int2e")

        context = CSUHFContext(
            overlap,
            kin,
            vnuc,
            eri,
            n_electrons=n_elec,
            p_guess="RHF",
            break_symm=True,
            max_iter=200,
        )
        CS_UHF_results = CS_UHF(context)
        Imp_RHF_eners[i] = CS_UHF_results.E_UHF.real

    assert np.mean(Imp_RHF_eners - reference_E), "H2 dissociation curve failed"


if __name__ == "__main__":
    test_N()
    # test_B_huge()
    # test_Cl()
    # H2_dissociation_test()
    pass
