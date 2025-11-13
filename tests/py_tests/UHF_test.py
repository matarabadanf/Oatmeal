import numpy as np
from py_mods.src.SCF.RHF import RHF
from Dev.CSUHF import CS_UHF
from py_mods.src.SCF.scf_utils import V_NN
from pathlib import Path

data_path = Path(__file__).parent / "data"

def test_B() -> None:
    S_aug_cc_pvqz_B    = np.loadtxt(f'{data_path}/B_S_aug-cc-pvqz.dat')
    T_aug_cc_pvqz_B    = np.loadtxt(f'{data_path}/B_kin_aug-cc-pvqz.dat')
    V_aug_cc_pvqz_B    = np.loadtxt(f'{data_path}/B_vnuc_aug-cc-pvqz.dat')
    eri_aug_cc_pvqz_B  = np.load(f'{data_path}/B_eri_aug-cc-pvqz.npy')
    E_hf_aug_cc_pvqz_B = np.load(f'{data_path}/B_e_hf_aug-cc-pvqz.npy')

    # test: SCF convergence for B in aug-cc-pvqz
    converged, E_hf, *_ = CS_UHF(S_aug_cc_pvqz_B, T_aug_cc_pvqz_B, V_aug_cc_pvqz_B, eri_aug_cc_pvqz_B, n_electrons=5, max_iter=500, threshold=1E-7, p_guess='core', verbose=True, conv_type='DIIS')
    assert converged, "Calculation did not converge"
    assert abs(E_hf.real - E_hf_aug_cc_pvqz_B) < 1E-8, f"SCF energy does not match reference value {E_hf.real} != {E_hf_aug_cc_pvqz_B}"

def test_N() -> None:
    S_aug_cc_pvqz_N    = np.loadtxt(f'{data_path}/N_S_cc-pvqz.dat')
    T_aug_cc_pvqz_N    = np.loadtxt(f'{data_path}/N_kin_cc-pvqz.dat')
    V_aug_cc_pvqz_N    = np.loadtxt(f'{data_path}/N_vnuc_cc-pvqz.dat')
    eri_aug_cc_pvqz_N  = np.load(f'{data_path}/N_eri_cc-pvqz.npy')
    E_hf_aug_cc_pvqz_N = np.load(f'{data_path}/N_e_hf_cc-pvqz.npy')

    # test: SCF convergence for B in cc-pvqz
    converged, E_hf, *_ = CS_UHF(S_aug_cc_pvqz_N, T_aug_cc_pvqz_N, V_aug_cc_pvqz_N, eri_aug_cc_pvqz_N, n_electrons=7, max_iter=500, threshold=1E-8, p_guess='core', verbose=True, conv_type='DIIS')
    assert converged, "Calculation did not converge"
    assert abs(E_hf.real - E_hf_aug_cc_pvqz_N) < 1E-8, f"SCF energy does not match reference value {E_hf.real} != {E_hf_aug_cc_pvqz_N}"

def test_Cl() -> None:
    S_aug_cc_pvqz_Cl    = np.loadtxt(f'{data_path}/Cl_S_cc-pvqz.dat')
    T_aug_cc_pvqz_Cl    = np.loadtxt(f'{data_path}/Cl_kin_cc-pvqz.dat')
    V_aug_cc_pvqz_Cl    = np.loadtxt(f'{data_path}/Cl_vnuc_cc-pvqz.dat')
    eri_aug_cc_pvqz_Cl  = np.load(f'{data_path}/Cl_eri_cc-pvqz.npy')
    E_hf_aug_cc_pvqz_Cl = np.load(f'{data_path}/Cl_e_hf_cc-pvqz.npy')

    # test: SCF convergence for B in cc-pvqz
    converged, E_hf, *_ = CS_UHF(S_aug_cc_pvqz_Cl, T_aug_cc_pvqz_Cl, V_aug_cc_pvqz_Cl, eri_aug_cc_pvqz_Cl, n_electrons=17, max_iter=500, threshold=1E-8, p_guess='core', verbose=True, conv_type='DIIS')
    assert converged, "Calculation did not converge"
    assert abs(E_hf.real - E_hf_aug_cc_pvqz_Cl) < 1E-8, f"SCF energy does not match reference value {E_hf.real} != {E_hf_aug_cc_pvqz_Cl}"

if __name__ == "__main__":
    test_Cl()
    pass
    