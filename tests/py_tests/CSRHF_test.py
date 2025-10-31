import numpy as np
from dev.CSUHF_dev import CS_RHF
from py_mods.src.SCF.RHF import RHF
from pathlib import Path

data_path = Path(__file__).parent / "data"

def test_theta_zero() -> None:
    S_sto3g_li    = np.loadtxt(f'{data_path}/Li_plus_S_6-31g.dat')
    T_sto3g_li    = np.loadtxt(f'{data_path}/Li_plus_kin_6-31g.dat')
    V_sto3g_li    = np.loadtxt(f'{data_path}/Li_plus_vnuc_6-31g.dat')
    eri_sto3g_li  = np.load(f'{data_path}/Li_plus_eri_6-31g.npy')
    E_hf_sto3g_li = np.load(f'{data_path}/Li_plus_e_hf_6-31g.npy')

    # test: SCF convergence for li in 6-31g, compared with the CS algorithm at theta = 0
    converged, E_hf, E_e_values, C_munu, P = RHF(S_sto3g_li, T_sto3g_li, V_sto3g_li, eri_sto3g_li, n_electrons=2, max_iter=100, threshold=1E-14, p_guess='core', verbose=False)
    assert converged, "Calculation did not converge"
    assert abs(E_hf - E_hf_sto3g_li) < 1E-8, f"SCF energy does not match reference value {E_hf} != {E_hf_sto3g_li}"
    converged, E_cs_hf, *_ = CS_RHF(S_sto3g_li, T_sto3g_li, V_sto3g_li, eri_sto3g_li, n_electrons=2, theta=0.0, max_iter=100, threshold=1E-14, p_guess='core', verbose=False)
    assert converged, "CS-RHF Calculation did not converge"
    assert abs(E_cs_hf - E_hf_sto3g_li) < 1E-8, f"CS-RHF energy does not match unscaled reference value {E_cs_hf} != {E_hf_sto3g_li}"

def test_theta_non_scaled() -> None:
    S_29s_He    = np.loadtxt(f'{data_path}/He_S_29s.dat')
    T_29s_He    = np.loadtxt(f'{data_path}/He_kin_29s.dat')
    V_29s_He    = np.loadtxt(f'{data_path}/He_vnuc_29s.dat')
    eri_29s_He  = np.load(f'{data_path}/He_eri_29s.npy')
    E_hf_29s_He = np.load(f'{data_path}/He_e_hf_29s.npy')

    # test: SCF convergence for He in 29s, compared with the CS algorithm at theta = 0
    converged, E_hf, E_e_values, C_munu, P = RHF(S_29s_He, T_29s_He, V_29s_He, eri_29s_He, n_electrons=2, max_iter=100, threshold=1E-12, p_guess='core', verbose=False)
    assert converged == True, "Calculation did not converge"
    assert abs(E_hf - E_hf_29s_He) < 1E-8, f"SCF energy does not match reference value {E_hf} != {E_hf_29s_He}"
    converged, E_cs_hf, *_ = CS_RHF(S_29s_He, T_29s_He, V_29s_He, eri_29s_He, n_electrons=2, theta=0.0, max_iter=100, threshold=1E-12, p_guess='core', verbose=False)
    assert converged == True, "CS-RHF Calculation did not converge"
    assert abs(E_cs_hf - E_hf_29s_He) < 1E-8, f"CS-RHF energy does not match unscaled reference value {E_cs_hf} != {E_hf_29s_He}"

def test_theta_18_scaled() -> None:
    S_29s_He    = np.loadtxt(f'{data_path}/He_S_29s.dat')
    T_29s_He    = np.loadtxt(f'{data_path}/He_kin_29s.dat')
    V_29s_He    = np.loadtxt(f'{data_path}/He_vnuc_29s.dat')
    eri_29s_He  = np.load(f'{data_path}/He_eri_29s.npy')
    E_hf_29s_He = -2.8616799930014833+0j

    # test: SCF convergence for He in 29s, compared with the CS algorithm at theta = 0
    converged, E_cs_hf, *_ = CS_RHF(S_29s_He, T_29s_He, V_29s_He, eri_29s_He, n_electrons=2, theta=0.0, max_iter=100, threshold=1E-12, p_guess='core', verbose=False)
    assert converged == True, "CS-RHF Calculation did not converge"
    assert abs(E_cs_hf - E_hf_29s_He) < 1E-8, f"CS-RHF energy does not match unscaled reference value {E_cs_hf} != {E_hf_29s_He}"

def test_theta_excited_non_scaled() -> None:
    S_29s_He    = np.loadtxt(f'{data_path}/He_S_29s.dat')
    T_29s_He    = np.loadtxt(f'{data_path}/He_kin_29s.dat')
    V_29s_He    = np.loadtxt(f'{data_path}/He_vnuc_29s.dat')
    eri_29s_He  = np.load(f'{data_path}/He_eri_29s.npy')
    E_hf_29s_He = -0.7126661655570355+0j

    # test: SCF convergence for He in 29s, compared with the CS algorithm at theta = 0
    occupation_determinant = np.array([0,2,0])
    converged, E_elec_comp, E_e_values, C_munu, P = CS_RHF(S_29s_He, T_29s_He, V_29s_He, eri_29s_He, 2, theta=0.0, occupation=occupation_determinant, max_iter=500, threshold=1E-12, p_guess='core', verbose=False)
    assert converged == True, "CS-RHF Calculation did not converge"
    assert abs(E_elec_comp - E_hf_29s_He) < 1E-8, f"CS-RHF energy does not match unscaled reference value {E_elec_comp} != {E_hf_29s_He}"

def test_theta_excited_non_scaled_huge_basis() -> None:
    '''This test takes about "5.93s user 17.29s system 909% cpu 2.552 total" seconds with the current implementation'''
    S_aug_5Z_He    = np.loadtxt(f'{data_path}/He_S_aug-cc-pv(5+d)z.dat')
    T_aug_5Z_He    = np.loadtxt(f'{data_path}/He_kin_aug-cc-pv(5+d)z.dat')
    V_aug_5Z_He    = np.loadtxt(f'{data_path}/He_vnuc_aug-cc-pv(5+d)z.dat')
    eri_aug_5Z_He  = np.load(f'{data_path}/He_eri_aug-cc-pv(5+d)z.npy')
    E_hf_aug_5Z_He = -0.7191606246115501+3.8786763672415536e-18j

    # test: SCF convergence for He in aug-cc-pv(5+d)z, compared with the CS algorithm at theta = 0
    occupation_determinant = np.array([0,2,0])
    converged, E_elec_comp, E_e_values, C_munu, P = CS_RHF(S_aug_5Z_He, T_aug_5Z_He, V_aug_5Z_He, eri_aug_5Z_He, 2, theta=0.00, occupation=occupation_determinant, max_iter=500, threshold=1E-12, p_guess='core', verbose=False)
    assert converged == True, "CS-RHF Calculation did not converge"
    assert abs(E_elec_comp - E_hf_aug_5Z_He) < 1E-8, f"CS-RHF energy does not match unscaled reference value {E_elec_comp} != {E_hf_aug_5Z_He}"

def test_theta_excited_18_scaled_huge_basis() -> None:
    '''This test takes about "5.56s user 17.50s system 898% cpu 2.567 total" seconds with the current implementation'''
    S_aug_5Z_He    = np.loadtxt(f'{data_path}/He_S_aug-cc-pv(5+d)z.dat')
    T_aug_5Z_He    = np.loadtxt(f'{data_path}/He_kin_aug-cc-pv(5+d)z.dat')
    V_aug_5Z_He    = np.loadtxt(f'{data_path}/He_vnuc_aug-cc-pv(5+d)z.dat')
    eri_aug_5Z_He  = np.load(f'{data_path}/He_eri_aug-cc-pv(5+d)z.npy')
    E_hf_aug_5Z_He = -0.7193108482175761-0.00015642424740663213j

    # test: SCF convergence for He in aug-cc-pv(5+d)z, compared with the CS algorithm at theta = 0
    occupation_determinant = np.array([0,2,0])
    converged, E_elec_comp, E_e_values, C_munu, P = CS_RHF(S_aug_5Z_He, T_aug_5Z_He, V_aug_5Z_He, eri_aug_5Z_He, 2, theta=0.05, occupation=occupation_determinant, max_iter=500, threshold=1E-12, p_guess='core', verbose=True)
    assert converged == True, "CS-RHF Calculation did not converge"
    assert abs(E_elec_comp - E_hf_aug_5Z_He) < 1E-8, f"CS-RHF energy does not match unscaled reference value {E_elec_comp} != {E_hf_aug_5Z_He}"


if __name__ == "__main__":
    pass