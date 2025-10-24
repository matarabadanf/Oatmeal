import numpy as np
from dev.CSUHF_dev import CS_RHF
from py_mods.src.RHF import RHF
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