import numpy as np
from py_mods.src.RHF import RHF

def test_Li():
    S_sto3g_li    = np.loadtxt('tests/data/Li_plus_S_6-31g.dat')
    T_sto3g_li    = np.loadtxt('tests/data/Li_plus_kin_6-31g.dat')
    V_sto3g_li    = np.loadtxt('tests/data/Li_plus_vnuc_6-31g.dat')
    eri_sto3g_li  = np.load('tests/data/Li_plus_eri_6-31g.npy')
    E_hf_sto3g_li = np.load('tests/data/Li_plus_e_hf_6-31g.npy')

    idn = np.identity(len(S_sto3g_li))

    # test 2: SCF convergence for li in 6-31g
    converged, E_hf, E_e_values, C_munu, P = RHF(S_sto3g_li, T_sto3g_li, V_sto3g_li, eri_sto3g_li, n_electrons=2, max_iter=100, threshold=1E-14, p_guess='core', verbose=True)
    assert converged, "Calculation did not converge"
    assert abs(E_hf - E_hf_sto3g_li) < 1E-8, f"SCF energy does not match reference value {E_hf} != {E_hf_sto3g_li}"

def test_Be():
    S_ccpvdz_Be    = np.loadtxt('tests/data/Be_S_ccpvdz.dat')
    T_ccpvdz_Be    = np.loadtxt('tests/data/Be_kin_ccpvdz.dat')
    V_ccpvdz_Be    = np.loadtxt('tests/data/Be_vnuc_ccpvdz.dat')
    eri_ccpvdz_Be  = np.load('tests/data/Be_eri_ccpvdz.npy')
    E_hf_ccpvdz_Be = np.load('tests/data/Be_e_hf_ccpvdz.npy')

    idn = np.identity(len(S_ccpvdz_Be))

    # test 2: SCF convergence for Be in ccpvdz
    converged, E_hf, E_e_values, C_munu, P = RHF(S_ccpvdz_Be, T_ccpvdz_Be, V_ccpvdz_Be, eri_ccpvdz_Be, n_electrons=4, max_iter=100, threshold=1E-14, p_guess='core', verbose=False)
    assert converged, "Calculation did not converge"
    assert abs(E_hf - E_hf_ccpvdz_Be) < 1E-8, f"SCF energy does not match reference value {E_hf} != {E_hf_ccpvdz_Be}"
