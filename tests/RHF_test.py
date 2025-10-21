import numpy as np
from py_mods.src.RHF import RHF
from py_mods.src.scf_utils import V_NN

def test_H2() -> None:
    S_sto3g_H2    = np.loadtxt('tests/data/H2_S_sto3g.dat')
    T_sto3g_H2    = np.loadtxt('tests/data/H2_kin_sto3g.dat')
    V_sto3g_H2    = np.loadtxt('tests/data/H2_vnuc_sto3g.dat')
    eri_sto3g_H2  = np.load('tests/data/H2_eri_sto3g.npy')
    E_hf_sto3g_H2 = np.load('tests/data/H2_e_hf_sto3g.npy')

    positions = np.array([
        [0. , 0. , 0. ],
        [0. , 0. , 1.4]
    ])
    nuc_nuc = V_NN(positions, np.array([1,1]), units='Bohr')

    # test : SCF convergence for H2 in STO-3G
    converged, E_elec, E_e_values, C_munu, P = RHF(S_sto3g_H2, T_sto3g_H2, V_sto3g_H2, eri_sto3g_H2, n_electrons=2, max_iter=100, threshold=1E-14, p_guess='core', verbose=False)
    assert converged, "Calculation did not converge"
    E_hf = E_elec + nuc_nuc
    assert abs(E_hf - E_hf_sto3g_H2) < 1E-6, f"SCF energy does not match reference value {E_hf} != {E_hf_sto3g_H2}"

def test_Li() -> None:
    S_sto3g_li    = np.loadtxt('tests/data/Li_plus_S_6-31g.dat')
    T_sto3g_li    = np.loadtxt('tests/data/Li_plus_kin_6-31g.dat')
    V_sto3g_li    = np.loadtxt('tests/data/Li_plus_vnuc_6-31g.dat')
    eri_sto3g_li  = np.load('tests/data/Li_plus_eri_6-31g.npy')
    E_hf_sto3g_li = np.load('tests/data/Li_plus_e_hf_6-31g.npy')

    # test: SCF convergence for li in 6-31g
    converged, E_hf, E_e_values, C_munu, P = RHF(S_sto3g_li, T_sto3g_li, V_sto3g_li, eri_sto3g_li, n_electrons=2, max_iter=100, threshold=1E-14, p_guess='core', verbose=False)
    assert converged, "Calculation did not converge"
    assert abs(E_hf - E_hf_sto3g_li) < 1E-8, f"SCF energy does not match reference value {E_hf} != {E_hf_sto3g_li}"

def test_Be() -> None:
    S_ccpvdz_Be    = np.loadtxt('tests/data/Be_S_ccpvdz.dat')
    T_ccpvdz_Be    = np.loadtxt('tests/data/Be_kin_ccpvdz.dat')
    V_ccpvdz_Be    = np.loadtxt('tests/data/Be_vnuc_ccpvdz.dat')
    eri_ccpvdz_Be  = np.load('tests/data/Be_eri_ccpvdz.npy')
    E_hf_ccpvdz_Be = np.load('tests/data/Be_e_hf_ccpvdz.npy')

    # test: SCF convergence for Be in ccpvdz
    converged, E_hf, E_e_values, C_munu, P = RHF(S_ccpvdz_Be, T_ccpvdz_Be, V_ccpvdz_Be, eri_ccpvdz_Be, n_electrons=4, max_iter=100, threshold=1E-14, p_guess='core', verbose=False)
    assert converged, "Calculation did not converge"
    assert abs(E_hf - E_hf_ccpvdz_Be) < 1E-8, f"SCF energy does not match reference value {E_hf} != {E_hf_ccpvdz_Be}"

if __name__ == "__main__":
    pass
    