import numpy as np
# from py_mods.src.SCF.CSRHF import CS_RHF, RHF_theta_traj, CS_RHF_ContextClass
from  py_mods.src.SCF.CSRHF import CS_RHF, RHF_theta_traj, CS_RHF_ContextClass
from py_mods.src.SCF.CSUHF import UHF_theta_traj, CS_UHF_ContextClass
from py_mods.src.SCF.RHF import RHF
from pathlib import Path

data_path = Path(__file__).parent / "data"
qchem_path = data_path / 'qchem'

def load_traj(filename):
    with open(filename) as f:
        cont = f.readlines()
    cont = [line.strip().replace('(', '').replace(')','') for line in cont]
    thetas = np.array([int(line.split(';')[0]) for line in cont])
    eners = [line.split(';')[1].strip().replace(' ', '').replace(',','+').replace('+-', '-') +'j' for line in cont]
    eners = np.array([complex(a) for a in eners])

    return thetas, eners

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

    Li_cxt = CS_RHF_ContextClass(
        S=S_sto3g_li,
        T=T_sto3g_li,
        V=V_sto3g_li,
        eri=eri_sto3g_li,
        n_electrons=2,
        max_iter=100,
        threshold=1E-14,
        p_guess='core',
        verbose=False
    )

    Li_results = CS_RHF(Li_cxt)
    assert converged, "CS-RHF Calculation did not converge"
    assert abs(Li_results.E_RHF - E_hf_sto3g_li) < 1E-8, f"CS-RHF energy does not match unscaled reference value {Li_results.E_RHF} != {E_hf_sto3g_li}"

def test_theta_non_scaled() -> None:
    S_29s_He    = np.loadtxt(f'{data_path}/He_S_29s.dat')
    T_29s_He    = np.loadtxt(f'{data_path}/He_kin_29s.dat')
    V_29s_He    = np.loadtxt(f'{data_path}/He_vnuc_29s.dat')
    eri_29s_He  = np.load(f'{data_path}/He_eri_29s.npy')
    E_hf_29s_He = np.load(f'{data_path}/He_e_hf_29s.npy')

    # test: SCF convergence for He in 29s, compared with the CS algorithm at theta = 0
    converged, E_hf, E_e_values, C_munu, P = RHF(S_29s_He, T_29s_He, V_29s_He, eri_29s_He, n_electrons=2, max_iter=400, threshold=1.2498E-07, p_guess='core', verbose=True)
    assert converged == True, "Calculation did not converge"
    assert abs(E_hf - E_hf_29s_He) < 1E-8, f"SCF energy does not match reference value {E_hf} != {E_hf_29s_He}"

    even_s29_ctx = CS_RHF_ContextClass(
        S=S_29s_He,
        T=T_29s_He,
        V=V_29s_He,
        eri=eri_29s_He,
        n_electrons=2,
        max_iter=100,
        threshold=6.2532E-08,
        p_guess='core',
        verbose=False
    )

    even_s29_res = CS_RHF(even_s29_ctx)
    assert even_s29_res.converged == True, "CS-RHF Calculation did not converge"
    assert abs(even_s29_res.E_RHF - E_hf_29s_He) < 1E-8, f"CS-RHF energy does not match unscaled reference value {even_s29_res.E_RHF} != {E_hf_29s_He}"


def test_theta_18_scaled() -> None:
    S_29s_He    = np.loadtxt(f'{data_path}/He_S_29s.dat')
    T_29s_He    = np.loadtxt(f'{data_path}/He_kin_29s.dat')
    V_29s_He    = np.loadtxt(f'{data_path}/He_vnuc_29s.dat')
    eri_29s_He  = np.load(f'{data_path}/He_eri_29s.npy')
    E_hf_29s_He = -2.8616799930014833+0j

    He_29s_context = CS_RHF_ContextClass(
        S=S_29s_He,
        T=T_29s_He,
        V=V_29s_He,
        eri=eri_29s_He,
        n_electrons=2,
        max_iter=100,
        threshold=6.2608E-08,
        p_guess='core',
        verbose=False
    )

    He_29s_context.theta = 0.18
    # test: SCF convergence for He in 29s, compared with the CS algorithm at theta = 0
    He_29s_res = CS_RHF(He_29s_context)
    assert He_29s_res.converged == True, "CS-RHF Calculation did not converge"
    assert abs(He_29s_res.E_RHF - E_hf_29s_He) < 1E-8, f"CS-RHF energy does not match unscaled reference value {He_29s_res.E_RHF} != {E_hf_29s_He}"

def test_theta_excited_non_scaled() -> None:
    S_29s_He    = np.loadtxt(f'{data_path}/He_S_29s.dat')
    T_29s_He    = np.loadtxt(f'{data_path}/He_kin_29s.dat')
    V_29s_He    = np.loadtxt(f'{data_path}/He_vnuc_29s.dat')
    eri_29s_He  = np.load(f'{data_path}/He_eri_29s.npy')
    E_hf_29s_He = -0.7126661655570355+0j

    He_29s_2s2 = CS_RHF_ContextClass(
        S=S_29s_He,
        T=T_29s_He,
        V=V_29s_He,
        eri=eri_29s_He,
        n_electrons=2,
        max_iter=100,
        threshold=1.7474E-08,
        p_guess='core',
        verbose=False
    )

    He_29s_2s2.occupation = np.array([0,2,0])

    # test: SCF convergence for He in 29s, compared with the CS algorithm at theta = 0
    He_29s_2s2_results = CS_RHF(He_29s_2s2)
    assert He_29s_2s2_results.converged == True, "CS-RHF Calculation did not converge"
    assert abs(He_29s_2s2_results.E_RHF - E_hf_29s_He) < 1E-8, f"CS-RHF energy does not match unscaled reference value {He_29s_2s2_results.E_RHF} != {E_hf_29s_He}"

def test_theta_excited_non_scaled_huge_basis() -> None:
    '''This test takes about "5.93s user 17.29s system 909% cpu 2.552 total" seconds with the current implementation'''
    S_aug_5Z_He    = np.loadtxt(f'{data_path}/He_S_aug-cc-pv(5+d)z.dat')
    T_aug_5Z_He    = np.loadtxt(f'{data_path}/He_kin_aug-cc-pv(5+d)z.dat')
    V_aug_5Z_He    = np.loadtxt(f'{data_path}/He_vnuc_aug-cc-pv(5+d)z.dat')
    eri_aug_5Z_He  = np.load(f'{data_path}/He_eri_aug-cc-pv(5+d)z.npy')
    E_hf_aug_5Z_He = -0.7191606246115501+3.8786763672415536e-18j

    He_29s_2s2 = CS_RHF_ContextClass(
        S=S_aug_5Z_He,
        T=T_aug_5Z_He,
        V=V_aug_5Z_He,
        eri=eri_aug_5Z_He,
        n_electrons=2,
        max_iter=500,
        threshold=1E-12,
        p_guess='core',
        verbose=False
    )

    He_29s_2s2.occupation = np.array([0,2,0])

    # test: excited SCF convergence for He in aug-cc-pv(5+d)z
    He_29s_2s2_results = CS_RHF(He_29s_2s2)
    assert He_29s_2s2_results.converged == True, "CS-RHF Calculation did not converge"
    assert abs(He_29s_2s2_results.E_RHF - E_hf_aug_5Z_He) < 1E-8, f"CS-RHF energy does not match unscaled reference value {He_29s_2s2_results.E_RHF} != {E_hf_aug_5Z_He}"

def test_theta_excited_18_scaled_huge_basis() -> None:
    '''This test takes about "5.56s user 17.50s system 898% cpu 2.567 total" seconds with the current implementation'''
    S_aug_5Z_He    = np.loadtxt(f'{data_path}/He_S_aug-cc-pv(5+d)z.dat')
    T_aug_5Z_He    = np.loadtxt(f'{data_path}/He_kin_aug-cc-pv(5+d)z.dat')
    V_aug_5Z_He    = np.loadtxt(f'{data_path}/He_vnuc_aug-cc-pv(5+d)z.dat')
    eri_aug_5Z_He  = np.load(f'{data_path}/He_eri_aug-cc-pv(5+d)z.npy')
    E_hf_aug_5Z_He = -0.7193108482175761-0.00015642424740663213j # theta = 0.05

    He_29s_2s2 = CS_RHF_ContextClass(
        S=S_aug_5Z_He,
        T=T_aug_5Z_He,
        V=V_aug_5Z_He,
        eri=eri_aug_5Z_He,
        n_electrons=2,
        max_iter=500,
        threshold=1E-12,
        p_guess='core',
        verbose=False
    )

    He_29s_2s2.occupation = np.array([0,2,0])
    He_29s_2s2.theta=0.05   

    # test: excited SCF convergence for He in aug-cc-pv(5+d)z with theta = 0.05
    He_29s_2s2_results = CS_RHF(He_29s_2s2)
    assert He_29s_2s2_results.converged == True, "CS-RHF Calculation did not converge"
    assert abs(He_29s_2s2_results.E_RHF - E_hf_aug_5Z_He) < 1E-8, f"CS-RHF energy does not match unscaled reference value {He_29s_2s2_results.E_RHF} != {E_hf_aug_5Z_He}"

def test_qchem_21s() -> None:
    '''1.62s user 1.69s system 312% cpu 1.057 total'''

    S_even_H2    = np.loadtxt(f'{data_path}/He_S_21s.dat')
    T_even_H2    = np.loadtxt(f'{data_path}/He_kin_21s.dat')
    V_even_H2    = np.loadtxt(f'{data_path}/He_vnuc_21s.dat')
    eri_even_H2  = np.load(f'{data_path}/He_eri_21s.npy')
    
    max_theta = 0.08 # because we have this data for reference
    n_points = 9

    w, k = load_traj(f'{qchem_path}/He_1s2_eventemp_qchem.dat')
    w, k2 = load_traj(f'{qchem_path}/He_2s2_eventemp_qchem.dat')


    # Test for the 1s2 case of both RHF and UHF 
    H2_RHF_context = CS_RHF_ContextClass(S_even_H2, T_even_H2, V_even_H2, eri_even_H2, 2, max_iter=1000, threshold=2E-10, conv_type='CROP')
    H2_context = CS_UHF_ContextClass(S_even_H2, T_even_H2, V_even_H2, eri_even_H2, 2, max_iter=1000, threshold=2E-10, p_guess='core', verbose=False, conv_type='CROP')

    traj_energies = RHF_theta_traj(max_theta, n_points, H2_RHF_context)
    assert abs(np.mean(traj_energies[1]-k)) < 1E-8+1E-8j, f'Mean error is too large in 1s2 RHF: {abs(np.mean(traj_energies[1]-k))}'
    
    traj_energies = UHF_theta_traj(max_theta, n_points, H2_context)
    assert abs(np.mean(traj_energies[1]-k)) < 1E-8+1E-8j, f'Mean error is too large in 1s2 UHF: {abs(np.mean(traj_energies[1]-k))}'
    
    # and for the Scaled 2s2 case RHF and UHF
    max_theta = 0.40 # because we have this data for reference
    n_points = 41

    H2_RHF_context.occupation = np.array([0,2,0])
    H2_context.occupation = (np.array([0,1,0]), np.array([0,1,0]))

    traj_energies = RHF_theta_traj(max_theta, n_points, H2_RHF_context)
    assert abs(np.mean(traj_energies[1]-k2)) < 1E-8+1E-8j, f'Mean error is too large in 2s2 RHF: {abs(np.mean(traj_energies[1]-k))}'

    traj_energies = UHF_theta_traj(max_theta, n_points, H2_context)
    assert abs(np.mean(traj_energies[1]-k2)) < 1E-8+1E-8j, f'Mean error is too large in 2s2 UHF: {abs(np.mean(traj_energies[1]-k))}'


def test_qchem_huge() -> None:
    '''This test takes about "5.56s user 17.50s system 898% cpu 2.567 total" seconds with the current implementation'''
    S_aug_5Z_He    = np.loadtxt(f'{data_path}/He_S_aug-cc-pv(5+d)z.dat')
    T_aug_5Z_He    = np.loadtxt(f'{data_path}/He_kin_aug-cc-pv(5+d)z.dat')
    V_aug_5Z_He    = np.loadtxt(f'{data_path}/He_vnuc_aug-cc-pv(5+d)z.dat')
    eri_aug_5Z_He  = np.load(f'{data_path}/He_eri_aug-cc-pv(5+d)z.npy')

    max_theta = 0.08 # because we have this data for reference
    n_points = 9

    w, k = load_traj(f'{qchem_path}/He_1s2_augqz_qchem.dat')
    w, k2 = load_traj(f'{qchem_path}/He_2s2_augqz_qchem.dat')

    cxt_He_5Z = CS_RHF_ContextClass(
        S=S_aug_5Z_He,
        T=T_aug_5Z_He,
        V=V_aug_5Z_He,
        eri=eri_aug_5Z_He,
        n_electrons=2,
        max_iter=1000,
        threshold=2E-10,
        p_guess='core',
        verbose=False,
        conv_type='CROP'
    )

    # test: SCF convergence for He in aug-cc-pv(5+d)z, compared with the CS algorithm at theta = 0
    traj_cls_ener = RHF_theta_traj(max_theta, n_points, cxt_He_5Z)
    assert np.mean(traj_cls_ener[1]-k) < 1E-8+1E-8j, f'Mean error is {np.mean(traj_cls_ener-k) }'

    # and for the excited state
    cxt_He_5Z.occupation = np.array([0,2,0])
    traj_cls_ener = RHF_theta_traj(max_theta, n_points, cxt_He_5Z)
    assert np.mean(traj_cls_ener[1]-k2) < 1E-8+1E-8j, f'Mean error is {np.mean(traj_cls_ener-k) }'

if __name__ == "__main__":
    test_theta_non_scaled()
    pass