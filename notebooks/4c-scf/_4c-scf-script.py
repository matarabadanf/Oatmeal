import numpy as np

from py_mods.src.SCF.linalg import transformation_matrix
from py_mods.src.external.DIRAC_ME import (
    build_4c_one_Fock_from_h5,
    build_S_V_W_T_from_h5,
    get_nuc_charge,
    full_eri_from_checkpoint,
)
from py_mods.src.SCF.scf_kernels import calc_p_matrix_comp


def J_LL(P_total, eri_LLLL, eri_LLSS, nL):
    J_LL = np.zeros_like(P_total[:nL, :nL])
    P_LL = P_total[:nL, :nL]
    P_SS = P_total[nL:, nL:]
    J_LL += np.einsum("mnsl, ls -> mn", eri_LLLL, P_LL)
    J_LL += np.einsum("mnsl, ls -> mn", eri_LLSS, P_SS)
    return J_LL


def J_SS(P_total, eri_SSSS, eri_SSLL, nL):
    J_SS = np.zeros_like(P_total[nL:, nL:])
    P_LL = P_total[:nL, :nL]
    P_SS = P_total[nL:, nL:]
    J_SS += np.einsum("mnsl, ls -> mn", eri_SSLL, P_LL)
    J_SS += np.einsum("mnsl, ls -> mn", eri_SSSS, P_SS)
    return J_SS


def g_matrix_4c(P, eri):
    n_bas = P.shape[0]
    n_bas_half = n_bas // 2

    P_aa = P[0:n_bas_half, 0:n_bas_half]
    P_bb = P[n_bas_half:n_bas, n_bas_half:n_bas]
    P_ab = P[0:n_bas_half, n_bas_half:n_bas]
    P_ba = P[n_bas_half:n_bas, 0:n_bas_half]

    P_total = P_aa + P_bb
    # J from total density like in UHF
    J = np.einsum("mnsl, ls -> mn", eri, P_total)

    K_aa = np.einsum("psrq, sr -> pq", eri, P_aa)
    K_bb = np.einsum("psrq, sr -> pq", eri, P_bb)
    K_ab = np.einsum("psrq, sr -> pq", eri, P_ab)
    K_ba = np.einsum("psrq, sr -> pq", eri, P_ba)

    G_aa = J - K_aa
    G_bb = J - K_bb
    # No J as no different spin coulomb

    G_ab = -K_ab
    G_ba = -K_ba

    G_full = np.zeros((n_bas, n_bas), dtype=np.complex128)
    G_full[0:n_bas_half, 0:n_bas_half] = G_aa
    G_full[n_bas_half:n_bas, n_bas_half:n_bas] = G_bb
    G_full[0:n_bas_half, n_bas_half:n_bas] = G_ab
    G_full[n_bas_half:n_bas, 0:n_bas_half] = G_ba

    return G_full


def g_matrix_4c2(P, eri, nL):
    n_bas = P.shape[0]
    n_bas_half = n_bas // 2

    eri_LLLL = eri[:nL, :nL, :nL, :nL]
    eri_LLSS = eri[:nL, :nL, nL:, nL:]
    eri_SSLL = eri[nL:, nL:, :nL, :nL]
    eri_SSSS = eri[nL:, nL:, nL:, nL:]

    P_aa = P[0:n_bas_half, 0:n_bas_half]
    P_bb = P[n_bas_half:n_bas, n_bas_half:n_bas]
    P_ab = P[0:n_bas_half, n_bas_half:n_bas]
    P_ba = P[n_bas_half:n_bas, 0:n_bas_half]

    P_total = P_aa + P_bb

    J_ll = J_LL(P_total, eri_LLLL, eri_LLSS, nL)
    J_ss = J_SS(P_total, eri_SSSS, eri_SSLL, nL)

    J = np.zeros((n_bas, n_bas), dtype=np.complex128)
    J[0:nL, 0:nL] = J[n_bas_half : n_bas_half + nL, n_bas_half : n_bas_half + nL] = J_ll
    J[nL:n_bas_half, nL:n_bas_half] = J[
        n_bas_half + nL : n_bas, n_bas_half + nL : n_bas
    ] = J_ss

    K_aa = np.einsum("psrq, sr -> pq", eri, P_aa)
    K_bb = np.einsum("psrq, sr -> pq", eri, P_bb)
    K_ab = np.einsum("psrq, sr -> pq", eri, P_ab)
    K_ba = np.einsum("psrq, sr -> pq", eri, P_ba)

    G_aa = J[:n_bas_half, :n_bas_half] - K_aa
    G_bb = J[n_bas_half:, n_bas_half:] - K_bb
    # No J as no different spin coulomb

    G_ab = -K_ab
    G_ba = -K_ba

    G_full = np.zeros((n_bas, n_bas), dtype=np.complex128)
    G_full[0:n_bas_half, 0:n_bas_half] = G_aa
    G_full[n_bas_half:n_bas, n_bas_half:n_bas] = G_bb
    G_full[0:n_bas_half, n_bas_half:n_bas] = G_ab
    G_full[n_bas_half:n_bas, 0:n_bas_half] = G_ba

    return G_full


def scf_iteration(F_1, X, nL, reference=None):
    F_p1 = X.T @ F_1 @ X

    e1, w1 = np.linalg.eigh(F_p1)

    idx = np.argsort(e1)
    e1 = e1[idx]
    w1 = w1[:, idx]

    e1_reduced = [ei for i, ei in enumerate(e1) if i % 2 == 0]

    if reference is not None:
        print(
            f"Eigenvalues are the same for the first iteration: {np.allclose(e1_reduced, reference)}"
        )
        print(
            f"Mean absolute error for the first iteration: {np.mean(np.abs(e1_reduced - reference))}"
        )
        print(
            f"Max absolute error for the first iteration: {np.max(np.abs(e1_reduced - reference))}"
        )
        print(
            f"Mean absolute error for the first iteration (pos eigvals): {np.mean(np.abs(reference[-nL:] - reference[-nL:]))}"
        )

    c_alpha_beta1 = X @ w1
    P_1 = calc_p_matrix_comp(c_alpha_beta1.conj().T, c_alpha_beta1, total_occ_det)
    # plot_map(P_1)

    return e1, w1, F_p1, P_1


def scf_steps(n_steps):
    energy_step = []
    for i in range(0, n_steps):
        if i == 0:
            G_new = np.zeros_like(H_core)
        else:
            G_new = g_matrix_4c(P_2, eri)
        F_new = H_core + G_new
        e_new, w_new, F_p_new, P_2 = scf_iteration(F_new, X, nL=9)
        e_scf = np.linalg.trace(P_2 @ H_core + 0.5 * P_2 @ G_new)
        print(f"\n\nSCF energy for iteration {i}: {e_scf.real} Hartree")
        energy_step.append(e_scf)
    
    return np.array(energy_step, dtype=np.complex128).real


if __name__ == "__main__":
    h5_filename = "files/He_checkpoint.h5"
    He_F0_eigvals = np.loadtxt("files/He_F_eigvals_1st_iter.dat")
    He_F1_eigvals = np.loadtxt("files/He_F_eigvals_2nd_iter.dat")
    He_F2_eigvals = np.loadtxt("files/He_F_eigvals_3rd_iter.dat")
    ref_e_scf = np.loadtxt("files/He_scf_energy.dat")

    F_0 = build_4c_one_Fock_from_h5(h5_filename)
    S, V, W, T = build_S_V_W_T_from_h5(h5_filename)
    nuc_charge = get_nuc_charge(h5_filename)
    eri = full_eri_from_checkpoint(h5_filename)

    H_core = T + V + W

    X = transformation_matrix(S)

    occ = np.zeros(S.shape[0])
    occ[:] = 0
    occ[-18] = occ[-17] = 1
    total_occ_det = occ

    scf_energies = scf_steps(15)

    print(f"difference in energies at each step")
    print(scf_energies[:len(ref_e_scf)] - ref_e_scf)


