from typing import Optional, Tuple, Union, List, Literal

import numpy as np
from numpy.typing import NDArray

from py_mods.src.integrals.UncontractedBasisSet import (
    UncontractedBasisSet,
    ERIs_Uncontracted,
)

from py_mods.src.SCF.scf_kernels import calc_p_matrix_comp


def full_eri_from_Uncontracted_Basis(UBS: UncontractedBasisSet):
    eri_tensor = ERIs_Uncontracted(UBS)

    return eri_tensor


def eri_classified(eri: NDArray[np.float64], nL: int) -> NDArray[np.float64]:
    eri_classess = np.zeros_like(eri, dtype=np.float64)

    eri_classess[:nL, :nL, :nL, :nL] = eri[:nL, :nL, :nL, :nL]  # LL-LL block
    eri_classess[:nL, :nL, nL:, nL:] = eri[:nL, :nL, nL:, nL:]  # LL-SS block
    eri_classess[nL:, nL:, :nL, :nL] = eri[nL:, nL:, :nL, :nL]  # SS-LL block
    eri_classess[nL:, nL:, nL:, nL:] = eri[nL:, nL:, nL:, nL:]  # SS-SS block

    return eri_classess


def occupation_4c(
    nS, nL, n_electrons, electronic_occ_det: Union[None, NDArray[np.int_]] = None
):
    occ = np.zeros(2 * (nS + nL), dtype=np.uint8)

    n_positron_states = 2 * nS

    if electronic_occ_det is None:
        occ[n_positron_states : n_positron_states + n_electrons] = 1
    else:
        assert (
            len(electronic_occ_det) == 2 * nL
        ), "Length of electronic occupation array must be equal to 2*nL"
        assert (
            sum(electronic_occ_det) == n_electrons
        ), "Sum of electronic occupation array must be equal to n_electrons"
        occ[n_positron_states:] = electronic_occ_det

    return occ


def g_matrix_4c(P, eri):
    n_bas = P.shape[0]
    n_bas_half = n_bas // 2

    P_aa = P[0:n_bas_half, 0:n_bas_half]
    P_bb = P[n_bas_half:n_bas, n_bas_half:n_bas]
    P_ab = P[0:n_bas_half, n_bas_half:n_bas]
    P_ba = P[n_bas_half:n_bas, 0:n_bas_half]

    P_total = P_aa + P_bb
    J = np.einsum("mnsl, ls -> mn", eri, P_total)

    K_aa = np.einsum("psrq, sr -> pq", eri, P_aa)
    K_bb = np.einsum("psrq, sr -> pq", eri, P_bb)
    K_ab = np.einsum("psrq, sr -> pq", eri, P_ab)
    K_ba = np.einsum("psrq, sr -> pq", eri, P_ba)

    G_aa = J - K_aa
    G_bb = J - K_bb
    G_ab = -K_ab
    G_ba = -K_ba

    G_full = np.zeros((n_bas, n_bas), dtype=np.complex128)
    # And we fill the matrix by blocks
    G_full[0:n_bas_half, 0:n_bas_half] = G_aa
    G_full[n_bas_half:n_bas, n_bas_half:n_bas] = G_bb
    G_full[0:n_bas_half, n_bas_half:n_bas] = G_ab
    G_full[n_bas_half:n_bas, 0:n_bas_half] = G_ba

    return G_full


def scf_iteration(F_1, X, total_occ_det):
    F_p1 = X.T @ F_1 @ X

    e1, w1 = np.linalg.eigh(F_p1)

    idx = np.argsort(e1)
    e1 = e1[idx]
    w1 = w1[:, idx]

    c_alpha_beta1 = X @ w1
    P_1 = calc_p_matrix_comp(c_alpha_beta1.conj().T, c_alpha_beta1, total_occ_det)

    return e1, w1, F_p1, P_1


def scf_steps(n_steps, H_core, eri, X, total_occ_det):
    energy_step = []
    P_old = np.zeros_like(H_core)

    for i in range(n_steps):
        if i == 0:
            G_new = np.zeros_like(H_core)
        else:
            G_new = g_matrix_4c(P_old, eri)

        if i > 0:
            e_scf = np.linalg.trace(P_old @ H_core + 0.5 * P_old @ G_new)
            energy_step.append(e_scf.real)

        F_new = H_core + G_new

        e_new, w_new, F_p_new, P_2 = scf_iteration(F_new, X, total_occ_det)

        if i == 0:
            e_scf = np.linalg.trace(P_2 @ H_core)
            energy_step.append(e_scf.real)

        P_old = P_2

    return energy_step
