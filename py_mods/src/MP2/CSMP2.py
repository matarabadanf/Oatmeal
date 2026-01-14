import numpy as np
from numpy.typing import NDArray
from typing import Literal, Union
from dataclasses import dataclass
from py_mods.src.SCF.types import CSRHFResults
from py_mods.src.SCF.CSUHF import CS_UHF_ResultsClass


@dataclass
class CS_MP2_Results(object):
    CS_MP2Context: Union[CSRHFResults, CS_UHF_ResultsClass]
    E_MP2: np.complex128
    E_corr: np.complex128
    MP_type: Literal["RMP2", "UMP2"]
    t2_abrs: NDArray[np.complex128] | None
    eris_mo: NDArray[np.complex128]


def CS_MP2(
    CS_MP2Context: Union[CSRHFResults, CS_UHF_ResultsClass],
) -> CS_MP2_Results:
    """Compute the MP2 energy correction using complex scaled UHF or RHF reference.

    Parameters
    ----------
    CS_RHF_Context: Union[CSRHFResults, CS_UHF_ResultsClass]
        Dataclass containing converged CS-RHF results.

    Returns
    -------
    returnClass: CS_MP2_Results
        Dataclass containing the MP2 energy correction.
    """
    if isinstance(CS_MP2Context, CSRHFResults):
        mp2_result = CS_MP2_RHF(CS_MP2Context)
    elif isinstance(CS_MP2Context, CS_UHF_ResultsClass):
        mp2_result = CS_MP2_UHF(CS_MP2Context)
    else:
        raise TypeError(
            f"CS_MP2Context must be either CSRHFResults or CS_UHF_ResultsClass. Type is {type(CS_MP2Context)}"
        )

    return mp2_result


def CS_MP2_RHF(CS_RHF_Context: CSRHFResults) -> CS_MP2_Results:
    """Compute the MP2 energy correction using complex scaled RHF reference.

    Parameters
    ----------
    CS_RHF_Context: CSRHFResults
        Dataclass containing converged CS-RHF results.

    Returns
    -------
    returnClass: CS_MP2_Results
        Dataclass containing the MP2 energy correction.
    """
    mp_type = "RMP2"

    # naive approach: no symm
    C_munu = CS_RHF_Context.C_munu

    if np.isclose(CS_RHF_Context.context.theta, 0.0):
        C_munu = C_munu.real

    # rest of info
    e_orb = CS_RHF_Context.e_orb
    eris_ao = CS_RHF_Context.scaled_eris
    n_occ: int = int(CS_RHF_Context.n_elec // 2)
    n_tot: int = len(CS_RHF_Context.context.S)
    n_virt: int = n_tot - n_occ

    # get the occupied and virtual indices:
    o_i = np.array([i for i, j in enumerate(CS_RHF_Context.det) if j == 2])
    v_i = np.array([i for i, j in enumerate(CS_RHF_Context.det) if j == 0])

    # calculate only (ovov) integrals
    eris_mo_chem = ao_to_ovov(C_munu, eris_ao, o_i, v_i)

    # <ij||kl> = <ij|kl> - <ij|lk>
    # <ij|ab> = (ia|jb)
    # <ij|kl> = (ia|jb) - (ja|ib)
    eris_mo_phys = eris_mo_chem.transpose(0, 2, 1, 3)  # oovv

    # get energies of virtual and occupied blocks

    e_a = e_orb[o_i]
    e_b = e_orb[o_i]
    e_r = e_orb[v_i]
    e_s = e_orb[v_i]

    # dim = n_occ, n_occ, n_virt, n_virt
    # what this does is that it "stretches" the smaller size axes and then
    # builds the "4D cube" where each entry is the operation r + s - a- b
    denom_abrs = (
        e_s[None, None, :, None]
        + e_r[None, None, None, :]
        - e_a[:, None, None, None]
        - e_b[None, :, None, None]
    )

    asym_eri = eris_mo_phys - eris_mo_phys.transpose(0, 1, 3, 2)

    t2_abrs = asym_eri / denom_abrs

    # for RHF: num = <ab|rs> [2 <rs|ab> - <rs|ba>]
    # for UHF num = (<ab||rs>)**2
    num = eris_mo_phys * (2 * eris_mo_phys - eris_mo_phys.transpose(0, 1, 3, 2))

    contribution = num / denom_abrs

    E_corr = -np.sum(num / denom_abrs)

    E_MP2 = E_corr + CS_RHF_Context.E_RHF

    returnClass = CS_MP2_Results(
        CS_RHF_Context, E_MP2, E_corr, mp_type, contribution, eris_mo_chem
    )

    return returnClass


def CS_MP2_UHF(CS_UHF_Context: CS_UHF_ResultsClass) -> CS_MP2_Results:
    """
    Compute the MP2 energy correction using complex scaled UHF reference.

    Parameters
    ----------
    CS_UHF_Context: CS_UHF_ResultsClass
        Dataclass containing converged CS-UHF results.

    Returns
    -------
    returnClass: CS_MP2_Results
        Dataclass containing the MP2 energy correction.
    """

    verbose = CS_UHF_Context.context.verbose

    mp_type = "RMP2"

    # naive approach: no symm
    C_alph = CS_UHF_Context.C_alpha
    C_beta = CS_UHF_Context.C_beta

    if np.isclose(CS_UHF_Context.context.theta, 0.0):
        C_alph = C_alph.real
        C_beta = C_beta.real

    n_spatorb = len(C_alph)
    n_spinorb = n_spatorb * 2

    # Build the whole C_ab matrix with alpha and beta blocks
    C_ab = np.zeros([n_spinorb, n_spinorb])
    C_ab[:n_spatorb, :n_spatorb] = C_alph
    C_ab[n_spatorb:, n_spatorb:] = C_beta

    # rest of info
    e_alph = CS_UHF_Context.e_alpha
    e_beta = CS_UHF_Context.e_beta
    eris_ao = CS_UHF_Context.scaled_eris
    det_a = CS_UHF_Context.det[0]
    det_b = CS_UHF_Context.det[1]

    if verbose:
        print("e_alpha:")
        print(e_alph)
        print("e_beta:")
        print(e_beta)

    # we get the occupied and virtual indices:
    o_ia = np.array([i for i, j in enumerate(CS_UHF_Context.det[0]) if j == 1])
    v_ia = np.array([i for i, j in enumerate(CS_UHF_Context.det[0]) if j == 0])
    o_ib = np.array([i for i, j in enumerate(CS_UHF_Context.det[1]) if j == 1])
    v_ib = np.array([i for i, j in enumerate(CS_UHF_Context.det[1]) if j == 0])

    # alpha-alpha ERI MO block
    aa_mo_chem = ao_to_ovov_generalized(
        eris_ao, C_alph[:, o_ia], C_alph[:, v_ia], C_alph[:, o_ia], C_alph[:, v_ia]
    )  # (aa|aa) = <aa|aa>. Indices must be OVOV in chemists notation.

    bb_mo_chem = ao_to_ovov_generalized(
        eris_ao, C_beta[:, o_ib], C_beta[:, v_ib], C_beta[:, o_ib], C_beta[:, v_ib]
    )  # (bb|bb) = <bb|bb>. Indices must be OVOV in chemists notation.

    ab_mo_chem = ao_to_ovov_generalized(
        eris_ao, C_alph[:, o_ia], C_alph[:, v_ia], C_beta[:, o_ib], C_beta[:, v_ib]
    )  # (aa|bb) = <ab|ab>. Indices must be OVOV in chemists notation.

    # Now we have three tensors of dimensions
    # oa, va, oa, va (aa|aa)
    # ob, vb, ob, vb (bb|bb)
    # oa, va, ob, vb (aa|bb)
    if verbose:
        print(f"Det a:\n {det_a}")
        print(f"Det b:\n {det_b}")
        print("Transformed orbital blocks:")
        print("(aa|aa) block shape: ", aa_mo_chem.shape)
        print("(bb|bb) block shape: ", bb_mo_chem.shape)
        print("(aa|bb) block shape: ", ab_mo_chem.shape)

    # <ij||kl> = <ij|kl> - <ij|lk>
    # <ij|ab> = (ia|jb)
    # <ij|kl> = (ia|jb) - (ja|ib)
    aa_mo_phys = aa_mo_chem.transpose(0, 2, 1, 3)  # oovv
    bb_mo_phys = bb_mo_chem.transpose(0, 2, 1, 3)  # oovv
    ab_mo_phys = ab_mo_chem.transpose(0, 2, 1, 3)  # oovv

    if verbose:
        print("Transformed orbital blocks:")
        print("<aa|aa> block shape: ", aa_mo_phys.shape)
        print("<bb|bb> block shape: ", bb_mo_phys.shape)
        print("<ab|ab> block shape: ", ab_mo_phys.shape)

    aa_asymm = aa_mo_phys - aa_mo_phys.transpose(0, 1, 3, 2)
    bb_asymm = bb_mo_phys - bb_mo_phys.transpose(0, 1, 3, 2)
    ab_asymm = ab_mo_phys

    # print(ab_mo_phys)
    # print('Transformed integrals shape (oovv): ', eris_mo_phys.shape)
    e_a = e_alph[o_ia]
    e_b = e_alph[o_ia]
    e_r = e_alph[v_ia]
    e_s = e_alph[v_ia]

    # dim = n_occ, n_occ, n_virt, n_virt
    # what this does is that it "stretches" the smaller size axes and then
    # builds the "4D cube" where each entry is the operation r + s - a- b
    aa_denom_abrs = (
        e_s[None, None, :, None]
        + e_r[None, None, None, :]
        - e_a[:, None, None, None]
        - e_b[None, :, None, None]
    )

    e_a = e_beta[o_ib]
    e_b = e_beta[o_ib]
    e_r = e_beta[v_ib]
    e_s = e_beta[v_ib]

    # dim = n_occ, n_occ, n_virt, n_virt
    # what this does is that it "stretches" the smaller size axes and then
    # builds the "4D cube" where each entry is the operation r + s - a- b
    bb_denom_abrs = (
        e_s[None, None, :, None]
        + e_r[None, None, None, :]
        - e_a[:, None, None, None]
        - e_b[None, :, None, None]
    )

    e_a = e_alph[o_ia]
    e_b = e_beta[o_ib]
    e_s = e_alph[v_ia]
    e_r = e_beta[v_ib]

    # dim = n_occ, n_occ, n_virt, n_virt
    # what this does is that it "stretches" the smaller size axes and then
    # builds the "4D cube" where each entry is the operation r + s - a- b
    ab_denom_abrs = (
        e_s[None, None, :, None]
        + e_r[None, None, None, :]
        - e_a[:, None, None, None]
        - e_b[None, :, None, None]
    )

    if verbose:
        print("aaaa denominator shape: ", aa_denom_abrs.shape)
        print("bbbb denominator shape: ", bb_denom_abrs.shape)
        print("abab denominator shape: ", ab_denom_abrs.shape)

    t2_aa_aa = aa_asymm / aa_denom_abrs
    t2_bb_bb = bb_asymm / bb_denom_abrs
    t2_ab_ab = ab_asymm / ab_denom_abrs

    if verbose:
        print("t2 amplitudes shape aa-aa: ", t2_aa_aa.shape)
        print("t2 amplitudes shape bb-bb: ", t2_bb_bb.shape)
        print("t2 amplitudes shape ab-ab: ", t2_ab_ab.shape)

    aa_mp2 = -np.sum(aa_asymm * t2_aa_aa) * 0.25
    bb_mp2 = -np.sum(bb_asymm * t2_bb_bb) * 0.25
    ab_mp2 = -np.sum(ab_mo_phys * t2_ab_ab)

    E_corr = aa_mp2 + bb_mp2 + ab_mp2

    E_MP2 = CS_UHF_Context.E_UHF - E_corr

    returnClass = CS_MP2_Results(CS_UHF_Context, E_MP2, E_corr, mp_type, None, None)

    return returnClass


def ao_to_ovov(
    C_munu: NDArray[np.complex128], eris_ao: NDArray[np.complex128], o: slice, v: slice
) -> NDArray[np.complex128]:
    # (ia|jb) = L_mi L_nj (mn|ls) R_la R_sb
    tmp = np.einsum("mP, mnls -> Pnls", C_munu[:, o], eris_ao)
    tmp = np.einsum("lR, PQls -> PQRs", C_munu[:, o], tmp)
    tmp = np.einsum("nQ, Pnls -> PQls", C_munu[:, v], tmp)
    eris_mo_ovov = np.einsum("sS, PQRs -> PQRS", C_munu[:, v], tmp)
    return eris_mo_ovov


def ao_to_ovov_generalized(
    eris_ao: NDArray[np.complex128],
    c1: NDArray[np.complex128],
    c2: NDArray[np.complex128],
    c3: NDArray[np.complex128],
    c4: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    # (ia|jb) =  L_mi  L_nj (mn|ls)  R_la  R_sb
    # (ia|jb) = c1_mi c2_nj (mn|ls) c3_la c4_sb

    tmp = np.einsum("mP, mnls -> Pnls", c1, eris_ao)
    tmp = np.einsum("nQ, Pnls -> PQls", c2, tmp)
    tmp = np.einsum("lR, PQls -> PQRs", c3, tmp)
    eris_mo_ovov = np.einsum("sS, PQRs -> PQRS", c4, tmp)

    return eris_mo_ovov
