import numpy as np
from numpy.typing import NDArray
from typing import Literal, Union
from dataclasses import dataclass
from py_mods.src.SCF.CSRHF import CS_RHF_ResultsClass
from py_mods.src.SCF.CSUHF import CS_UHF_ResultsClass
from py_mods.src.SCF.plot_utilities import plot_map
from time import time

@dataclass
class CS_MP2_Results(object):
    CS_MP2Context: Union[CS_RHF_ResultsClass, CS_UHF_ResultsClass]
    E_MP2: np.complex128
    E_corr: np.complex128
    MP_type: Literal['RMP2', 'UMP2']
    eris_mo: NDArray[np.complex128]


def CS_MP2(CS_MP2Context: Union[CS_RHF_ResultsClass, CS_UHF_ResultsClass]) -> CS_MP2_Results:
    """Compute the MP2 energy correction using complex scaled UHF or RHF reference.

    Parameters
    ----------
    CS_RHF_Context: Union[CS_RHF_ResultsClass, CS_UHF_ResultsClass]
        Dataclass containing converged CS-RHF results.

    Returns
    -------
    returnClass: CS_MP2_Results
        Dataclass containing the MP2 energy correction.
    """
    if isinstance(CS_MP2Context, CS_RHF_ResultsClass):
        mp2_result = CS_MP2_RHF(CS_MP2Context)
    elif isinstance(CS_MP2Context, CS_UHF_ResultsClass):
        mp2_result = CS_MP2_UHF(CS_MP2Context)
    else:
        raise TypeError(f"CS_MP2Context must be either CS_RHF_ResultsClass or CS_UHF_ResultsClass. Type is {type(CS_MP2Context)}")

    return mp2_result


def CS_MP2_RHF(CS_RHF_Context: CS_RHF_ResultsClass) -> CS_MP2_Results:
    """Compute the MP2 energy correction using complex scaled RHF reference.

    Parameters
    ----------
    CS_RHF_Context: CS_RHF_ResultsClass
        Dataclass containing converged CS-RHF results.

    Returns
    -------
    returnClass: CS_MP2_Results
        Dataclass containing the MP2 energy correction.
    """
    mp_type = 'RMP2'

    # naive approach: no symm
    R_munu = CS_RHF_Context.R_munu

    if np.isclose(CS_RHF_Context.context.theta, 0.0):
        R_munu = R_munu.real

    #rest of info
    e_orb = CS_RHF_Context.e_orb
    eris_ao = CS_RHF_Context.context.eri
    n_occ : int  = int(CS_RHF_Context.n_elec // 2 )
    n_tot : int  = len(CS_RHF_Context.context.S)
    n_virt : int = n_tot - n_occ

    # # we get the occupied and virtual indices:
    # o = slice(0, n_occ)
    # v = slice(n_occ, None)

    o_i = np.array([i for i,j in enumerate(CS_RHF_Context.det) if j == 2])
    v_i = np.array([i for i,j in enumerate(CS_RHF_Context.det) if j == 0])

    # print(f'Number of occupied orbitals: {n_occ} ({o})')
    # print(f'Number of virtual orbitals: {n_virt} ({v})')
    # print(f'Number of total orbitals: {n_tot} ')

    # more appropriate approach, calculate only (ovov) integrals
    t_start = time()
    eris_mo_chem = ao_to_ovov(R_munu, eris_ao, o_i, v_i, n_occ)
    t_end = time()
    # print(f'Time taken for (ovov) AO to MO integral transformation: {t_end - t_start}')

    # <ij||kl> = <ij|kl> - <ij|lk>
    # <ij|ab> = (ia|jb)
    # <ij|kl> = (ia|jb) - (ja|ib)
    eris_mo_phys = eris_mo_chem.transpose(0,2,1,3) # oovv

    # print('Transformed integrals shape (oovv): ', eris_mo_phys.shape)

    mp2_ener = 0.

    e_a = e_orb[o_i]
    e_b = e_orb[o_i]
    e_r = e_orb[v_i]
    e_s = e_orb[v_i]


    # dim = n_occ, n_occ, n_virt, n_virt
    # what this does is that it "stretches" the smaller size axes and then
    # builds the "4D cube" where each entry is the operation r + s - a- b
    denom_abrs =  (
        e_s[None, None, :, None]
      + e_r[None, None, None, :]  
      - e_a[:, None, None, None] 
      - e_b[None, :, None, None]
    )


    # for RHF: num = <ab|rs> [2 <rs|ab> - <rs|ba>]
    # for UHF num = (<ab||rs>)**2
    num = eris_mo_phys * ( 2 * eris_mo_phys - eris_mo_phys.transpose(0,1,3,2) )

    E_corr = -np.sum(num/denom_abrs)

    E_MP2 = E_corr + CS_RHF_Context.E_RHF 

    returnClass = CS_MP2_Results(CS_RHF_Context, E_MP2, E_corr, mp_type, eris_mo_chem)

    return returnClass

def CS_MP2_UHF(CS_UHF_Context: CS_UHF_ResultsClass) -> CS_MP2_Results:
    print('Not implemented yet')

    mp_type = 'RMP2'

    # naive approach: no symm
    R_alph = CS_UHF_Context.R_alpha
    R_beta = CS_UHF_Context.R_beta 

    if np.isclose(CS_UHF_Context.context.theta, 0.0):
        R_alph = R_alph.real
        R_beta = R_beta.real

    n_spatorb = len(R_alph)
    n_spinorb = n_spatorb * 2

    C_ab = np.zeros([n_spinorb, n_spinorb])

    C_ab[:n_spatorb, :n_spatorb] = R_alph
    C_ab[n_spatorb:, n_spatorb:] = R_beta

    plot_map(C_ab)

    #rest of info
    e_alph = CS_UHF_Context.e_alpha
    e_beta = CS_UHF_Context.e_beta
    print('e_alpha:')
    print(e_alph)
    print('e_beta:')
    print(e_beta)
    e_orb = np.concatenate((e_alph, e_beta), 0)
    print(e_orb)
    eris_ao = CS_UHF_Context.context.eri
    alpha_occ : int  = CS_UHF_Context.n_alpha 
    beta_occ : int   = CS_UHF_Context.n_beta
    n_occ = alpha_occ + beta_occ
    det_a = CS_UHF_Context.det[0]
    det_b = CS_UHF_Context.det[1]
    n_virt : int = n_spinorb - n_occ

    # we get the occupied and virtual indices:
    oa = slice(0, alpha_occ)
    ob = slice(n_spatorb, n_spatorb + beta_occ)
    va = slice(n_occ, n_spatorb)
    vb = slice(n_spatorb + beta_occ, None)

    o_ia = np.array([i for i,j in enumerate(CS_UHF_Context.det[0]) if j == 1])
    v_ia = np.array([i for i,j in enumerate(CS_UHF_Context.det[0]) if j == 0])
    o_ib = np.array([i for i,j in enumerate(CS_UHF_Context.det[1]) if j == 1])
    v_ib = np.array([i for i,j in enumerate(CS_UHF_Context.det[1]) if j == 0])

    o_i = np.concatenate((o_ia, o_ib + n_spatorb))
    v_i = np.concatenate((v_ia, v_ib + n_spatorb))

    # alpha-alpha ERI MO block
    aa_mo_chem = ao_to_ovov_generalized(
        eris_ao, 
        R_alph[:, o_ia], 
        R_alph[:, v_ia], 
        R_alph[:, o_ia], 
        R_alph[:, v_ia]
    ) # (aa|aa) = <aa|aa>. Indices must be OVOV in chemists notation. 

    bb_mo_chem = ao_to_ovov_generalized(
        eris_ao,
        R_beta[:, o_ib],
        R_beta[:, v_ib],
        R_beta[:, o_ib],
        R_beta[:, v_ib]
    ) # (bb|bb)

    ab_mo_chem = ao_to_ovov_generalized(
        eris_ao,
        R_alph[:, o_ia],
        R_alph[:, v_ia],
        R_beta[:, o_ib],
        R_beta[:, v_ib]
    ) # (aa|bb) = <ab|ab>

    # Now we have three tensors of dimensions
    # oa, va, oa, va (aa|aa)
    # ob, vb, ob, vb (bb|bb)
    # oa, va, ob, vb (aa|bb)

    print(f'Det a:\n {det_a}')
    print(f'Det b:\n {det_b}')
    print('Transformed orbital blocks:')
    print('(aa|aa) block shape: ', aa_mo_chem.shape)
    print('(bb|bb) block shape: ', bb_mo_chem.shape)
    print('(aa|bb) block shape: ', ab_mo_chem.shape)

    # <ij||kl> = <ij|kl> - <ij|lk>
    # <ij|ab> = (ia|jb)
    # <ij|kl> = (ia|jb) - (ja|ib)
    aa_mo_phys = aa_mo_chem.transpose(0,2,1,3) # oovv
    bb_mo_phys = bb_mo_chem.transpose(0,2,1,3) # oovv
    ab_mo_phys = ab_mo_chem.transpose(0,2,1,3) # oovv

    # print('Transformed integrals shape (oovv): ', eris_mo_phys.shape)

    mp2_ener = 0.

    e_a = e_orb[o_i]
    e_b = e_orb[o_i]
    e_r = e_orb[v_i]
    e_s = e_orb[v_i]


    # dim = n_occ, n_occ, n_virt, n_virt
    # what this does is that it "stretches" the smaller size axes and then
    # builds the "4D cube" where each entry is the operation r + s - a- b
    denom_abrs =  (
        e_s[None, None, :, None]
      + e_r[None, None, None, :]  
      - e_a[:, None, None, None] 
      - e_b[None, :, None, None]
    )

    print('Denominator shape: ', denom_abrs.shape)


    # for RHF: num = <ab|rs> [2 <rs|ab> - <rs|ba>]
    # for UHF num = (<ab||rs>)**2
    num = eris_mo_phys * ( 2 * eris_mo_phys - eris_mo_phys.transpose(0,1,3,2) )

    E_corr = -np.sum(num/denom_abrs)

    E_MP2 = E_corr + CS_RHF_Context.E_RHF 

    returnClass = CS_MP2_Results(CS_RHF_Context, E_MP2, E_corr, mp_type, eris_mo_chem)

    return returnClass
















def ao_to_ovov(C_munu, eris_ao, o: slice, v: slice):
    # (ia|jb) = L_mi L_nj (mn|ls) R_la R_sb
    tmp = np.einsum("mP, mnls -> Pnls", C_munu[:, o], eris_ao)
    tmp = np.einsum("lR, PQls -> PQRs", C_munu[:, o], tmp)
    tmp = np.einsum("nQ, Pnls -> PQls", C_munu[:, v], tmp)
    eris_mo_ovov = np.einsum("sS, PQRs -> PQRS", C_munu[:, v], tmp)
    return eris_mo_ovov

def ao_to_ovov_generalized(eris_ao, c1, c2, c3, c4):
    # (ia|jb) =  L_mi  L_nj (mn|ls)  R_la  R_sb
    # (ia|jb) = c1_mi c2_nj (mn|ls) c3_la c4_sb

    tmp = np.einsum("mP, mnls -> Pnls", c1, eris_ao)
    tmp = np.einsum("nQ, Pnls -> PQls", c2, tmp)
    tmp = np.einsum("lR, PQls -> PQRs", c3, tmp)
    eris_mo_ovov = np.einsum("sS, PQRs -> PQRS", c4, tmp)
    
    return eris_mo_ovov