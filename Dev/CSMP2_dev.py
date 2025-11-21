from re import S
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, Union
from dataclasses import dataclass
from Dev.CSRHF_dev import CS_RHF_ResultsClass
from py_mods.src.SCF.CSUHF import CS_UHF_ResultsClass

@dataclass
class CS_MP2_Results(object):
    CS_MP2Context: Union[CS_RHF_ResultsClass, CS_UHF_ResultsClass]
    E_MP2: np.complex128
    MP_type: Literal['RMP2', 'UMP2']


def CS_MP2(CS_MP2Context: Union[CS_RHF_ResultsClass, CS_UHF_ResultsClass]):
    """Compute the MP2 energy correction using complex scaled RHF or UHF reference.

    Args:
        CS_MP2Context (Union[CS_RHF_ResultsClass, CS_UHF_ResultsClass]): Context containing converged CS-RHF or CS-UHF results.

    Returns:
        CS_MP2_Results: Dataclass containing the MP2 energy correction.
    """
    if isinstance(CS_MP2Context, CS_RHF_ResultsClass):
        mp2_result = CS_MP2_RHF(CS_MP2Context)
    # elif isinstance(CS_MP2Context, CS_UHF_ResultsClass):
    #     mp2_result = CS_MP2_UHF(CS_MP2Context)
    else:
        raise TypeError(f"CS_MP2Context must be either CS_RHF_ResultsClass or CS_UHF_ResultsClass. Type is {type(CS_MP2Context)}")

    return mp2_result


def CS_MP2_RHF(CS_RHF_Context: CS_RHF_ResultsClass):
    """Compute the MP2 energy correction using complex scaled RHF reference.

    Args:
        CS_RHF_Context (CS_RHF_ResultsClass): Context containing converged CS-RHF results.

    Returns:
        CS_RHF_ResultsClass: Dataclass containing the MP2 energy correction.
    """
    mp_type = 'RMP2'

    # naive approach: no symm
    R_munu = CS_RHF_Context.R_munu
    L_munu = CS_RHF_Context.L_munu
    e_orb = CS_RHF_Context.e_orb
    eris_ao = CS_RHF_Context.context.eri
    n_occ = CS_RHF_Context.n_elec // 2 
    n_virt = len(CS_RHF_Context.context.S) - n_occ
    n_tot = len(CS_RHF_Context.context.S)

    # get integrals in MO basis
    tmp = np.einsum("mu, mnls -> unls", R_munu, eris_ao)
    tmp = np.einsum("nv, unls -> uvls", R_munu, tmp)
    tmp = np.einsum("lr, uvls -> uvrs", R_munu, tmp)
    eris_mo = np.einsum("ts, uvrs -> uvrt", R_munu, tmp)

    print('Transformed integrals')

    o = slice(0, n_occ)
    v = slice(n_occ, n_tot)

    # Done like this because of the possibility of L and R coefficients. 
    eri_ai_bj = eris_mo[o, v, o, v]
    eri_aj_bi = eris_mo[v, v, o, o]

    antisymm_ab_ij = eri_ai_bj - eri_aj_bi

    antisymm_sr = np.abs(antisymm_ab_ij)**2

    mp2_ener = 0 
    for i in range(n_occ):
        for j in range(n_occ):
            for a in range(n_virt):
                for b in range(n_virt):

                    denom = (
                        e_orb[i] + e_orb[j]
                        - e_orb[n_occ + a] - e_orb[n_occ + b]
                    )

                    if abs(denom) < 1e-12:
                        continue  

                    mp2_ener += 0.25 * antisymm_sr[i,j,a,b] / denom


    returnClass = CS_MP2_Results(CS_RHF_Context, mp2_ener, mp_type)

    return returnClass