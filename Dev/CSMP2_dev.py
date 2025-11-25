from re import S
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, Union
from dataclasses import dataclass
from Dev.CSRHF_dev import CS_RHF_ResultsClass
from py_mods.src.SCF.CSUHF import CS_UHF_ResultsClass
import random

@dataclass
class CS_MP2_Results(object):
    CS_MP2Context: Union[CS_RHF_ResultsClass, CS_UHF_ResultsClass]
    E_MP2: np.complex128
    E_corr: np.complex128
    MP_type: Literal['RMP2', 'UMP2']


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
    # elif isinstance(CS_MP2Context, CS_UHF_ResultsClass):
    #     mp2_result = CS_MP2_UHF(CS_MP2Context)
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
    L_munu = CS_RHF_Context.L_munu
    # L_munu = R_munu.conj().T # use this definition to test the non-scaled case 

    #rest of info
    e_orb = CS_RHF_Context.e_orb
    eris_ao = CS_RHF_Context.context.eri
    n_occ = CS_RHF_Context.n_elec // 2 
    n_tot = len(CS_RHF_Context.context.S)
    n_virt = n_tot - n_occ

    # (ab|cd) = L_am L_bn (mn|ls) R_cl R_ds

    # get integrals in MO basis
    tmp = np.einsum("ma, mnls -> anls", L_munu, eris_ao)
    tmp = np.einsum("nb, anls -> abls", L_munu, tmp)
    tmp = np.einsum("lc, abls -> abcs", R_munu, tmp)
    eris_mo = np.einsum("sd, abcs -> abcd", R_munu, tmp)

    print('Transformed integrals')

    for i in range(10):
        i = random.randint(0, n_tot-1)
        j = random.randint(0, n_tot-1)
        k = random.randint(0, n_tot-1)
        l = random.randint(0, n_tot-1)

        print(f'[{i:2d},{j:2d},{k:2d},{l:2d}]: {eris_mo[i,j,k,l]:16.4E},     [{j:2d},{i:2d},{k:2d},{l:2d}]:{eris_mo[j,i,k,l]:8.4E},        {eris_mo[i,j,k,l]-eris_mo[j,i,k,l]:8.4E}')

    mp2_ener = 0 

    # occupied 
    for i in range(n_occ):
        ei = e_orb[i]
        for j in range(n_occ):
            ej = e_orb[j]

            # virtual 
            for a in range(n_occ, n_tot):
                ea = e_orb[a]
                for b in range(n_occ, n_tot):
                    eb = e_orb[b]

                    # ij||kl = ij|ab - ji|ab
                    # ij|ab = (ia|jb)
                    # ij|kl = (ia|jb) - (ja|ib)

                    iajb = eris_mo[i,a,j,b]
                    jaib = eris_mo[j,a,i,b]

                    antisym = (iajb - jaib)**2

                    denom = ei + ej - ea - eb

                    mp2_ener += 1/4 * antisym / denom

    E_corr = mp2_ener 

    mp2_ener = CS_RHF_Context.E_RHF - E_corr

    returnClass = CS_MP2_Results(CS_RHF_Context, mp2_ener, E_corr, mp_type)

    return returnClass