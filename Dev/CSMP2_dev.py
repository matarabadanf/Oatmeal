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
    L_munu = R_munu.T # use this definition to test the non-scaled case 

    #rest of info
    e_orb = CS_RHF_Context.e_orb
    eris_ao = CS_RHF_Context.context.eri
    n_occ = CS_RHF_Context.n_elec // 2 
    n_tot = len(CS_RHF_Context.context.S)
    n_virt = n_tot - n_occ

    print(f'Number of occupied orbitals: {n_occ}')
    print(f'Number of virtual orbitals: {n_virt}')
    print(f'Number of total orbitals: {n_tot}')

    # (ab|cd) = L_ma L_nb (mn|ls) R_lc R_sd
    tmp = np.einsum("ma, mnls -> anls", L_munu, eris_ao)
    tmp = np.einsum("nb, anls -> abls", L_munu, tmp)
    tmp = np.einsum("lc, abls -> abcs", R_munu, tmp)
    eris_mo = np.einsum("sd, abcs -> abcd", R_munu, tmp).real # real for now 

    print('Transformed integrals')

    # print('\n\nRandom check of symmetry of integrals in MO basis:')

    # for i in range(5):
    #     i = random.randint(0, n_tot-1)
    #     j = random.randint(0, n_tot-1)
    #     k = random.randint(0, n_tot-1)
    #     l = random.randint(0, n_tot-1)

    #     print(f'[{i:2d},{j:2d},{k:2d},{l:2d}] == [{i:2d},{j:2d},{l:2d},{k:2d}]: {eris_mo[i,j,k,l]-eris_mo[j,i,k,l] < 1E-10}')

    mp2_ener = 0.

    # occupied are a,b virtual are r,s 

    # occupied
    for a in range(n_occ):
        ea = e_orb[a]
        for b in range(n_occ):
            eb = e_orb[b]
            # virtual 
            for r in range(n_occ, n_tot):
                er = e_orb[r]
                for s in range(n_occ, n_tot):
                    es = e_orb[s]

                    # <ij||kl> = <ij|ab> - <ji|ab>
                    # <ij|ab> = (ia|jb)
                    # <ij|kl> = (ia|jb) - (ja|ib)

                    abrs = eris_mo[a,r,b,s] # o,v,o,v
                    rsba = eris_mo[r,b,s,a] # v,o,v,o
                    rsab = eris_mo[r,a,s,b] # v,o,v,o

                    # for RHF: num = <ab|rs> [2 <rs|ab> - <rs|ba>]
                    # for UHF num = (<ab||rs>)**2

                    num = abrs * ( 2 * rsab - rsba)
                    denom = (er + es - ea - eb).real 
                    mp2_ener -= num / denom # 1/4 * num / denom

                    # as in szabo a,b are virtual and r and s occupied
                    # i,j will be virtual and k,l will be occupied in our loop a, b occ, r, s virt
                    # i,j = r,s
                    # k,l = a,b
                    # ei, ej, ek, el = er, es, ea, eb

                    # # # the abrs is klij -> kilj; rsab -> ijkl; rsba -> ijlk
                    # abrs = eris_mo[k,i,l,j]                
                    # rsab = eris_mo[i,k,j,l]
                    # rsba = eris_mo[i,l,j,k]
                    # # num = (abrs - absr)**2
                    # # num =  abrs * rsab - abrs * rsba
                    # denom = (ei + ej - ek - el).real 
                    # mp2_ener -= .25 * num / denom # 1/4 * num / denom

    E_corr = mp2_ener 

    mp2_ener = E_corr + CS_RHF_Context.E_RHF 

    returnClass = CS_MP2_Results(CS_RHF_Context, mp2_ener, E_corr, mp_type)

    # to see later, the exact use of slices in this https://pycrawfordprogproj.readthedocs.io/en/latest/Project_04/Project_04.html

    return returnClass