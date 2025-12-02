from re import S
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Union
from dataclasses import dataclass
from py_mods.src.SCF.CSRHF import CS_RHF_ResultsClass
from py_mods.src.SCF.CSUHF import CS_UHF_ResultsClass
from py_mods.src.SCF.plot_utilities import plot_map

@dataclass
class CS_MP2_Results(object):
    CS_MP2Context: Union[CS_RHF_ResultsClass, CS_UHF_ResultsClass]
    E_MP2: np.complex128
    E_corr: np.complex128
    MP_type: Literal['RMP2', 'UMP2']


def CS_MP2(CS_MP2Context: Union[CS_RHF_ResultsClass, CS_UHF_ResultsClass], eris_mo) -> CS_MP2_Results:
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
        mp2_result = CS_MP2_RHF(CS_MP2Context, eris_mo)
    # elif isinstance(CS_MP2Context, CS_UHF_ResultsClass):
    #     mp2_result = CS_MP2_UHF(CS_MP2Context)
    else:
        raise TypeError(f"CS_MP2Context must be either CS_RHF_ResultsClass or CS_UHF_ResultsClass. Type is {type(CS_MP2Context)}")

    return mp2_result


def CS_MP2_RHF(CS_RHF_Context: CS_RHF_ResultsClass, eris_mo) -> CS_MP2_Results:
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
    R_munu = CS_RHF_Context.R_munu.real
    L_munu = CS_RHF_Context.L_munu.real
    # L_munu = R_munu.T # use this definition to test the non-scaled case 

    #rest of info
    e_orb = CS_RHF_Context.e_orb
    eris_ao = CS_RHF_Context.context.eri.real
    n_occ : int  = int(CS_RHF_Context.n_elec // 2 )
    n_tot : int  = len(CS_RHF_Context.context.S)
    n_virt : int = n_tot - n_occ

    print(f'Number of occupied orbitals: {n_occ}')
    print(f'Number of virtual orbitals: {n_virt}')
    print(f'Number of total orbitals: {n_tot}')

    eris_mo = eris_mo.reshape([n_tot, n_tot,n_tot,n_tot])
    eris_mo_2 = ao_to_mo(R_munu, eris_ao)
    eris_mo_3 = ao_to_mo_biorthogonal(L_munu.T, R_munu, eris_ao)

    eris_mo = eris_mo

    print(f'Maximum difference between building MO eris with C and L.T, R: {np.max(eris_mo_2-eris_mo_3)}')


    # print(eris_mo[0,0,:,:]-eris_mo_2[0,0,:,:])

    # lets build the T2 amplitudes ourselves

    t2 = np.zeros([n_occ,n_occ,n_virt,n_virt])

    # t_ab^rs = - <rs||ab>/ er+es-ea-eb = - [<rs|ab>-<rs|ba>] / ener

    for a in range(n_occ):
        ea = e_orb[a]
        for b in range(n_occ):
            eb = e_orb[b]
            # virtual 
            for r in range(n_virt):
                er = e_orb[r + n_occ]
                for s in range(n_occ, n_virt):
                    es = e_orb[s + n_occ]

                    rsab = eris_mo[r,a,s,b]
                    rsba = eris_mo[r,b,s,a]

                    # t2[a,b,r,s] -= (rsab-rsba) / (er + es - ea - eb)
                    t2[a,b,r,s] -= (rsab) / (er + es - ea - eb)

    plot_map(t2[0,0,:,:])
    # plot_map(t2_2[0,0,:,:])    
    # plot_map(t2[0,1,:,:])
    # plot_map(t2_2[0,1,:,:])
    # plot_map(t2[1,0,:,:])
    # plot_map(t2[1,1,:,:])

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

                    # <ij||kl> = <ij|kl> - <ij|lk>
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

def ao_to_mo(C_munu, eris_ao):

    # (pq|rs) = L_mP L_nQ (mn|ls) R_lR R_sS
    tmp = np.einsum("mP, mnls -> Pnls", C_munu, eris_ao)
    tmp = np.einsum("nQ, Pnls -> PQls", C_munu, tmp)
    tmp = np.einsum("lR, PQls -> PQRs", C_munu, tmp)
    eris_mo_2 = np.einsum("sS, PQRs -> PQRS", C_munu, tmp)
    return eris_mo_2

def ao_to_mo_biorthogonal(L_munu, R_munu, eris_ao):

    # (pq|rs) = L_mP L_nQ (mn|ls) R_lR R_sS
    tmp = np.einsum("mP, mnls -> Pnls", L_munu, eris_ao)
    tmp = np.einsum("nQ, Pnls -> PQls", L_munu, tmp)
    tmp = np.einsum("lR, PQls -> PQRs", R_munu, tmp)
    eris_mo_2 = np.einsum("sS, PQRs -> PQRS", R_munu, tmp)
    return eris_mo_2