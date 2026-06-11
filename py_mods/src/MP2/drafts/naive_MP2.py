import numpy as np
from numpy.typing import NDArray
from typing import Literal, Union
from dataclasses import dataclass
from py_mods.src.SCF.CSRHF import CSRHFResults
from py_mods.src.SCF.CSUHF import CSUHFResults


@dataclass
class CS_MP2_Results(object):
    CS_MP2Context: Union[CSRHFResults, CSUHFResults]
    E_MP2: np.complex128
    E_corr: np.complex128
    MP_type: Literal["RMP2", "UMP2"]
    eris_mo: NDArray[np.complex128]


def CS_MP2(
    CS_MP2Context: Union[CSRHFResults, CSUHFResults],
) -> CS_MP2_Results:
    """Compute the MP2 energy correction using complex scaled UHF or RHF reference.

    Parameters
    ----------
    CS_RHF_Context: Union[CSRHFResults, CSUHFResults]
        Dataclass containing converged CS-RHF results.

    Returns
    -------
    returnClass: CS_MP2_Results
        Dataclass containing the MP2 energy correction.
    """
    if isinstance(CS_MP2Context, CSRHFResults):
        mp2_result = CS_MP2_RHF(CS_MP2Context)
    # elif isinstance(CS_MP2Context, CS_UHF_ResultsClass):
    #     mp2_result = CS_MP2_UHF(CS_MP2Context)
    else:
        raise TypeError(
            f"CS_MP2Context must be either CSRHFResults or CSUHFResults. Type is {type(CS_MP2Context)}"
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
    mp_type: Literal["RMP2", "UMP2"] = "RMP2"

    # naive approach: no symm
    C_munu = CS_RHF_Context.C_munu

    if np.isclose(CS_RHF_Context.context.theta, 0.0):
        C_munu = np.array(C_munu.real, dtype=np.complex128)

    # rest of info
    e_orb = CS_RHF_Context.e_orb
    eris_ao = CS_RHF_Context.context.eri.real
    n_occ: int = int(CS_RHF_Context.n_elec // 2)
    n_tot: int = len(CS_RHF_Context.context.S)
    n_virt: int = n_tot - n_occ

    # print(f'Number of occupied orbitals: {n_occ}')
    # print(f'Number of virtual orbitals: {n_virt}')
    # print(f'Number of total orbitals: {n_tot}')

    eris_mo = ao_to_mo(C_munu, eris_ao)

    mp2_ener = np.complex128(0.0)

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

                    abrs = eris_mo[a, r, b, s]  # o,v,o,v
                    rsba = eris_mo[r, b, s, a]  # v,o,v,o
                    rsab = eris_mo[r, a, s, b]  # v,o,v,o

                    # for RHF: num = <ab|rs> [2 <rs|ab> - <rs|ba>]
                    # for UHF num = (<ab||rs>)**2

                    num = abrs * (2 * rsab - rsba)
                    denom = er + es - ea - eb
                    mp2_ener -= num / denom  # 1/4 * num / denom

    E_corr = mp2_ener

    mp2_ener = E_corr + CS_RHF_Context.E_RHF

    returnClass = CS_MP2_Results(CS_RHF_Context, mp2_ener, E_corr, mp_type, eris_mo)

    return returnClass


def ao_to_mo(C_munu, eris_ao):

    # (pq|rs) = L_mP L_nQ (mn|ls) R_lR R_sS
    tmp = np.einsum("mP, mnls -> Pnls", C_munu, eris_ao)
    tmp = np.einsum("nQ, Pnls -> PQls", C_munu, tmp)
    tmp = np.einsum("lR, PQls -> PQRs", C_munu, tmp)
    eris_mo_2 = np.einsum("sS, PQRs -> PQRS", C_munu, tmp)
    return eris_mo_2
