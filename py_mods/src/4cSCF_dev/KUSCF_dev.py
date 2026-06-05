from typing import Optional, Tuple, Union, List, Literal

import numpy as np
from numpy.typing import NDArray

from py_mods.src.integrals.UncontractedBasisSet import (
    UncontractedBasisSet,
    ERIs_Uncontracted,
)


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
