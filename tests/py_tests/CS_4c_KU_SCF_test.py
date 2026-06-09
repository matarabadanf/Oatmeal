import numpy as np
from pathlib import Path
import pytest

from py_mods.src.SCF_4c_dev.types_4c import CS_4c_KU_SCF_Context
from py_mods.src.external.DIRAC_ME import (
    build_S_V_W_T_from_h5,
    full_eri_from_h5,
    build_uncontracted_basis_from_h5,
)
from py_mods.src.SCF_4c_dev.KUSCF_dev import (
    occupation_4c,
    eri_classified,
    _kuscf_kernel,
)

notebook_path = Path(__file__).parent.parent.parent / "notebooks" / "4c-scf" / "data"


def test_CS_4c_KU_SCF_kernel_He():

    h5_filename = f"{notebook_path}/He_checkpoint.h5"

    S, V, W, T = build_S_V_W_T_from_h5(h5_filename)

    _, nL, nS = build_uncontracted_basis_from_h5(h5_filename)
    eri = full_eri_from_h5(h5_filename)
    eri = eri_classified(eri, nL)

    occ_det = occupation_4c(nS, nL, 2)

    test_ctx = CS_4c_KU_SCF_Context(
        nL,
        nS,
        S,
        T,
        V,
        W,
        eri,
        2,
        theta=0.00,
        occ=occ_det,
        verbose=False,
        threshold=1e-10,
    )

    results = _kuscf_kernel(test_ctx)

    assert results.converged == True, "The 4c SCF Kernel failed to converge."

    expected_energy = -2.8612850992180254
    assert np.isclose(
        results.E_SCF.real, expected_energy, atol=1e-8
    ), f"Energy mismatch: expected {expected_energy}, got {results.E_SCF.real}"

@pytest.mark.massive
def test_CS_4c_KU_SCF_kernel_Ne():

    h5_filename = f"{notebook_path}/Ne_checkpoint.h5"

    S, V, W, T = build_S_V_W_T_from_h5(h5_filename)

    _, nL, nS = build_uncontracted_basis_from_h5(h5_filename)
    eri = full_eri_from_h5(h5_filename)
    eri = eri_classified(eri, nL)

    occ_det = occupation_4c(nS, nL, 10)

    test_ctx = CS_4c_KU_SCF_Context(
        nL,
        nS,
        S,
        T,
        V,
        W,
        eri,
        10,
        theta=0.00,
        occ=occ_det,
        verbose=False,
        threshold=1e-10,
    )

    results = _kuscf_kernel(test_ctx)

    assert results.converged == True, "The 4c SCF Kernel failed to converge for Ne."

    expected_energy = -128.68579576537914
    assert np.isclose(
        results.E_SCF.real, expected_energy, atol=1e-8
    ), f"Energy mismatch: expected {expected_energy}, got {results.E_SCF.real}"
