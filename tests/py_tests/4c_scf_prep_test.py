from pathlib import Path

import numpy as np
import scipy

from py_mods.src.SCF.linalg import transformation_matrix
from py_mods.src.external.DIRAC_ME import (
    build_4c_one_Fock_from_h5,
    build_S_V_W_T_from_h5,
)

from py_mods.src.external.DIRAC_ME import (
    build_S_V_W_T_from_h5,
    generate_primitive_KUSCFContext_from_h5,
)

from py_mods.src.SCF_4c_dev.scf_4c_kernels import (
    scf_steps,
)

data_path = Path(__file__).parent / "data"
notebook_path = (
    Path(__file__).parent.parent.parent / "notebooks" / "4c-scf" / "data"
)


checkpoint_files = [
    f"{data_path}/H_checkpoint.h5",
    f"{data_path}/He_checkpoint.h5",
    f"{data_path}/Ne_checkpoint.h5",
]

eigenvalue_files = [
    f"{notebook_path}/H_F_eigvals_1st_iter.dat",
    f"{notebook_path}/He_F_eigvals_1st_iter.dat",
    f"{notebook_path}/Ne_F_eigvals_1st_iter.dat",
]


def test_F0():
    """
    Test the first iteration eigenvalues of against reference eigenvalues.
    """
    for efile, checkpoint_file in zip(eigenvalue_files, checkpoint_files):
        F_0 = build_4c_one_Fock_from_h5(checkpoint_file)
        S, *_ = build_S_V_W_T_from_h5(checkpoint_file)

        X = transformation_matrix(S)

        F_p = X.T @ F_0 @ X

        e, w = scipy.linalg.eigh(F_p)

        idx = np.argsort(e)
        e = e[idx]
        w = w[:, idx]

        e = [val for i, val in enumerate(e) if i % 2 == 0]

        ref_eigvals = np.loadtxt(efile)

        assert np.allclose(ref_eigvals, e)


def test_He_SCF():
    h5_filename = f"{notebook_path}/He_checkpoint.h5"
    He_ctx = generate_primitive_KUSCFContext_from_h5(h5_filename)

    H_core = He_ctx.T + He_ctx.V + He_ctx.W

    X = transformation_matrix(He_ctx.S)

    scf_energies = scf_steps(
        15, H_core, He_ctx.eri_classess, X, He_ctx.occ
    )
    scf_reference_energies = np.loadtxt(f"{notebook_path}/He_scf_energy.dat")

    print(f"SCF energies at each iteration: {scf_energies}")
    print(
        f"Difference at each iteration of SCF energies: {scf_energies[:len(scf_reference_energies)] - scf_reference_energies}"
    )
