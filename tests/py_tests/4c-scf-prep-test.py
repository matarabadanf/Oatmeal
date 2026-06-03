from pathlib import Path

import numpy as np
import scipy

from py_mods.src.SCF.linalg import transformation_matrix
from py_mods.src.external.DIRAC_ME import (
    build_4c_one_Fock_from_h5,
    build_S_V_W_T_from_h5,
)

data_path = Path(__file__).parent / "data"


def test_F0():
    checkpoint_files = [
        f"{data_path}/H_checkpoint.h5",
        f"{data_path}/He_checkpoint.h5",
        f"{data_path}/Ne_checkpoint.h5",
    ]

    eigenvalue_files = [
        f"{data_path}/H_F_eigvals_1st_iter.dat",
        f"{data_path}/He_F_eigvals_1st_iter.dat",
        f"{data_path}/Ne_F_eigvals_1st_iter.dat",
    ]

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
