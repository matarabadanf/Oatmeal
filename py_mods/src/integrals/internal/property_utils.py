from scipy.stats import f
from ST_utils import obara_saika_bottom_up, S_1D
from typing import Literal
import numpy as np
from numpy.typing import NDArray


def dipole_integral(
    i: int,
    j: int,
    k: int,
    Ax: NDArray[np.float64],
    a: float,
    l: int,
    m: int,
    n: int,
    Bx: NDArray[np.float64],
    b: float,
    dimension: Literal[0, 1, 2],
):
    """
    Compute dipole integral
    """

    if dimension not in [0, 1, 2]:
        raise ValueError(
            "Only dimensions x (0), y (1) or z (2) can be used for dipolar moment calculation"
        )

    ax, ay, az = Ax
    bx, by, bz = Bx

    match dimension:
        case 0:
            d_ij = S_1D(ax, ax, a, b, i + 1, l) + ax * S_1D(ax, bx, a, b, i, l)
            s_component_1 = S_1D(ay, by, a, b, j, m)
            s_component_2 = S_1D(az, bz, a, b, k, n)

        case 1:
            d_ij = S_1D(ay, ay, a, b, j + 1, m) + ay * S_1D(ay, by, a, b, j, m)
            s_component_1 = S_1D(ax, bx, a, b, i, l)
            s_component_2 = S_1D(az, bz, a, b, k, n)

        case 2:
            d_ij = S_1D(az, az, a, b, k + 1, n) + az * S_1D(az, bz, a, b, k, n)
            s_component_1 = S_1D(ax, bx, a, b, i, l)
            s_component_2 = S_1D(ay, by, a, b, j, m)

    return d_ij * s_component_1 * s_component_2


def OS_multipole_tensor(Ax, Bx, Cx, a, b, i, j, max_e):
    """
    Compute multipole integral using OS recurrence relations
    """

    max_dim = max(i, j) + 1

    if i == j:
        max_dim += 1

    d_ij_e_mat = np.zeros([max_dim, max_dim, max_e + 1])

    X_ab = Bx - Ax
    X_ac = Cx - Ax

    p = a + b
    X_pa = b / p * X_ab
    X_pb = -a / p * X_ab
    X_pc = -a / p * X_ac

    # Base case for ground floor: i and j layer, e = 0
    d_ij_e_mat[:, :, 0] = obara_saika_bottom_up(Ax, Bx, a, b, i, j)

    # compute the [0,0,e] entries
    for e in range(1, max_e):
        d_ij_e_mat[0, 0, e] += X_pc * d_ij_e_mat[0, 0, e - 1]
        d_ij_e_mat[0, 0, e] += 1 / (2 * p) * ((e - 1) * d_ij_e_mat[0, 0, e - 2])

    return d_ij_e_mat


def s_i_j_eplus(i, j, d_ij_e_mat, p, X_pc, e):
    """
    Compute s_i_j_eplus term for OS recurrence relations
    """

    return X_pc * d_ij_e_mat[i, j, e - 1] + 1 / (2 * p) * (
        (i) * d_ij_e_mat[i - 1, j, e - 1]
        + (j) * d_ij_e_mat[i, j - 1, e - 1]
        + (e - 1) * d_ij_e_mat[i, j, e - 2]
    )


def s_i_jplus_e(j, d_ij_e_mat, p, X_pb, e, total):

    if j < 1 or j > d_ij_e_mat.shape[1] or e < 0 or e > d_ij_e_mat.shape[2]:
        return 0
    return X_pb * d_ij_e_mat[total, j - 1, e] + 1 / (2 * p) * (
        (total) * d_ij_e_mat[total - 1, j - 1, e]
        + (j - 1) * d_ij_e_mat[total, j - 2, e]
        + (e) * d_ij_e_mat[total, j - 1, e - 1]
    )


def s_iplus_j_e(i, d_ij_e_mat, p, X_pa, e, total):
    return X_pa * d_ij_e_mat[i - 1, total, e] + 1 / (2 * p) * (
        (i - 1) * d_ij_e_mat[i - 2, total, e]
        + (total) * d_ij_e_mat[i - 1, total - 1, e]
        + (e) * d_ij_e_mat[i - 1, total, e - 1]
    )
