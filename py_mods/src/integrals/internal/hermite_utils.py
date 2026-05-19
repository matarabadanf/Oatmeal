from typing import Union
import numpy as np
import math
from numpy.typing import NDArray
from py_mods.src.integrals.internal.special_functions import boys_hypergeom
from py_mods.src.integrals.internal.array_utils import IDX3DC


def R_tuv_n(
    p: float,
    R_pc: NDArray[np.float64],
    t_max: int,
    u_max: int,
    v_max: int,
    k_hyper: int,
) -> NDArray[np.float64]:
    """
    Calculate R(t,u,v,n) auxiliary integrals for electron repulsion.

    Parameters
    ----------
    p : float
        Gaussian exponent product
    R_pc : NDArray[np.float64], shape (3,)
        Vector between Gaussian center P and nuclear center C
    t_max : int
        Maximum t index
    u_max : int
        Maximum u index
    v_max : int
        Maximum v index
    k_hyper : int
        Value of k in the series definition of the Boys function

    Returns
    -------
    NDArray[np.float64]
        Array of R(t,u,v,n) integrals with shape (t_max+1, u_max+1, v_max+1, n_max)
        where n_max = t_max + u_max + v_max
    """
    R_2: float = R_pc[0] ** 2 + R_pc[1] ** 2 + R_pc[2] ** 2
    X_pc, Y_pc, Z_pc = R_pc

    n_max = t_max + u_max + v_max + 1
    R_tuv_n_array = np.zeros([t_max + 1, u_max + 1, v_max + 1, n_max])

    # Initialize t,u,v = 0 terms
    for n in range(0, n_max):
        R_tuv_n_array[0, 0, 0, n] = (-2 * p) ** n * boys_hypergeom(n, p * R_2, k_hyper)

    # Recursion in t
    for t in range(0, t_max):
        for n in range(n_max - 1):
            component_1 = t * R_tuv_n_array[t - 1, 0, 0, n + 1] if t >= 1 else 0
            component_2 = X_pc * R_tuv_n_array[t, 0, 0, n + 1]
            R_tuv_n_array[t + 1, 0, 0, n] = component_1 + component_2

    # Recursion in u
    for u in range(u_max):
        for t in range(0, t_max + 1):
            for n in range(n_max - 1):
                component_1 = u * R_tuv_n_array[t, u - 1, 0, n + 1] if u >= 1 else 0
                component_2 = Y_pc * R_tuv_n_array[t, u, 0, n + 1]
                R_tuv_n_array[t, u + 1, 0, n] = component_1 + component_2

    # Recursion in v
    for v in range(v_max):
        for u in range(u_max + 1):
            for t in range(0, t_max + 1):
                for n in range(n_max - 1):
                    component_1 = v * R_tuv_n_array[t, u, v - 1, n + 1] if v >= 1 else 0
                    component_2 = Z_pc * R_tuv_n_array[t, u, v, n + 1]
                    R_tuv_n_array[t, u, v + 1, n] = component_1 + component_2

    return R_tuv_n_array


def E_bottoms_up(
    A: NDArray[np.float64],
    a: float,
    i_max: int,
    B: NDArray[np.float64],
    b: float,
    j_max: int,
    t_max: int,
    out: Union[None, NDArray] = None,
) -> NDArray[np.float64]:
    """
    Bottoms-up calculation of Hermite expansion coefficients E_{ij}^t. In the
    same way OS relations can be built upwards instead of recursive calling,
    the same principle is applied here: build the zero cases and then
    recursively build the whole E^{AB}_{t} Hermite polynomial coefficients
    tensor.

    Parameters
    ----------
    A: NDArray[float], of size (3,)
        Cartesian coordinates of centre A.
    a: float
        Exponent of Gaussian centered at A.
    i_max: int
        Maximum angular momentum projection of A.
    B: NDArray[float], of size (3,)
        Cartesian coordinates of centre B.
    b: float
        Exponent of Gaussian centered at b.
    i_max: int
        Maximum angular momentum projection of B.
    t_max: int
        Maximum value t of the Hermite polynomial. t <= i + j

    Returns
    -------
    E_ab_t_array: NDArray[float] of size (i_max, j_max, t)
        Complete Hermite polynomial coefficient tensor
    """

    E_ab_t_array = np.zeros((i_max + 1, j_max + 1, t_max + 1), dtype=np.float64)

    X_ab = B - A

    p = a + b
    mu = (a * b) / p
    X_pa = b / p * X_ab
    X_pb = -a / p * X_ab

    # Base case i = j = t = 0 (Helgaker 9.5.8)
    E_ab_t_array[0, 0, 0] = np.exp(-mu * X_ab**2)

    # Compute the [i,0,0] entries (Helgaker 9.5.20)
    for i in range(1, i_max + 1):
        E_ab_t_array[i, 0, 0] += X_pa * E_ab_t_array[i - 1, 0, 0]
        E_ab_t_array[i, 0, 0] += 1 / (2 * p) * ((i - 1) * E_ab_t_array[i - 2, 0, 0])

    # Compute the [0,j,0] entries (Helgaker 9.5.21)
    for j in range(1, j_max + 1):
        E_ab_t_array[0, j, 0] += X_pb * E_ab_t_array[0, j - 1, 0]
        E_ab_t_array[0, j, 0] += 1 / (2 * p) * ((j - 1) * E_ab_t_array[0, j - 2, 0])

    # Compute the [i,0,t] entries (Helgaker 9.5.18)
    for i in range(1, i_max + 1):
        for t in range(1, min(i + 1, t_max + 1)):
            E_ab_t_array[i, 0, t] = (
                (2 * p) ** (-t) * math.comb(i, t) * E_ab_t_array[i - t, 0, 0]
            )

    # Compute the [0,j,t] entries (Helgaker 9.5.19)
    for j in range(1, j_max + 1):
        for t in range(1, min(j + 1, t_max + 1)):
            E_ab_t_array[0, j, t] = (
                (2 * p) ** (-t) * math.comb(j, t) * E_ab_t_array[0, j - t, 0]
            )

    # Compute the [i, j, 0] entries (Helgaker 9.5.20)
    for i in range(1, i_max + 1):
        for j in range(1, j_max + 1):
            E_ab_t_array[i, j, 0] = X_pa * E_ab_t_array[i - 1, j, 0]
            E_ab_t_array[i, j, 0] += (1 / (2 * p)) * (
                (i - 1) * E_ab_t_array[i - 2, j, 0]
                + (j) * E_ab_t_array[i - 1, j - 1, 0]
            )

    # Compute the rest entries (Helgaker 9.5.17)
    for i in range(1, i_max + 1):
        for j in range(1, j_max + 1):
            for t in range(1, t_max + 1):
                E_ab_t_array[i, j, t] = (
                    1
                    / (2 * p * t)
                    * (
                        i * E_ab_t_array[i - 1, j, t - 1]
                        + j * E_ab_t_array[i, j - 1, t - 1]
                    )
                )

    return E_ab_t_array


def E_bottoms_up_flat(
    A: NDArray[np.float64],
    a: float,
    i_max: int,
    B: NDArray[np.float64],
    b: float,
    j_max: int,
    t_max: int,
    out: Union[None, NDArray] = None,
) -> NDArray[np.float64]:
    """
    Bottoms-up calculation of Hermite expansion coefficients E_{ij}^t. In the
    same way OS relations can be built upwards instead of recursive calling,
    the same principle is applied here: build the zero cases and then
    recursively build the whole E^{AB}_{t} Hermite polynomial coefficients
    tensor.

    The difference with the legacy one is that this operates in flat dimensions,
    and can allow passing a buffer array to avoid memory allocation.

    Parameters
    ----------
    A: NDArray[float], of size (3,)
        Cartesian coordinates of centre A.
    a: float
        Exponent of Gaussian centered at A.
    i_max: int
        Maximum angular momentum projection of A.
    B: NDArray[float], of size (3,)
        Cartesian coordinates of centre B.
    b: float
        Exponent of Gaussian centered at b.
    i_max: int
        Maximum angular momentum projection of B.
    t_max: int
        Maximum value t of the Hermite polynomial. t <= i + j

    Returns
    -------
    E_ab_t_array: NDArray[float] of size (i_max, j_max, t)
        Complete Hermite polynomial coefficient tensor
    """
    if out is not None:
        E_ab_t_array = out
        E_ab_t_array[:] = 0
    else:
        E_ab_t_array = np.zeros(
            [(i_max + 1) * (j_max + 1) * (t_max + 1)], dtype=np.float64
        )

    s_i = i_max + 1
    s_j = j_max + 1
    s_t = t_max + 1

    X_ab = B - A

    p = a + b
    mu = (a * b) / p
    X_pa = b / p * X_ab
    X_pb = -a / p * X_ab

    # Base case i = j = t = 0 (Helgaker 9.5.8)
    E_ab_t_array[IDX3DC(0, 0, 0, s_i, s_j, s_t)] = np.exp(-mu * X_ab**2)

    # Compute the [i,0,0] entries (Helgaker 9.5.20)
    for i in range(1, i_max + 1):
        E_ab_t_array[IDX3DC(i, 0, 0, s_i, s_j, s_t)] += (
            X_pa * E_ab_t_array[IDX3DC(i - 1, 0, 0, s_i, s_j, s_t)]
        )
        E_ab_t_array[IDX3DC(i, 0, 0, s_i, s_j, s_t)] += (
            1 / (2 * p) * ((i - 1) * E_ab_t_array[IDX3DC(i - 2, 0, 0, s_i, s_j, s_t)])
        )

    # Compute the [0,j,0] entries (Helgaker 9.5.21)
    for j in range(1, j_max + 1):
        E_ab_t_array[IDX3DC(0, j, 0, s_i, s_j, s_t)] += (
            X_pb * E_ab_t_array[IDX3DC(0, j - 1, 0, s_i, s_j, s_t)]
        )
        E_ab_t_array[IDX3DC(0, j, 0, s_i, s_j, s_t)] += (
            1 / (2 * p) * ((j - 1) * E_ab_t_array[IDX3DC(0, j - 2, 0, s_i, s_j, s_t)])
        )

    # Compute the [i,0,t] entries (Helgaker 9.5.18)
    for i in range(1, i_max + 1):
        for t in range(1, min(i + 1, t_max + 1)):
            E_ab_t_array[IDX3DC(i, 0, t, s_i, s_j, s_t)] = (
                (2 * p) ** (-t)
                * math.comb(i, t)
                * E_ab_t_array[IDX3DC(i - t, 0, 0, s_i, s_j, s_t)]
            )

    # Compute the [0,j,t] entries (Helgaker 9.5.19)
    for j in range(1, j_max + 1):
        for t in range(1, min(j + 1, t_max + 1)):
            E_ab_t_array[IDX3DC(0, j, t, s_i, s_j, s_t)] = (
                (2 * p) ** (-t)
                * math.comb(j, t)
                * E_ab_t_array[IDX3DC(0, j - t, 0, s_i, s_j, s_t)]
            )

    # Compute the [i, j, 0] entries (Helgaker 9.5.20)
    for i in range(1, i_max + 1):
        for j in range(1, j_max + 1):
            E_ab_t_array[IDX3DC(i, j, 0, s_i, s_j, s_t)] = (
                X_pa * E_ab_t_array[IDX3DC(i - 1, j, 0, s_i, s_j, s_t)]
            )
            E_ab_t_array[IDX3DC(i, j, 0, s_i, s_j, s_t)] += (1 / (2 * p)) * (
                (i - 1) * E_ab_t_array[IDX3DC(i - 2, j, 0, s_i, s_j, s_t)]
                + (j) * E_ab_t_array[IDX3DC(i - 1, j - 1, 0, s_i, s_j, s_t)]
            )

    # Compute the rest entries (Helgaker 9.5.17)
    for i in range(1, i_max + 1):
        for j in range(1, j_max + 1):
            for t in range(1, t_max + 1):
                E_ab_t_array[IDX3DC(i, j, t, s_i, s_j, s_t)] = (
                    1
                    / (2 * p * t)
                    * (
                        i * E_ab_t_array[IDX3DC(i - 1, j, t - 1, s_i, s_j, s_t)]
                        + j * E_ab_t_array[IDX3DC(i, j - 1, t - 1, s_i, s_j, s_t)]
                    )
                )

    return E_ab_t_array[: s_i * s_j * s_t].reshape(s_i, s_j, s_t)


# =============================================================================
# LEGACY CODE
# =============================================================================


def _E_legacy(
    Ax: NDArray[np.float64],
    a: float,
    i: int,
    Bx: NDArray[np.float64],
    b: float,
    j: int,
    t: int,
    dim: int,
) -> float:
    """
    Calculate Hermite expansion coefficients for Gaussian overlap.

    Parameters
    ----------
    basis_1 : Primitive
        First Gaussian primitive
    i : int
        Angular momentum index for basis_1
    basis_2 : Primitive
        Second Gaussian primitive
    j : int
        Angular momentum index for basis_2
    t : int
        Expansion order
    dim : int
        Spatial dimension (0,1,2 for x,y,z)

    Returns
    -------
    float
        Hermite expansion coefficient E_{ij}^t
    """
    A_coord = Ax[dim]
    B_coord = Bx[dim]

    X_ab = B_coord - A_coord

    p = a + b
    mu = (a * b) / p
    X_pa = b / p * X_ab
    X_pb = -a / p * X_ab

    # Base cases
    if i < 0 or j < 0 or t < 0 or t > (i + j):
        return 0
    elif i == 0 and j == 0 and t == 0:
        return np.exp(-mu * X_ab**2)

    # Recursive cases
    if t > 0:
        return (
            i * _E_legacy(Ax, a, i - 1, Bx, b, j, t - 1, dim)
            + j * _E_legacy(Ax, a, i, Bx, b, j - 1, t - 1, dim)
        ) / (2.0 * p * t)
    if t == 0 and i > 0:
        return X_pa * _E_legacy(Ax, a, i - 1, Bx, b, j, 0, dim) + _E_legacy(
            Ax, a, i - 1, Bx, b, j, 1, dim
        )
    if t == 0 and j > 0:
        return X_pb * _E_legacy(Ax, a, i, Bx, b, j - 1, 0, dim) + _E_legacy(
            Ax, a, i, Bx, b, j - 1, 1, dim
        )
    else:
        return 0.0
