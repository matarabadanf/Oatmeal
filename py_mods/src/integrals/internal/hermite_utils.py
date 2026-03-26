import numpy as np
from numpy.typing import NDArray
from py_mods.src.integrals.internal.special_functions import boys_hypergeom


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


def E(
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
            i * E(Ax, a, i - 1, Bx, b, j, t - 1, dim)
            + j * E(Ax, a, i, Bx, b, j - 1, t - 1, dim)
        ) / (2.0 * p * t)
    if t == 0 and i > 0:
        return X_pa * E(Ax, a, i - 1, Bx, b, j, 0, dim) + E(
            Ax, a, i - 1, Bx, b, j, 1, dim
        )
    if t == 0 and j > 0:
        return X_pb * E(Ax, a, i, Bx, b, j - 1, 0, dim) + E(
            Ax, a, i, Bx, b, j - 1, 1, dim
        )
    else:
        return 0.0
