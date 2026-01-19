import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from py_mods.src.integrals.internal.special_functions import boys_hypergeom
from py_mods.src.integrals.primitive import Primitive


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
    R_2 = R_pc[0] ** 2 + R_pc[1] ** 2 + R_pc[2] ** 2
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
    basis_1: Primitive, i: int, basis_2: Primitive, j: int, t: int, dim: int
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
    Ax = basis_1.R[dim]
    Bx = basis_2.R[dim]
    a = basis_1.exp
    b = basis_2.exp

    X_ab = Bx - Ax
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
            i * E(basis_1, i - 1, basis_2, j, t - 1, dim)
            + j * E(basis_1, i, basis_2, j - 1, t - 1, dim)
        ) / (2.0 * p * t)
    if t == 0 and i > 0:
        return X_pa * E(basis_1, i - 1, basis_2, j, 0, dim) + (1.0 / (2.0 * p)) * E(
            basis_1, i - 1, basis_2, j, 1, dim
        )
    if t == 0 and j > 0:
        return X_pb * E(basis_1, i, basis_2, j - 1, 0, dim) + E(
            basis_1, i, basis_2, j - 1, 1, dim
        )


def E_ab(
    basis_1: Primitive,
    projection_1: Tuple[int, int, int],
    basis_2: Primitive,
    projection_2: Tuple[int, int, int],
    t: int,
    u: int,
    v: int,
) -> float:
    """
    Calculate product of Hermite coefficients for 3D Gaussian overlap.

    Parameters
    ----------
    basis_1 : Primitive
        First Gaussian primitive
    projection_1 : Tuple[int, int, int]
        Angular momentum indices (i,k,m) for basis_1
    basis_2 : Primitive
        Second Gaussian primitive
    projection_2 : Tuple[int, int, int]
        Angular momentum indices (j,l,n) for basis_2
    t : int
        x-component expansion order
    u : int
        y-component expansion order
    v : int
        z-component expansion order

    Returns
    -------
    float
        Product E_x * E_y * E_z of Hermite expansion coefficients
    """
    i, k, m = projection_1
    j, l, n = projection_2

    E_1 = E(basis_1, i, basis_2, j, t, 0)
    E_2 = E(basis_1, k, basis_2, l, u, 1)
    E_3 = E(basis_1, m, basis_2, n, v, 2)

    return E_1 * E_2 * E_3
