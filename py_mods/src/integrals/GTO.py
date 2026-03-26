import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Tuple
from py_mods.src.integrals.internal.ST_utils import kinetic_energy_integrals
from py_mods.src.integrals.internal.hermite_utils import R_tuv_n, E
from py_mods.src.integrals.internal.ST_utils import S_1D


@dataclass
class GTO:
    """
    Gaussian primitive basis function.

    Attributes
    ----------
    R : NDArray[np.float64]
        Center coordinates (3D vector)
    exp : float
        Gaussian exponent
    total_L : int
        Total angular momentum quantum number
    l_projections : NDArray[np.int32]
        Array of angular momentum projections (n_projections x 3)
    normalization_constants : NDArray[np.float64]
        Normalization constants for each projection
    charge : float
        Charge associated with the primitive

    Notes
    -----
    - The l_projections array contains all valid (l_x, l_y, l_z) combinations
      such that l_x + l_y + l_z = total_L.
    - The normalization_constants array contains the normalization factors
      corresponding to each angular momentum projection.
    """

    R: NDArray[np.float64]
    exp: float
    total_L: int
    l_projections: NDArray[np.int32]  # of dimensions (n_projections, 3)
    normalization_constants: NDArray[np.float64]
    charge: float = 1


def create_GTO(R: NDArray[np.float64], exp: float, total_L: int) -> GTO:
    """
    Factory function to create a GTO object with computed angular momentum projections
    and normalization constants.

    Parameters

    ----------
    R : NDArray[np.float64]
        Center coordinates (3D vector)
    exp : float
        Gaussian exponent
    total_L : int
        Total angular momentum quantum number

    Returns
    -------
    GTO
        GTO object
    """
    # Generate all angular momentum projections for total_L
    l_projections = generate_angular_momentum_projections(total_L)

    normalization_constants = np.zeros(len(l_projections)) + 1

    prim = GTO(
        R=R,
        exp=exp,
        total_L=total_L,
        l_projections=l_projections,
        normalization_constants=normalization_constants,
    )

    prim.normalization_constants = 1 / np.sqrt(self_overlap(Prim=prim))

    return prim


def generate_angular_momentum_projections(total_L):
    l_projections = []
    for l_x in range(total_L, -1, -1):
        for l_y in range(total_L - l_x, -1, -1):
            l_z = total_L - l_x - l_y
            l_projections.append([l_x, l_y, l_z])
    return np.array(l_projections, dtype=np.int32)


def self_overlap(Prim: GTO) -> NDArray[np.float64]:
    """
    Compute the self-overlap of a GTO function.

    Parameters
    ----------
    Prim : GTO
        GTO function

    Returns
    -------
    NDArray[np.float64]
        Self-overlap values for each angular momentum projection
    """
    overlaps = np.zeros(len(Prim.l_projections))
    for i, proj in enumerate(Prim.l_projections):
        overlaps[i] = S_3D(
            Prim,
            proj,
            Prim.normalization_constants[i],
            Prim,
            proj,
            Prim.normalization_constants[i],
        )

    return overlaps


def normalize_GTO(Prim: GTO) -> None:
    """
    Normalize the primitive basis function in place by updating its normalization constants.

    Parameters
    ----------
    Prim : GTO
        The primitive basis function to be normalized. Its normalization_constants attribute
        will be updated in place.
    """
    for i, projection in enumerate(Prim.l_projections):
        N = 1 / np.sqrt(
            S_3D(
                Prim,
                projection,
                Prim.normalization_constants[i],
                Prim,
                projection,
                Prim.normalization_constants[i],
            )
        )
        Prim.normalization_constants[i] = N


def create_normalized_GTO(
    R: NDArray[np.float64], exp: float, total_L: int, charge: float = 1
) -> GTO:
    """
    Factory function to create a GTO object with computed angular momentum projections
    and normalization constants.

    Parameters

    ----------
    R : NDArray[np.float64]
        Center coordinates (3D vector)
    exp : float
        Gaussian exponent
    total_L : int
        Total angular momentum quantum number

    Returns
    -------
    GTO
        GTO object
    """
    # Generate all angular momentum projections for total_L
    l_projections = []
    for l_x in range(total_L, -1, -1):
        for l_y in range(total_L - l_x, -1, -1):
            l_z = total_L - l_x - l_y
            l_projections.append([l_x, l_y, l_z])

    l_projections = np.array(l_projections, dtype=np.int32)

    # Compute normalization constants for each projection
    normalization_constants = np.zeros(len(l_projections)) + 1

    prim = GTO(
        R=R,
        exp=exp,
        total_L=total_L,
        l_projections=l_projections,
        normalization_constants=normalization_constants,
        charge=charge,
    )

    normalize_GTO(prim)

    return prim


# --- Hermite coefficients product for 3D overlap ---
def E_ab(
    basis_1: GTO,
    projection_1: Tuple[int, int, int],
    basis_2: GTO,
    projection_2: Tuple[int, int, int],
    t: int,
    u: int,
    v: int,
) -> float:
    """
    Calculate product of Hermite coefficients for 3D Gaussian overlap.

    Parameters
    ----------
    basis_1 : GTO
        First Gaussian primitive
    projection_1 : Tuple[int, int, int]
        Angular momentum indices (i,k,m) for basis_1
    basis_2 : GTO
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

    E_1 = E(basis_1.R, basis_1.exp, i, basis_2.R, basis_2.exp, j, t, 0)
    E_2 = E(basis_1.R, basis_1.exp, k, basis_2.R, basis_2.exp, l, u, 1)
    E_3 = E(basis_1.R, basis_1.exp, m, basis_2.R, basis_2.exp, n, v, 2)

    return E_1 * E_2 * E_3


# --- Overlap 3D with primitives ---


def S_3D(
    basis_1: GTO,
    projection_1,
    N_A: float,
    basis_2: GTO,
    projection_2,
    N_B: float,
) -> float:
    """
    Calculate the product ofthe three overlap integral components.

    Parameters
    ------
    basis_1 : GTO
        First primitive; must provide attributes:
          - R : array-like of length 3, center coordinates (R_x, R_y, R_z)
          - exp : float, Gaussian exponent (alpha)
          - angular_momentum : int, total angular momentum (l)
    projection_1 : numpy.ndarray
        Length-3 integer array with projection quantum numbers for basis_1 (m_x, m_y, m_z)
    basis_2 : GTO
        Second primitive; same requirements as basis_1
    projection_2 : numpy.ndarray
        Length-3 integer array with projection quantum numbers for basis_2

    Returns
    ------
        float: The product of the three overlap components (S_ab, S_cd, S_ef).
    """
    S_ab, S_cd, S_ef = S_3D_components(basis_1, projection_1, basis_2, projection_2)

    return S_ab * S_cd * S_ef * N_A * N_B


# --- Kinetic with primitives ---
def T_1D(
    coord_1: float, coord_2: float, exp_1: float, exp_2: float, proj_1, proj_2
) -> float:
    t = kinetic_energy_integrals(
        coord_1,
        coord_2,
        exp_1,
        exp_2,
        proj_1,
        proj_2,
    )
    return t


def T_3D(prim_1, proj_1, N_a, prim_2, proj_2, N_b):
    # X-component
    T_x = T_1D(prim_1.R[0], prim_2.R[0], prim_1.exp, prim_2.exp, proj_1[0], proj_2[0])
    S_x = S_1D(prim_1.R[0], prim_2.R[0], prim_1.exp, prim_2.exp, proj_1[0], proj_2[0])

    # Y-component
    T_y = T_1D(prim_1.R[1], prim_2.R[1], prim_1.exp, prim_2.exp, proj_1[1], proj_2[1])
    S_y = S_1D(prim_1.R[1], prim_2.R[1], prim_1.exp, prim_2.exp, proj_1[1], proj_2[1])

    # Z-component
    T_z = T_1D(prim_1.R[2], prim_2.R[2], prim_1.exp, prim_2.exp, proj_1[2], proj_2[2])
    S_z = S_1D(prim_1.R[2], prim_2.R[2], prim_1.exp, prim_2.exp, proj_1[2], proj_2[2])

    return N_a * N_b * (T_x * S_y * S_z + S_x * T_y * S_z + S_x * S_y * T_z)


# --- Overlap 3D with primitives ---
def S_3D_components(
    basis_1: GTO,
    projection_1: np.ndarray,
    basis_2: GTO,
    projection_2: np.ndarray,
) -> np.ndarray:
    """
    Compute the three Cartesian components of the 3D overlap between two primitive functions.

    To ensure orthogonality if the scalar product is 0 (they dont share a
    component in the same projection) and the individual l is nonzero, the
    function returns [0,0,0].

    Parameters
    ----------
    basis_1 : GTO
        First primitive; must provide attributes:
          - R : array-like of length 3, center coordinates (R_x, R_y, R_z)
          - exp : float, Gaussian exponent (alpha)
          - angular_momentum : int, total angular momentum (l)
    projection_1 : numpy.ndarray
        Length-3 integer array with projection quantum numbers for basis_1 (m_x, m_y, m_z)
    basis_2 : GTO
        Second primitive; same requirements as basis_1
    projection_2 : numpy.ndarray
        Length-3 integer array with projection quantum numbers for basis_2

    Returns
    -------
    overlap_components : numpy.ndarray of shape (3,)
        1D numpy array of shape (3,) with the overlap components [S_x, S_y, S_z].
        If the projection vectors are orthogonal and both l1 and l2 are nonzero,
        returns numpy.array([0, 0, 0]).

    Notes
    -------
        - We will do it this way for now, since we have to test for d functions.
        It is true that the calculation of the three 1d overlaps might be redundant,
        but it must be checked out.
    """
    # If there is overlap calculate it
    R_a = basis_1.R
    R_b = basis_2.R

    alpha = basis_1.exp
    beta = basis_2.exp

    overlap_components = np.zeros(3)

    for comp, _ in enumerate(overlap_components):
        overlap_components[comp] = S_1D(
            R_a[comp], R_b[comp], alpha, beta, projection_1[comp], projection_2[comp]
        )

    return overlap_components


# --- Electron-nuclear attraction integral between two Gaussian basis functions ---
def h_ab_Z(
    basis_1,
    projection_1,
    basis_2,
    projection_2,
    charge_atom: float,
    coord_atom,
    k_hyper: int = 80,
) -> float:
    coord_atom = np.array(coord_atom).reshape(-1)

    a = basis_1.exp
    b = basis_2.exp

    r_A = basis_1.R
    r_B = basis_2.R

    i, k, m = projection_1
    j, l, n = projection_2

    t_max = i + j + 1
    u_max = k + l + 1
    v_max = m + n + 1

    p = a + b
    r_P = (a * r_A + b * r_B) / p

    h_ab_total: float = 0.0

    r_PC = r_P - coord_atom
    # print(r_PC)
    R_tuv_n_array = R_tuv_n(p, r_PC, t_max, u_max, v_max, k_hyper)
    charge = charge_atom

    for t in range(t_max + 1):
        for u in range(u_max + 1):
            for v in range(v_max + 1):

                coefficient = E_ab(
                    basis_1, projection_1, basis_2, projection_2, t, u, v
                )

                # print(coefficient)

                hermite_integral: float = R_tuv_n_array[t, u, v, 0]

                # print(f"{t}, {u}, {v}, {0}: {coefficient} {charge} {hermite_integral}")
                # print(f"{coefficient} ")

                h_ab_total += coefficient * hermite_integral

    return -(2 * np.pi / p) * charge_atom * h_ab_total


def V_3D(
    basis_1: GTO,
    projection_1: Tuple[int, int, int],
    N_a: float,
    basis_2: GTO,
    projection_2: Tuple[int, int, int],
    N_b: float,
    charge_atom: float,
    coord_atom: NDArray[np.float64],
    k_hyper: int = 80,
):

    V_unnorm = h_ab_Z(
        basis_1, projection_1, basis_2, projection_2, charge_atom, coord_atom, k_hyper
    )

    return N_a * N_b * V_unnorm


##--- Two-electron repulsion integral between four Gaussian basis functions ---
def g_abcd(
    basis_1: GTO,
    p1: Tuple[int, int, int],
    basis_2: GTO,
    p2: Tuple[int, int, int],
    basis_3: GTO,
    p3: Tuple[int, int, int],
    basis_4: GTO,
    p4: Tuple[int, int, int],
    k_hyper: int = 80,
) -> float:
    """
    Calculate two-electron repulsion integral between four Gaussian basis functions.

    Parameters
    ----------
    basis_1, basis_2, basis_3, basis_4 : GTO
        Gaussian primitive basis functions
    p1, p2, p3, p4 : Tuple[int, int, int]
        Angular momentum indices (i,k,m) for each basis function
    k_hyper : int, optional
        Order of Boys function expansion, by default 80

    Returns
    -------
    float
        Two-electron repulsion integral value (ab|cd)

    Notes
    -----
    Implements the two-electron repulsion integral using Hermite Gaussian
    functions following the McMurchie-Davidson scheme. The integral is computed
    as (ab|cd) where the notation indicates electron 1 between a,b and
    electron 2 between c,d.
    """

    a = basis_1.exp
    b = basis_2.exp
    c = basis_3.exp
    d = basis_4.exp

    r_A = basis_1.R
    r_B = basis_2.R
    r_C = basis_3.R
    r_D = basis_4.R

    i, k, m = p1
    j, l, n = p2
    ii, kk, mm = p3
    jj, ll, nn = p4

    t_max = i + j + 1
    u_max = k + l + 1
    v_max = m + n + 1

    tau_max = ii + jj + 1
    nu_max = kk + ll + 1
    phi_max = mm + nn + 1

    p = a + b
    r_P = (a * r_A + b * r_B) / p

    q = c + d
    r_Q = (c * r_C + d * r_D) / q

    r_PQ = r_P - r_Q

    alpha = p * q / (p + q)

    Hermite_integral = R_tuv_n(
        alpha, r_PQ, t_max + tau_max, u_max + nu_max, v_max + phi_max, k_hyper
    )

    g_abcd = 0

    for t in range(t_max):
        for u in range(u_max):
            for v in range(v_max):
                for tau in range(tau_max):
                    for nu in range(nu_max):
                        for phi in range(phi_max):
                            coefficient_1 = E_ab(basis_1, p1, basis_2, p2, t, u, v)
                            coefficient_2 = E_ab(basis_3, p3, basis_4, p4, tau, nu, phi)
                            integral = Hermite_integral[t + tau, u + nu, v + phi, 0]

                            sign = (-1.0) ** (tau + nu + phi)
                            g_abcd += coefficient_1 * coefficient_2 * integral * sign

    return 2 * np.power(np.pi, 2.5) / (p * q * np.sqrt(p + q)) * g_abcd


def eri(
    basis_1: GTO,
    p1: Tuple[int, int, int],
    N_a: float,
    basis_2: GTO,
    p2: Tuple[int, int, int],
    N_b: float,
    basis_3: GTO,
    p3: Tuple[int, int, int],
    N_c: float,
    basis_4: GTO,
    p4: Tuple[int, int, int],
    N_d: float,
    k_hyper: int = 80,
) -> float:

    return (
        N_a
        * N_b
        * N_c
        * N_d
        * g_abcd(
            basis_1,
            p1,
            basis_2,
            p2,
            basis_3,
            p3,
            basis_4,
            p4,
            k_hyper,
        )
    )
