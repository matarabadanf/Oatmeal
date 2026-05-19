import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Tuple, List
from py_mods.src.integrals.internal.ST_utils import (
    _kinetic_energy_integrals_legacy,
    obara_saika_bottom_up,
    T_ab_1d_matrix,
)
from py_mods.src.integrals.internal.hermite_utils import (
    R_tuv_n,
    _E_legacy,
    E_bottoms_up,
)

from py_mods.src.integrals.internal.ST_utils import _S_1D_legacy
from py_mods.src.integrals.internal.coulomb_utils import _h_ab_Z_legacy
from scipy.special import factorial2


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
    l_dim: int
    normalization_constants: NDArray[np.float64]
    charge: float = 1


# =============================================================================
# GTO construction and normalization
# =============================================================================


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
    l_projections = _generate_angular_momentum_projections(total_L)

    normalization_constants = np.zeros(len(l_projections)) + 1

    prim = GTO(
        R=R,
        exp=exp,
        total_L=total_L,
        l_projections=l_projections,
        normalization_constants=normalization_constants,
        l_dim=len(l_projections),
    )

    prim.normalization_constants = 1 / np.sqrt(_self_overlap(Prim=prim))

    return prim


def _generate_angular_momentum_projections(total_L: int) -> NDArray[np.int32]:
    """
    Generate all valid angular momentum projections for a given total angular momentum.

    Parameters
    ----------
    total_L : int
        Total angular momentum quantum number

    Returns
    -------
    NDArray[np.int32]
        Array of shape (N, 3) containing all valid (l_x, l_y, l_z) combinations
    """
    l_projections = []
    for l_x in range(total_L, -1, -1):
        for l_y in range(total_L - l_x, -1, -1):
            l_z = total_L - l_x - l_y
            l_projections.append([l_x, l_y, l_z])
    return np.array(l_projections, dtype=np.int32)


def _self_overlap(Prim: GTO) -> NDArray[np.float64]:
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
        overlaps[i] = _S_3D_legacy(
            Prim,
            proj,
            Prim.normalization_constants[i],
            Prim,
            proj,
            Prim.normalization_constants[i],
        )

    return overlaps


def _normalize_GTO(Prim: GTO, hermit_normalize: bool = False) -> None:
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
            _S_3D_legacy(
                Prim,
                projection,
                Prim.normalization_constants[i],
                Prim,
                projection,
                Prim.normalization_constants[i],
            )
        )
        Prim.normalization_constants[i] = N
        if hermit_normalize:
            Prim.normalization_constants[i] *= _hermit_norm_coefficient(*projection)


def create_normalized_GTO(
    R: NDArray[np.float64],
    exp: float,
    total_L: int,
    charge: float = 1,
    hermit_norm: bool = False,
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
    charge : float, optional
        Charge associated with the primitive, by default 1
    hermit_norm : bool, optional
        Whether to apply Hermite normalization to the GTO, by default False

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
        l_dim=len(l_projections),
    )

    _normalize_GTO(prim, hermit_normalize=hermit_norm)

    # if hermit_norm:
    #     hermit_normalize([prim])

    return prim


# --- Overlap 3D with primitives ---


def _S_3D_legacy(
    basis_1: GTO,
    projection_1: np.ndarray,
    N_A: float,
    basis_2: GTO,
    projection_2: np.ndarray,
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
    S_ab, S_cd, S_ef = _S_3D_components_legacy(
        basis_1, projection_1, basis_2, projection_2
    )

    return S_ab * S_cd * S_ef * N_A * N_B


# --- Overlap 3D with primitives ---
def _S_3D_components_legacy(
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
        overlap_components[comp] = _S_1D_legacy(
            R_a[comp], R_b[comp], alpha, beta, projection_1[comp], projection_2[comp]
        )

    return overlap_components


# =============================================================================
# GTO Shell-based matrix element computation
# =============================================================================


def S_ab_shell(gto1: GTO, gto2: GTO) -> NDArray[np.float64]:
    """
    Calculate the overlap integral matrix between two shell basis functions.

    Parameters
    ----------
    gto1 : GTO
        First Gaussian primitive shell
    gto2 : GTO
        Second Gaussian primitive shell

    Returns
    -------
    NDArray[np.float64]
        Overlap integral matrix for the given shells
    """
    ax, ay, az = gto1.R
    bx, by, bz = gto2.R

    a = gto1.exp
    b = gto2.exp

    i = gto1.total_L
    j = gto2.total_L

    s_ab_x = obara_saika_bottom_up(ax, bx, a, b, i, j)
    s_ab_y = obara_saika_bottom_up(ay, by, a, b, i, j)
    s_ab_z = obara_saika_bottom_up(az, bz, a, b, i, j)

    S_ab = np.zeros([gto1.l_dim, gto2.l_dim])

    for p1, proj_1 in enumerate(gto1.l_projections):
        for p2, proj_2 in enumerate(gto2.l_projections):
            i, k, m = proj_1
            j, l, n = proj_2
            S_ab[p1, p2] = s_ab_x[i, j] * s_ab_y[k, l] * s_ab_z[m, n]

    norm_tensor = np.einsum(
        "i,j->ij", gto1.normalization_constants, gto2.normalization_constants
    )
    S_ab *= norm_tensor

    return S_ab


def T_ab_shell(gto1: GTO, gto2: GTO) -> NDArray[np.float64]:

    ax, ay, az = gto1.R
    bx, by, bz = gto2.R

    a = gto1.exp
    b = gto2.exp

    i = gto1.total_L
    j = gto2.total_L

    T_ab_x = T_ab_1d_matrix(ax, bx, a, b, i, j)
    T_ab_y = T_ab_1d_matrix(ay, by, a, b, i, j)
    T_ab_z = T_ab_1d_matrix(az, bz, a, b, i, j)

    s_ab_x = obara_saika_bottom_up(ax, bx, a, b, i, j)
    s_ab_y = obara_saika_bottom_up(ay, by, a, b, i, j)
    s_ab_z = obara_saika_bottom_up(az, bz, a, b, i, j)

    T_ab = np.zeros([gto1.l_dim, gto2.l_dim])


    for p1, proj_1 in enumerate(gto1.l_projections):
        for p2, proj_2 in enumerate(gto2.l_projections):
            ix, ky, mz = proj_1
            jx, ly, nz = proj_2
            
            # print(i,j,k)
            # print(k,l,m)

            # print(repr(T_ab_x))

            T1 = T_ab_x[ix, jx] * s_ab_y[ky, ly] * s_ab_z[mz, nz]
            T2 = s_ab_x[ix, jx] * T_ab_y[ky, ly] * s_ab_z[mz, nz]
            T3 = s_ab_x[ix, jx] * s_ab_y[ky, ly] * T_ab_z[mz, nz]
            T_ab[p1, p2] = T1 + T2 + T3

    norm_tensor = np.einsum(
        "i,j->ij", gto1.normalization_constants, gto2.normalization_constants
    )
    T_ab *= norm_tensor

    return T_ab


def g_abcd_shell(
    gto1: GTO, gto2: GTO, gto3: GTO, gto4: GTO, k_hyper: int = 90
) -> NDArray[np.float64]:
    """
    Shell-based ERI tensor calculator. The idea here is that instead of
    working with projections as in the original g_abcd function, this
    function reuses intermediates. The E_AB coefficients are calculated
    once and then read from this, instead of recomputing them in the innermost
    loop.

    Recall that Gaussian overlap distributions over cartesian coordinates
    can be expanded as a LC of Hermite Gaussians with fixed Hermite polynomial
    coefficients E^{ab}_{t} which are much easier to manipulate. (Helgaker sec 9.5)

    Parameters
    ----------
    gto1, gto2, gto3, gto4 : GTO
        Gaussian primitive basis functions

    Returns
    -------
    NDArray[np.float64]
        ERI tensor for the given shells
    """

    a = gto1.exp
    b = gto2.exp
    c = gto3.exp
    d = gto4.exp

    r_A = gto1.R
    r_B = gto2.R
    r_C = gto3.R
    r_D = gto4.R

    p = a + b
    r_P = (a * r_A + b * r_B) / p

    q = c + d
    r_Q = (c * r_C + d * r_D) / q

    r_PQ = r_P - r_Q

    alpha = p * q / (p + q)

    # We compute over + 1 dimension due to the way the recurrent function works.
    t_max = u_max = v_max = gto1.total_L + gto2.total_L + 1
    tau_max = nu_max = phi_max = gto3.total_L + gto4.total_L + 1

    # Compute the hermite auxiliary integral
    Hermite_integral = R_tuv_n(
        alpha, r_PQ, t_max + tau_max, u_max + nu_max, v_max + phi_max, k_hyper
    )

    # Compute coefficients of A and B for each cartesian projection
    E_AB_x = E_bottoms_up(r_A[0], a, gto1.total_L, r_B[0], b, gto2.total_L, t_max)
    E_AB_y = E_bottoms_up(r_A[1], a, gto1.total_L, r_B[1], b, gto2.total_L, u_max)
    E_AB_z = E_bottoms_up(r_A[2], a, gto1.total_L, r_B[2], b, gto2.total_L, v_max)

    E_AB_full = np.zeros((gto1.l_dim, gto2.l_dim, t_max, u_max, v_max))

    # Since there are different projections, the hole AB tensor for each projection pair
    # is obtained by contracting over the angular momenta of each projection in each
    # cartesian coordinate.
    for p1, proj_1 in enumerate(gto1.l_projections):
        for p2, proj_2 in enumerate(gto2.l_projections):
            i, k, m = proj_1
            j, l, n = proj_2

            # This way, to each projection vector, there is associated an index p1 or p2.
            # Then t, u, v indicate the degree of the hermite polynomial t <= i+j...
            # By contracting this way, the individual angular momenta indices are removed,
            # and only the total Hermite coefficient between two projections with specific
            # t,u,v is stored.
            E_AB_full[p1, p2, :, :, :] = np.einsum(
                "t, u, v -> tuv",
                E_AB_x[i, j, :t_max],
                E_AB_y[k, l, :u_max],
                E_AB_z[m, n, :v_max],
            )

    # This is repeated for C and D
    E_CD_x = E_bottoms_up(r_C[0], c, gto3.total_L, r_D[0], d, gto4.total_L, tau_max)
    E_CD_y = E_bottoms_up(r_C[1], c, gto3.total_L, r_D[1], d, gto4.total_L, nu_max)
    E_CD_z = E_bottoms_up(r_C[2], c, gto3.total_L, r_D[2], d, gto4.total_L, phi_max)

    E_CD_full = np.zeros((gto3.l_dim, gto4.l_dim, tau_max, nu_max, phi_max))
    for p3, proj_3 in enumerate(gto3.l_projections):
        for p4, proj_4 in enumerate(gto4.l_projections):
            ii, kk, mm = proj_3
            jj, ll, nn = proj_4
            E_CD_full[p3, p4, :, :, :] = np.einsum(
                "t, u, v -> tuv",
                E_CD_x[ii, jj, :tau_max],
                E_CD_y[kk, ll, :nu_max],
                E_CD_z[mm, nn, :phi_max],
            )

    eri_block = np.zeros((gto1.l_dim, gto2.l_dim, gto3.l_dim, gto4.l_dim))

    # Now the eri tensor is defined as (Helgaker 9.9.33):
    # factor * sum_{t,u,v,tau,nu,phi} E^{AB}_{t,u,v} E^{CD}_{tau,nu,phi} * R_{..}(alpha, R_{PQ})
    for t in range(t_max):
        for u in range(u_max):
            for v in range(v_max):
                for tau in range(tau_max):
                    for nu in range(nu_max):
                        for phi in range(phi_max):
                            sign = (-1.0) ** (tau + nu + phi)
                            integral = (
                                Hermite_integral[t + tau, u + nu, v + phi, 0] * sign
                            )
                            eri_block += integral * np.einsum(
                                "ij,kl->ijkl",
                                E_AB_full[:, :, t, u, v],
                                E_CD_full[:, :, tau, nu, phi],
                            )

    # And the normalization constants are built as a tensor product of the normalization
    # Constant vectors
    norm_tensor = np.einsum(
        "i,j,k,l->ijkl",
        gto1.normalization_constants,
        gto2.normalization_constants,
        gto3.normalization_constants,
        gto4.normalization_constants,
    )

    # And normalization happens as a elementwise product
    eri_block *= norm_tensor

    # And the factor (Helgaker 9.9.33) is applied here
    return 2 * np.power(np.pi, 2.5) / (p * q * np.sqrt(p + q)) * eri_block


def _hermit_norm_coefficient(i: int, j: int, k: int) -> float:
    """
    Calculate the Hermite normalization coefficient for a specific projection.

    Parameters
    ----------
    i, j, k : int
        Angular momentum indices for x, y, z coordinates

    Returns
    -------
    float
        The computed normalization factor
    """
    Fa = factorial2(2 * i - 1) if i > 0 else 1
    Fb = factorial2(2 * j - 1) if j > 0 else 1
    Fc = factorial2(2 * k - 1) if k > 0 else 1
    return np.sqrt(Fa * Fb * Fc)


# =============================================================================
# LEGACY CODE
# =============================================================================


# --- Hermite coefficients product for 3D overlap ---
def _E_ab_legacy(
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

    E_1 = _E_legacy(basis_1.R, basis_1.exp, i, basis_2.R, basis_2.exp, j, t, 0)
    E_2 = _E_legacy(basis_1.R, basis_1.exp, k, basis_2.R, basis_2.exp, l, u, 1)
    E_3 = _E_legacy(basis_1.R, basis_1.exp, m, basis_2.R, basis_2.exp, n, v, 2)

    return E_1 * E_2 * E_3


# --- Kinetic with primitives ---
def _T_1D_legacy(
    coord_1: float, coord_2: float, exp_1: float, exp_2: float, proj_1: int, proj_2: int
) -> float:
    """
    Calculate 1D kinetic energy integral between primitive Gaussians.

    Parameters
    ----------
    coord_1, coord_2 : float
        Center coordinates
    exp_1, exp_2 : float
        Gaussian exponents
    proj_1, proj_2 : int
        Angular momentum projection for this dimension

    Returns
    -------
    float
        1D kinetic energy integral
    """
    t = _kinetic_energy_integrals_legacy(
        coord_1,
        coord_2,
        exp_1,
        exp_2,
        proj_1,
        proj_2,
    )
    return t


def _T_3D_legacy(
    prim_1: GTO,
    proj_1: np.ndarray,
    N_a: float,
    prim_2: GTO,
    proj_2: np.ndarray,
    N_b: float,
) -> float:
    """
    Calculate the 3D kinetic energy integral between two primitive GTO projections.

    Parameters
    ----------
    prim_1, prim_2 : GTO
        Gaussian primitive basis functions
    proj_1, proj_2 : np.ndarray
        Angular momentum projection vectors
    N_a, N_b : float
        Normalization constants

    Returns
    -------
    float
        3D kinetic energy integral
    """
    # X-component
    T_x = _T_1D_legacy(
        prim_1.R[0], prim_2.R[0], prim_1.exp, prim_2.exp, proj_1[0], proj_2[0]
    )
    S_x = _S_1D_legacy(
        prim_1.R[0], prim_2.R[0], prim_1.exp, prim_2.exp, proj_1[0], proj_2[0]
    )

    # Y-component
    T_y = _T_1D_legacy(
        prim_1.R[1], prim_2.R[1], prim_1.exp, prim_2.exp, proj_1[1], proj_2[1]
    )
    S_y = _S_1D_legacy(
        prim_1.R[1], prim_2.R[1], prim_1.exp, prim_2.exp, proj_1[1], proj_2[1]
    )

    # Z-component
    T_z = _T_1D_legacy(
        prim_1.R[2], prim_2.R[2], prim_1.exp, prim_2.exp, proj_1[2], proj_2[2]
    )
    S_z = _S_1D_legacy(
        prim_1.R[2], prim_2.R[2], prim_1.exp, prim_2.exp, proj_1[2], proj_2[2]
    )

    return N_a * N_b * (T_x * S_y * S_z + S_x * T_y * S_z + S_x * S_y * T_z)


def _V_3D_legacy(
    basis_1,
    projection_1,
    N_a,
    basis_2,
    projection_2,
    N_b,
    charge_atom,
    coord_atom,
    k_hyper: int = 80,
):

    V_unnorm = _h_ab_Z_legacy(
        basis_1, projection_1, basis_2, projection_2, charge_atom, coord_atom, k_hyper
    )

    return N_a * N_b * V_unnorm


##--- Two-electron repulsion integral between four Gaussian basis functions ---
def _g_abcd_legacy(
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
                            coefficient_1 = _E_ab_legacy(
                                basis_1, p1, basis_2, p2, t, u, v
                            )
                            coefficient_2 = _E_ab_legacy(
                                basis_3, p3, basis_4, p4, tau, nu, phi
                            )
                            integral = Hermite_integral[t + tau, u + nu, v + phi, 0]

                            sign = (-1.0) ** (tau + nu + phi)
                            g_abcd += coefficient_1 * coefficient_2 * integral * sign

    return 2 * np.power(np.pi, 2.5) / (p * q * np.sqrt(p + q)) * g_abcd


def _eri_legacy(
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
    """
    Calculate the two-electron repulsion integral with normalized primitives.

    Parameters
    ----------
    basis_1, basis_2, basis_3, basis_4 : GTO
        Gaussian primitive basis functions
    p1, p2, p3, p4 : Tuple[int, int, int]
        Angular momentum projection vectors
    N_a, N_b, N_c, N_d : float
        Normalization constants
    k_hyper : int, optional
        Order of Boys function expansion, by default 80

    Returns
    -------
    float
        Normalized ERI value
    """

    return (
        N_a
        * N_b
        * N_c
        * N_d
        * _g_abcd_legacy(
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


def _S_3D_uncontracted_GTO_list_legacy(
    gto_list: List[GTO],
) -> NDArray[np.float64]:
    """
    Compute the full overlap matrix for a list of uncontracted GTOs.

    Parameters
    ----------
    gto_list : List[GTO]
        List of GTO objects representing the basis functions.

    Returns
    -------
    NDArray[np.float64], shape(total_projections, total_projections)
        Overlap matrix.
    """
    l_projections = [len(gto.l_projections) for gto in gto_list]
    total_size = sum(l_projections)
    S_matrix = np.zeros((total_size, total_size))

    basis_start_index = [sum(l_projections[0:i]) for i in range(len(l_projections))]

    for mu_idx, mu in enumerate(gto_list):
        for nu_idx, nu in enumerate(gto_list):
            for mu_proj_idx, muproj in enumerate(mu.l_projections):
                for nu_proj_idx, nuproj in enumerate(nu.l_projections):
                    idx_mu_and_proj = basis_start_index[mu_idx] + mu_proj_idx
                    idx_nu_and_proj = basis_start_index[nu_idx] + nu_proj_idx

                    S_matrix[idx_mu_and_proj, idx_nu_and_proj] = _S_3D_legacy(
                        mu,
                        muproj,
                        mu.normalization_constants[mu_proj_idx],
                        nu,
                        nuproj,
                        nu.normalization_constants[nu_proj_idx],
                    )

    return S_matrix


if __name__ == "__main__":
    pass
