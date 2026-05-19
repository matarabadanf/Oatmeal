import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Tuple, Union
from py_mods.src.integrals.internal.hermite_utils import (
    R_tuv_n,
    E_bottoms_up,
    _E_legacy,
)


@dataclass
class _dummy_GTO:
    """
    NOT INTENDED FOR USE. For typing purposes.
    """

    R: NDArray[np.float64]
    exp: float
    total_L: int
    l_projections: NDArray[np.int32]  # of dimensions (n_projections, 3)
    l_dim: int
    normalization_constants: NDArray[np.float64]
    charge: float = 1


def V_ab_Z_shell(
    gto1,
    gto2,
    charge_atom: float,
    coord_atom,
    k_hyper: int = 90,
) -> NDArray[np.float64]:
    """
    Calculate electron-nuclear attraction integral between two _dummy_GTOs.

    Parameters
    ----------
    gto1 : _dummy_GTO
        First Gaussian primitive
    gto2 : _dummy_GTO
        Second Gaussian primitive
    charge_atom : float
        Nuclear charge
    coord_atom : NDArray[np.float64]
        Nuclear coordinates, shape (3,)
    k_hyper : int, optional
        Order of Boys function expansion, by default 80

    Returns
    -------
    NDArray[np.float64]
        Electron-nuclear attraction integral matrix elements, shape (gto1.l_dim, gto2.l_dim)
    """
    coord_atom = np.array(coord_atom).reshape(-1)

    a: float = gto1.exp
    b: float = gto2.exp

    r_A = gto1.R
    r_B = gto2.R

    t_max = u_max = v_max = gto1.total_L + gto2.total_L + 1
    p = a + b
    r_P = (a * r_A + b * r_B) / p

    r_PC = r_P - coord_atom
    # Compute the hermite auxiliary integral
    R_tuv_n_array = R_tuv_n(p, r_PC, t_max, u_max, v_max, k_hyper)
    charge = charge_atom

    # Recalling the definition of the Hermite expansion of the overlap distribution,
    # any Gaussian overlap distribution can be expressed as a linear combination of
    # Hermite gaussians, where the required number of Hermite gaussians is determined
    # by t_max = i + j. Since we require all those, per cartesian cordinate, we compute
    # all for t,u,v.
    E_AB_x = E_bottoms_up(r_A[0], a, gto1.total_L, r_B[0], b, gto2.total_L, t_max)
    E_AB_y = E_bottoms_up(r_A[1], a, gto1.total_L, r_B[1], b, gto2.total_L, u_max)
    E_AB_z = E_bottoms_up(r_A[2], a, gto1.total_L, r_B[2], b, gto2.total_L, v_max)

    E_AB_full = np.zeros((gto1.l_dim, gto2.l_dim, t_max, u_max, v_max))

    # After the E_AB coefficients in each cartesian dimension have been computed,
    # The total E_AB_tuv[:,:,:] is required to compute the electrostatic potential.
    # This quantity in the end depends on the total accumulation over these axis.
    # Therefore, we can contract over t,u,v, resulting in a 5 dimensional tensor:
    for p1, proj_1 in enumerate(gto1.l_projections):
        for p2, proj_2 in enumerate(gto2.l_projections):
            i, k, m = proj_1
            j, l, n = proj_2

            E_AB_full[p1, p2, :, :, :] = np.einsum(
                "t, u, v -> tuv",
                E_AB_x[i, j, :t_max],
                E_AB_y[k, l, :u_max],
                E_AB_z[m, n, :v_max],
            )

    # Finally, this tensor is accumulated over the auxiliary integral (Helgaker 9.9.32)
    h_ab_tensor = np.einsum(
        "ijtuv, tuv -> ij", E_AB_full, R_tuv_n_array[:t_max, :u_max, :v_max, 0]
    )

    # And normalization is carried as elementwise product
    norm_tensor = np.einsum(
        "i,j->ij", gto1.normalization_constants, gto2.normalization_constants
    )
    h_ab_tensor *= norm_tensor

    return -(2 * np.pi / p) * charge_atom * h_ab_tensor


# =============================================================================
# LEGACY CODE
# =============================================================================


def _eri_legacy(
    basis_1: _dummy_GTO,
    p1: Tuple[int, int, int],
    N_a: float,
    basis_2: _dummy_GTO,
    p2: Tuple[int, int, int],
    N_b: float,
    basis_3: _dummy_GTO,
    p3: Tuple[int, int, int],
    N_c: float,
    basis_4: _dummy_GTO,
    p4: Tuple[int, int, int],
    N_d: float,
    k_hyper: int = 80,
) -> float:

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


# --- Hermite coefficients product for 3D overlap ---


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


def _g_abcd_legacy(
    basis_1: _dummy_GTO,
    p1: Tuple[int, int, int],
    basis_2: _dummy_GTO,
    p2: Tuple[int, int, int],
    basis_3: _dummy_GTO,
    p3: Tuple[int, int, int],
    basis_4: _dummy_GTO,
    p4: Tuple[int, int, int],
    k_hyper: int = 80,
) -> float:
    """
    Calculate two-electron repulsion integral between four Gaussian basis functions.

    Parameters
    ----------
    basis_1, basis_2, basis_3, basis_4 : _dummy_GTO
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
                            g_abcd += coefficient_1 * coefficient_2 * integral

    return 2 * np.power(np.pi, 2.5) / (p * q * np.sqrt(p + q)) * g_abcd


def _E_ab_legacy(
    basis_1: _dummy_GTO,
    projection_1: Tuple[int, int, int],
    basis_2: _dummy_GTO,
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


def _h_ab_Z_legacy(
    basis_1: _dummy_GTO,
    projection_1: Tuple[int, int, int],
    basis_2: _dummy_GTO,
    projection_2: Tuple[int, int, int],
    charge_atom: Union[int, float],
    coord_atom: NDArray[np.float64],
    k_hyper: int = 80,
) -> float:
    """
    Calculate electron-nuclear attraction integral between two Gaussian basis functions.

    Parameters
    ----------
    basis_1 : _dummy_GTO
        First Gaussian primitive basis function
    projection_1 : Tuple[int, int, int]
        Angular momentum indices (i,k,m) for basis_1
    basis_2 : _dummy_GTO
        Second Gaussian primitive basis function
    projection_2 : Tuple[int, int, int]
        Angular momentum indices (j,l,n) for basis_2
    n_atoms : int
        Number of atoms in system
    charge_atom : Union[int, float]
        Nuclear charge
    coord_atom : NDArray[np.float64]
        Nuclear coordinates, shape (3,)
    k_hyper : int, optional
        Order of Boys function expansion, by default 80

    Returns
    -------
    float
        Electron-nuclear attraction integral value

    Notes
    -----
    Implements the electron-nuclear attraction integral using Hermite Gaussian
    functions following the McMurchie-Davidson scheme.
    """
    coord_atom = np.array(coord_atom).reshape(-1)

    a = basis_1.exp
    b = basis_2.exp

    r_A = basis_1.R.reshape(-1)
    r_B = basis_2.R.reshape(-1)

    i, k, m = projection_1
    j, l, n = projection_2

    t_max = i + j + 1
    u_max = k + l + 1
    v_max = m + n + 1

    p = a + b
    r_P = (a * r_A + b * r_B) / p

    h_ab_total = 0

    r_PC = r_P - coord_atom
    # print(r_PC)
    R_tuv_n_array = R_tuv_n(p, r_PC, t_max, u_max, v_max, k_hyper)
    charge = charge_atom

    for t in range(t_max + 1):
        for u in range(u_max + 1):
            for v in range(v_max + 1):

                coefficient = _E_ab_legacy(
                    basis_1, projection_1, basis_2, projection_2, t, u, v
                )

                hermite_integral = R_tuv_n_array[t, u, v, 0]

                # print(f"{t}, {u}, {v}, {0}: {coefficient} {charge} {hermite_integral}")
                # print(f"{coefficient} ")

                h_ab_total += coefficient * charge * hermite_integral

    return -(2 * np.pi / p) * charge_atom * h_ab_total


if __name__ == "__main__":
    pass
