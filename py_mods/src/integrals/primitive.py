import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from py_mods.src.integrals.internal.ST_utils import S_1D


@dataclass
class Primitive:
    """
    Gaussian primitive basis function.

    Attributes
    ----------
    R : NDArray[np.float64]
        Center coordinates (3D vector)
    exp : float
        Gaussian exponent
    angular_momentum : int
        Angular momentum quantum number
    norm : float
        Normalization constant
    """

    R: NDArray[np.float64]
    exp: float
    total_L: int
    l_projections: NDArray[np.int32]  # of dimensions (n_projections, 3)
    normalization_constants: NDArray[np.float64]


def create_primitive(R: NDArray[np.float64], exp: float, total_L: int) -> Primitive:
    """
    Factory function to create a Primitive object with computed angular momentum projections
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
    Primitive
        Primitive object
    """
    # Generate all angular momentum projections for total_L
    l_projections = []
    for l_x in range(total_L + 1):
        for l_y in range(total_L - l_x + 1):
            l_z = total_L - l_x - l_y
            l_projections.append([l_x, l_y, l_z])
    l_projections = np.array(l_projections, dtype=np.int32)

    # Compute normalization constants for each projection
    normalization_constants = np.zeros(len(l_projections)) + 1

    prim = Primitive(
        R=R,
        exp=exp,
        total_L=total_L,
        l_projections=l_projections,
        normalization_constants=normalization_constants,
    )

    prim.normalization_constants = 1 / np.sqrt(self_overlap(Prim=prim))

    return prim


def self_overlap(Prim: Primitive) -> NDArray[np.float64]:
    """
    Compute the self-overlap of a Primitive function.

    Parameters
    ----------
    Prim : Primitive
        Primitive function

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


def S_3D(
    basis_1: Primitive, projection_1, N_A, basis_2: Primitive, projection_2, N_B
) -> float:
    """
    Calculate the product ofthe three overlap integral components.

    Parameters
    ------
    basis_1 : Primitive
        First primitive; must provide attributes:
          - R : array-like of length 3, center coordinates (R_x, R_y, R_z)
          - exp : float, Gaussian exponent (alpha)
          - angular_momentum : int, total angular momentum (l)
    projection_1 : numpy.ndarray
        Length-3 integer array with projection quantum numbers for basis_1 (m_x, m_y, m_z)
    basis_2 : Primitive
        Second primitive; same requirements as basis_1
    projection_2 : numpy.ndarray
        Length-3 integer array with projection quantum numbers for basis_2

    Returns
    ------
        float: The product of the three overlap components (S_ab, S_cd, S_ef).
    """
    S_ab, S_cd, S_ef = (
        S_3D_components(basis_1, projection_1, basis_2, projection_2) * N_A**3 * N_B**3
    )

    return S_ab * S_cd * S_ef


# --- Overlap 3D with primitives ---
def S_3D_components(
    basis_1: Primitive,
    projection_1: np.ndarray,
    basis_2: Primitive,
    projection_2: np.ndarray,
) -> np.ndarray:
    """
    Compute the three Cartesian components of the 3D overlap between two primitive functions.

    To ensure orthogonality if the scalar product is 0 (they dont share a
    component in the same projection) and the individual l is nonzero, the
    function returns [0,0,0].

    Parameters
    ----------
    basis_1 : Primitive
        First primitive; must provide attributes:
          - R : array-like of length 3, center coordinates (R_x, R_y, R_z)
          - exp : float, Gaussian exponent (alpha)
          - angular_momentum : int, total angular momentum (l)
    projection_1 : numpy.ndarray
        Length-3 integer array with projection quantum numbers for basis_1 (m_x, m_y, m_z)
    basis_2 : Primitive
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
