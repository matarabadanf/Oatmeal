import numpy as np
from numpy.typing import NDArray
from basis_utils import Primitive, Contracted

def orthogonal(i:int, j:int) -> bool:
    """
    Check if two angular momenta are the same.

    Parameters
    ----------
    i : int
        First index to compare.
    j : int
        Second index to compare.

    Returns
    -------
    bool
        False if the indices are equal (not orthogonal), True otherwise.
    """
    if i == j:
        return False
    else:
        return True

def obara_saika_bottom_up(Ax: float, Bx: float, a: float, b: float, i: int, j: int) -> NDArray[np.float64]:
    """
    Compute 1D overlap integrals between two Cartesian Gaussian primitives
    using the Obara–Saika bottom-up recurrence relations.

    The Gaussians are defined as:

        f(x) = x^i * exp[-a * (x - A_x)^2]
        g(x) = x^j * exp[-b * (x - B_x)^2]

    Instead of recursive calls or caching entries, the simplest was to build
    the matrix from the bottom up. Max dimension of the matrix is chosen to be
    a square matrix. Construction of The matrix starts by building the [i,0]
    and [0,j] entries as their recurrence do not require mixed [i,j] indices.

    The whole matrix is returned due to the possible use of entries later. 

    Parameters
    ----------
    Ax : float
        Center of the first Gaussian.
    Bx : float
        Center of the second Gaussian.
    a : float
        Exponent of the first Gaussian.
    b : float
        Exponent of the second Gaussian.
    i : int
        Angular momentum of the first Gaussian.
    j : int
        Angular momentum of the second Gaussian.

    Returns
    -------
    S_ij_mat : numpy.ndarray of shape (max_dim, max_dim)
        Matrix containing the overlap integrals for angular momentum pairs
        up to (i, j).

    Notes
    -----
    Implements the recursive relations described in Helgaker, Jørgensen &
    Olsen, *Molecular Electronic-Structure Theory*, Ch. 9.
    """
    max_dim = max(i,j)

    if i == j:
        max_dim += 1

    S_ij_mat = np.zeros([max_dim, max_dim])

    X_ab = (Bx-Ax)
    p = a + b
    X_pa = b/p * X_ab
    X_pb = -a/p * X_ab

    # Base case for i == j == 0
    S_ij_mat[0,0] = (np.pi/p)**0.5 * np.exp(-(a * b)/p * X_ab**2)

    # Compute the [i,0] entries 
    for i in range(1, max_dim):
        S_ij_mat[i,0] +=  X_pa * S_ij_mat[i-1,0] 
        S_ij_mat[i,0] +=  1/(2*p)*((i-1) * S_ij_mat[i-2,0])

    # Compute the [0,j] entries 
    for j in range(1, max_dim):
        S_ij_mat[0,j] +=  X_pb * S_ij_mat[0,j-1] 
        S_ij_mat[0,j] +=  1/(2*p)*((j-1) * S_ij_mat[0,j-2])

    # Compute the rest of entries using recursion
    for total in range(1, max_dim):
        for i in range(total, max_dim):
            S_ij_mat[i,total] +=  X_pa * S_ij_mat[i-1,total] 
            S_ij_mat[i,total] += 1/(2*p)*((i-1) * S_ij_mat[i-2,total])
            S_ij_mat[i,total] += 1/(2*p)*((total) * S_ij_mat[i-1,total-1])

        for j in range(total, max_dim):
            S_ij_mat[total,j] +=  X_pb * S_ij_mat[total, j-1] 
            S_ij_mat[total,j] += 1/(2*p)*((total) * S_ij_mat[total-1,j-1])
            S_ij_mat[total,j] += 1/(2*p)*((j-1) * S_ij_mat[total,j-2])

    return S_ij_mat

def S_1D(Ax: float, Bx: float, a: float, b: float, i: int, j: int) -> float:
    """
    Calculate overlap integral S between two Gaussian basis functions in 1D.

    Computes overlap between two cartesian Gaussians by calling Obara-Saika.
    Returns the last element of the overlap matrix. 

    Parameters
    ----------
    Ax : float
        Center of the first Gaussian.
    Bx : float
        Center of the second Gaussian.
    a : float
        Exponent of the first Gaussian.
    b : float
        Exponent of the second Gaussian.
    i : int
        Angular momentum of the first Gaussian.
    j : int
        Angular momentum of the second Gaussian.
    
    Returns
    -------
    S_ij : float
        Overlap in 1D of two Gaussian functions.
    
    Notes
    -----
    Includes a check to avoid computing the whole matrix if the angular moment
    is not the same. For example an s function and a p function have 0 overlap
    due to orthogonality. 
    """
    if i*j == 0 and i != j:
        return 0

    return float(obara_saika_bottom_up(Ax, Bx, a, b, i, j)[-1][-1])

def S_3D_components(basis_1: Primitive, projection_1: np.ndarray, basis_2: Primitive, projection_2: np.ndarray) -> list:
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
    """
    scalar_product = np.dot(projection_1, projection_2)
    l1 = basis_1.angular_momentum
    l2 = basis_2.angular_momentum

    if scalar_product == 0 and l1 != 0 and l2 != 0:
        return np.array([0,0,0])

    # If there is overlap calculate it
    R_a = basis_1.R
    R_b = basis_2.R

    alpha = basis_1.exp
    beta = basis_2.exp

    a, c, e = projection_1
    b, d, f = projection_2

    overlap_components = np.zeros(3)

    for comp, _ in enumerate(overlap_components):
        if not orthogonal(projection_1[comp], projection_2[comp]):
            overlap_components[comp] = S_1D(R_a[comp], R_b[comp], alpha, beta, projection_1[comp], projection_2[comp])

    return overlap_components

def S_3D(basis_1: Primitive, projection_1, basis_2: Primitive, projection_2) -> float:
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
    S_ab, S_cd, S_ef =  S_3D_components(basis_1, projection_1, basis_2, projection_2)

    return S_ab * S_cd * S_ef

def N_const(basis: Primitive) -> float:
    """
    Calculate normalization constant of a Primitive.

    Takes the value of a single projection directly as the normalization
    constant is equal in any projection for a certain total l. 

    Parameters
    ------
    basis_1 : Primitive
        First primitive; must provide attributes:
          - R : array-like of length 3, center coordinates (R_x, R_y, R_z)
          - exp : float, Gaussian exponent (alpha)
          - angular_momentum : int, total angular momentum (l)

    Returns
    ------
        N_A : float
            normalization constant.
    """
    projection = np.array([basis.angular_momentum,0,0])

    S_3d = S_3D(basis, projection, basis, projection)
    N_A = 1.0 / np.sqrt(S_3d)

    return N_A


def normalize(basis: Primitive) -> None:
    """
    Calculate and assign the normalization constant of a primitive. 
    
    Parameters
    ------
    basis : Primitive
        First primitive; must provide attributes:
          - R : array-like of length 3, center coordinates (R_x, R_y, R_z)
          - exp : float, Gaussian exponent (alpha)
          - angular_momentum : int, total angular momentum (l)
    
    Returns
    ------
        None
    """
    norm = N_const(basis)
    basis.normalization_constant = norm


if __name__ == '__main__':

    # Primitives for tests
    basis_1 = Primitive(np.array([0,0,0]), 0.5, 0, 1)
    basis_2 = Primitive(np.array([0,0,0]), 0.5, 0, 1)
    normalize(basis_1)
    normalize(basis_2)

    # Test 1: self overlap 
    self_overlap = basis_1.normalization_constant ** 2 * S_3D(basis_1, np.array([0,0,0]), basis_1, np.array([0,0,0]))
    assert(abs(self_overlap - 1) < 0.000001)

    # Test 2: different projection overlap 
    diff_l = basis_1.normalization_constant ** 2 * S_3D(basis_1, np.array([0,0,0]), basis_1, np.array([1,0,0]))
    assert(diff_l == 0)