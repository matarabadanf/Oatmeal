import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from py_mods.src.integrals.internal.coulomb_utils import h_ab_Z
from py_mods.src.integrals.primitive import Primitive
from py_mods.src.integrals.internal.ST_utils import S_1D, kinetic_energy_integrals
from typing import Tuple
# from py_mods.src.integrals.basis_set import Contracted

# --- Normalization and projection utilities ---

def N_const(basis: Primitive) -> float: # TODO: the issue is here
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
    projection = np.array([basis.angular_momentum, 0, 0])

    S_3d = S_3D(basis, projection, basis, projection)
    N_A = 1.0 / np.sqrt(S_3d)

    return N_A


def project(l: int) -> list[list[int]]:
    """
    Return projections with total angular momentum l.


    Parameters
    ------
    l : int
        total angular momentum.

    Returns
    ------
    projections : list[list[int]]
        all possible projections with total angular momentum l.
    """
    if l == 0:
        return [[0,0,0]]
    elif l == 1:
        return [[1,0,0], [0,1,0], [0,0,1]]
    elif l == 2:
        return [[2,0,0], [1,1,0], [0,2,0], [0,1,1], [0,0,2], [1,0,1]]
    else:
        return []

def project_dim(l: int) -> int:
    """
    Return number of projections with total angular momentum l.

    Parameters
    ------
    l : int
        total angular momentum.

    Returns
    ------
    projections : int
        Number of projections for total angular momentum l.
    """
    if l == 0:
        return 1
    elif l == 1:
        return 3
    elif l == 2:
        return 6
    else:
        return -1

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


# --- Overlap 3D with primitives ---
def S_3D_components(basis_1: Primitive, projection_1: np.ndarray, basis_2: Primitive, projection_2: np.ndarray) -> np.ndarray:
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


# --- Kinetic 3D with primitives ---

def T_3D(basis_1: Primitive, projection_1: np.ndarray, basis_2: Primitive, projection_2: np.ndarray) -> float:
    """
    Calculate the product of the three kinetic energy integral components.

    Computes the kinetic energy integrals between two primitive functions
    for each Cartesian component and calculates the total kinetic energy with: 

    T_{ab} = T_{ij} S_{kl} S_{mn} + S_{ij} T_{kl} S_{mn} + S_{ij} S_{kl} T_{mn}

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
    total_kinetic : float
        The product of the three kinetic energy components (T_ab, T_cd, T_ef).
    
    Notes
    ------
        - Calls  S_3D_components.
        - No orthogonality check takes place but could be implemented.
    """
    R_a = basis_1.R
    R_b = basis_2.R

    alpha = basis_1.exp
    beta  = basis_2.exp

    a, c, e = projection_1
    b, d, f = projection_2

    S_ab, S_cd, S_ef = S_3D_components(basis_1, projection_1, basis_2, projection_2)

    T_comp = np.zeros(3)

    T_ab = kinetic_energy_integrals(R_a[0], R_b[0], alpha, beta, a, b)
    T_cd = kinetic_energy_integrals(R_a[1], R_b[1], alpha, beta, c, d)
    T_ef = kinetic_energy_integrals(R_a[2], R_b[2], alpha, beta, e, f)

    total_kinetic = (T_ab * S_cd * S_ef) + (S_ab * T_cd  * S_ef) +  (S_ab * S_cd * T_ef)

    return total_kinetic 


def calculate_primitive_dimension(basis_list: list[Primitive]):
    return [project_dim(basis.angular_momentum) for basis in basis_list]


# --- Overlap and Kinetic 3D primitive matrices ---

# --- Potential 3D primitive matrices ---
def calc_V_uncontracted(contracted_basis_1: list[Primitive], contracted_basis_2: list[Primitive], charge, atom_position):
    # for now we will asume that the selection rule responsability is elsewhere
    # and focus on the projection aspect rather.
    # therefore the basis introduced in this functions must have the same l

    projections = project(contracted_basis_1[0].angular_momentum)

    angular_dimension = len(projections)

    dim_1 = len(projections*len(contracted_basis_1))

    V_prim_mat = np.zeros([dim_1, dim_1])

    for p1, projection_1 in enumerate(projections):
        for p2, projection_2 in enumerate(projections):

            # print(projection_1, projection_2)

            l_index_1 = p1 * angular_dimension
            l_index_2 = p2 * angular_dimension

            for i, primitive in enumerate(contracted_basis_1):
                for j, primitive_2 in enumerate(contracted_basis_2):
                    V_prim_mat[l_index_1 + i][l_index_2 + j] = h_ab_Z(primitive, projection_1, primitive_2, projection_2, 1, charge, atom_position)

    return V_prim_mat

if __name__ == '__main__':
    pass 

