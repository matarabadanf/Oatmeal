from calendar import c
from email import message
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Union
from py_mods.src.integrals.internal.coulomb_utils import h_ab_Z
from py_mods.src.integrals.uncontracted import S_3D, T_3D, project, project_dim
from py_mods.src.integrals.primitive import Primitive
from py_mods.src.integrals.basis_set import Contracted

def mu_to_primitive_map(l_dim):
    map_list = []
    for i , dim in enumerate(l_dim):
        map_list.extend([i] * dim)
    return map_list



def check_orthogonality(mu: NDArray[np.int64]) -> NDArray[np.bool]:
    """
    Check the orthogonality between basis functions based on their angular momentum.
    
    Parameters
    ----------
    mu : NDArray[np.int64]
        Array representing the angular momentum of basis functions.
    Returns
    -------
    truth_matrix : NDArray[np.bool] of shape(len(mu), len(mu))
        A boolean matrix indicating orthogonality between basis functions.
    """
    nu = np.copy(mu)
    dim = len(nu)

    truth_matrix = np.zeros([dim,dim], dtype=bool)

    for i in range(dim):
        for j in range(dim):
            if mu[i] == nu[j]:
                truth_matrix[i,j] = True
    
    return truth_matrix

def check_ortho_projection(l_projs: List[List[int]], l_truth_matrix: NDArray[np.bool]) -> NDArray[np.bool]:

    l_trans = l_projs
    dim = len(l_trans)

    truth_matrix = np.zeros([dim,dim], dtype=bool)

    for i in range(dim):
        for j in range(dim):
            if l_truth_matrix[i,j]:
                if np.dot(l_trans[i].T, l_trans[j]) == 0:
                    if sum(l_trans[i]) == sum(l_trans[j]) == 0:
                        truth_matrix[i,j] = True
                else:
                    truth_matrix[i,j] = True

    return truth_matrix

def plot_truth_matrix(truth_matrix: NDArray[np.bool], lab='l') -> None:
    import matplotlib.pyplot as plt

    plt.imshow(truth_matrix, cmap='gray', interpolation='nearest')
    plt.title(f'{lab} orthogonality Truth Matrix')
    plt.xlabel('Basis Function Index')
    plt.ylabel('Basis Function Index')
    plt.colorbar(label='Orthogonal (True/False)')
    plt.show()

################################################
# --- Contracted Overlap and Kinetic energy ---#
################################################

def ST_contracted(basis_or_data: Union[
        List[Contracted],
        Tuple[List[int], List[List[Primitive]], List[List[float]]],
    ],
    graph=False
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the contracted overlap and kinetic energy matrices.

    Parameters
    ----------
    n_primitives : list[int]
        List containing the number of primitives for each contracted basis function.
    primitives : list[list[Primitive]]
        List of lists of Primitive objects for each contracted basis function.
    contraction_coefficients : list[list[float]]
        List of lists of contraction coefficients for each contracted basis function.
    
    Returns
    -------
    Tuple[NDArray[np.float64] , NDArray[np.float64]] of shape (n_contracted, n_contracted)
        A tuple containing the contracted overlap matrix and the contracted kinetic energy matrix.
        
    """
    if isinstance(basis_or_data, list) and all(isinstance(b, Contracted) for b in basis_or_data):
        n_primitives = [b.n_primitives for b in basis_or_data]
        primitives = [b.primitives for b in basis_or_data]
        contraction_coefficients = [b.c_coeff for b in basis_or_data]
    
    elif isinstance(basis_or_data, tuple) and len(basis_or_data) == 3:
        n_primitives, primitives, contraction_coefficients = basis_or_data
    
    else:
        raise TypeError('Arguments for ST_contracted must be either a list of Contracted or a tuple')

    # get the dimensions of each contracted basis. This is that s will be 1, p 3, d 6, ...
    l_dim = [project_dim(primitives[i][0].angular_momentum) for i in range(len(n_primitives))]
    
    # get the projections of each contracted basis. Same as before, because we will need them.
    l_projs = [project(primitives[i][0].angular_momentum) for i in range(len(n_primitives))]

    # Previous is a list of lists, now it is a flat list, associated with a certain l_dim
    l_projs = np.array([np.array(item) for sublist in l_projs for item in sublist])

    # this takes space, but it is nice to have the matrix. can be later removed.
    l_mu = np.array([sum(proj) for proj in l_projs], dtype=np.int64)
    calculate_element = check_orthogonality(l_mu)
    if graph:
        plot_truth_matrix(calculate_element)
    # same here
    calculate_element = check_ortho_projection(l_projs, calculate_element)
    if graph:
        plot_truth_matrix(calculate_element, 'projection')
    # The contracted matrix will be as large as the total number of projections. 
    size_mat = sum(l_dim)

    contraction_guides = [contracted.coeff_guide for contracted in basis_or_data]
    contraction_guides = [item for sublist in contraction_guides for item in sublist]

    S_contracted_matrix = np.zeros([size_mat, size_mat])
    T_contracted_matrix = np.zeros([size_mat, size_mat])

    # this function is a helper. Since we have expanded the dimension to the projection
    # number, this function generates a list that maps each mu to the dimension. 
    # for example with 1s, 2s, 2p, the map is [0, 1, 2, 2, 2] indicating that 
    # first index corresponds to first contractedm second to second and last 
    # three to the different projections of the third contracted. 
    prim_map = mu_to_primitive_map(l_dim)

    # Start to do pairs. 
    for mu in range(size_mat):
        for nu in range(size_mat):

            # Determine if evaluate element
            if not calculate_element[mu,nu]:
                continue

            # If the ME has to be calculated, determine the primitive that it corresponds to
            i = prim_map[mu]
            j = prim_map[nu]

            # get what are the projections
            proj_1 = np.array(l_projs[mu])
            proj_2 = np.array(l_projs[nu])

            contracted_1 = basis_or_data[i]
            contracted_2 = basis_or_data[j]

            guide_1 = contraction_guides[mu]
            guide_2 = contraction_guides[nu]

            s, t = ST_sub_matrix_contracted(contracted_1, proj_1, guide_1, contracted_2, proj_2, guide_2)
            S_contracted_matrix[mu][nu] = s
            T_contracted_matrix[mu][nu] = t

    return S_contracted_matrix, T_contracted_matrix


def ST_sub_matrix_contracted(cont_1: Contracted, proj_1: NDArray[np.int64], guide_1: int, cont_2: Contracted, proj_2: NDArray[np.int64], guide_2: int) -> float:
    """
    Calculate the contracted overlap and kinetic energy matrix elements.

    Parameters
    ----------
    contracted_basis_1 : list[Primitive]
        List of Primitive objects for the first contracted basis function.
    contracted_basis_2 : list[Primitive]
        List of Primitive objects for the second contracted basis function.
    c_mu : list[float]
        Contraction coefficients for the first contracted basis function.
    c_nu : list[float]
        Contraction coefficients for the second contracted basis function.
    
    Returns
    -------
    tuple

    """
    c_mu = cont_1.c_coeff[guide_1]
    c_nu = cont_2.c_coeff[guide_2]

    prim_mu = cont_1.primitives
    prim_nu = cont_2.primitives

    overlap = 0.0
    kinetic = 0.0  

    for i, prim_i in enumerate(prim_mu):
        for j, prim_j in enumerate(prim_nu):

            S_ij = S_3D(prim_i, proj_1, prim_j, proj_2) # TODO: there is a bug for d 
            T_ij = T_3D(prim_i, proj_1, prim_j, proj_2) 

            overlap += c_mu[i] * c_nu[j] * S_ij
            kinetic += c_mu[i] * c_nu[j] * T_ij

    return overlap, kinetic

################################################
# ---   Contracted Coulomb nuclear energy   ---#
################################################

def V_contracted(
        basis_or_data: Union[
            List[Contracted],
            Tuple[List[int], List[List[Primitive]], List[List[float]]]
        ],
        nuclear_charge, 
        atom_position,
        graph=False,
) -> NDArray[np.float64]:
    
    if isinstance(basis_or_data, list) and all(isinstance(b, Contracted) for b in basis_or_data):
        n_primitives = [b.n_primitives for b in basis_or_data]
        primitives = [b.primitives for b in basis_or_data]
        contraction_coefficients = [b.c_coeff for b in basis_or_data]
    
    elif isinstance(basis_or_data, tuple) and len(basis_or_data) == 3:
        n_primitives, primitives, contraction_coefficients = basis_or_data
    
    else:
        raise TypeError('Arguments for ST_contracted must be either a list of Contracted or a tuple')

    # get the dimensions of each contracted basis. This is that s will be 1, p 3, d 6, ...
    l_dim = [project_dim(primitives[i][0].angular_momentum) for i in range(len(n_primitives))]
    
    # get the projections of each contracted basis. Same as before, because we will need them.
    l_projs = [project(primitives[i][0].angular_momentum) for i in range(len(n_primitives))]

    # Previous is a list of lists, now it is a flat list, associated with a certain l_dim
    l_projs = np.array([np.array(item) for sublist in l_projs for item in sublist])

    # this takes space, but it is nice to have the matrix. can be later removed.
    l_mu = np.array([sum(proj) for proj in l_projs], dtype=np.int64)
    calculate_element = check_orthogonality(l_mu)
    if graph:
        plot_truth_matrix(calculate_element)
    # same here
    calculate_element = check_ortho_projection(l_projs, calculate_element)
    if graph:
        plot_truth_matrix(calculate_element, 'projection')
    # The contracted matrix will be as large as the total number of projections. 
    size_mat = sum(l_dim)

    contraction_guides = [contracted.coeff_guide for contracted in basis_or_data]
    contraction_guides = [item for sublist in contraction_guides for item in sublist]

    V_contracted_matrix = np.zeros([size_mat, size_mat])

    # this function is a helper. Since we have expanded the dimension to the projection
    # number, this function generates a list that maps each mu to the dimension. 
    # for example with 1s, 2s, 2p, the map is [0, 1, 2, 2, 2] indicating that 
    # first index corresponds to first contractedm second to second and last 
    # three to the different projections of the third contracted. 
    prim_map = mu_to_primitive_map(l_dim)

    # Start to do pairs. 
    for mu in range(size_mat):
        for nu in range(size_mat):

            # Determine if evaluate element
            if not calculate_element[mu,nu]:
                continue

            # If the ME has to be calculated, determine the primitive that it corresponds to
            i = prim_map[mu]
            j = prim_map[nu]

            # get what are the projections
            proj_1 = np.array(l_projs[mu])
            proj_2 = np.array(l_projs[nu])

            contracted_1 = basis_or_data[i]
            contracted_2 = basis_or_data[j]

            guide_1 = contraction_guides[mu]
            guide_2 = contraction_guides[nu]

            v = V_sub_matrix_contracted(contracted_1, proj_1, guide_1, contracted_2, proj_2, guide_2, nuclear_charge, atom_position)
            V_contracted_matrix[mu][nu] = v

    return V_contracted_matrix

def V_sub_matrix_contracted(cont_1: Contracted, proj_1: NDArray[np.int64], guide_1: int, cont_2: Contracted, proj_2: NDArray[np.int64], guide_2: int, charge, atom_position):

    c_mu = cont_1.c_coeff[guide_1]
    c_nu = cont_2.c_coeff[guide_2]

    prim_mu = cont_1.primitives
    prim_nu = cont_2.primitives

    V = 0 

    for i, prim_i in enumerate(prim_mu):
        for j, prim_j in enumerate(prim_nu):

            V_ij = h_ab_Z(prim_i, proj_1, prim_j, proj_2, 1, charge, atom_position) 

            V += c_mu[i] * c_nu[j] * V_ij

    return V

