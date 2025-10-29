import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Union
from py_mods.src.integrals.internal.coulomb_utils import h_ab_Z
from py_mods.src.integrals.uncontracted import S_3D, T_3D, project, project_dim
from py_mods.src.integrals.primitive import Primitive
from py_mods.src.integrals.basis_set import Contracted

################################################
# ---       Utilities for contraction       ---#
################################################
def mu_to_primitive_map(l_dim: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    Generate map from mu to the corresponding primitive.

    When a primitive has angular momentum higher than 0, the matrices have
    dimensions larger than the number of primitives. Therefore, with mu
    it is not sufficient to determine the primitive, and so this map is 
    made. This map points to the primitive for things such as normalization
    constants.

    Parameters
    ----------
    l_dim : list[int]
        list with the total number of projections of each primitive.

    Returns 
    -------
    map_list : NDArray[np.int64] of shape (sum(l_dim), )
        map that associates mu index with primitive.
    """
    map_list = []
    for i , dim in enumerate(l_dim):
        map_list.extend([i] * dim)
    return np.array(map_list)

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
    """
    Check the orthogonality between basis functions based on their projection.
    
    Parameters
    ----------
    mu : NDArray[np.int64] of shape (mu, )
        Array representing the angular momentum of basis functions.
    l_truth_matrix :  NDArray[np.int64] of shape (mu, mu)
        Truth matrix computed for l orthogonality. 
    Returns
    -------
    truth_matrix : NDArray[np.bool] of shape(mu, mu)
        A boolean matrix indicating orthogonality between basis functions.
    """
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
    '''Generate plot of orthogonality matrix.
    
    Parameters
    ----------
    truth_matrix : NDArray[np.bool] of shape (mu, mu)
        truth matrix computed previously
    lab, optional : str
        Label for the title
    
    Returns
    -------
    None
    
    Notes
    -----
    - Lazy import of matplotlib because this function will not be generally used
    '''
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

def ST_contracted(
        contracted_list: List[Contracted], 
        graph=False
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the contracted overlap and kinetic energy matrices.

    Parameters
    ----------
    contracted_list : list[Contracted]
        List containing the number Contracted basis

    Returns
    -------
    Tuple[NDArray[np.float64] , NDArray[np.float64]] of shape (n_contracted, n_contracted)
        A tuple containing the contracted overlap matrix and the contracted kinetic energy matrix.
        
    """
    if isinstance(contracted_list, list) and all(isinstance(b, Contracted) for b in contracted_list):
        n_primitives = [b.n_primitives for b in contracted_list]
        primitives = [b.primitives for b in contracted_list]
        contraction_guides = [contracted.coeff_guide for contracted in contracted_list]
        contraction_guides = [item for sublist in contraction_guides for item in sublist]
    else:
        raise TypeError('Arguments for ST_contracted must be a list of Contracted')

    # get necessary informatiojn to chose the matrix elements to compute
    l_dim, l_projs, calculate_element, contr_map = matrix_element_setup(primitives, n_primitives, graph=graph)

    # The contracted matrix will be as large as the total number of projections. 
    size_mat = sum(l_dim)
    S_contracted_matrix = np.zeros([size_mat, size_mat])
    T_contracted_matrix = np.zeros([size_mat, size_mat])

    # Start to do pairs. 
    for mu in range(size_mat):
        for nu in range(size_mat):

            # Determine if evaluate element
            if not calculate_element[mu,nu]:
                continue

            # If the ME has to be calculated, determine the primitive that it corresponds to
            i = contr_map[mu]
            j = contr_map[nu]

            # Get associated projection
            proj_mu = np.array(l_projs[mu])
            proj_nu = np.array(l_projs[nu])

            # Get associated contracted
            contracted_mu = contracted_list[i]
            contracted_nu = contracted_list[j]

            # Select normalization constants
            coeffs_mu = contraction_guides[mu]
            coeffs_nu = contraction_guides[nu]

            ST = ST_sub_matrix_contracted(contracted_mu, proj_mu, coeffs_mu, contracted_nu, proj_nu, coeffs_nu)
            S_contracted_matrix[mu][nu] = ST[0]
            T_contracted_matrix[mu][nu] = ST[1]

    return S_contracted_matrix, T_contracted_matrix


def ST_sub_matrix_contracted(cont_1: Contracted, proj_1: NDArray[np.int64], guide_1: int, cont_2: Contracted, proj_2: NDArray[np.int64], guide_2: int) -> float:
    """
    Calculate the contracted overlap and kinetic energy matrix elements.

    Parameters
    ----------
    cont_1 : Contracted
        Contracted Basis 1
    proj_1 : NDArray[np.int64] of size (3, )
        Projection of the basis for specific matrix element
    guide_1 : int
        Index of contraction coefficients to be used. 
    cont_2 : Contracted
        Contracted Basis 1
    proj_2 : NDArray[np.int64] of size (3, )
        Projection of the basis for specific matrix element
    guide_2 : int
        Index of contraction coefficients to be used. 
    charge : float
        Charge of atom to compute V
    atom_position : NDArray[np.float64] of size (3, )
        Position of the atom to compute V

    Returns
    -------
    NDArray[np.float64] of shape (2, )
        Contracted S and T matrix elements. 
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

    return np.array([overlap, kinetic])

################################################
# ---   Contracted Coulomb nuclear energy   ---#
################################################

def V_contracted(
        contracted_list: List[Contracted],
        nuclear_charge: float, 
        atom_position: NDArray[np.float64],
        graph=False,
) -> NDArray[np.float64]:
    """
    Calculate the contracted overlap and kinetic energy matrices.

    Parameters
    ----------
    contracted_list : list[Contracted]
        List containing the number Contracted basis
    nuclear_charfe : float
        Charge of atom to compute V matrix
    atom_position : 
        Atom position to compute V matrix 
    graph, optional : Bool (default False)
        Graph orthogonality matrix. 

    Returns
    -------
    NDArray[np.float64] (n_contracted, n_contracted)
        Contracted Coulomb potential matrix.
    """
    if isinstance(contracted_list, list) and all(isinstance(b, Contracted) for b in contracted_list):
        n_primitives = [b.n_primitives for b in contracted_list]
        primitives = [b.primitives for b in contracted_list]
        contraction_guides = [contracted.coeff_guide for contracted in contracted_list]
        contraction_guides = [item for sublist in contraction_guides for item in sublist]
    else:
        raise TypeError('Arguments for ST_contracted must be a list of Contracted')

    # get necessary informatiojn to chose the matrix elements to compute
    l_dim, l_projs, calculate_element, contr_map = matrix_element_setup(primitives, n_primitives, graph=graph)

    # The contracted matrix will be as large as the total number of projections. 
    size_mat = sum(l_dim)

    V_contracted_matrix = np.zeros([size_mat, size_mat])
    # Start to do pairs. 
    for mu in range(size_mat):
        for nu in range(size_mat):

            # Determine if evaluate element
            if not calculate_element[mu,nu]:
                continue

            # Get index of associated contracted
            i = contr_map[mu]
            j = contr_map[nu]

            # Get associated projection
            proj_mu = np.array(l_projs[mu])
            proj_nu = np.array(l_projs[nu])

            # Get associated contracted
            contracted_mu = contracted_list[i]
            contracted_nu = contracted_list[j]

            # Select normalization constants
            coeffs_mu = contraction_guides[mu]
            coeffs_nu = contraction_guides[nu]

            # Calculate matrix element
            v = V_sub_matrix_contracted(contracted_mu, proj_mu, coeffs_mu, contracted_nu, proj_nu, coeffs_nu, nuclear_charge, atom_position)
            V_contracted_matrix[mu][nu] = v

    return V_contracted_matrix

def V_sub_matrix_contracted(cont_1: Contracted, proj_1: NDArray[np.int64], guide_1: int, cont_2: Contracted, proj_2: NDArray[np.int64], guide_2: int, charge, atom_position):
    """
    Calculate Coulomb contracted matrix element of two basis with a given projection. 

    Parameters
    ----------
    cont_1 : Contracted
        Contracted Basis 1
    proj_1 : NDArray[np.int64] of size (3, )
        Projection of the basis for specific matrix element
    guide_1 : int
        Index of contraction coefficients to be used. 
    cont_2 : Contracted
        Contracted Basis 1
    proj_2 : NDArray[np.int64] of size (3, )
        Projection of the basis for specific matrix element
    guide_2 : int
        Index of contraction coefficients to be used. 
    charge : float
        Charge of atom to compute V
    atom_position : NDArray[np.float64] of size (3, )
        Position of the atom to compute V

    Returns
    -------
    V : float
        Contracted Coulomb potential matrix element. 
    """
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

def matrix_element_setup(primitives, n_primitives, graph=False):
    # get the dimensions of each contracted basis. This is that s will be 1, p 3, d 6, ...
    l_dim = [project_dim(primitives[i][0].angular_momentum) for i in range(len(n_primitives))]
    
    # get the projections of each contracted basis. Same as before, because we will need them.
    l_projs = [project(primitives[i][0].angular_momentum) for i in range(len(n_primitives))]
    l_projs = np.array([np.array(item) for sublist in l_projs for item in sublist]) #flatten
    
    # Projs to later determine orthogonality 
    l_mu = np.array([sum(proj) for proj in l_projs], dtype=np.int64)

    # Determine orthogonality
    calculate_element = check_orthogonality(l_mu)
    if graph:
        plot_truth_matrix(calculate_element)

    calculate_element = check_ortho_projection(l_projs, calculate_element)
    if graph:
        plot_truth_matrix(calculate_element, 'projection')

    # This function is a helper. See it's definition.
    prim_map = mu_to_primitive_map(l_dim)

    return l_dim, l_projs, calculate_element, prim_map

if __name__ == '__main__':
    pass 

