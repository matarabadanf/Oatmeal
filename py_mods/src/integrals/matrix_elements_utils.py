import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from py_mods.src.integrals.coulomb import h_ab_Z

@dataclass
class Primitive:
    """Represents a primitive Cartesian Gaussian function."""
    R: np.ndarray # of len 3
    exp: float
    angular_momentum: int
    normalization_constant: float

@dataclass
class Contracted:
    """Represents a contracted Gaussian function."""
    n_primitives: int
    angular_momentum: int
    normalization_constants: list[float]
    primitives: list[Primitive]
    c_coeff: list[float]

    def __init__(self, R: np.ndarray, exps: list[float], c_coeff: list[float], angular_momentum: int) -> None:
        """
        Parameters
        ----------
        R : np.ndarray
            Center of the basis function (length 3).
        exps : List[float]
            Gaussian exponents (a_i) for each primitive.
        c_coeff : List[float]
            Contraction coefficients (d_i) for each primitive.
        angular_momentum : int
            Total angular momentum l.
        """
        self.n_primitives = len(exps)
        self.angular_momentum = angular_momentum
        self.c_coeff = c_coeff

        self.normalization_constants = [1.0 for _ in exps]

        # Create Primitive instances
        self.primitives = [
            Primitive(R=np.array(R, dtype=float),
                      exp=exp,
                      angular_momentum=angular_momentum,
                      normalization_constant=norm)
            for exp, norm in zip(exps, self.normalization_constants)
        ]

        for i, prim in enumerate(self.primitives):
            N_a = N_const(prim)
            prim.normalization_constant = N_a
            self.normalization_constants[i] = N_a
            self.c_coeff[i] = N_a * c_coeff[i]


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
            S_ij_mat[i,total] =  X_pa * S_ij_mat[i-1,total] +  1/(2*p)*((i-1) * S_ij_mat[i-2,total] + (total) * S_ij_mat[i-1,total-1])
        for j in range(total, max_dim):
            S_ij_mat[total,j] =  X_pb * S_ij_mat[total, j-1] +  1/(2*p)*((total) * S_ij_mat[total-1,j-1] +(j-1) * S_ij_mat[total,j-2])

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

    return float(obara_saika_bottom_up(Ax, Bx, a, b, i, j)[i][j])

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
        # if not orthogonal(projection_1[comp], projection_2[comp]):
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

def kinetic_energy_integrals(Ax: float, Bx: float, a: float, b: float, ii: int, jj: int) -> float:
    """
    Calculate overlap integral T between two Gaussian basis functions in 1D.

    Computes overlap between two cartesian Gaussians by calling Obara-Saika.
    As in the overlap integral recurrence, instead of cacheing or calling 
    recursivelym the whole matrix is build from the independent (ii,0) and 
    (0,jj) cases. 

    When calling the OS for the overlap, a dimension of max+1 is called 
    due to the requirement in the recurrence relations. 

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
    ii : int
        Angular momentum of the first Gaussian.
    jj : int
        Angular momentum of the second Gaussian.
    
    Returns
    -------
    T_ij : float
        Kinetic energy in 1D of two Gaussian functions.

        Notes
    -----
        - Calls obara_saika_bottom_up.
        - The recurrence relations require overlap integrals S_[i+1,j] and S_[i,j+1],
          so the function computes integrals up to max(ii, jj) + 1.
        - The algorithm fills the kinetic energy matrix in the following order:
            1. Base case T_00.
            2. First row [i, 0] and first column [0, j].
            3. Mixed indices. 
            4. Last element T_ij[max,max].
    """
    X_ab =  (Bx-Ax) 
    p    =  a + b
    X_pa =  b/p * X_ab
    X_pb = -a/p * X_ab

    # Max dim + 1 and for it to be a square matrix
    max_dim = max(ii+1, jj+1)
    kinetic_energy = np.zeros([max_dim, max_dim])

    # Compute the overlap matrix terms 
    recurrence_integrals = obara_saika_bottom_up(Ax, Bx, a, b, max_dim, max_dim)

    # Fill the T_00 case
    kinetic_energy[0,0] = (a -2 * a **2 *(X_pa ** 2 + 1/(2*p))) * recurrence_integrals[0,0]

    if ii == jj == 0:
        return kinetic_energy[0,0]

    # First row and column
    for i in range(0, max_dim-1):
        f_term = X_pa * kinetic_energy[i,0]
        s_term =  1/(2*p)*(i * kinetic_energy[i-1,0])

        t_term_1 =  b/p * (2*a * recurrence_integrals[i+1,0])
        t_term_2 =  b/p * (- i * recurrence_integrals[i-1,0])
        t_term = t_term_1 + t_term_2

        kinetic_energy[i+1,0] = f_term + s_term + t_term

    for j in range(0, max_dim-1):

        f_term = X_pb * kinetic_energy[0, j]
        s_term =  1/(2*p)*(j * kinetic_energy[0, j-1])

        t_term_1 =  a/p * (2 * b * recurrence_integrals[0, j+1])
        t_term_2 =  a/p * (- j *recurrence_integrals[0, j-1])
        t_term = t_term_1 + t_term_2

        kinetic_energy[0,j+1] = f_term + s_term + t_term

    # Iteration over the rows and columns
    for total in range(0, max_dim-1):
        j = total
        for i in range(0, max_dim-1):
            f_term = X_pa * kinetic_energy[i,j]
            s_term =  1/(2*p)*(i * kinetic_energy[i-1,j] + j * kinetic_energy[i, j-1])

            t_term_1 =  b/p * (2 * a * recurrence_integrals[i+1,j])
            t_term_2 =  b/p * (- i * recurrence_integrals[i-1,j])
            t_term = t_term_1 + t_term_2

            kinetic_energy[i+1, j] = f_term + s_term + t_term

        i = total
        for j in range(0, max_dim-1):
            f_term = X_pb * kinetic_energy[i, j]
            s_term =  1/(2*p)*(i*kinetic_energy[i-1, j] + j * kinetic_energy[i, j-1])

            t_term_1 =  a/p * (2 * b * recurrence_integrals[i, j+1])
            t_term_2 =  a/p * (- j * recurrence_integrals[i, j-1])
            t_term = t_term_1 + t_term_2

            kinetic_energy[i,j+1] = f_term + s_term + t_term

    # Corner case
    if ii == jj and ii == max_dim-1:

        i = max_dim -1
        j = max_dim -1

        f_term = X_pa * kinetic_energy[i-1,j]
        s_term =  1/(2*p)*((i-1) * kinetic_energy[i-2,j] + j * kinetic_energy[i-1, j-1])

        t_term_1 =  b/p * (2 * a * recurrence_integrals[i,j])
        t_term_2 =  b/p * (- (i-1) * recurrence_integrals[i-2,j])
        t_term = t_term_1 + t_term_2

        kinetic_energy[max_dim-1, max_dim-1] =  f_term + s_term + t_term

    T_ij = kinetic_energy[ii][jj]

    return float(T_ij)


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

if __name__ == '__main__':

    # Primitives for tests
    basis_1 = Primitive(np.array([0,0,0]), 0.5, 0, 1)
    basis_2 = Primitive(np.array([0,0,0]), 0.5, 0, 1)
    normalize(basis_1)
    normalize(basis_2)

    # Test 1: self overlap 
    self_overlap = basis_1.normalization_constant ** 2 * S_3D(basis_1, np.array([0,0,0]), basis_1, np.array([0,0,0]))
    assert abs(self_overlap - 1) < 0.000001, f"Self overlap test failed: value is {self_overlap}, should be 1"

    # Test 2: different projection overlap 
    diff_l = basis_1.normalization_constant ** 2 * S_3D(basis_1, np.array([0,0,0]), basis_1, np.array([1,0,0]))
    assert diff_l == 0, f"Different l overlap test failed: value is {diff_l}, should be 0"

    # New primitives for further tests:
    basis_1 = Primitive(np.array([0,0,0]), 0.5, 1, 1)
    basis_2 = Primitive(np.array([0,0,0]), 0.5, 1, 1)
    # normalize(basis_1)
    # normalize(basis_2)

    # Test 3: Kinetic energy with different l:
    t_test = T_3D(basis_1, np.array([1,0,0]), basis_2, np.array([0,0,0]))
    assert abs(t_test-0) < 0.0000001, f"Kinetic energy different l test failed: value is {t_test}, should be 0"

    # Test 4: Kinetic energy with same l:
    t_test = T_3D(basis_1, np.array([1,0,0]), basis_2, np.array([1,0,0]))
    assert abs(t_test-3.4802049980198166) < 0.0000001 , f"Kinetic energy same l test failed: value is {t_test}, should be 3.4802049980198166"

    # test contraction 
    alphas = [3.42525091, 0.62391373, 0.16885540]
    d =      [0.15432897, 0.53532814, 0.44463454]

    sto_3g_1s_h1 = Contracted(np.array([0,0,0]), alphas, d, 0)

    # print(sto_3g_1s_h1.c_coeff)

def calculate_primitive_dimension(basis_list: list[Primitive]):
    return [project_dim(basis.angular_momentum) for basis in basis_list]


def calc_S_uncontracted(contracted_basis_1: list[Primitive], contracted_basis_2: list[Primitive]):
    # for now we will asume that the selection rule responsability is elsewhere
    # and focus on the projection aspect rather.
    # therefore the basis introduced in this functions must have the same l

    projections = project(contracted_basis_1[0].angular_momentum)

    angular_dimension = len(projections)

    dim_1 = len(projections*len(contracted_basis_1))

    S_prim_mat = np.zeros([dim_1, dim_1])
    T_prim_mat = np.zeros([dim_1, dim_1])


    for p1, projection_1 in enumerate(projections):
        for p2, projection_2 in enumerate(projections):

            # print(projection_1, projection_2)

            if projection_1 != projection_2:
                continue

            l_index_1 = p1 * angular_dimension
            l_index_2 = p2 * angular_dimension

            for i, primitive in enumerate(contracted_basis_1):
                for j, primitive_2 in enumerate(contracted_basis_2):
                    s_ab, s_cd, s_ef = S_3D_components(primitive, projection_1, primitive_2, projection_2)
                    S_prim_mat[l_index_1 + i][l_index_2 + j] = s_ab * s_cd * s_ef
                    T_prim_mat[l_index_1 + i][l_index_2 + j] = T_3D(primitive, projection_1, primitive_2, projection_2)

    return S_prim_mat, T_prim_mat

def contracted_matrix_element(uncontracted_matrix, c_mu, c_nu):
# Now looping over the size of the matrices:
    contr_prop = 0.0

    max_p = len(c_mu)
    max_q = len(c_nu)

    # print(f'max_p = {max_p}, max_q = {max_q}')
    # print(uncontracted_matrix)

    for p in range(max_p):
        for q in range(max_q):
            contr_prop += c_mu[p] * c_nu[q] * uncontracted_matrix[p][q]

    return contr_prop

def extend_contraction_coefficients(c_mu, projection_dim:int):
    extended_c_mu = []
    for dim in range(projection_dim):
        # print(f'Since dimension is {project_dim}')
        sublist = c_mu
        # print(f'I will add {sublist}')
        extended_c_mu.extend(sublist)
    return extended_c_mu

def ST_sub_matrix_contracted(contracted_basis_1: list[Primitive], contracted_basis_2: list[Primitive], c_mu: list[float], c_nu: list[float]):

    S_sub_matrix_uncontracted = calc_S_uncontracted(contracted_basis_1, contracted_basis_2)

    l_projections = project_dim(contracted_basis_1[0].angular_momentum)

    dimension = len(contracted_basis_1) * l_projections

    # print(f'Angular momentum is {contracted_basis_1[0].angular_momentum}')
    # print(f'Dimension is {dimension}')

    c_mu_extended = extend_contraction_coefficients(c_mu, l_projections)
    c_nu_extended = extend_contraction_coefficients(c_nu, l_projections)

    # print(S_sub_matrix_uncontracted[0])
    # print(c_mu_extended)
    # print(c_nu_extended)

    return contracted_matrix_element(S_sub_matrix_uncontracted[0], c_mu_extended, c_nu_extended), contracted_matrix_element(S_sub_matrix_uncontracted[1], c_mu_extended, c_nu_extended)

def ST_contracted(n_primitives: list[int], primitives: list[list[Primitive]], contraction_coefficients: list[list[float]]):


    l_dim = [project_dim(primitives[i][0].angular_momentum) for i in range(len(n_primitives))]
    l_projs = [project(primitives[i][0].angular_momentum) for i in range(len(n_primitives))]

    l_projs = [item for sublist in l_projs for item in sublist]


    #print(l_projs)

    #print(f'ldim is {l_dim}')
    size_mat = sum(l_dim)

    #print(f'size is {size_mat}')

    S_contracted_matrix = np.zeros([size_mat, size_mat])
    T_contracted_matrix = np.zeros([size_mat, size_mat])

    #print(S_contracted_matrix)

    prim_map = mu_to_primitive_map(l_dim)

    # print(f'prim_map is {prim_map}\n\n\n')

    # here we have implemented ALL selection rules. It can be improved but
    for mu in range(size_mat):
        for nu in range(size_mat):

            i = prim_map[mu]
            j = prim_map[nu]

            if primitives[i][0].angular_momentum != primitives[j][0].angular_momentum:
                continue

            p1 = np.array(l_projs[mu])
            p2 = np.array(l_projs[nu])

            sum_1 = sum(p1)
            sum_2 = sum(p2)

            # print(f'p1 is {p1}, p2 is {p2}')
            # print(f'i is {i}, j is {j}')

            scalar_product = np.dot(p1, p2)

            # print(f'scalar product is {scalar_product}')
            # print(f'sum_1 is {sum_1}, sum_2 is {sum_2}')

            if scalar_product == 0 and not sum_1 == sum_2 == 0:
                continue

            else:
                S_contracted_matrix[mu][nu] = ST_sub_matrix_contracted(primitives[i], primitives[j], contraction_coefficients[i], contraction_coefficients[j])[0]
                T_contracted_matrix[mu][nu] = ST_sub_matrix_contracted(primitives[i], primitives[j], contraction_coefficients[i], contraction_coefficients[j])[1]

                # I believe that this should work, however it would be ideal
                # to avoid wasting time in calculating here three (or 6 or so on)
                # per matrix element

                S_contracted_matrix[mu][nu] /= l_dim[prim_map[mu]]
                T_contracted_matrix[mu][nu] /= l_dim[prim_map[mu]]

    return S_contracted_matrix, T_contracted_matrix

def mu_to_primitive_map(l_dim):
    map_list = []
    for i , dim in enumerate(l_dim):
        map_list.extend([i] * dim)
    return map_list

def V_contracted(n_primitives: list[int], primitives: list[list[Primitive]], contraction_coefficients: list[list[float]], nuclear_charge, position):


    l_dim = [project_dim(primitives[i][0].angular_momentum) for i in range(len(n_primitives))]
    l_projs = [project(primitives[i][0].angular_momentum) for i in range(len(n_primitives))]

    l_projs = [item for sublist in l_projs for item in sublist]


    #print(l_projs)

    #print(f'ldim is {l_dim}')
    size_mat = sum(l_dim)

    #print(f'size is {size_mat}')

    V_contracted_matrix = np.zeros([size_mat, size_mat])

    #print(S_contracted_matrix)

    prim_map = mu_to_primitive_map(l_dim)

    # print(f'prim_map is {prim_map}\n\n\n')

    # here we have implemented ALL selection rules. It can be improved but
    for mu in range(size_mat):
        for nu in range(size_mat):

            i = prim_map[mu]
            j = prim_map[nu]

            if primitives[i][0].angular_momentum != primitives[j][0].angular_momentum:
                continue

            p1 = np.array(l_projs[mu])
            p2 = np.array(l_projs[nu])

            sum_1 = sum(p1)
            sum_2 = sum(p2)

            # print(f'p1 is {p1}, p2 is {p2}')
            # print(f'i is {i}, j is {j}')

            scalar_product = np.dot(p1, p2)

            # print(f'scalar product is {scalar_product}')
            # print(f'sum_1 is {sum_1}, sum_2 is {sum_2}')

            if scalar_product == 0 and not sum_1 == sum_2 == 0:
                continue

            else:
                V_contracted_matrix[mu][nu] = V_sub_matrix_contracted(primitives[i], primitives[j], contraction_coefficients[i], contraction_coefficients[j], nuclear_charge, position)

                # Same as before. Should work, but there must be a better way

                V_contracted_matrix[mu][nu] /= l_dim[prim_map[mu]]

    return V_contracted_matrix

def V_sub_matrix_contracted(contracted_basis_1: list[Primitive], contracted_basis_2: list[Primitive], c_mu: list[float], c_nu: list[float], charge, atom_position):

    V_sub_matrix_uncontracted = calc_V_uncontracted(contracted_basis_1, contracted_basis_2, charge, atom_position)

    l_projections = project_dim(contracted_basis_1[0].angular_momentum)

    dimension = len(contracted_basis_1) * l_projections

    # print(f'Angular momentum is {contracted_basis_1[0].angular_momentum}')
    # print(f'Dimension is {dimension}')

    c_mu_extended = extend_contraction_coefficients(c_mu, l_projections)
    c_nu_extended = extend_contraction_coefficients(c_nu, l_projections)

    # print(S_sub_matrix_uncontracted[0])
    # print(c_mu_extended)
    # print(c_nu_extended)

    return contracted_matrix_element(V_sub_matrix_uncontracted, c_mu_extended, c_nu_extended)

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