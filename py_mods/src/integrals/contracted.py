import numpy as np
from py_mods.src.integrals.uncontracted import calc_S_uncontracted, calc_V_uncontracted, project, project_dim
from py_mods.src.integrals.primitive import Primitive

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
