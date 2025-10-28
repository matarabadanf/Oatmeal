import numpy as np
from py_mods.src.integrals.special_functions import boys_hypergeom
from py_mods.src.integrals.matrix_elements_utils import Primitive

def R_tuv_n(p, R_pc, t_max, u_max, v_max, k_hyper):


    #rint(R_pc)

    R_2= R_pc[0]**2 + R_pc[1]**2 +R_pc[2]**2
    X_pc, Y_pc, Z_pc = R_pc

    # the dimensions of this object are [t_max+1, u_max+1 v_max+1, n_max+1]
    # where n_max = t+u+v
    # However, when later performing the summation i must do it up to n_max, not n_max+1
    # The direct summation works here because it is initialized in np-zeros, however, it is better to consider that later in the summation function

    n_max =  t_max + u_max + v_max + 1

    R_tuv_n_array = np.zeros([t_max+1, u_max+1, v_max+1, n_max])

    for n in range(0, n_max-1):
        R_tuv_n_array[0,0,0,n] = (-2*p)**n * boys_hypergeom(n, p * R_2, k_hyper)



    # now lets get into recursion
    for t in range(0, t_max):
        for n in range(n_max-1):
            component_1 = t * R_tuv_n_array[t-1,0,0,n+1] if t >= 1 else 0
            component_2 = X_pc * R_tuv_n_array[t,0,0,n+1]
            R_tuv_n_array[t+1,0,0,n] = component_1 + component_2


    for u in range(u_max):
        for t in range(0, t_max+1):
            for n in range(n_max-1):
                component_1 = u * R_tuv_n_array[t,u-1,0,n+1] if u >= 1 else 0
                component_2 = Y_pc * R_tuv_n_array[t,u,0,n+1]
                R_tuv_n_array[t,u+1,0,n] = component_1 + component_2

    # return R_tuv_n_array

    for v in range(v_max):
        for u in range(u_max+1):
            for t in range(0, t_max+1):
                for n in range(n_max-1):
                    component_1 = v * R_tuv_n_array[t,u,v-1,n+1] if v >= 1 else 0
                    component_2 = Z_pc * R_tuv_n_array[t,u,v,n+1]
                    R_tuv_n_array[t,u,v+1,n] = component_1 + component_2
    return R_tuv_n_array


def E(basis_1: Primitive, i, basis_2: Primitive, j, t, dim):
    # Calculates the Hermite coefficients of the expansion

    Ax = basis_1.R[dim]
    Bx = basis_2.R[dim]
    a = basis_1.exp
    b = basis_2.exp

    max_dim = max(i+1,j+1)

    X_ab = (Bx-Ax)
    p = a + b
    mu = (a*b)/p
    X_pa = b/p * X_ab
    X_pb = -a/p * X_ab

    #edge cases:
    if i < 0 or j < 0 or t < 0 or t > (i + j):
        return 0

    elif i==0 and j == 0 and t == 0:
        return np.exp(-mu*X_ab**2)

    if t > 0:
        return (i * E(basis_1, i - 1, basis_2, j, t - 1, dim) +
                j * E(basis_1, i, basis_2, j - 1, t - 1, dim)) / (2.0 * p * t)

    if t == 0 and i > 0:
        return X_pa * E(basis_1, i - 1, basis_2, j, 0, dim) + E(basis_1, i - 1, basis_2, j, 1, dim)
    if t == 0 and j > 0:
        return X_pb * E(basis_1, i, basis_2, j - 1, 0, dim) + E(basis_1, i, basis_2, j - 1, 1, dim)

'''
    # recursions
    if t > 0:
        return (i * E(i-1, j, t-1, Ax, Bx, a, b) + j * E(i, j-1, t-1, Ax, Bx, a, b))/(2*p*t)

    elif t == 0 and i > 0:
        return X_pa*E(i-1, j, t, Ax, Bx, a, b) + E(i-1, j, 1, Ax, Bx, a, b)
    elif t == 0 and j > 0:
        return X_pb*E(i, j-1, t, Ax, Bx, a, b) + E(i, j-1, 1, Ax, Bx, a, b)
'''

def E_ab(basis_1:Primitive, projection_1, basis_2: Primitive, projection_2, t, u, v):

    i, k, m = projection_1
    j, l, n = projection_2

    E_1 = E(basis_1, i, basis_2, j, t, 0)
    E_2 = E(basis_1, k, basis_2, l, u, 1)
    E_3 = E(basis_1, m, basis_2, n, v, 2)

    return E_1 * E_2 * E_3