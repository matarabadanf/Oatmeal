import numpy as np 
from py_mods.src.integrals.matrix_elements_utils import Primitive
from py_mods.src.integrals.hermite import R_tuv_n, E_ab

def h_ab_Z(basis_1: Primitive, projection_1, basis_2: Primitive, projection_2, n_atoms:int, charge_atom:int, coord_atom:np.ndarray, k_hyper=80):

    a = basis_1.exp
    b = basis_2.exp

    r_A = basis_1.R
    r_B = basis_2.R

    i, k, m = projection_1
    j, l, n = projection_2

    t_max = i + j + 1
    u_max = k + l + 1
    v_max = m + n + 1

    p = a+b
    r_P = (a * r_A + b * r_B)/p

    h_ab_total = 0

    r_PC = r_P - coord_atom
    # print(r_PC)
    R_tuv_n_array = R_tuv_n(p, r_PC, t_max, u_max, v_max, k_hyper)
    charge = charge_atom

    for t in range(t_max):
        for u in range(u_max):
            for v in range(v_max):

                coefficient = E_ab(basis_1, projection_1, basis_2, projection_2, t, u, v)
                hermite_integral = R_tuv_n_array[t, u, v, 0]

                # print(f"{t}, {u}, {v}, {0}: {coefficient} {charge} {hermite_integral}")
                # print(f"{coefficient} ")

                h_ab_total += coefficient * charge * hermite_integral

    return (-1)**(t_max+u_max+v_max)*2 * np.pi / p * h_ab_total

def g_abcd(basis_1, p1, basis_2, p2, basis_3, p3, basis_4, p4, k_hyper=80):

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


    p = a+b
    r_P = (a * r_A + b * r_B)/p

    q = c+d
    r_Q = (c * r_C + d * r_D)/q

    r_PQ = r_P - r_Q

    alpha = p*q/(p+q)

    Hermite_integral = R_tuv_n(alpha, r_PQ, t_max + tau_max, u_max + nu_max, v_max + phi_max, k_hyper)

    g_abcd = 0

    for t in range(t_max):
        for u in range(u_max):
            for v in range(v_max):
                for tau in range(tau_max):
                    for nu in range(nu_max):
                        for phi in range(phi_max):
                            coefficient_1 = E_ab(basis_1, p1, basis_2, p2, t, u, v)
                            coefficient_2 = E_ab(basis_3, p3, basis_4, p4, tau, nu, phi)
                            integral = Hermite_integral[t + tau, u + nu, v + phi, 0]
                            g_abcd += coefficient_1 * coefficient_2 * integral

    return 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))* g_abcd