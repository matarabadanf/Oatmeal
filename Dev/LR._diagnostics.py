from pyscf import gto, scf
import numpy as np 
from Dev.CSRHF_dev import CS_RHF_ContextClass, CS_RHF
import matplotlib.pyplot as plt 
from py_mods.src.SCF.RHF import plot_map

n_elec = [2,2,4,4,6,6,8,8,10,10]
elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']
charges = [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0,]
basis = 'cc-pvdz'

P_diags_ns = []
P_diags_cs = []
LR_diags_ns = []
LR_diags_cs = []
ediffs_ns = []
ediffs_cs = []

for element, n_elec, charge in zip(elements, n_elec, charges):
    mol_He= gto.M(atom = f'{element} 0 0 0', spin=0, charge=charge, basis=basis) 
    mol_He.build()

    kin = mol_He.intor('int1e_kin')
    vnuc = mol_He.intor('int1e_nuc')
    overlap = mol_He.intor('int1e_ovlp')
    eri = mol_He.intor('int2e')

    # prepare UHF calculation
    H2_context = CS_RHF_ContextClass(overlap, kin, vnuc, eri, n_electrons=n_elec, verbose=True)
    H2_context.theta = 0
    # unscaled calculations
    print(f'\n\n\n Case 1s2 {element} {basis}, theta = 0')
    one_s2_theta0 = CS_RHF(H2_context)

    if one_s2_theta0.converged:
        P_diags_ns.append(one_s2_theta0.LR_diagnostics.P_herm)
        LR_diags_ns.append(one_s2_theta0.LR_diagnostics.LR_herm)
        ediffs_ns.append(one_s2_theta0.LR_diagnostics.E_RHF_LR-one_s2_theta0.LR_diagnostics.E_RHF_RR)

    # scaled calculations
    theta = 0.08
    H2_context.theta = theta
    print(f'\n\n\n Case {element} {basis}, theta = {theta}')
    one_s2_theta1 = CS_RHF(H2_context)
    if one_s2_theta1.converged:
        P_diags_cs.append(one_s2_theta1.LR_diagnostics.P_herm)
        LR_diags_cs.append(one_s2_theta1.LR_diagnostics.LR_herm)
        ediffs_cs.append(one_s2_theta1.LR_diagnostics.E_RHF_LR-one_s2_theta0.LR_diagnostics.E_RHF_RR)


ediffs_ns = np.array(ediffs_ns, dtype=np.complex128)
ediffs_cs = np.array(ediffs_cs, dtype=np.complex128)
# print(ediffs_cs * ediffs_cs)
# print(repr(one_s2_theta1.F_final.reshape(one_s2_theta1.X.shape)))
invx = np.linalg.inv(one_s2_theta0.X)
# plot_map(one_s2_theta0.LR_diagnostics.LR_diff.real)
# plot_map((invx @ one_s2_theta0.R_munu - invx @ one_s2_theta0.R_munu).real)

inv = np.linalg.inv(one_s2_theta0.C_prime)
caalcinv = np.conj(one_s2_theta0.C_prime).T

# plot_map((inv-caalcinv).real)

# plt.scatter(LR_diags_ns, ediffs_ns.real, label='Re(Delta_E), NS')
# plt.scatter(LR_diags_cs, ediffs_cs.real, label='Re(Delta_E), CS')
# plt.scatter(LR_diags_cs, ediffs_cs.imag, label='Im(Delta_E), CS')
# # plt.xlim([-1E-19,2E-18])
# plt.xlabel('LR Hermiticity diagnostic value / arb. units.')
# plt.ylabel('LR-RR $\Delta E$')
# plt.legend()
# plt.show()


# plt.scatter(P_diags_ns, ediffs_ns.real, label='Re(Delta_E), NS')
# plt.scatter(P_diags_cs, ediffs_cs.real, label='Re(Delta_E), CS')
# plt.scatter(P_diags_cs, ediffs_cs.imag, label='Im(Delta_E), CS')
# # plt.xlim([-1E-19,2E-18])
# plt.xlabel('LR Hermiticity diagnostic value / arb. units.')
# plt.ylabel('LR-RR $\Delta E$')
# plt.legend()
# plt.show()
