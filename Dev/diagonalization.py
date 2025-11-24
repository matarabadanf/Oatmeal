import scipy 
import numpy as np 
from py_mods.src.SCF.RHF import plot_map

non_herm_mat = np.array([
    [1, 2+1j, 3],
    [4, 5, 6+2j],
    [7+3j, 8, 9]
])

plot_map(non_herm_mat.real-non_herm_mat.real.T)
plot_map((non_herm_mat).imag)

ev, L, R = scipy.linalg.eig(non_herm_mat, left=True, right=True)

idx = ev.argsort()
e_values = ev[idx]
L = L[:, idx]
R = R[:, idx]

norms = np.einsum('ij,ij->j', L.conj(), R)
L = L / norms

diag = L.conj().T @ non_herm_mat @ R

plot_map((L.conj().T @ R).real)

