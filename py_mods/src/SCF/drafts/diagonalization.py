import numpy as np 
import scipy

herm_matrix = np.array([[2, 1+1j], [1-1j, 3]])

eigenvalues, eigenvectors = scipy.linalg.eigh(herm_matrix)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

print("U @ U:\n")
print(eigenvectors.T.conj() @ eigenvectors )

mult = eigenvectors.T.conj() @ herm_matrix @ eigenvectors

print("\nDiagonalized Matrix:\n")
print(mult)

# non-hermitian case
print("\nNon-Hermitian Case:\n")
non_herm_matrix = np.array([[2, 1+1j], [3+1j, 3]])
non_herm_matrix = np.array([
    [-1.55863874-0.43112485j, -0.8399457 +0.33611679j],
    [-0.8399457 +0.33611679j, -0.6612901 -0.01106972j]
])

eigenvalues, eigenvectors = scipy.linalg.eig(non_herm_matrix)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

print("U @ U:\n")
print(eigenvectors.T.conj() @ eigenvectors )

mult = eigenvectors.T.conj() @ herm_matrix @ eigenvectors

print("\nDiagonalized Matrix:\n")
print(mult)

print('Inverse of P:\n')
print(np.linalg.inv(eigenvectors))

mult = np.linalg.inv(eigenvectors) @ non_herm_matrix @ eigenvectors

print("\nDiagonalized Matrix using Inverse:\n")
print(mult)