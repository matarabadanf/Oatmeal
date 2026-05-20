import numpy as np
from numpy.typing import NDArray

def quaternion_to_matrix(c0: NDArray[np.float64], c1: NDArray[np.float64], c2: NDArray[np.float64], c3: NDArray[np.float64]) -> NDArray[np.complex128]:
    Z = np.zeros_like(c0, dtype=np.complex128)
    Z.real = c0
    Z.imag = c1
    W = np.zeros_like(c0, dtype=np.complex128)
    W.real = c2
    W.imag = c3
    block_size = len(c1[0])
    Q = np.zeros([2 * block_size, 2 * block_size], dtype=np.complex128)
    Q[0:block_size, 0:block_size] = Z
    Q[block_size:, block_size:] = np.conj(Z)
    Q[0:block_size, block_size:] = W
    Q[block_size:, 0:block_size] = -W
    return Q
