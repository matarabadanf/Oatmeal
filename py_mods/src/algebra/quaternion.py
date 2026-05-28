import numpy as np
from numpy.typing import NDArray


def quaternion_to_matrix(
    c0: NDArray[np.float64],
    c1: NDArray[np.float64],
    c2: NDArray[np.float64],
    c3: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """ 
    Transform a quaternion matrix expressed as 4 real matrices (c0, c1, c2, c3) into its complex matrix representation.

    Parameters
    ----------
    c0 : NDArray[np.float64]
        Real part of the quaternion matrix.
    c1 : NDArray[np.float64]
        Coefficient of the i component of the quaternion matrix.
    c2 : NDArray[np.float64]
        Coefficient of the j component of the quaternion matrix.
    c3 : NDArray[np.float64]
        Coefficient of the k component of the quaternion matrix.

    Returns
    -------
    Q : NDArray[np.complex128]
        Complex matrix representation of the quaternion matrix.
    
    Notes
    -----
    - Recall that the complex quaternion representationx matrix satisfies the formula:
      $$Q = \\begin{pmatrix}Z & W \\ -W^* & Z^*}\\end{pmatrix}$$
      where Z = c0 + i c1 and W = c2 + i c3.
    - The resulting matrix Q will have dimensions (2N, 2N) where N is the dimension of the input matrices c0, c1, c2, c3.
    
    """
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
    Q[block_size:, 0:block_size] = -np.conj(W)
    return Q
