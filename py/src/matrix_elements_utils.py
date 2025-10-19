import numpy as np
from numpy.typing import NDArray

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
    angular_momentum_matrix : numpy.ndarray of shape [max_dim, max_dim]
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

    angular_momentum_matrix = np.zeros([max_dim, max_dim])

    X_ab = (Bx-Ax)
    p = a + b
    X_pa = b/p * X_ab
    X_pb = -a/p * X_ab


    angular_momentum_matrix[0,0] =  (np.pi/p)**0.5 * np.exp(-(a * b)/p * X_ab**2)

    # Compute the [i,0] entries 
    for i in range(1, max_dim):
        angular_momentum_matrix[i,0] =  X_pa * angular_momentum_matrix[i-1,0] +  1/(2*p)*((i-1) * angular_momentum_matrix[i-2,0])

    # Compute the [0,j] entries 
    for j in range(1, max_dim):
        angular_momentum_matrix[0,j] =  X_pb * angular_momentum_matrix[0,j-1] +  1/(2*p)*((j-1) * angular_momentum_matrix[0,j-2])

    # Compute the rest of entries using recursion
    for total in range(1, max_dim):
        for i in range(total, max_dim):
            angular_momentum_matrix[i,total] =  X_pa * angular_momentum_matrix[i-1,total] +  1/(2*p)*((i-1) * angular_momentum_matrix[i-2,total] + (total) * angular_momentum_matrix[i-1,total-1])
        for j in range(total, max_dim):
            angular_momentum_matrix[total,j] =  X_pb * angular_momentum_matrix[total, j-1] +  1/(2*p)*((total) * angular_momentum_matrix[total-1,j-1] +(j-1) * angular_momentum_matrix[total,j-2])

    return angular_momentum_matrix