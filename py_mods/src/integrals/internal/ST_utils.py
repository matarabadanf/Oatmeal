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
    S_ij_mat : numpy.ndarray of shape (max_dim, max_dim)
        Matrix containing the overlap integrals for angular momentum pairs
        up to (i, j).

    Notes
    -----
    Implements the recursive relations described in Helgaker, Jørgensen &
    Olsen, *Molecular Electronic-Structure Theory*, Ch. 9.
    """
    max_dim = max(i,j) + 1

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
    # if i*j == 0 and i != j:
    #     return 0
    
    result = float(obara_saika_bottom_up(Ax, Bx, a, b, i, j)[i][j]) # TODO: something fishy here

    return result


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

if __name__ == '__main__':
    pass 