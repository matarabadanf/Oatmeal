import numpy as np
from numpy.typing import NDArray

def transformation_matrix(S_munu: np.ndarray) -> np.ndarray:
    """
    Calculate The normalization transformation matrix X.

    Uses symmetric orthogonalization. This is obtaining the matrix S^{-1/2}
    by obtaining the diagonal form of S, s. 
    
    U^{dagger} @ S @ U = s
    
    The diagonal matrix s^{-1/2} is easily computed and to obtain S^{-1/2} we
    use the transformation:
    
    X = S^{-1/2} = U @ s^{-1/2} @ U^{dagger} 

    Parameters
    ------
    S_munu : np.ndarray of square dimensions.
    
    Returns
    ------
    X : np.ndarray of same shape as S_munu
        Transformation matrix X.

    Notes 
    ------
    The operation S^{-1/2} @ S @ S^{-1/2} = Identity must always hold.
    """
    dim = len(S_munu)

    # diagonalize U.T @ S @ U = s
    s, U = np.linalg.eig(S_munu)

    s_root = np.zeros([dim, dim])

    for index, eigenvalue in enumerate(s):
        s_root[index,index] = 1/np.sqrt(eigenvalue)

    X = U @ s_root @ U.T

    return X

if __name__ == "__main__":
    S_sto3g_Li = np.array([
        [ 3.57678642, -0.02023738,  0.        ,  0.        ,  0.        ],
        [-0.02023738,  0.10216337,  0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.31968167,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.31968167,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.31968167]
    ])

    idn = np.identity(5)

    # test 1: successful transformation matrix
    X = transformation_matrix(S_sto3g_Li)
    transformed = X @ S_sto3g_Li @ X      # sould be the identity    
    assert(np.allclose(transformed, idn))