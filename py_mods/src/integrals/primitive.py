import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

@dataclass
class Primitive:
    """
    Gaussian primitive basis function.
    
    Attributes
    ----------
    R : NDArray[np.float64]
        Center coordinates (3D vector)
    exp : float
        Gaussian exponent
    angular_momentum : int
        Angular momentum quantum number
    norm : float
        Normalization coefficient
    """
    def __init__(self, R: NDArray[np.float64], exp: float,  angular_momentum: int, norm: float):
        self.exp = float(exp)
        self.R = np.asarray(R, dtype=np.float64)
        self.angular_momentum = angular_momentum
        self.norm = float(norm)
        
        if self.R.shape != (3,):
            raise ValueError("R must be a 3D vector")
        

