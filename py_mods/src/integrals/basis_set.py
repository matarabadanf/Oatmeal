from dataclasses import dataclass
import numpy as np
from py_mods.src.integrals.uncontracted import N_const
from py_mods.src.integrals.primitive import Primitive

@dataclass
class Contracted:
    """Represents a contracted Gaussian function."""
    n_primitives: int
    angular_momentum: int
    normalization_constants: list[float]
    primitives: list[Primitive]
    c_coeff: list[float]

    def __init__(self, R: np.ndarray, exps: list[float], d_coeff: list[float], angular_momentum: int) -> None:
        """
        Parameters
        ----------
        R : np.ndarray
            Center of the basis function (length 3).
        exps : List[float]
            Gaussian exponents (a_i) for each primitive.
        d_coeff : List[float]
            Contraction coefficients (d_i) for each primitive.
        angular_momentum : int
            Total angular momentum l.
        """
        assert len(exps) == len(d_coeff), 'Number of exponents and expansion coefficients must be the same.'

        self.n_primitives = len(exps)
        self.angular_momentum = angular_momentum
        self.d_coeff = d_coeff

        self.normalization_constants = [1.0 for _ in exps]

        # Create Primitive instances
        self.primitives = [
            Primitive(R=np.array(R, dtype=float),
                      exp=exp,
                      angular_momentum=angular_momentum,
                      norm=norm)
            for exp, norm in zip(exps, self.normalization_constants)
        ]

        # enforce normalization in the inner primitives for later function calls
        self.normalization_constants = np.array([N_const(basis) for basis in self.primitives])
        for i, _ in enumerate(self.primitives):
            self.primitives[i].norm = self.normalization_constants[i]

        # define the linear expansion coefficients 
        self.c_coeff = np.array([self.d_coeff[i] * self.normalization_constants[i] for i in range(self.n_primitives)])



def normalize(basis: Primitive) -> None:
    """
    Calculate and assign the normalization constant of a primitive. 
    
    Parameters
    ------
    basis : Primitive
        First primitive; must provide attributes:
          - R : array-like of length 3, center coordinates (R_x, R_y, R_z)
          - exp : float, Gaussian exponent (alpha)
          - angular_momentum : int, total angular momentum (l)
    
    Returns
    ------
        None
    """
    norm = N_const(basis)
    basis.normalization_constant = norm
