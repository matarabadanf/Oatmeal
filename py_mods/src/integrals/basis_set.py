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

    def __init__(self, R: np.ndarray, exps: list[float], c_coeff: list[float], angular_momentum: int) -> None:
        """
        Parameters
        ----------
        R : np.ndarray
            Center of the basis function (length 3).
        exps : List[float]
            Gaussian exponents (a_i) for each primitive.
        c_coeff : List[float]
            Contraction coefficients (d_i) for each primitive.
        angular_momentum : int
            Total angular momentum l.
        """
        self.n_primitives = len(exps)
        self.angular_momentum = angular_momentum
        self.c_coeff = c_coeff

        self.normalization_constants = [1.0 for _ in exps]

        # Create Primitive instances
        self.primitives = [
            Primitive(R=np.array(R, dtype=float),
                      exp=exp,
                      angular_momentum=angular_momentum,
                      normalization_constant=norm)
            for exp, norm in zip(exps, self.normalization_constants)
        ]

        for i, prim in enumerate(self.primitives):
            N_a = N_const(prim)
            prim.normalization_constant = N_a
            self.normalization_constants[i] = N_a
            self.c_coeff[i] = N_a * c_coeff[i]


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
