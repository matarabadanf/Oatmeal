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

        consts = []

        for primitive in self.primitives:
            red_projs = _reduced_projections(primitive.angular_momentum) # _reduced_projections(primitive.angular_momentum)
            for index, proj in enumerate(red_projs):
                const =  N_const(primitive, proj) 
                consts.append(const) 
                primitive.norm[index] = const

        # enforce normalization in the inner primitives for later function calls
        self.normalization_constants = np.array(consts).reshape(self.n_primitives, -1).T

        # we define the c_coefficients for the different internal angular momentum 
        # posibilities. For example, in the case of l = 2, there are two 
        # cartesian types of combinations: [2,0,0] and [1,1,0]. Therefore we need 
        # both to define correctly the overlaps contractions and so on 
        self.c_coeff = self.normalization_constants * self.d_coeff

        # this array will help because later when we calculate matrix elements
        # we can map the mu nu to the expansion coefficient. 
        self.coeff_guide = _coeff_indices(self.angular_momentum)


def _coeff_indices(l: int) -> list[int]:
    if l == 0:
        return [0]
    elif l == 1:
        return [0,0,0]
    elif l == 2: 
        return [0,0,0,1,1,1]
    elif l == 3:
        return [0,0,0,1,1,1,1,1,1,2]

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

def _reduced_projections(l):
    if l == 0:
        return [[0,0,0]]
    elif l == 1:
        return [[1,0,0]]
    elif l == 2:
        return [[2,0,0], [1,1,0]]
    elif l == 3:
        return [[3,0,0], [2,1,0], [1,1,1]]
    else:
        return []

if __name__ == '__main__':
    pass 
