from typing import Union
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class _primitive_KUSCFContext:
    n_bas: int
    nL: int
    nS: int
    S: NDArray[np.float64]
    T: NDArray[np.complex128]
    V: NDArray[np.float64]
    W: NDArray[np.float64]
    eri_classess: NDArray[np.float64]
    n_electrons: int
    H_core: NDArray[np.complex128]

    # Optional
    occupation: Union[int, NDArray[np.uint8], None] = None
