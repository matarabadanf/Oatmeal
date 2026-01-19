import numpy as np
from py_mods.src.integrals.uncontracted import S_3D
from py_mods.src.integrals.primitive import Primitive


def normalize_primitive(prim: Primitive) -> None:
    overlap = S_3D(prim, prim)
    normalization_constant = 1 / np.sqrt(overlap)
    prim.norm = normalization_constant
