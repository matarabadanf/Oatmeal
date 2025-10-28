import ctypes
from pathlib import Path
from py_mods.src._interfaces._c_importer import get_lib_ext

lib_folder = Path(__file__).parent.parent / "c_libs"
ext = get_lib_ext()

_gaussian_utils = ctypes.CDLL(str(lib_folder / f"libgaussians{ext}"))

_gaussian_utils.cartesian_gaussian.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]
_gaussian_utils.cartesian_gaussian.restype = ctypes.c_double

def gaussian_from_c(ii: int, A_x: float, a: float, x: float) -> float:
    val = _gaussian_utils.cartesian_gaussian(ii, A_x, a, x)
    return val

w = (gaussian_from_c(2, 0, 0.5, 1.7))
print(w)