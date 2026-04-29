import ctypes as ct
import numpy as np
from numpy.ctypeslib import ndpointer

lib = ct.CDLL("./inter_py_lib.so")

lib.interest_prim_py.argtypes = [
    ct.c_double,                                                # factor
    ndpointer(dtype=np.float64, ndim=1, shape=(1000,)),         # fint_inp

    ct.c_int,                                                   # la_inp
    ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,

    ct.c_int,                                                   # lb_inp
    ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,

    ct.c_int,                                                   # lc_inp
    ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,

    ct.c_int,                                                   # ld_inp
    ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
]
lib.interest_prim_py.restype = None

fint = np.zeros(1000, dtype=np.float64)
fact =  1

la = 1
lb = 1 
lc = 1 
ld = 1 

lib.interest_prim_py(
    fact, fint,
    la, 1, 0.0, 0.0, 0.0, 1.42541094,
    lb, 1, 0.0, 0.0, 0.0, 1.42541094,
    lc, 1, 0.0, 0.0, 1.4, 1.42541094,
    ld, 1, 0.0, 0.0, 1.4, 1.42541094,
)

projs = [int((l) * (l + 1) / 2) for l in [la, lb, lc, ld]]

tot_projs = int(projs[0] * projs[1] * projs[2] * projs[3])

print(fint[0:tot_projs].reshape(projs[0], projs[1], projs[2], projs[3]))
