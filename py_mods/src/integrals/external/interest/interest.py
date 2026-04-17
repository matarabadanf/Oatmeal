from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from numpy.ctypeslib import ndpointer

import ctypes as ct

from py_mods.src.integrals.GTO import GTO, create_GTO
from py_mods.src.integrals.CGTO import Eri_GTO_tensor

import pathlib

_current_dir = pathlib.Path(__file__).parent.resolve()

_buffer_size = 10000

try:
    _lib = ct.CDLL(f"{_current_dir}/src/inter_py_lib.so")
except OSError:
    raise RuntimeError(
        f"The inter_py_lib.so has not been found in {_current_dir}/src/. \
        Please recompile from the {_current_dir}/src/ folder with gfortran \
        -shared -fPIC *.f90 -o inter_py_lib.so (or other compiler)."
    )

_lib.interest_prim_py.argtypes = [
    ct.c_double,                                                # factor
    ndpointer(dtype=np.float64, ndim=1, shape=(_buffer_size,), flags="C_CONTIGUOUS"),         # fint_inp

    ct.c_int,                                                   # la_inp
    ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,

    ct.c_int,                                                   # lb_inp
    ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,

    ct.c_int,                                                   # lc_inp
    ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,

    ct.c_int,                                                   # ld_inp
    ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
]
_lib.interest_prim_py.restype = None

def interest_shell_tensor(
    gto1: GTO, gto2: GTO, gto3: GTO, gto4: GTO, working_array
) -> None:
    fact =  1

    la = gto1.total_L + 1
    lb = gto2.total_L + 1
    lc = gto3.total_L + 1
    ld = gto4.total_L + 1

    ax, ay, az = gto1.R
    bx, by, bz = gto2.R
    cx, cy, cz = gto3.R
    dx, dy, dz = gto4.R

    a_exp = gto1.exp 
    b_exp = gto2.exp 
    c_exp = gto3.exp 
    d_exp = gto4.exp 

    _lib.interest_prim_py(
        fact, working_array,
        la, a_exp, ax, ay, az, 1,
        lb, b_exp, bx, by, bz, 1,
        lc, c_exp, cx, cy, cz, 1,
        ld, d_exp, dx, dy, dz, 1,
    )

_buffer1 = np.zeros(_buffer_size)

def interest_full_tensor(gto_list: List[GTO], buffer_1 = None) -> NDArray[np.float64]:
    projections = [len(primitive.l_projections) for primitive in gto_list]
    shell_start = [sum(projections[0:i]) for i in range(len(projections))]
    tensor_size = sum(projections)

    eri_tensor = np.zeros([tensor_size for _ in range(4)])

    # We chose this lkji order just because DIRAC's code does it too. To be fair I am not
    # sure I understand the need for this ordering. Also, The internal transpositions on
    # Interest.
    for L, L_gto in enumerate(gto_list):  # l
        L_size, L_start, L_end, L_consts = _unpack_primitive(projections, shell_start, L, L_gto)

        for K, K_gto in enumerate(gto_list):  # k
            K_size, K_start, K_end, K_consts = _unpack_primitive(projections, shell_start, K, K_gto)

            for J, J_gto in enumerate(gto_list):  # j
                J_size, J_start, J_end, J_consts = _unpack_primitive(projections, shell_start, J, J_gto)
                
                for I, I_gto in enumerate(gto_list):  # i
                    I_size, I_start, I_end, I_consts = _unpack_primitive(projections, shell_start, I, I_gto)

                    shell_total_size = I_size * J_size * K_size * L_size
                    _buffer1[:shell_total_size] = 0.0

                    interest_shell_tensor(K_gto, L_gto, I_gto, J_gto, _buffer1)

                    tmp = _buffer1[:shell_total_size].reshape(
                        (L_size, K_size, J_size, I_size)
                    )

                    # So since what we have built is eri[K,L,I,J], we need to transpose to get IJKL
                    block = tmp.transpose(3, 2, 1, 0)

                    # And now that we are on IJKL, we just operate all normally with this indexing.
                    # We contract the normalization
                    norm_tensor = np.einsum(
                        "i,j,k,l->ijkl", I_consts, J_consts, K_consts, L_consts
                    )
                    block *= norm_tensor

                    # And fill the block
                    eri_tensor[
                        I_start:I_end, J_start:J_end, K_start:K_end, L_start:L_end
                    ] = block

    eri = eri_tensor
    return eri


def _unpack_primitive(projections, shell_start, index, gto):
    L_size = projections[index]
    L_start = shell_start[index]
    L_end = shell_start[index] + L_size
    L_consts = gto.normalization_constants
    return L_size, L_start, L_end, L_consts
