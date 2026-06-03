from typing import Optional, Tuple, Union, List, Literal

import h5py
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from py_mods.src.algebra.quaternion import quaternion_to_matrix
from py_mods.src.integrals.GTO import create_GTO
from py_mods.src.integrals.UncontractedBasisSet import (
    UncontractedBasisSet,
    create_UncontractedBasisSet,
    ERIs_Uncontracted,
)

c = 137.035999177


def show_h5cont(filename: str) -> List[str]:
    """
    Utility function to display the contents of an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to .h5 file

    Returns
    -------
    availiable_in_h5 : List[str]
        h5 availiable keys.
    """
    availiable_in_h5 = []

    def _show(name, obj):
        if isinstance(obj, h5py.Group):
            availiable_in_h5.append(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            availiable_in_h5.append(f"Dataset: {name} {obj.shape}")

    with h5py.File(f"{filename}", "r") as f:
        f.visititems(_show)

    return availiable_in_h5


def retriangularize(
    array: NDArray[np.float64], n_basis: int, antisymmetric: bool = False
) -> NDArray[np.float64]:
    """
    Convert a packed triangular array back into a full square matrix.

    Parameters
    ----------
    array : NDArray[np.float64]
        Triangular packed array
    n_basis : int
        Size of square matrix
    antisymmetric : bool, optional
        Flag defining if A = -A^T. Defaults to False.

    Returns
    -------
    M : NDArray[np.float64]
        Square matrix reconstructed from the packed array.

    """
    M = np.zeros((n_basis, n_basis), dtype=array.dtype)
    k = 0

    for j in range(n_basis):
        for i in range(j + 1):
            M[i, j] = array[k]
            M[j, i] = array[k] if not antisymmetric else -array[k]
            k += 1
    return M


def plot_trinagular_packed(
    array: NDArray[np.float64],
    n: int,
    antisymmetric: bool = False,
    LC_size: Optional[Union[int, None]] = None,
    title: str = "Packed matrix heatmap",
    cmap: str = "viridis",
    only_LC: bool = False,
) -> NDArray[np.float64]:
    """
    Visualize a packed triangular array as a heatmap.

    Parameters
    ----------
    array : NDArray[np.float64]
        Triangular packed array
    n : int
        Size of square matrix
    antisymmetric : bool, optional
        Flag defining if A = -A^T. Defaults to False.
    LC_size : Optional[Union[int, None]], optional
        Size of the LC basis. If provided, draws lines separating LC and SC blocks. Defaults to None.
    title : str, optional
        Title of the plot. Defaults to "Packed matrix heatmap".
    cmap : str, optional
        Colormap for the heatmap. Defaults to "viridis".
    only_LC : bool, optional
        If True, limits the plot to only the LC block. Defaults to False.

    Returns
    -------
    M : NDArray[np.float64]
        Square matrix reconstructed from the packed array.
    """
    M = retriangularize(array, n, antisymmetric)
    Mplot = M.real if np.iscomplexobj(M) else M

    fig, ax = plt.subplots(figsize=(8, 7))
    if only_LC:
        ax.set_xlim(0, LC_size)
        ax.set_ylim(0, LC_size)

    im = ax.imshow(Mplot, cmap=cmap, origin="upper", aspect="auto")
    fig.colorbar(im, ax=ax, label="Matrix element")

    ax.set_xlabel("AO index")
    ax.set_ylabel("AO index")
    ax.set_title(title)

    if LC_size is not None:
        ax.axvline(x=LC_size - 0.5, color="black", linewidth=1.5)
        ax.axhline(y=LC_size - 0.5, color="black", linewidth=1.5)

    plt.tight_layout()
    plt.show()

    return M


def get_nuc_charge(h5filename: str) -> int:
    """Extract the total nuclear charge from the HDF5 file.

    Parameters
    ----------
    h5filename : str
        Path to the HDF5 file containing the matrices.

    Returns
    -------
    int
        Total nuclear charge.
    """
    with h5py.File(h5filename, "r") as f:
        nuc_charge = np.asarray(f["input/molecule/nuc_charge"][()])

    return int(nuc_charge)


def extract_arrays_from_h5(
    h5filename,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Extracts the necessary arrays from the HDF5 file for constructing the S, V, W, and T matrices.

    Parameters
    ----------
    h5filename : str
        Path to the HDF5 file containing the matrices.

    Returns
    -------
    kinarray : NDArray[np.float64]
        Quaternion Triangular packed relativistic kinetic energy matrix.
    molfield : NDArray[np.float64]
        Triangular packed molecular field matrix.
    overlap : NDArray[np.float64]
        Triangular packed overlap matrix.
    betamatarr : NDArray[np.float64]
        Triangular packed beta matrix.
    fockarra : NDArray[np.float64]
        Quaternion packed one-electron Fock matrix.
    """
    with h5py.File(f"{h5filename}", "r") as f:
        kinarray = np.asarray(f["result/operators/ao_matrices/RELKINEN"][()])
        molfield = np.asarray(f["result/operators/ao_matrices/MOLFIELDTFFT"][()])
        fockarra = np.asarray(f["result/operators/ao_matrices/ONEFOCK"][()])
        betamatarr = np.asarray(f["result/operators/ao_matrices/BETAMAT FFFT"][()])
        overlap = np.asarray(f["result/operators/ao_matrices/OVERLAP TFFT"][()])

    return kinarray, molfield, overlap, betamatarr, fockarra


def unflatten_S_V_W(
    molfield: NDArray[np.float64],
    overlap: NDArray[np.float64],
    betamatarr: NDArray[np.float64],
    n_bas: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert Triangular packed arrays to square matrix forms.

    Parameters
    ----------
    molfield : NDArray[np.float64]
        Triangular packed molecular field matrix.
    overlap : NDArray[np.float64]
        Triangular packed overlap matrix.
    betamatarr : NDArray[np.float64]
        Triangular packed beta matrix.
    n_bas : int
        Number of basis functions.

    Returns
    -------
    S : NDArray[np.float64]
        Square overlap matrix.
    V : NDArray[np.float64]
        Square molecular field matrix.
    W : NDArray[np.float64]
        Square beta matrix scaled by -2*c^2.
    """
    S = retriangularize(overlap, n_bas)
    V = retriangularize(molfield, n_bas)
    W = -retriangularize(betamatarr, n_bas) * c**2 * 2

    return S, V, W


def extend_S_V_W(
    S: NDArray[np.float64], V: NDArray[np.float64], W: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Extend the S, V, and W matrices to 4c form by creating block diagonal matrices.

    Parameters
    ----------
    S : NDArray[np.float64]
        Square overlap matrix.
    V : NDArray[np.float64]
        Square molecular field matrix.
    W : NDArray[np.float64]
        Square beta matrix scaled by -2*c^2.

    Returns
    -------
    S_new : NDArray[np.float64]
        Extended 4c overlap matrix.
    V_new : NDArray[np.float64]
        Extended 4c molecular field matrix.
    W_new : NDArray[np.float64]
        Extended 4c beta matrix.
    """
    dim = 2 * len(S[0])
    dim_half = len(S[0])
    V_new = np.zeros((dim, dim), dtype=np.float64)
    W_new = np.zeros((dim, dim), dtype=np.float64)
    S_new = np.zeros((dim, dim), dtype=np.float64)

    V_new[0:dim_half, 0:dim_half] = V_new[dim_half:, dim_half:] = V
    W_new[0:dim_half, 0:dim_half] = W_new[dim_half:, dim_half:] = W
    S_new[0:dim_half, 0:dim_half] = S_new[dim_half:, dim_half:] = S

    return S_new, V_new, W_new


def unflatten_rel_T(
    kinarray: NDArray[np.float64], n_bas: int
) -> NDArray[np.complex128]:
    """
    Convert quaternion triangular packed rel c(a*p) to full 4c matrix.

    Parameters
    ----------
    kinarray : NDArray[np.float64]
        Quaternion Triangular packed relativistic kinetic energy matrix.
    n_bas : int
        Number of basis functions.

    Returns
    -------
    T : NDArray[np.complex128]
        Full 4c relativistic kinetic energy matrix.
    """
    n_tri = n_bas * (n_bas + 1) // 2

    assert len(kinarray) == 4 * n_tri, (
        f"kinarray length {len(kinarray)} != 4×{n_tri}={4*n_tri} " f"for n_bas={n_bas}"
    )

    c0 = retriangularize(kinarray[0:n_tri], n_bas, True)
    c1 = retriangularize(kinarray[n_tri : n_tri * 2], n_bas, True)
    c2 = retriangularize(kinarray[n_tri * 2 : n_tri * 3], n_bas, True)
    c3 = retriangularize(kinarray[3 * n_tri :], n_bas, True)

    T = quaternion_to_matrix(c0, c1, c2, c3)
    return T


def build_S_V_W_T_from_h5(h5filename: str):
    """
    Build the 4c S, V, W, and T matrices from the HDF5 file.

    Parameters
    ----------
    h5filename : str
        Path to the HDF5 file containing the matrices.

    Returns
    -------
    S : NDArray[np.float64]
        Full 4c overlap matrix.
    V : NDArray[np.float64]
        Full 4c molecular field matrix.
    W : NDArray[np.float64]
        Full 4c beta matrix.
    T : NDArray[np.complex128]
        Full 4c relativistic kinetic energy matrix.
    """
    kinarray, molfield, overlap, betamatarr, _ = extract_arrays_from_h5(h5filename)
    n_bas = int((np.sqrt(1 + 8 * len(overlap)) - 1) / 2)

    S, V, W = unflatten_S_V_W(molfield, overlap, betamatarr, n_bas)

    S, V, W = extend_S_V_W(S, V, W)

    T = unflatten_rel_T(kinarray, n_bas)

    return S, V, W, T


def build_4c_one_Fock_from_h5(h5filename: str, guess: str='barenuc') -> NDArray[np.complex128]:
    """
    Build the full 4c one-electron Fock matrix from the HDF5 file.

    Parameters
    ----------
    h5filename : str
        Path to the HDF5 file containing the matrices.

    Returns
    -------
    Fock_4c : NDArray[np.complex128]
        Full 4c one-electron Fock matrix.

    Notes
    ----
    The current implementation assumes the 
    """

    S, V, W, T = build_S_V_W_T_from_h5(h5filename)

    if guess == "barenuc":
        Fock_4c = T + V + W
    else:
        raise RuntimeError('"barenuc" is the only guess allowed in the current implementation')

    _, _, overlap, _, fockarra = extract_arrays_from_h5(h5filename)

    n_bas = int((np.sqrt(1 + 8 * len(overlap)) - 1) / 2)

    focsquare = fockarra.reshape(4, n_bas, n_bas)  # or manually, but this is convenient
    f0, f1, f2, f3 = focsquare[0], focsquare[1], focsquare[2], focsquare[3]
    Qf = quaternion_to_matrix(f0, f1, f2, f3)

    assert np.all(Fock_4c == Qf)

    return Fock_4c


def extract_basis_data(
    h5filename: str, component: Literal["Large", "Small"] = "Large"
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int16]]:

    if component not in ["Large", "Small"]:
        raise ValueError("Component must be either 'Large' or 'Small'")

    basis_number = "1" if component == "Large" else "2"

    with h5py.File(h5filename, "r") as f:
        R_array = np.asarray(f[f"input/aobasis/{basis_number}/center"][()], dtype=np.float64)
        R_array = R_array.reshape(-1, 3)
        exps_array = np.asarray(f[f"input/aobasis/{basis_number}/exponents"][()], dtype=np.float64)
        l_array = np.asarray(f[f"input/aobasis/{basis_number}/orbmom"][()], dtype=np.int16)

    return R_array, exps_array, l_array


def build_uncontracted_basis_from_checkpoint(h5filename) -> UncontractedBasisSet:
    Ldata = extract_basis_data(h5filename, "Large")
    Sdata = extract_basis_data(h5filename, "Small")

    LC_list = []

    for R, alpha, l in zip(*Ldata):
        gto_instance = create_GTO(R, alpha, l - 1)
        LC_list.append(gto_instance)

    SC_list = []
    for R, alpha, l in zip(*Sdata):
        gto_instance = create_GTO(R, alpha, l - 1)
        SC_list.append(gto_instance)

    total_basis = LC_list + SC_list

    Unc_bas_set = create_UncontractedBasisSet(total_basis)

    return Unc_bas_set


def full_eri_from_checkpoint(h5filename) -> NDArray[np.float64]:
    h_basis = build_uncontracted_basis_from_checkpoint(h5filename)
    eri_tensor = ERIs_Uncontracted(h_basis)

    return eri_tensor
