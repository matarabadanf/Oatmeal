import numpy as np
from numpy.typing import NDArray


def IDX2DC(i: int, j: int, i_s: int, j_s: int) -> int:
    """
    Obtain the flat index of a $(i_s, j_s)$ matrix at $(i, j)$, assuming C-style ordering (row-major).

    Parameters
    ----------
    i : int
        Row index.
    j : int
        Column index.
    i_s : int
        Size of rows.
    j_s : int
        Size of columns.

    Returns
    -------
    int
        Flat index.

    """
    return i * j_s + j


def IDX3DC(i: int, j: int, k: int, i_s: int, j_s: int, k_s: int) -> int:
    """
    Obtain the flat index of a (i_s, j_s, k_s) tensor at (i, j, k), assuming C-style ordering (row-major).

    Parameters
    ----------
    i : int
        First index.
    j : int
        Second index.
    k : int
        Third index.
    i_s : int
        Size of first dimension.
    j_s : int
        Size of second dimension.
    k_s : int
        Size of third dimension.

    Returns
    -------
    int
        Flat index.

    """
    return i * j_s * k_s + j * k_s + k


def IDX4DC(
    i: int, j: int, k: int, l: int, i_s: int, j_s: int, k_s: int, l_s: int
) -> int:
    """
    Obtain the flat index of a (i_s, j_s, k_s, l_s) tensor at (i, j, k, l), assuming C-style ordering (row-major).

    Parameters
    ----------
    i : int
        First index.
    j : int
        Second index.
    k : int
        Third index.
    l : int
        Fourth index.
    i_s : int
        Size of first dimension.
    j_s : int
        Size of second dimension.
    k_s : int
        Size of third dimension.
    l_s : int
        Size of fourth dimension.

    Returns
    -------
    int
        Flat index.

    """
    return i * j_s * k_s * l_s + j * k_s * l_s + k * l_s + l


def IDX5DC(
    i: int,
    j: int,
    k: int,
    l: int,
    m: int,
    i_s: int,
    j_s: int,
    k_s: int,
    l_s: int,
    m_s: int,
) -> int:
    """
    Obtain the flat index of a (i_s, j_s, k_s, l_s, m_s) tensor at (i, j, k, l, m), assuming C-style ordering (row-major).

    Parameters
    ----------
    i : int
        First index.
    j : int
        Second index.
    k : int
        Third index.
    l : int
        Fourth index.
    m : int
        Fifth index.
    i_s : int
        Size of first dimension.
    j_s : int
        Size of second dimension.
    k_s : int
        Size of third dimension.
    l_s : int
        Size of fourth dimension.
    m_s : int
        Size of fifth dimension.

    Returns
    -------
    int
        Flat index.

    """
    return i * j_s * k_s * l_s * m_s + j * k_s * l_s * m_s + k * l_s * m_s + l * m_s + m


def IDXnDC(indices: NDArray[np.int64], sizes: NDArray[np.int64], ndim: int) -> int:
    """
    Obtain the flat index of a (sizes[0], sizes[1], ..., sizes[ndim-1]) tensor at (indices[0], indices[1], ..., indices[ndim-1]), assuming C-style ordering (row-major).

    Parameters
    ----------
    indices : NDArray[np.int64]
        Indices of the tensor.
    sizes : NDArray[np.int64]
        Sizes of the tensor.
    ndim : int
        Number of dimensions.

    Returns
    -------
    int
        Flat index.

    """
    assert len(indices) == len(sizes) == ndim

    final_index = 0

    for i in range(ndim - 1):
        product = 1
        for j in range(i + 1, ndim):
            product *= sizes[j]
        final_index += indices[i] * product
    final_index += indices[-1]

    return final_index


def IDX2DF(i: int, j: int, i_s: int, j_s: int) -> int:
    """
    Obtain the flat index of a (i_s, j_s) matrix at (i, j), assuming Fortran-style ordering (column-major).

    Parameters
    ----------
    i : int
        Row index.
    j : int
        Column index.
    i_s : int
        Size of rows.
    j_s : int
        Size of columns.

    Returns
    -------
    int
        Flat index.

    """
    return j * i_s + i


def IDX3DF(i: int, j: int, k: int, i_s: int, j_s: int, k_s: int) -> int:
    """
    Obtain the flat index of a (i_s, j_s, k_s) tensor at (i, j, k), assuming Fortran-style ordering (column-major).

    Parameters
    ----------
    i : int
        First index.
    j : int
        Second index.
    k : int
        Third index.
    i_s : int
        Size of first dimension.
    j_s : int
        Size of second dimension.
    k_s : int
        Size of third dimension.

    Returns
    -------
    int
        Flat index.

    """
    return i + j * i_s + k * i_s * j_s


def IDX4DF(
    i: int, j: int, k: int, l: int, i_s: int, j_s: int, k_s: int, l_s: int
) -> int:
    """
    Obtain the flat index of a (i_s, j_s, k_s, l_s) tensor at (i, j, k, l), assuming Fortran-style ordering (column-major).

    Parameters
    ----------
    i : int
        First index.
    j : int
        Second index.
    k : int
        Third index.
    l : int
        Fourth index.
    i_s : int
        Size of first dimension.
    j_s : int
        Size of second dimension.
    k_s : int
        Size of third dimension.
    l_s : int
        Size of fourth dimension.

    Returns
    -------
    int
        Flat index.

    """
    return i + j * i_s + k * i_s * j_s + l * i_s * j_s * k_s


def IDX5DF(
    i: int,
    j: int,
    k: int,
    l: int,
    m: int,
    i_s: int,
    j_s: int,
    k_s: int,
    l_s: int,
    m_s: int,
) -> int:
    """
    Obtain the flat index of a (i_s, j_s, k_s, l_s, m_s) tensor at (i, j, k, l, m), assuming Fortran-style ordering (column-major).

    Parameters
    ----------
    i : int
        First index.
    j : int
        Second index.
    k : int
        Third index.
    l : int
        Fourth index.
    m : int
        Fifth index.
    i_s : int
        Size of first dimension.
    j_s : int
        Size of second dimension.
    k_s : int
        Size of third dimension.
    l_s : int
        Size of fourth dimension.
    m_s : int
        Size of fifth dimension.

    Returns
    -------
    int
        Flat index.

    """
    return i + j * i_s + k * i_s * j_s + l * i_s * j_s * k_s + m * i_s * j_s * k_s * l_s


def IDXnDF(indices: NDArray[np.int64], sizes: NDArray[np.int64], ndim: int) -> int:
    """
    Obtain the flat index of a (sizes[0], sizes[1], ..., sizes[ndim-1]) tensor at (indices[0], indices[1], ..., indices[ndim-1]), assuming Fortran-style ordering (column-major).

    Parameters
    ----------
    indices : NDArray[np.int64]
        Indices of the tensor.
    sizes : NDArray[np.int64]
        Sizes of the tensor.
    ndim : int
        Number of dimensions.

    Returns
    -------
    int
        Flat index.

    """
    assert len(indices) == len(sizes) == ndim

    final_index = 0

    for i in range(ndim - 1):
        product = 1
        for j in range(i + 1, ndim):
            product *= sizes[j]
        final_index += indices[i] * product
    final_index += indices[-1]

    return final_index


if __name__ == "__main__":
    pass
