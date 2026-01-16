from re import I
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Literal, Tuple, Union
import numpy as np


# --- Plot Utilities ---
def plot_map(
    matrix: Union[NDArray[np.float64], NDArray[np.complex128]],
    plot_range=None,
    title=None,
    filename: Union[None, str] = None,
    returnable: bool = False,
):
    assert len(matrix) == len(
        matrix[0]
    ), "Mismatch of matrix dimensions, must be a square matrix"

    if matrix.dtype == np.float64:
        plotObject = _plot_map_real(matrix, plot_range, title)
    elif matrix.dtype == np.complex128:
        plotObject = _plot_map_imag(matrix, plot_range, title)
    else:
        raise TypeError(
            "Type must be Ndarray of shape (n, n) of types float or complex"
        )

    if filename is not None:
        plotObject.savefig(filename, dpi=600)

    if not returnable:
        plotObject.show()
    else:
        return plotObject


def _plot_map_real(matrix, plot_range=None, title=None):

    plt.imshow(matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Value")
    if not isinstance(title, str):
        plt.title("Matrix Heatmap")
    else:
        plt.title(title)
    if plot_range is not None:
        plt.xlim(plot_range[0])
        plt.ylim(plot_range[1])
    return plt


def _plot_map_imag(matrix, plot_range=None, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im_real = axes[0].imshow(matrix.real, cmap="viridis", interpolation="nearest")
    axes[0].set_title("Real")
    fig.colorbar(im_real, ax=axes[0], label="Value")

    im_imag = axes[1].imshow(matrix.imag, cmap="viridis", interpolation="nearest")
    axes[1].set_title("Imag")
    fig.colorbar(im_imag, ax=axes[1], label="Value")

    if title:
        fig.suptitle(title)
    else:
        fig.suptitle("Complex Matrix Heatmap")

    if plot_range is not None:
        for ax in axes:
            ax.set_xlim(plot_range[0])
            ax.set_ylim(plot_range[1])

    plt.tight_layout()
    return plt


def plot_mo_analysis(C1, E1, C2, E2, titles=None):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 6),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [10, 1]},
        sharex="col",
    )

    # axes
    (ax_mat1, ax_mat2), (ax_eig1, ax_eig2) = axes

    # pair data
    data_pairs = [
        (C1, E1, ax_mat1, ax_eig1, titles[0] if titles else "System 1"),
        (C2, E2, ax_mat2, ax_eig2, titles[1] if titles else "System 2"),
    ]

    # global min and max to share scale
    c_min = min(C1.min(), C2.min())
    c_max = max(C1.max(), C2.max())

    # plot in loop both
    for C, E, ax_m, ax_e, title in data_pairs:

        # Heatmap
        im_c = ax_m.imshow(
            C,
            cmap="viridis",
            interpolation="nearest",
            vmin=c_min,
            vmax=c_max,
            aspect="auto",
        )
        ax_m.set_title(title)
        ax_m.set_ylabel("Basis Functions (AO)")
        fig.colorbar(im_c, ax=ax_m, label="Coeff Value")

        # Eigenvalues
        E_reshaped = E[np.newaxis, :]

        im_e = ax_e.imshow(
            E_reshaped, cmap="plasma", interpolation="nearest", aspect="auto"
        )

        ax_e.set_xlabel("Molecular Orbitals (MO)")
        ax_e.set_yticks([])
        ax_e.set_ylabel("E")

        fig.colorbar(im_e, ax=ax_e, label="Energy")

    return fig


def _plot_3_maps_real(matrices, plot_range=None, titles=None):
    """
    Plots 3 matrices side-by-side.

    Parameters
    ----------
    matrices: list of 3 NDArray[np.float64]
        List containing the three matrices to plot.
    plot_range: tuple, optional
        Tuple containing x and y limits for the plots.
    titles: list of str, optional
        List containing titles for each subplot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    if titles is None:
        titles = ["Matrix 1", "Matrix 2", "Matrix 3"]

    for ax, matrix, title in zip(axes, matrices, titles):

        im = ax.imshow(matrix, cmap="viridis", interpolation="nearest")
        fig.colorbar(im, ax=ax, label="Value", fraction=0.046, pad=0.04)

        ax.set_title(title)

        if plot_range is not None:
            ax.set_xlim(plot_range[0])
            ax.set_ylim(plot_range[1])

    return fig


if __name__ == "__main__":
    pass
