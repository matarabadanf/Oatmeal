import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Union, List
import numpy as np
from py_mods.src.SCF.CS_SCF_types import CSUHFResults


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


def _plot_UHF_MO_energies(
    e_alpha: NDArray[np.complex128],
    e_beta: NDArray[np.complex128],
    homo_index: int,
    n_alpha: int,
    n_beta: int,
    units: str = "Hartree",
    ylim: Union[None, List] = None,
):
    lumo_index = homo_index + 1
    ylim_str = ""

    pos_alpha = 1
    pos_beta = 1.5
    line_width = 0.4

    occ_alpha_eners = e_alpha[:n_alpha]
    occ_beta_eners = e_beta[:n_beta]

    x_alpha_occ = np.zeros_like(occ_alpha_eners) + pos_alpha
    x_beta_occ = np.zeros_like(occ_beta_eners) + pos_beta

    plt.scatter(x_alpha_occ, occ_alpha_eners, marker="^")
    plt.scatter(x_beta_occ, occ_beta_eners, marker="v")

    for e in e_alpha:
        plt.hlines(
            y=e,
            xmin=pos_alpha - line_width / 2,
            xmax=pos_alpha + line_width / 2,
            colors="blue",
            linewidth=2,
        )

    for e in e_beta:
        plt.hlines(
            y=e,
            xmin=pos_beta - line_width / 2,
            xmax=pos_beta + line_width / 2,
            colors="red",
            linewidth=2,
        )

    plt.ylabel(f"Energy {units}")
    plt.xticks([pos_alpha, pos_beta], ["Alpha", "Beta"])

    plt.tick_params(axis="x", length=0)

    if ylim is not None:
        ylim_str = f"({str(ylim).replace("'", '').replace('"', '')})"

        if len(ylim) != 2:
            raise ValueError("ylim must be a list of two elements: [ymin, ymax]")

        if isinstance(ylim[0], str):
            if ylim[0] == "core":
                ylim_index = 0

            elif "HOMO" in ylim[0]:
                ylim_index = eval(ylim[0].replace("HOMO", f"{homo_index}"))

            elif "LUMO" in ylim[0]:
                ylim_index = eval(ylim[0].replace("LUMO", f"{lumo_index}"))

            ylim[0] = float(min(e_alpha[ylim_index].real, e_beta[ylim_index].real))
            ylim[0] += ylim[0] * 0.1

        if isinstance(ylim[1], str):
            if "HOMO" in ylim[1]:
                ylim_index = eval(ylim[1].replace("HOMO", f"{homo_index}"))

            elif "LUMO" in ylim[1]:
                ylim_index = eval(ylim[1].replace("LUMO", f"{lumo_index}"))

            ylim[1] = float(min(e_alpha[ylim_index].real, e_beta[ylim_index].real))
            ylim[1] += ylim[1] * 0.1

        ylim_dist = abs(ylim[1] - ylim[0])

        plt.ylim(ylim)

    plt.title(f"MO energies {ylim_str}")

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.show()


def plot_UHF_MO_energies(
    unf_results: CSUHFResults, units: str = "Hartree", ylim: Union[None, List] = None
):

    _plot_UHF_MO_energies(
        unf_results.e_alpha,
        unf_results.e_alpha,
        unf_results.homo_index,
        int(unf_results.n_alpha.real),
        int(unf_results.n_beta.real),
        units=units,
        ylim=ylim,
    )


def plot_theta_traj(energies):
    """
    Plot complex energy trajectory.

    Parameters
    ----------
    energies : sequence
        Complex energies.
    """
    reals = [energy.real for energy in energies]
    imags = [energy.imag for energy in energies]
    plt.plot(reals, imags, marker="o")
    plt.xlabel("Re(E)")
    plt.ylabel("Im(E)")
    plt.title("Complex Scaled RHF Energy vs Theta")
    plt.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    plt.ticklabel_format(style="sci")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_theta_orbital_energies(energies, theta, xrange=[0, 0]):
    """
    Scatter plot orbital energies.

    Parameters
    ----------
    energies : sequence
        Orbital energies.
    theta : float
        Current angle.
    xrange : list
        X-axis limits.
    """
    reals = [energy.real for energy in energies]
    imags = [energy.imag for energy in energies]
    if xrange != [0, 0]:
        plt.xlim(xrange)
        reals = [re for re in reals if re < xrange[1]]
        imags = imags[0 : len(reals)]

    plt.scatter(reals, imags, marker="o")
    plt.xlabel("Re(Orbital Energies)")
    plt.ylabel("Im(Orbital Energies)")
    plt.ticklabel_format(style="sci")
    plt.title(f"Complex Scaled RHF Orbital Energies at Theta={theta}")
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    pass
