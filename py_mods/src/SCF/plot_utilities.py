import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Literal, Tuple, Union
import numpy as np

# --- Plot Utilities ---
def plot_map(matrix: Union[NDArray[np.float64], NDArray[np.complex128]], plot_range=None, title=None):
    if matrix.dtype == np.float64:
        _plot_map_real(matrix, plot_range, title)
    elif matrix.dtype == np.complex128:
        _plot_map_imag(matrix, plot_range, title)

def _plot_map_real(matrix, plot_range=None, title=None):

    plt.imshow(matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Value")
    if not isinstance(title, str):
        plt.title("Matrix Heatmap")
    else:
        plt.title(title)
    if range is not None: 
        plt.xlim(plot_range[0])
        plt.ylim(plot_range[1])
    plt.show()

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
    plt.show()
