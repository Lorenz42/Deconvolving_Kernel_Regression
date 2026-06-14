# Please plot the 3 kernels: deconvolving_kernel_rectangular_support, epanechnikov_kernel and deconvolving_kernel_round_support for the following parameter
#
# I want 3 plots that show the 1000 kernel weights corresponding to the 1000 differences
# and then I would like to have one plot where I can see all the functions in one plot.
# The plots should be all in one figure and this figure should be stored in the folder /home/lh9809/Images_from_deconvolving_kernel_package

import os

import numpy as np
import matplotlib.pyplot as plt

from .kernels import deconvolving_kernel_rectangular_support, epanechnikov_kernel
from .fourier_transformation_kernels import deconvolving_kernel_round_support


# ----------------------------------------------------------------------------------
# Parameters
#
# The 1000 query points lie on a line and there is a single data point at the origin,
# so this is a one dimensional (D = 1) setup. The covariance matrix is therefore the
# 1x1 identity (unit variance), matching the diagonal of the identity given in the
# task description.

query_points = np.linspace(-10, 10, 1000)[:, np.newaxis]   # Shape (M, D) = (1000, 1)
data_points = np.array([[0.0]])                            # Shape (N, D) = (1, 1)
covariance_matrices = np.array([[[1.0]]])                  # Shape (N, D, D) = (1, 1, 1)
bandwidth = 2
bandwidth_scaling = False
rescaled_norm = False
eigenvalue_calculation = True
discrete_fourier_transformation_grid_size = 1000

output_folder = "/home/lh9809/Images_from_deconvolving_kernel_package"


# ----------------------------------------------------------------------------------
# Evaluate the kernels
#
# Every kernel returns an (M, N) = (1000, 1) array of kernel weights. Since there is a
# single data point we take column 0 to get the 1000 weights for the 1000 differences.

kernels = {
    "deconvolving_kernel_rectangular_support": deconvolving_kernel_rectangular_support,
    "epanechnikov_kernel": epanechnikov_kernel,
    "deconvolving_kernel_round_support": deconvolving_kernel_round_support,
}

# The x axis shows the differences query_point - data_point (the data point is at 0).
differences = (query_points - data_points[0]).ravel()

kernel_weights = {}
for name, kernel in kernels.items():
    values = kernel(query_points,
                    data_points,
                    covariance_matrices,
                    bandwidth,
                    bandwidth_scaling,
                    rescaled_norm,
                    eigenvalue_calculation,
                    discrete_fourier_transformation_grid_size)
    kernel_weights[name] = values[:, 0]


# ----------------------------------------------------------------------------------
# Plot: three individual panels plus one combined panel, all in one figure

colors = ["tab:blue", "tab:orange", "tab:green"]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.ravel()

# Three individual plots
for ax, color, (name, weights) in zip(axes[:3], colors, kernel_weights.items()):
    ax.plot(differences, weights, color=color)
    ax.set_title(name)
    ax.set_xlabel("difference  (query_point - data_point)")
    ax.set_ylabel("kernel weight")
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.3)

# Combined plot with all three kernels
for color, (name, weights) in zip(colors, kernel_weights.items()):
    axes[3].plot(differences, weights, color=color, label=name)
axes[3].set_title("All kernels")
axes[3].set_xlabel("difference  (query_point - data_point)")
axes[3].set_ylabel("kernel weight")
axes[3].axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
axes[3].grid(True, alpha=0.3)
axes[3].legend()

fig.suptitle(f"Kernel weights (bandwidth = {bandwidth})", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.97))

os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "kernels.png")
fig.savefig(output_path, dpi=150)
print(f"Figure saved to {output_path}")
