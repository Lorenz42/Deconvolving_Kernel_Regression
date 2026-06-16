# This script plots the deconvolving kernel for multiple values of $\sigma$ and $h=1$

import os
import numpy as np
import matplotlib.pyplot as plt
from deconvolving_kernel.kernels import deconvolving_kernel_rectangular_support

query_points = np.linspace(-10, 10, 1000)[:, np.newaxis]   # Shape (M, D) = (1000, 1)
data_points = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])                            # Shape (N, D) = (1, 1)
small_covariance_values = np.array([[[0.25]], [[0.5]], [[0.75]],[[1.0]], [[1.25]],[[1.5]]])     
large_covariance_values = np.array([[[1]], [[2]], [[3]],[[4]], [[5]],[[6]]])              # Shape (N, D, D) = (1, 1, 1)
bandwidth = 1
bandwidth_scaling = False
rescaled_norm = False
eigenvalue_calculation = True

n = data_points.shape[0]

# Set the outputfolder to the figures folder of the project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_folder = os.path.join(project_root, "figures")
os.makedirs(output_folder, exist_ok=True)

kernel_values_small = deconvolving_kernel_rectangular_support(query_points,
                                                        data_points,
                                                        small_covariance_values,
                                                        bandwidth,
                                                        bandwidth_scaling,
                                                        rescaled_norm,
                                                        eigenvalue_calculation)

kernel_values_large = deconvolving_kernel_rectangular_support(query_points,
                                                        data_points,
                                                        large_covariance_values,
                                                        bandwidth,
                                                        bandwidth_scaling,
                                                        rescaled_norm,
                                                        eigenvalue_calculation)


# Plot configuration
plt.rcParams.update({
    "text.usetex": True,            # Use LaTeX to render all text
    "font.family": "serif",         # Match standard LaTeX serif font
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 11,           # Font size for x and y labels
    "font.size": 10,                # Base font size
    "legend.fontsize": 9,           # Font size for legends
    "xtick.labelsize": 9,           # Font size for tick labels
    "ytick.labelsize": 9,
    "figure.titlesize": 12
})
fig_width_inches = 5.5
fig_height_inches = fig_width_inches * 0.618  # Golden ratio
fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
colors = plt.cm.viridis(np.linspace(0, 0.8, n))

# --- 4. Plotting Loop ---
for i in range(n):
    # Using raw strings (r"...") for clean LaTeX rendering in the label
    label = rf"$\sigma^2 = {small_covariance_values[i,0,0]:.2f}$" 
    ax.plot(
        query_points, 
        kernel_values_small[:, i], 
        label=label, 
        color=colors[i], 
        linewidth=1.5
    )
ax.set_xlabel(r"$x$")  
ax.set_ylabel(r"$K_{U, 1}(x)$") 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
ax.legend(loc="upper right", frameon=False)
plt.tight_layout()
output_path = os.path.join(output_folder, "plot_deconvolving_kernel_different_variances_small.pdf")
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close(fig)

# --- Start a second, independent figure for the large-variance kernels ---
fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
for i in range(n):
    # Using raw strings (r"...") for clean LaTeX rendering in the label
    label = rf"$\sigma^2 = {large_covariance_values[i,0,0]:.2f}$"
    ax.plot(
        query_points, 
        kernel_values_large[:, i], 
        label=label, 
        color=colors[i], 
        linewidth=1.5
    )
ax.set_xlabel(r"$x$")  
ax.set_ylabel(r"$K_{U, 1}(x)$") 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
ax.legend(loc="upper right", frameon=False)
plt.tight_layout()
output_path = os.path.join(output_folder, "plot_deconvolving_kernel_different_variances_large.pdf")
plt.savefig(output_path, bbox_inches='tight', dpi=300)