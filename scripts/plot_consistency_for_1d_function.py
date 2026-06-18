import os
import numpy as np
import matplotlib.pyplot as plt
from deconvolving_kernel.kernels import epanechnikov_kernel, deconvolving_kernel_rectangular_support, kernel_regression

def test_function(x_values):
    return np.sin(x_values) + np.cos(3*x_values) + 0.1*x_values   

def generate_data(test_function, n_samples, x_variance, y_variance):
    x_values = np.random.uniform(-np.pi, np.pi, n_samples)
    y_values = test_function(x_values) + np.random.normal(0, np.sqrt(y_variance), n_samples)
    x_values = x_values + np.random.normal(0, np.sqrt(x_variance), n_samples)
    return x_values[:, np.newaxis], y_values

def mse(dataset_1, dataset_2):
    return np.mean((dataset_1 - dataset_2)**2)

def find_best_solution(kernel, test_function, query_points, data_points, y_values_data_points, covariance_matrices, bandwidth_grid):
    # Ravel so the (M,) regression output and the ground truth are compared
    # element-wise instead of broadcasting to an (M, M) array.
    true_solution = test_function(query_points).ravel()
    best_solution = np.array([])
    best_error = 1e+20
    best_bandwidth = -1

    for bandwidth in bandwidth_grid:
        current_solution = kernel_regression(kernel,
                                            query_points,
                                            data_points,
                                            y_values_data_points,
                                            covariance_matrices,
                                            bandwidth)
        if mse(current_solution, true_solution) < best_error:
            best_error = mse(current_solution, true_solution)
            best_bandwidth = bandwidth
            best_solution = current_solution
    return best_solution, best_bandwidth



# Set the outputfolder to the figures folder of the project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_folder = os.path.join(project_root, "figures")
os.makedirs(output_folder, exist_ok=True)




query_points = np.linspace(-np.pi, np.pi, 100)[:, np.newaxis]
x_variance = 0.75
y_variance = 0.1
n_samples_list = [100, 1000, 10000, 100000]
solutions_epanechnikov = []
bandwidths_epanechnikov = []
solutions_deconvolving = []
bandwidths_deconvolving = []
solutions_ground_truth = []
solutions_convolved = []

for n_samples in n_samples_list:
    
    data_points, y_values_data_points =  generate_data(test_function, n_samples, x_variance, y_variance)                        # Shape (N, D) = (1, 1)
    covariance_matrices = np.ones((n_samples, 1, 1)) * x_variance
    bandwidth_grid = np.linspace(0.01, 1, 50)
    bandwidth_grid_epanechnikov = np.linspace(0.2, 1, 50)




    solution_epanechnikov, bandwidth_epanechnikov = find_best_solution(epanechnikov_kernel, 
                                                                        test_function, 
                                                                        query_points, 
                                                                        data_points, 
                                                                        y_values_data_points, 
                                                                        covariance_matrices, 
                                                                        bandwidth_grid_epanechnikov)
    solutions_epanechnikov.append(solution_epanechnikov)
    bandwidths_epanechnikov.append(bandwidth_epanechnikov)

    solution_deconvolving, bandwidth_deconvolving = find_best_solution(deconvolving_kernel_rectangular_support, 
                                                                        test_function, 
                                                                        query_points, 
                                                                        data_points, 
                                                                        y_values_data_points, 
                                                                        covariance_matrices, 
                                                                        bandwidth_grid)
    solutions_deconvolving.append(solution_deconvolving)
    bandwidths_deconvolving.append(bandwidth_deconvolving)


    true_solution = test_function(query_points)
    solutions_ground_truth.append(true_solution)



    # Blur the true function with the measurement-noise density N(0, x_variance).
    # This is the convolved function that the deconvolving kernel aims to recover, so
    # convolve with the (normalized) Gaussian density evaluated on the grid rather than
    # with a single random noise realization.
    grid_spacing = query_points[1, 0] - query_points[0, 0]
    half_width = min(int(np.ceil(4 * np.sqrt(x_variance) / grid_spacing)),
                    (len(true_solution) - 1) // 2)
    offsets = np.arange(-half_width, half_width + 1) * grid_spacing
    noise_density = np.exp(-offsets**2 / (2 * x_variance))
    noise_density /= noise_density.sum()
    convolved_function = np.convolve(true_solution.ravel(), noise_density, mode='same')
    solutions_convolved.append(convolved_function)


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
# One distinct colour per curve (Epanechnikov, deconvolving, ground truth, convolved).
colors = plt.cm.viridis(np.linspace(0, 0.8, 4))

# 2 x 2 grid of subplots, one per sample size, sharing axes for easy comparison.
fig_width_inches = 7.5
fig_height_inches = fig_width_inches * 0.75
fig, axes = plt.subplots(2, 2, figsize=(fig_width_inches, fig_height_inches),
                         sharex=True, sharey=True)

for ax, n_samples, solution_epanechnikov, solution_deconvolving, true_solution, convolved_function in zip(
        axes.ravel(),
        n_samples_list,
        solutions_epanechnikov,
        solutions_deconvolving,
        solutions_ground_truth,
        solutions_convolved):

    ax.plot(query_points, true_solution, label=r"$f(x)=\sin(x) + \cos(3x) + 0.1x$",
            color=colors[0], linewidth=0.75, linestyle="--")
    ax.plot(query_points, convolved_function, label=r"$(f \ast U)(x)$",
            color=colors[3], linewidth=0.75, linestyle="--")  

    ax.plot(query_points, solution_deconvolving, label="Kernel Regression (Deconvolving kernel)",
            color=colors[0], linewidth=1.5) 
    ax.plot(query_points, solution_epanechnikov, label="Kernel Regression (Epanechnikov kernel)",
            color=colors[3], linewidth=1.5)



    ax.set_title(rf"$n = {n_samples}$")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)

# Axis labels only on the outer edges to avoid clutter.
for ax in axes[-1, :]:
    ax.set_xlabel(r"$x$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$y$")

# A single shared legend for the whole figure.
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.02))

fig.tight_layout(rect=(0, 0, 1, 0.96))
output_path = os.path.join(output_folder, "plot_consistency.pdf")
fig.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close(fig)
