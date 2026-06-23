import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy.interpolate import RegularGridInterpolator
from deconvolving_kernel.kernels import epanechnikov_kernel, deconvolving_kernel_rectangular_support, kernel_regression
from deconvolving_kernel.fourier_transformation_kernels import deconvolving_kernel_round_support


#----------------------------------------------------------------------------------
# First, generate the data and define the function for choosing the optimal solution.
def product_of_cos_function(x_values):
    x1 = x_values[:, 0]
    x2 = x_values[:, 1]
    return 1- np.cos(2*x1)*np.cos(2*x2)  

def absolute_value_function(x_values):
    x1 = x_values[:, 0]
    return 1 - np.abs(x1)/np.pi

def cone_function(x_values):
    return 1 - np.linalg.norm(x_values, axis=1)

cameraman_512 = np.flipud(data.camera()) / 255.0   # normalize to [0, 1]; flip rows so it is not upside down with origin='lower'
cameraman_grid = np.linspace(-np.pi, np.pi, 512)
f_cameraman = RegularGridInterpolator(
    (cameraman_grid, cameraman_grid),
    cameraman_512,
    method='linear',
    bounds_error=False,
    fill_value=0.5,
)

def cameraman_function(x_values):
    return f_cameraman(x_values)

def generate_data(test_function, n_samples, x_covariance, y_variance):
    x_values = np.random.uniform(-np.pi, np.pi, (n_samples, 2))
    y_values = test_function(x_values) + np.random.normal(0, np.sqrt(y_variance), n_samples)
    x_values = x_values + np.random.multivariate_normal(np.zeros(2), x_covariance, n_samples)
    return x_values, y_values

def mse(dataset_1, dataset_2):
    return np.mean((dataset_1 - dataset_2)**2)

def find_best_solution(kernel, test_function, query_points, data_points, y_values_data_points, covariance_matrices, bandwidth_grid, discrete_fourier_transformation_grid_size=1000):
    # Ravel so the (M,) regression output and the ground truth are compared
    # element-wise instead of broadcasting to an (M, M) array.
    true_solution = test_function(query_points).ravel()
    best_solution = np.array([])
    best_error = 1e+20
    best_bandwidth = -1
    error_list =[]

    for bandwidth in bandwidth_grid:
        current_solution = kernel_regression(kernel,
                                            query_points,
                                            data_points,
                                            y_values_data_points,
                                            covariance_matrices,
                                            bandwidth,
                                            discrete_fourier_transformation_grid_size=discrete_fourier_transformation_grid_size)
        error_list.append(mse(current_solution, true_solution))

        if mse(current_solution, true_solution) < best_error:
            best_error = mse(current_solution, true_solution)
            best_bandwidth = bandwidth
            best_solution = current_solution
    return best_solution, best_bandwidth, error_list, best_error



# Set the outputfolder to the figures folder of the project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_folder = os.path.join(project_root, "figures")
os.makedirs(output_folder, exist_ok=True)


res = 100
X, Y = np.mgrid[-np.pi:np.pi:res*1j, -np.pi:np.pi:res*1j]
query_points = np.column_stack([X.ravel(), Y.ravel()])
x_covariance = np.array([[0.5, 0], [0, 0.5]])
y_variance = 0.05
n_samples = 2000


def make_plot_and_store(function, n_samples, x_covariance):
    data_points, y_values_data_points =  generate_data(function, 
                                                    n_samples, 
                                                    x_covariance, 
                                                    y_variance)                        # Shape (N, D) = (1, 1)
    covariance_matrices = x_covariance[np.newaxis, :, :].repeat(n_samples, axis=0)
    bandwidth_grid = np.linspace(0.15, 1, 100)
    round_dft_grid_size = 256

    solution_epanechnikov, bandwidth_epanechnikov, error_epanechnikov, mse_epanechnikov = find_best_solution(epanechnikov_kernel, 
                                                                        function, 
                                                                        query_points, 
                                                                        data_points, 
                                                                        y_values_data_points, 
                                                                        covariance_matrices, 
                                                                        bandwidth_grid)

    solution_deconvolving_rectangular, bandwidth_deconvolving_rectangular, error_deconvolving_rectangular, mse_deconvolving_rectangular = find_best_solution(deconvolving_kernel_rectangular_support, 
                                                                        function, 
                                                                        query_points, 
                                                                        data_points, 
                                                                        y_values_data_points, 
                                                                        covariance_matrices, 
                                                                        bandwidth_grid)


    solution_deconvolving_round, bandwidth_deconvolving_round, error_deconvolving_round, mse_deconvolving_round = find_best_solution(deconvolving_kernel_round_support,
                                                                        function,
                                                                        query_points,
                                                                        data_points,
                                                                        y_values_data_points,
                                                                        covariance_matrices,
                                                                        bandwidth_grid,
                                                                        discrete_fourier_transformation_grid_size=round_dft_grid_size)


    true_solution = function(query_points)

    
    # Store all computed raw data to an .npz file for later reuse.
    """
    data_folder = "/scratch/gpfs/GILLES/lh9809/data"
    os.makedirs(data_folder, exist_ok=True)
    data_path = os.path.join(data_folder, f"raw_data_{function.__name__}_{n_samples}.npz")
    np.savez(
        data_path,
        query_points=query_points,
        data_points=data_points,
        y_values_data_points=y_values_data_points,
        covariance_matrices=covariance_matrices,
        bandwidth_grid=bandwidth_grid,
        round_dft_grid_size=round_dft_grid_size,
        true_solution=true_solution,
        solution_epanechnikov=solution_epanechnikov,
        bandwidth_epanechnikov=bandwidth_epanechnikov,
        error_epanechnikov=error_epanechnikov,
        solution_deconvolving_rectangular=solution_deconvolving_rectangular,
        bandwidth_deconvolving_rectangular=bandwidth_deconvolving_rectangular,
        error_deconvolving_rectangular=error_deconvolving_rectangular,
        solution_deconvolving_round=solution_deconvolving_round,
        bandwidth_deconvolving_round=bandwidth_deconvolving_round,
        error_deconvolving_round=error_deconvolving_round,
    )
    """


    # Plotting

    plt.rcParams.update({
        "text.usetex": True,            # Use LaTeX to render all text
        "text.latex.preamble": r"\usepackage{amsmath}",  # for \lVert, \rVert, etc.
        "font.family": "serif",         # Match standard LaTeX serif font
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 16,           # Font size for x and y labels
        "axes.titlesize": 16,           # Font size for subplot titles
        "font.size": 15,                # Base font size
        "legend.fontsize": 14,          # Font size for legends
        "xtick.labelsize": 13,          # Font size for tick labels
        "ytick.labelsize": 13,
        "figure.titlesize": 18
    })
    # One distinct colour per curve (Epanechnikov, deconvolving, ground truth, convolved).
    colors = plt.cm.viridis(np.linspace(0, 0.8, 4))

    vmin = min(true_solution.min(), solution_deconvolving_rectangular.min(), 
            solution_deconvolving_round.min(), solution_epanechnikov.min())
    vmax = max(true_solution.max(), solution_deconvolving_rectangular.max(), 
            solution_deconvolving_round.max(), solution_epanechnikov.max())

    # Two equally sized blocks side by side: the 2 x 2 grid of estimated functions on
    # the left and the mean-squared-error curves on the right. Using subfigures makes
    # the right-hand plot exactly the same size as the whole 2 x 2 block.
    block_width_inches = 7.5
    block_height_inches = block_width_inches * 0.75
    fig = plt.figure(figsize=(2 * block_width_inches, block_height_inches), layout='constrained')
    fig.suptitle(f"n= {n_samples}")
    subfig_left, subfig_right = fig.subfigures(1, 2, wspace=0.04)

    # --- Left block: 2 x 2 grid of the estimated functions ---
    axes = subfig_left.subplots(2, 2, sharex=True, sharey=True)

    im1 = axes[0, 0].imshow(true_solution.reshape(res, res),extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower',
                        cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0,0].set_title(r'$f(x)$')

    im2 = axes[1, 0].imshow(solution_epanechnikov.reshape(res, res),extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower',
                        cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1,0].set_title('Kernel Regression \n (Epanechnikov kernel)')

    im3 = axes[0, 1].imshow(solution_deconvolving_rectangular.reshape(res, res),extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower',
                        cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Kernel Regression \n (Support: Circle)')

    im4 = axes[1, 1].imshow(solution_deconvolving_round.reshape(res, res),extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower',
                        cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('Kernel Regression \n (Support: Square)')

    # A single shared colorbar for all four panels (they use the same vmin/vmax).
    subfig_left.colorbar(im1, ax=axes, shrink=0.9)

    # --- Right block: mean squared error vs bandwidth ---
    # Each error curve is plotted against the bandwidth grid it was computed on (the
    # round-support kernel uses a coarser grid than the closed-form kernels).
    ax_mse = subfig_right.subplots(1, 1)
    mse_colors = plt.cm.viridis(np.linspace(0, 0.8, 3))

    ax_mse.plot(bandwidth_grid, error_epanechnikov,
                label=f"Epanechnikov kernel (MSE = {mse_epanechnikov:.3f})", color=mse_colors[0], linewidth=1.5)
    ax_mse.plot(bandwidth_grid, error_deconvolving_rectangular,
                label=f"Support: Square (MSE = {mse_deconvolving_rectangular:.3f})", color=mse_colors[1], linewidth=1.5)
    ax_mse.plot(bandwidth_grid, error_deconvolving_round,
                label=f"Support: Circle (MSE = {mse_deconvolving_round:.3f})", color=mse_colors[2], linewidth=1.5)

    ax_mse.set_title('Mean squared error')
    ax_mse.set_xlabel(r"Bandwidth $h$")
    ax_mse.set_ylabel("MSE")
    # Clip the view to [0, 1]: the error explodes for small bandwidths, so cap the
    # y-axis to keep the interesting 0-1 range clearly visible.
    ax_mse.set_ylim(0, 1)
    ax_mse.spines['top'].set_visible(False)
    ax_mse.spines['right'].set_visible(False)
    ax_mse.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
    ax_mse.legend(frameon=False)

    output_path = os.path.join(output_folder, f"round_vs_rectangular_plot_{function.__name__}_{n_samples}.pdf")
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


# Registry mapping function names (used on the command line and in the output file
# names) to the corresponding callables.
FUNCTIONS = {
    "product_of_cos_function": product_of_cos_function,
    "absolute_value_function": absolute_value_function,
    "cameraman_function": cameraman_function,
    "cone_function": cone_function,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the kernel comparison for one function and one sample size.")
    parser.add_argument("--function", required=True, choices=sorted(FUNCTIONS),
                        help="Name of the test function to use.")
    parser.add_argument("--n_samples", type=int, required=True,
                        help="Number of (noisy) data points to generate.")
    args = parser.parse_args()

    make_plot_and_store(FUNCTIONS[args.function], args.n_samples, x_covariance)