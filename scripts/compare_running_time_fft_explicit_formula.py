import time

import numpy as np

from deconvolving_kernel.kernels import deconvolving_kernel_rectangular_support
from deconvolving_kernel.fourier_transformation_kernels import deconvolving_kernel_round_support


# --- Setup ----------------------------------------------------------------
np.random.seed(0)

n_points = 1_000_000
bandwidth = 1.0
discrete_fourier_transformation_grid_size = 32

# 1,000,000 random 1D data points, shape (N, D) = (1e6, 1).
data_points = np.random.uniform(-5.0, 5.0, (n_points, 1))

# 1,000,000 random covariance matrices, shape (N, D, D) = (1e6, 1, 1). In 1D a
# covariance matrix is just a positive variance.
covariance_matrices = np.random.uniform(0.05, 0.5, (n_points, 1, 1))

# A single query point at which we evaluate the kernels, shape (M, D) = (1, 1).
query_points = np.zeros((1, 1))


# --- Time the explicit-formula (rectangular support) kernel ---------------
start = time.perf_counter()
kernel_values_rectangular = deconvolving_kernel_rectangular_support(
    query_points,
    data_points,
    covariance_matrices,
    bandwidth,
    discrete_fourier_transformation_grid_size=discrete_fourier_transformation_grid_size,
)
time_rectangular = time.perf_counter() - start


# --- Time the FFT (round support) kernel ----------------------------------
start = time.perf_counter()
kernel_values_round = deconvolving_kernel_round_support(
    query_points,
    data_points,
    covariance_matrices,
    bandwidth,
    discrete_fourier_transformation_grid_size=discrete_fourier_transformation_grid_size,
)
time_round = time.perf_counter() - start


# --- Report ---------------------------------------------------------------
print(f"Number of points / covariance matrices: {n_points:,}")
print(f"deconvolving_kernel_rectangular_support (explicit formula): {time_rectangular:.4f} s")
print(f"deconvolving_kernel_round_support       (FFT):             {time_round:.4f} s")

if time_rectangular < time_round:
    faster, slower = "rectangular_support (explicit formula)", "round_support (FFT)"
    speedup = time_round / time_rectangular
else:
    faster, slower = "round_support (FFT)", "rectangular_support (explicit formula)"
    speedup = time_rectangular / time_round

print(f"\nFaster method: deconvolving_kernel_{faster}")
print(f"It is {speedup:.1f}x faster than deconvolving_kernel_{slower}.")
