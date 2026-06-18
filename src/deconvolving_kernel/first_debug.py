# Please do kernel regression as in first_code.py with the 4 kernels epanechnikov_kernel, deconvolving_kernel_rectangular_support, deconvolving_kernel_round_support and deconvolving_kernel_heterogeneous_noise
# You should try all combinations of the switches         bandwidth_scaling, rescaled_norm and eigenvalue_calculation
# if you get an not implemented error, that is intended and not a problem
#
# if there are any problems, please report them to me
#
# Run from the src directory with:  python -m deconvolving_kernel.first_debug

import itertools

import numpy as np

from .kernels import (
    kernel_regression,
    epanechnikov_kernel,
    deconvolving_kernel_rectangular_support,
)
from .fourier_transformation_kernels import (
    deconvolving_kernel_round_support,
    deconvolving_kernel_heterogeneous_noise,
)


# ----------------------------------------------------------------------------------
# Generating data (same style as first_code.py, but small so the Fourier based
# kernels, which build one DFT grid per data point, stay fast)

def function_evaluation(points):
    # Extract x1 (first column) and x2 (second column)
    x1 = points[:, 0]
    x2 = points[:, 1]
    return 1 - np.cos(x1 * 4) * np.cos(x2 * 4)


def generate_data(n_samples, function, y_var, x_cov_range=np.array([0.05, 0.15]), x_noise=True):
    x_val = np.random.uniform(-1.5, 1.5, (n_samples, 2))
    y_val = function(x_val)
    noise = np.random.normal(0, y_var, n_samples)
    y_val = y_val + noise
    x_cov = np.random.choice([x_cov_range[0], x_cov_range[1]], (n_samples, 2))

    if x_noise:
        for i in range(n_samples):
            x_val[i] += np.random.multivariate_normal(np.array([0, 0]), np.diag(x_cov[i]))

    return x_val, y_val, x_cov


# ----------------------------------------------------------------------------------
# Build a small problem

np.random.seed(0)

N_SAMPLES = 20
DFT_GRID_SIZE = 128        # smaller than the default 1000 to keep the test fast
BANDWIDTH = 0.3

x_data, y_data, x_cov_list = generate_data(N_SAMPLES, function_evaluation, 0.1)

# x_cov_list holds per-point diagonal variances of shape (N, D); the kernels expect
# full covariance matrices of shape (N, D, D). Expand to diagonals, matching how the
# input noise was generated (np.diag(x_cov[i])).
covariance_matrices = np.stack([np.diag(c) for c in x_cov_list])

# A small query grid (8 x 8 = 64 points) on which we evaluate the regression.
axis = np.linspace(-1, 1, 8)
X, Y = np.meshgrid(axis, axis)
query_points = np.column_stack([X.ravel(), Y.ravel()])

M = query_points.shape[0]


# ----------------------------------------------------------------------------------
# Run every kernel with every combination of the three switches

kernels = {
    "epanechnikov_kernel": epanechnikov_kernel,
    "deconvolving_kernel_rectangular_support": deconvolving_kernel_rectangular_support,
    "deconvolving_kernel_round_support": deconvolving_kernel_round_support,
    "deconvolving_kernel_heterogeneous_noise": deconvolving_kernel_heterogeneous_noise,
}

switch_names = ["bandwidth_scaling", "rescaled_norm", "eigenvalue_calculation"]

problems = []
print(f"Testing {len(kernels)} kernels x {2 ** len(switch_names)} switch combinations "
      f"(M={M} query points, N={N_SAMPLES} data points)\n")

for kernel_name, kernel in kernels.items():
    print(f"=== {kernel_name} ===")
    for combo in itertools.product([False, True], repeat=len(switch_names)):
        switches = dict(zip(switch_names, combo))
        label = ", ".join(f"{name}={value}" for name, value in switches.items())
        try:
            prediction = kernel_regression(
                kernel,
                query_points,
                x_data,
                y_data,
                covariance_matrices,
                BANDWIDTH,
                discrete_fourier_transformation_grid_size=DFT_GRID_SIZE,
                **switches,
            )

            # Check the output is shaped and valued as we would like.
            if prediction.shape != (M,):
                status = f"PROBLEM: wrong shape {prediction.shape}, expected ({M},)"
                problems.append((kernel_name, label, status))
            elif not np.all(np.isfinite(prediction)):
                status = "PROBLEM: prediction contains non-finite values (nan/inf)"
                problems.append((kernel_name, label, status))
            else:
                status = "ok"
        except NotImplementedError as error:
            # Intended for switch combinations a kernel does not support.
            status = f"not implemented (intended): {error}"
        except Exception as error:
            status = f"PROBLEM: {type(error).__name__}: {error}"
            problems.append((kernel_name, label, status))

        print(f"  [{label}] -> {status}")
    print()


# ----------------------------------------------------------------------------------
# Report

print("=" * 70)
if problems:
    print(f"Found {len(problems)} problem(s):\n")
    for kernel_name, label, status in problems:
        print(f"  - {kernel_name} [{label}]: {status}")
else:
    print("No problems found: all kernels worked for every switch combination "
          "(aside from intended not-implemented cases).")
