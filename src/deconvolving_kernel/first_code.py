import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so figures are written to disk
import matplotlib.pyplot as plt
import kernels as my_kernels

# Folder where all generated figures are stored
OUTPUT_DIR = "/home/lh9809/Images_from_deconvolving_kernel_package"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------
# Generating data

def function_evaluation(points):
    # Extract x1 (first column) and x2 (second column)
    x1 = points[:, 0]
    x2 = points[:, 1]

    # Perform the calculation
    return 1 - np.cos(x1*4) *np.cos(x2*4)

# Create data
def generate_data(n_samples, function, y_var, x_cov_range = np.array([0.05, 0.15]), x_noise = True):
  x_val = np.random.uniform(-1.5, 1.5, (n_samples, 2))
  y_val = function(x_val)
  noise = np.random.normal(0, y_var, n_samples)
  y_val = y_val + noise
  # x_cov = np.random.uniform(x_cov_range[0], x_cov_range[1], (n_samples, 2))
  x_cov = np.random.choice([x_cov_range[0], x_cov_range[1]], (n_samples, 2))

  if x_noise:
    for i in range(n_samples):
      x_val[i] += np.random.multivariate_normal(np.array([0,0]), np.diag(x_cov[i]))

  return x_val, y_val, x_cov

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
x_grid = np.column_stack([X.ravel(), Y.ravel()])

def select_min(h_values,liste):
  print(f"minimal error:", min(liste))
  print(f"best h:", h_values[liste.index(min(liste))])
  return h_values[liste.index(min(liste))]


def combine_datasets(weighted_average, normalization):
  weighted_average = np.sum(weighted_average, axis = 0)
  normalization = np.sum(normalization, axis = 0)
  return weighted_average/(normalization + 10**(-10))


def get_optimal_function(h, h_list, stored_function):
    # Ensure h_list is a numpy array for indexing
    h_list = np.asarray(h_list)

    # Find the index where h_list equals h
    # .item() converts a single-element array to a standard Python scalar
    try:
        h_idx = np.where(h_list == h)[0][0]
        return stored_function[h_idx, :]
    except IndexError:
        raise ValueError(f"Value {h} not found in h_list")

def test_kde_estimator(x_grid, x_data, y_data, x_cov_list):
  h_values  = np.linspace(0.15, 0.45, 50)
  true_function = function_evaluation(x_grid)
  n = len(x_cov_list)

  # x_cov_list holds per-point diagonal variances of shape (N, D); the kernels
  # expect full covariance matrices of shape (N, D, D). Expand to diagonals,
  # matching how the input noise was generated (np.diag(x_cov[i])).
  cov_matrices = np.stack([np.diag(c) for c in x_cov_list])

  error_epi = []
  error_deconv = []
  h_used = []

  epi_predi = np.zeros((len(h_values), len(x_grid)))
  dec_predi = np.zeros((len(h_values), len(x_grid)))


  counter = 0
  for h in h_values:

    epi_prediction = my_kernels.kernel_regression(my_kernels.epanechnikov_kernel,
                                                   x_grid,
                                                   x_data,
                                                   y_data,
                                                   cov_matrices,
                                                   h,
                                                   eigenvalue_calculation=False)

    epi_predi[counter, :] = epi_prediction
    dec_prediction = my_kernels.kernel_regression(my_kernels.deconvolving_kernel_rectangular_support,
                                                   x_grid,
                                                   x_data,
                                                   y_data,
                                                   cov_matrices,
                                                   h)
    dec_predi[counter, :] = dec_prediction

    counter += 1

    tmp_epi = np.mean((epi_prediction - true_function)**2)
    tmp_dec = np.mean((dec_prediction - true_function)**2)

    error_epi.append(tmp_epi)
    error_deconv.append(tmp_dec)

    h_used.append(h)

  h_epi = select_min(h_used,error_epi)
  h_dec = select_min(h_used,error_deconv)

  plt.figure()
  plt.plot(h_used, [min(x, 1) for x in error_epi], label = "epi")
  plt.plot(h_used, [min(x,1) for x in error_deconv], label = "deconv")
  plt.xlabel("h")
  plt.ylabel("MSE")
  plt.legend()
  plt.savefig(os.path.join(OUTPUT_DIR, "mse_vs_bandwidth.png"), dpi=150, bbox_inches="tight")
  plt.close()

    # Here comes the plotting area :)
  true_density_map = true_function.reshape(100, 100)
  epi_density = get_optimal_function(h_epi, h_values, epi_predi)
  epi_density_map = epi_density.reshape(100, 100)
  dec_density = get_optimal_function(h_dec, h_values, dec_predi)
  dec_density_map = dec_density.reshape(100, 100)

  # 3. Calculate the difference
  diff_epi = true_density_map - epi_density_map
  diff_dec = true_density_map - dec_density_map
  diff_min = min(diff_epi.min(), diff_dec.min())
  diff_max = max(diff_epi.max(), diff_dec.max())

  # 4. Determine a common color scale for the first two functions
  vmin = min(true_density_map.min(), epi_density_map.min(), dec_density_map.min())
  vmax = max(true_density_map.max(), epi_density_map.max(), dec_density_map.max())

  # 5. Create the plots
  fig, axes = plt.subplots(2, 3, figsize=(18, 10))

  # Plot 1: Function 1
  im1 = axes[0, 0].imshow(true_density_map, extent=[-5, 5, -5, 5], origin='lower',
                      cmap='viridis', vmin=vmin, vmax=vmax)
  axes[0,0].set_title('True Function')
  fig.colorbar(im1, ax=axes[0,0])

  # Plot 2: Function 2
  im2 = axes[0, 1].imshow(epi_density_map, extent=[-5, 5, -5, 5], origin='lower',
                      cmap='viridis', vmin=vmin, vmax=vmax)
  axes[0,1].set_title('Estimated Function using Epi')
  fig.colorbar(im2, ax=axes[0,1])

  # Plot 3: Difference (F1 - F2)
  # We use a diverging colormap 'RdBu_r' for the difference
  im3 = axes[0, 2].imshow(diff_epi, extent=[-5, 5, -5, 5], origin='lower',
                      cmap='RdBu_r', vmin = diff_min, vmax = diff_max)
  axes[0, 2].set_title('Heterogeneous Kernel')
  fig.colorbar(im3, ax=axes[0,2])



  # Plot 1: Function 1
  im1 = axes[1, 0].imshow(true_density_map, extent=[-5, 5, -5, 5], origin='lower',
                      cmap='viridis', vmin=vmin, vmax=vmax)
  axes[1,0].set_title('True Function')
  fig.colorbar(im1, ax=axes[1,0])

  # Plot 2: Function 2
  im2 = axes[1, 1].imshow(dec_density_map, extent=[-5, 5, -5, 5], origin='lower',
                      cmap='viridis', vmin=vmin, vmax=vmax)
  axes[1,1].set_title('Estimated Function using the deconvoluted kernel')
  fig.colorbar(im2, ax=axes[1, 1])

  # Plot 3: Difference (F1 - F2)
  # We use a diverging colormap 'RdBu_r' for the difference
  im3 = axes[1, 2].imshow(diff_dec, extent=[-5, 5, -5, 5], origin='lower',
                      cmap='RdBu_r', vmin = diff_min, vmax = diff_max)
  axes[1, 2].set_title('Difference')
  fig.colorbar(im3, ax=axes[1, 2])

  fig.tight_layout()
  fig.savefig(os.path.join(OUTPUT_DIR, "function_estimates.png"), dpi=150, bbox_inches="tight")
  plt.close(fig)

  return h_epi, h_dec


x_data, y_data, x_cov_list = generate_data(2000, function_evaluation, 0.1, x_cov_range = np.array([0.05, 0.15]), x_noise = True)
test_kde_estimator(x_grid, x_data, y_data, x_cov_list)