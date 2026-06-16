# In this file we would like to define
# the final version of the kernels used
# in this work.

import numpy as np
from scipy.special import wofz, gamma
from scipy.fft import fftshift, ifftshift, ifftn
from scipy.interpolate import RegularGridInterpolator


#---------------------------------------------------------------------------
# Definition of the one dimensional kernels

# This is the one dimensional kernel that results from the kernel 
# $\varphi_K(x) = 1_[|x|\leq 1}, the noise $U \sim \mathcal{N}(0, \sigma^2)$
# and the bandwidth $h$. lambda_val determines the shape of this kernel
# and it expects the $lambda_val = h/\sigma$ as input

def deconvolving_kernel_1d(x, lambda_val):
    # Calculate the complex argument for the Faddeeva function w(z)
    # z = (i * lambda * |x| - 1/lambda) / sqrt(2)
    z = (1j * lambda_val * np.abs(x) - (1 / lambda_val)) / np.sqrt(2)

    # Evaluate the Faddeeva function (efficient numerical implementation)
    w_z = wofz(z)

    # Combine with the exponential term: exp(-i|x|) * w(z)
    term_to_extract = np.exp(-1j * np.abs(x)) * w_z
    imag_part = np.imag(term_to_extract)

    constant = (-lambda_val / np.sqrt(2 * np.pi)) * np.exp(1 / (2 * lambda_val**2))

    return constant * imag_part


# ----------------------------------------------------------------------------------

# Here we will define some auxiliary functions

def check_dimensions(
    query_point: np.ndarray,
    data_point: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Checks that the dimensions of the input arrays are mutually consistent.
    
    Args:
        query_point (np.ndarray): Shape (M, D)
        data_point (np.ndarray): Shape (N, D)
        covariance_matrix (np.ndarray): Shape (N, D, D)
        
    Raises:
        ValueError: If any of the corresponding dimensions (M or D) do not match,
                    or if the covariance matrix is not square.
    """
    # Unpack the shapes
    M, D = query_point.shape
    N, D_1 = data_point.shape
    N_1, D_2, D_3 = covariance_matrix.shape

    # 1. Check that the feature dimension D matches between query and data points
    if D != D_1:
        raise ValueError(
            f"Dimension mismatch: query_point has feature dimension D={D}, "
            f"but data_point has feature dimension D_1={D_1}. They must match."
        )

    # 2. Check that the number of data points N matches the number of covariance matrices
    if N != N_1:
        raise ValueError(
            f"Dimension mismatch: data_point has N={N} points, "
            f"but covariance_matrix has N_1={N_1} matrices. Every data point must have a covariance matrix."
        )

    # 3. Check that the covariance matrices match the feature dimension D
    if D != D_2 or D != D_3:
        raise ValueError(
            f"Dimension mismatch: Expected covariance matrices of shape ({N}, {D}, {D}) "
            f"to match feature dimension D={D}. Got trailing dimensions ({D_2}, {D_3})."
        )
        
    # 4. Double check that the covariance matrices are actually square (D_2 == D_3)
    if D_2 != D_3:
        raise ValueError(
            f"Invalid Shape: Covariance matrices must be square (D_2 == D_3). "
            f"Got shape ({N_1}, {D_2}, {D_3})."
        )
    return M, N, D


def calculate_difference_matrix(
    query_points: np.ndarray,
    data_points: np.ndarray,
    covariance_matrices: np.ndarray,
    bandwidth: float,
    bandwidth_scaling: bool=False,
    rescaled_norm: bool=False,
    eigenvalue_calculation: bool=True,

):
    """
    Args:
        query_points (np.ndarray): Shape (M, D) representing M query points in D dimensions.
        data_points (np.ndarray): Shape (N, D) representing N data points in D dimensions.
        covariance_matrices (np.ndarray): Shape (N, D, D) or (D, D) covariance matrices.
        bandwidth (float): bandwidth parameter
        
    Returns:
        differences (np.ndarray): Shape (M, N, D) representing the differences.
        h (np.ndarray): Shape (N,)
        lambda_val (np.ndarray): Shape (N,D) representing the vector of lambda values for each data point.
    """

    M = query_points.shape[0]
    N = data_points.shape[0]
    D = query_points.shape[1]

    differences = query_points[:, np.newaxis, :] - data_points[np.newaxis, :, :]
    eigenvalues = np.zeros((N, D))
    h = np.full((N,D), bandwidth)

    if eigenvalue_calculation:
        # Ensure the Hermitian symmetry
        covariance_matrices_sym = 0.5 * (covariance_matrices + covariance_matrices.transpose(0, 2, 1))

        # This code returns the eigenvalues (N, D) and the orthogonal
        # matrices corresponding to the eigenvectors (N, D, D)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrices_sym)

        # This Einstein summation transposes the matrices that contain the
        # eigenvectors in its columns and performs the matrix vector multiplication
        # at the same time.
        differences = np.einsum('mnd, ndj -> mnj', differences, eigenvectors)

        if rescaled_norm:
            h = h * np.sqrt(np.abs(eigenvalues))

    if bandwidth_scaling:
        traces = np.trace(covariance_matrices, axis1=1, axis2=2)
        scaling = traces /np.mean(traces)
        h = h * scaling[:, np.newaxis]


    differences = differences / h[np.newaxis, :, :]

    # calculating lambda_val
    sigmas = np.sqrt(np.abs(eigenvalues))
    lambda_val = h[:, :]/ (sigmas+ 1e-10)

    return differences, h, lambda_val


def epanechnikov_normalization_factor(dim: int) -> float:
    """Calculates the analytical integral of (1 - ||x||^2) * 1{||x|| <= 1}
    over an arbitrary D-dimensional space.

    Args:
        dim (int): The dimension of the vector space (D >= 1).

    Returns:
        float: The total volume (integral) of the function.
    """
    if dim < 1:
        raise ValueError("Dimension must be an integer greater than or equal to 1.")
        
    # Calculate the surface area of a unit (D-1)-sphere
    surface_area = (2 * (np.pi ** (dim / 2))) / gamma(dim / 2)
    
    # Calculate the radial integral chunk: 2 / (D * (D + 2))
    radial_integral = 2 / (dim * (dim + 2))
    
    return float(surface_area * radial_integral)
    




# ----------------------------------------------------------------------------------

# In this section, we define the kernels implemented in this section

def deconvolving_kernel_rectangular_support(
        query_points: np.ndarray,
        data_points: np.ndarray,
        covariance_matrices: np.ndarray,
        bandwidth: float,
        bandwidth_scaling: bool=False,
        rescaled_norm: bool=False,
        eigenvalue_calculation: bool=True,
        discrete_fourier_transformation_grid_size: int=1000,
) -> np.ndarray:
    """
    Args:
        query_points (np.ndarray): Shape (M, D) representing M query points in D dimensions.
        data_points (np.ndarray): Shape (N, D) representing N data points in D dimensions.
        covariance_matrices (np.ndarray): Shape (N, D, D) covariance matrices.
        bandwidth (float): The smoothing bandwidth parameter.
        bandwidth_scaling (bool): Whether to scale the bandwidth dynamically. Defaults to False.
        rescaled_norm (bool): Whether we use the norm $\lVert \cdot \lVert_{\Sigma_i^{-1}}$. Defaults to False.
        eigenvalue_calculation (bool): Whether to rotate the coordinate system. Defaults to True.
        discrete_fourier_transformation_grid_size (int): The number of points per dimension for the DFT. Defaults to 1000.


    Returns:
        kernel_values (np.ndarray): Shape (M, N) representing the kernel values.
    """


    # check that the input dimensions match
    M, N, D = check_dimensions(query_points, data_points, covariance_matrices)

    if eigenvalue_calculation == False and rescaled_norm == True:
        raise NotImplementedError("This combination of switches does not work. Please set eigenvalue_calculation to True.")

    differences, h, lambda_val = calculate_difference_matrix(query_points, 
                                                          data_points, 
                                                          covariance_matrices, 
                                                          bandwidth, 
                                                          bandwidth_scaling,
                                                          rescaled_norm, 
                                                          eigenvalue_calculation)
    
    kernel_values = deconvolving_kernel_1d(differences, lambda_val[np.newaxis, :,:])
    kernel_values = np.prod(kernel_values, axis=2)
    bandwidth_normalization = np.prod(h, axis=1)
    kernel_values = kernel_values/(bandwidth_normalization[np.newaxis, :])

    return kernel_values



def epanechnikov_kernel(
        query_points: np.ndarray,
        data_points: np.ndarray,
        covariance_matrices: np.ndarray,
        bandwidth: float,
        bandwidth_scaling: bool=False,
        rescaled_norm: bool=False,
        eigenvalue_calculation: bool=True,
        discrete_fourier_transformation_grid_size: int=1000,
) -> np.ndarray:
    """
    Args:
        query_points (np.ndarray): Shape (M, D) representing M query points in D dimensions.
        data_points (np.ndarray): Shape (N, D) representing N data points in D dimensions.
        covariance_matrices (np.ndarray): Shape (N, D, D) covariance matrices.
        bandwidth (float): The smoothing bandwidth parameter.
        bandwidth_scaling (bool): Whether to scale the bandwidth dynamically. Defaults to False.
        rescaled_norm (bool): Whether we use the norm $\lVert \cdot \lVert_{\Sigma_i^{-1}}$. Defaults to False.
        eigenvalue_calculation (bool): Whether to rotate the coordinate system. Defaults to True.
        discrete_fourier_transformation_grid_size (int): The number of points per dimension for the DFT. Defaults to 1000.


    Returns:
        kernel_values (np.ndarray): Shape (M, N) representing the kernel values.
    """

    M, N, D = check_dimensions(query_points, data_points, covariance_matrices)

    if eigenvalue_calculation == False and rescaled_norm == True:
        raise NotImplementedError("This combination of switches does not work. Please set eigenvalue_calculation to True.")

    differences, h, _ = calculate_difference_matrix(query_points,
                                                 data_points,
                                                 covariance_matrices,
                                                 bandwidth,
                                                 bandwidth_scaling,
                                                 rescaled_norm,
                                                 eigenvalue_calculation) 
    
    norms = np.linalg.norm(differences, axis=2)
    kernel_values = np.zeros_like(norms)
    mask = norms <= 1
    kernel_values[mask] = epanechnikov_normalization_factor(D) * (1 - norms[mask]**2)
    bandwidth_normalization = np.prod(h, axis=1)
    kernel_values = kernel_values/(bandwidth_normalization[np.newaxis, :])
    return kernel_values




# --------------------------------------------------------------------------------
# In this section, we calculate the kernel regression function for an arbitrary kernel

# The kernel functions build intermediate arrays of shape (batch_M, N, D), and the
# deconvolving kernel in particular allocates several complex128 (16 bytes/element)
# temporaries at once. This factor is a conservative estimate (in bytes) of the peak
# memory used per element of the (batch_M, N, D) working set, so that the automatically
# chosen batch size leaves enough head room for those temporaries.
_BYTES_PER_WORKING_ELEMENT = 16 * 8  # ~8 simultaneous complex128 temporaries


def _available_memory_bytes() -> int:
    """Returns the amount of currently available physical memory in bytes.

    Tries ``psutil`` first and falls back to ``os.sysconf`` on POSIX systems.
    If neither is available, returns a conservative 1 GiB default.
    """
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        import os
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 1024 ** 3  # 1 GiB fallback


def _choose_query_batch_size(M: int,
                             N: int,
                             D: int,
                             max_memory_bytes=None,
                             memory_fraction: float=0.8) -> int:
    """Chooses how many query points to process at once so that the kernel
    computation fits into the available memory.

    Args:
        M (int): Total number of query points.
        N (int): Number of data points.
        D (int): Feature dimension.
        max_memory_bytes (int, optional): Memory budget in bytes. If ``None``,
            ``memory_fraction`` of the currently available memory is used.
        memory_fraction (float): Fraction of available memory to use when
            ``max_memory_bytes`` is ``None``. Defaults to 0.8.

    Returns:
        int: The number of query points per batch (at least 1, at most M).
    """
    if max_memory_bytes is None:
        max_memory_bytes = int(_available_memory_bytes() * memory_fraction)

    # Memory needed per query point for the (1, N, D) working set.
    bytes_per_query = max(1, N * D * _BYTES_PER_WORKING_ELEMENT)

    batch_size = int(max_memory_bytes // bytes_per_query)
    # Always make progress with at least one query point per batch and never
    # use a larger batch than the number of query points we actually have.
    return max(1, min(M, batch_size))


def kernel_regression(kernel,
                      query_points,
                      data_points,
                      y_values_data_points,
                      covariance_matrices,
                      bandwidth,
                      bandwidth_scaling=False,
                      rescaled_norm=False,
                      eigenvalue_calculation=True,
                      discrete_fourier_transformation_grid_size=1000,
                      batch_size=None,
                      max_memory_bytes=None,
                      memory_fraction=0.8):
    """Performs kernel regression, batching over query points to bound memory use.

    Computing all kernel values at once requires arrays of shape (M, N, D), which
    can exceed the available memory when both the number of query points M and the
    number of data points N are large. To avoid this, the query points are split
    into batches.
    Args:
        kernel (callable): A kernel function (e.g. ``deconvolving_kernel_rectangular_support``
            or ``epanechnikov_kernel``) returning an (M, N) array of kernel values.
        query_points (np.ndarray): Shape (M, D) query points.
        data_points (np.ndarray): Shape (N, D) data points.
        y_values_data_points (np.ndarray): Shape (N,) response values at the data points.
        covariance_matrices (np.ndarray): Shape (N, D, D) covariance matrices.
        bandwidth (float): The smoothing bandwidth parameter.
        bandwidth_scaling (bool): Whether to scale the bandwidth dynamically. Defaults to False.
        rescaled_norm (bool): Whether to use the rescaled norm. Defaults to False.
        eigenvalue_calculation (bool): Whether to rotate the coordinate system. Defaults to True.
        discrete_fourier_transformation_grid_size (int): DFT grid size. Defaults to 1000.
        batch_size (int, optional): Number of query points to process per batch. If
            ``None`` (default), a batch size that fits into memory is chosen automatically.
        max_memory_bytes (int, optional): Memory budget in bytes used to pick the batch
            size automatically. If ``None``, ``memory_fraction`` of the currently
            available memory is used. Ignored when ``batch_size`` is given.
        memory_fraction (float): Fraction of available memory to use when sizing the
            batches automatically. Defaults to 0.8.

    Returns:
        np.ndarray: Shape (M,) the kernel regression prediction at each query point.
    """
    query_points = np.asarray(query_points)
    M = query_points.shape[0]
    N = data_points.shape[0]
    D = query_points.shape[1]

    if batch_size is None:
        batch_size = _choose_query_batch_size(
            M, N, D,
            max_memory_bytes=max_memory_bytes,
            memory_fraction=memory_fraction,
        )
    else:
        batch_size = max(1, int(batch_size))

    epsilon = 1e-10
    predictions = np.empty(M, dtype=float)

    for start in range(0, M, batch_size):
        stop = min(start + batch_size, M)
        query_batch = query_points[start:stop]

        kernel_values = kernel(query_batch,
                              data_points,
                              covariance_matrices,
                              bandwidth,
                              bandwidth_scaling,
                              rescaled_norm,
                              eigenvalue_calculation,
                              discrete_fourier_transformation_grid_size)

        weighted_average = np.sum(kernel_values * y_values_data_points, axis=1)
        normalization_factor = np.sum(kernel_values, axis=1)

        predictions[start:stop] = weighted_average / (normalization_factor + epsilon)

    return predictions











