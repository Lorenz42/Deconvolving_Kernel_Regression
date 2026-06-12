# In this file we would like to define
# the final version of the kernels used
# in this work.

import numpy as np
from scipy.special import wofz


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

    # 2. Check that the number of data points M matches the number of covariance matrices
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
    coordinate_system_rotation: bool=True,
) -> np.ndarray:
    """
    Args:
        query_points (np.ndarray): Shape (M, D) representing N query points in D dimensions.
        data_points (np.ndarray): Shape (N, D) representing M data points in D dimensions.
        covariance_matrices (np.ndarray): Shape (N, D, D) or (D, D) covariance matrices.
        bandwidth (float): bandwidth parameter
        
    Returns:
        differences (np.ndarray): Shape (M, N, D) representing the differences.
        lambda_val: (np.ndarray): Shape (N,D) representing the vector of lambda values for each data point.
    """
    
    





# ----------------------------------------------------------------------------------

# In this section, we define the kernels implemented in this section

def deconvolving_kernel_rectangular_support(
        query_points: np.ndarray,
        data_points: np.ndarray,
        covariance_matrices: np.ndarray,
        bandwidth: float,
        bandwidth_scaling: bool=False,
        kernel_support_scaled_fourier_domain: bool=False,
        coordinate_system_rotation: bool=True,
        discrete_fourier_transformation_grid_size: int=1000,
) -> np.ndarray:
    """
    Args:
        query_points (np.ndarray): Shape (M, D) representing N query points in D dimensions.
        data_points (np.ndarray): Shape (N, D) representing M data points in D dimensions.
        covariance_matrices (np.ndarray): Shape (N, D, D) or (D, D) covariance matrices.
        bandwidth (float): The smoothing bandwidth parameter.
        bandwidth_scaling (bool): Whether to scale the bandwidth dynamically. Defaults to False.
        kernel_support_scaled_fourier_domain (bool): Whether we use the norm $\lVert \cdot \lVert_{\Sigma_i^{-1}}$. Defaults to False.
        coordinate_system_rotation (bool): Whether to rotate the coordinate system. Defaults to True.
        discrete_fourier_transformation_grid_size (int): The number of points per dimension for the DFT. Defaults to 1000.


    Returns:
        kernel_values (np.ndarray): Shape (M, N) representing the kernel values.
    """


    # check that the input dimensions match
    M, N, D = check_dimensions(query_points, data_points, covariance_matrices)






