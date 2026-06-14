# -------------------------------------------------------------------------------------------
# In this file, we will define the kernels that rely on explicit Fourier 
# Transformation to be calculated
import numpy as np
from scipy.fft import fftshift, ifftshift, ifftn
from scipy.interpolate import RegularGridInterpolator
from .kernels import calculate_difference_matrix, check_dimensions


# ----------------------------------------------------------------------------------

# Hee we define the circular mask used.

def circular_mask(*K_grids, covariance_matrix=None):
    """Generalized N-dimensional support mask in Fourier space.

    Returns a boolean mask selecting the unit ball {||k|| <= 1}. When a
    covariance matrix is provided the ball turns into the ellipsoid
    {k^T @ covariance_matrix^{-1} @ k <= 1}.

    Args:
        *K_grids: Arbitrary number of D coordinate grids (KX, KY, KZ, ...), one
            per dimension, each of identical shape.
        covariance_matrix (np.ndarray, optional): Shape (D, D). If None (or the
            identity matrix) a spherical mask is used. Defaults to None.

    Returns:
        np.ndarray: A boolean mask of the same shape as the input grids.
    """
    number_of_dimensions = len(K_grids)

    if covariance_matrix is None or np.allclose(covariance_matrix, np.eye(number_of_dimensions)):
        # --- Case 1: Spherical mask (no covariance / identity matrix) ---
        # Sum the squares of all coordinate grids dynamically
        squared_radius = sum(K**2 for K in K_grids)
        radius = np.sqrt(squared_radius)
    else:
        # --- Case 2: Ellipsoidal mask (with covariance) ---
        # Flatten and stack the grids to perform the matrix multiplication
        # efficiently. The shape will be (D, total_number_of_points).
        flat_grids = [K.ravel() for K in K_grids]
        K_vectors = np.vstack(flat_grids)

        inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

        # Compute the quadratic form k^T @ covariance_matrix^{-1} @ k for every point
        transformed_distances = np.sum(K_vectors * (inverse_covariance_matrix @ K_vectors), axis=0)

        # Reshape the flattened result back to the original grid shape
        radius = np.sqrt(transformed_distances).reshape(K_grids[0].shape)

    mask = (radius <= 1)
    return mask


def charc_func_normal(*K_grids, bandwidth, covariance_matrix, list_of_all_covariance_matrices=None):
    """
    Evaluates exp(0.5 * (k / bandwidth)^T @ covariance_matrix @ (k / bandwidth)),

    Args:
        *K_grids: Arbitrary number of D coordinate grids (KX, KY, KZ, ...), one
            per dimension, each of identical shape.
        bandwidth (float or np.ndarray): The smoothing bandwidth.
        covariance_matrix (np.ndarray): Shape (D, D) noise covariance matrix.
        list_of_all_covariance_matrices (np.ndarray, optional): Unused here, kept
            so that every Fourier space function shares the same call signature.

    Returns:
        np.ndarray: An array of the same shape as the input grids.
    """
    grid_shape = K_grids[0].shape

    # 1. Flatten and stack grids into a (D, total_number_of_points) matrix of vectors
    flat_grids = [K.ravel() for K in K_grids]
    K_vectors = np.vstack(flat_grids)

    # 2. Divide every frequency by the bandwidth
    K_vectors = K_vectors / np.reshape(bandwidth, (-1, 1))

    # 3. Compute the quadratic form (k / h)^T @ covariance_matrix @ (k / h) for all points
    norm = np.einsum('ji,jk,ki->i', K_vectors, covariance_matrix, K_vectors)

    # 4. Reshape the flat result back into the original grid structure
    return np.exp(0.5 * norm).reshape(grid_shape)



def psi(*K_grids, bandwidth, covariance_matrix, list_of_all_covariance_matrices):
    """
    
    Args:
        *K_grids: Arbitrary number of multidimensional coordinate arrays (KX, KY, KZ, ...).
        bandwidth: Scaling parameter.
        covariance_matrix: Target covariance matrix of shape (D, D).
        list_of_all_covariance_matrices: A list of covariance matrices used to calculate the normalization denominator.
        
    Returns:
        An array of the same shape as the input grids.
    """
    grid_shape = K_grids[0].shape

    def phi(*grids, covariance):
        # 1. Flatten and stack grids to form a matrix of vectors: shape (ndim, total_points)
        flat_grids = [G.ravel() for G in grids]
        K_vectors = np.vstack(flat_grids)
        
        # 2. Compute the quadratic form: k^T * covariance * k dynamically for N-dimensions
        norm = np.einsum('ji,jk,ki->i', K_vectors, covariance, K_vectors)
        norm = norm / (bandwidth**2)
        
        # 3. Reshape the flat result back into the original grid structure
        # Note the negative sign (-1/2) matching your original function definition
        return np.exp(-0.5 * norm).reshape(grid_shape)

    normalization = np.zeros_like(K_grids[0])
    for covariance in list_of_all_covariance_matrices:
        normalization += phi(*K_grids, covariance=covariance)**2    
    normalization += 1e-10

    negative_grids = [-1 * K for K in K_grids]
    numerator = phi(*negative_grids, covariance=covariance_matrix)

    return numerator / normalization


# ----------------------------------------------------------------------------------

# Here we define the N-dimensional discrete Fourier transformer that turns a
# Fourier space function and a support mask into an interpolatable real space kernel

class FourierTransformerND:
    def __init__(self,
                 function,
                 mask_of_support,
                 number_of_dimensions=2,
                 bandwidth=1,
                 covariance_matrix=None,
                 list_of_all_covariance_matrices=None,
                 number_of_grid_points_per_dimension=1024,
                 length_per_dimension=20.0,
                 rescaled_norm=False):
        """Generalized N-dimensional discrete Fourier transformer.

        Args:
            function (callable): Function to evaluate in Fourier space.
            mask_of_support (callable): Mask function filtering Fourier space.
            number_of_dimensions (int): Number of dimensions D. Defaults to 2.
            bandwidth (float or np.ndarray): The smoothing bandwidth passed to
                ``function``. Defaults to 1.
            covariance_matrix (np.ndarray, optional): Covariance matrix of the
                current data point of shape (D, D). Defaults to None.
            list_of_all_covariance_matrices (np.ndarray, optional): All covariance
                matrices of shape (N, D, D). Defaults to None.
            number_of_grid_points_per_dimension (int): Number of grid points per
                dimension. Defaults to 1024.
            length_per_dimension (float): Total length of the domain per dimension
                in Fourier space. Defaults to 20.0.
            rescaled_norm (bool): Whether to use the ellipsoidal support mask.
                Defaults to False.
        """
        
        self.function = function
        self.mask_of_support = mask_of_support
        self.number_of_dimensions = number_of_dimensions
        self.bandwidth = bandwidth
        self.covariance_matrix = covariance_matrix
        self.list_of_all_covariance_matrices = list_of_all_covariance_matrices
        self.number_of_grid_points_per_dimension = number_of_grid_points_per_dimension
        self.length_per_dimension = length_per_dimension

        self.dk = self.length_per_dimension / self.number_of_grid_points_per_dimension
        self.dx = (2 * np.pi) / self.length_per_dimension

        # 1. Generate the N-dimensional grid for Fourier space
        k_vals = np.linspace(-self.length_per_dimension / 2,
                             self.length_per_dimension / 2,
                             self.number_of_grid_points_per_dimension,
                             endpoint=False)
        # Create a list of 1D coordinate arrays for meshgrid
        k_grids = [k_vals] * self.number_of_dimensions
        K_grids = np.meshgrid(*k_grids, indexing='ij')

        if rescaled_norm:
            mask = mask_of_support(*K_grids, covariance_matrix=self.covariance_matrix)
        else:
            mask = mask_of_support(*K_grids, covariance_matrix=None)

        fourier_space = np.zeros(K_grids[0].shape)
        # Extract the masked coordinates as a tuple of flattened arrays
        masked_coords = tuple(grid[mask] for grid in K_grids)

        fourier_space[mask] = function(*masked_coords,
                                       bandwidth=self.bandwidth,
                                       covariance_matrix=self.covariance_matrix,
                                       list_of_all_covariance_matrices=self.list_of_all_covariance_matrices)
        self.fourier_space = fourier_space

        # 3. Inverse N-dimensional Fourier transform
        f_real = fftshift(ifftn(ifftshift(self.fourier_space)))

        # 4. Scaling for the continuous inverse Fourier transform in D dimensions:
        #    (dk / 2pi)^D * (number_of_grid_points_per_dimension)^D
        scaling = (self.dk / (2 * np.pi))**self.number_of_dimensions \
            * (self.number_of_grid_points_per_dimension**self.number_of_dimensions)
        self.real_space_grid = f_real.real * scaling

        # 5. Interpolator setup for D dimensions
        coord_axis = np.linspace(-self.number_of_grid_points_per_dimension / 2,
                                 self.number_of_grid_points_per_dimension / 2,
                                 self.number_of_grid_points_per_dimension,
                                 endpoint=False) * self.dx
        self.coords = tuple(coord_axis for _ in range(self.number_of_dimensions))

        self._interp = RegularGridInterpolator(
            self.coords,
            self.real_space_grid,
            bounds_error=False,
            fill_value=0
        )

    def __call__(self, points):
        """
        Args:
            points (np.ndarray): Shape (..., D) representing coordinates.

        Returns:
            np.ndarray: The interpolated real space values at those points.
        """
        return self._interp(points)


# ----------------------------------------------------------------------------------

# In this section, we define the kernel implemented in this file

def deconvolving_kernel_round_support(
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

    differences, h, _ = calculate_difference_matrix(query_points,
                                                 data_points,
                                                 covariance_matrices,
                                                 bandwidth,
                                                 bandwidth_scaling=bandwidth_scaling,
                                                 rescaled_norm=False,
                                                 eigenvalue_calculation=eigenvalue_calculation)

    # Build one deconvolving kernel per data point, since each data point carries
    # its own covariance matrix, and evaluate it at the corresponding differences.
    kernel_values = np.zeros((M, N))
    for i in range(N):
        kernel_function = FourierTransformerND(charc_func_normal,
                                               circular_mask,
                                               number_of_dimensions=D,
                                               bandwidth=h[i,0],
                                               covariance_matrix=covariance_matrices[i],
                                               list_of_all_covariance_matrices=covariance_matrices,
                                               number_of_grid_points_per_dimension=discrete_fourier_transformation_grid_size,
                                               length_per_dimension=20.0,
                                               rescaled_norm=rescaled_norm)
        kernel_values[:, i] = kernel_function(differences[:, i, :])

    bandwidth_normalization = np.prod(h, axis=1)
    kernel_values = kernel_values / (bandwidth_normalization[np.newaxis, :])

    return kernel_values


def deconvolving_kernel_heterogeneous_noise(
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

    differences, h, _ = calculate_difference_matrix(query_points,
                                                 data_points,
                                                 covariance_matrices,
                                                 bandwidth,
                                                 bandwidth_scaling=bandwidth_scaling,
                                                 rescaled_norm=False,
                                                 eigenvalue_calculation=eigenvalue_calculation)

    # Build one deconvolving kernel per data point, since each data point carries
    # its own covariance matrix, and evaluate it at the corresponding differences.
    kernel_values = np.zeros((M, N))
    for i in range(N):
        kernel_function = FourierTransformerND(psi,
                                               circular_mask,
                                               number_of_dimensions=D,
                                               bandwidth=h[i,0],
                                               covariance_matrix=covariance_matrices[i],
                                               list_of_all_covariance_matrices=covariance_matrices,
                                               number_of_grid_points_per_dimension=discrete_fourier_transformation_grid_size,
                                               length_per_dimension=20.0,
                                               rescaled_norm=rescaled_norm)
        kernel_values[:, i] = kernel_function(differences[:, i, :])

    bandwidth_normalization = np.prod(h, axis=1)
    kernel_values = kernel_values / (bandwidth_normalization[np.newaxis, :])

    return kernel_values

