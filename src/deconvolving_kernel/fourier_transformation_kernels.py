# -------------------------------------------------------------------------------------------
# Fourier transformation
import numpy as np
import matplotlib.pyplot as plt
from kernels import calculate_difference_matrix, check_dimensions
from scipy.fft import fftshift, ifftshift, ifftn
from scipy.interpolate import RegularGridInterpolator



def circular_mask(*K_grids, covariance_matrix=None):
    """
    Generalized N-dimensional mask function. 
    Transforms into an ellipsoidal mask if a covariance matrix is provided.
    
    Args:
        *K_grids: Arbitrary number of multidimensional coordinate arrays (KX, KY, KZ, ...).
        radius: The cutoff radius for the mask.
        cov: Covariance matrix of shape (ndim, ndim). If None, defaults to identity (spherical).
        
    Returns:
        A boolean mask array of the same shape as the input grids.
    """
    ndim = len(K_grids)
    
    if covariance_matrix is None or np.allclose(covariance_matrix, np.eye(ndim)):
        # --- Case 1: Sphmask = mask_of_support(*K_grids, covariance_matrix=self.covariance_matrix)erical Mask (No covariance / Identity matrix) ---
        # Sum the squares of all coordinate grids dynamically
        k_squares = sum(K**2 for K in K_grids)
        K_radius = np.sqrt(k_squares)
    else:
        # --- Case 2: Ellipsoidal Mask (With covariance) ---
        # Flatten and stack the grids to perform matrix multiplication efficiently
        # Shape will be (ndim, total_number_of_points)
        flat_grids = [K.ravel() for K in K_grids]
        K_vectors = np.vstack(flat_grids)
        
        # The inverse covariance scales/rotates the Fourier space axes
        inv_cov = np.linalg.inv(covariance_matrix)
        
        # Compute the quadratic form: K^T * Sigma^-1 * K for every point
        # This is equivalent to dot(inv_cov, K_vectors) and summing element-wise with K_vectors
        transformed_distances = np.sum(K_vectors * (inv_cov @ K_vectors), axis=0)
        
        # Reshape the flattened results back to the original grid shape
        K_radius = np.sqrt(transformed_distances).reshape(K_grids[0].shape)
        
    # The factor of 2*np.pi applies if 'radius' is defined in angular frequency (w) 
    # instead of ordinary frequency (f). Adjust the threshold here if needed.
    mask = (K_radius <= 1)
    return mask


def charc_func_normal(*K_grids, h, cov):
    """
    Generalized N-dimensional characteristic function of a normal distribution.
    Computes exp(0.5 * k^T * cov * k / h^2) across an arbitrary number of dimensions.
    
    Args:
        *K_grids: Arbitrary number of multidimensional coordinate arrays (KX, KY, KZ, ...).
        h: Scaling parameter.
        cov: Covariance matrix of shape (ndim, ndim).
        cov_list: Optional list of covariance matrices (unused here, kept for API matching).
        
    Returns:
        An array of the same shape as the input grids containing the evaluated function.
    """
    ndim = len(K_grids)
    grid_shape = K_grids[0].shape
    
    # 1. Flatten and stack grids to form a matrix of vectors: shape (ndim, total_points)
    flat_grids = [K.ravel() for K in K_grids]
    K_vectors = np.vstack(flat_grids)
    
    # 2. Compute the quadratic form: k^T * cov * k for all points simultaneously
    # 'ij,ji->i' multiplies cov by each vector, then takes the dot product with the vector again
    # This replaces your manual: cov[0,0]*KX**2 + (cov[0,1] + cov[1,0])*KX*KY + ...
    norm = np.einsum('ji,jk,ki->i', K_vectors, cov, K_vectors)
    
    # 3. Apply the scaling factor h
    norm = norm / (h**2)
    
    # 4. Reshape the flat result back into the original grid structure
    return np.exp(0.5 * norm).reshape(grid_shape)

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
        """
        Generalized N-Dimensional Discrete Fourier Transformer.
        
        Args:
            func: Function to evaluate in Fourier space.
            mask_func: Mask function filtering Fourier space.
            ndim: Number of dimensions (n).
            radius: Radius parameter passed to mask_func.
            h: Height/scale parameter passed to func.
            cov: Default covariance matrix (defaults to identity matrix of size ndim).
            cov_list: Optional list of covariance matrices.
            N: Number of grid points per dimension.
            L: Total length of the domain per dimension in Fourier space.
        """
        self.number_of_dimensions = number_of_dimensions
        self.function = function
        self.mask_of_support = mask_of_support
        self.number_of_grid_points_per_dimension = number_of_grid_points_per_dimension
        self.length_per_dimension = length_per_dimension
        self.bandwidth = bandwidth
        self.covariance_matrix = covariance_matrix
        self.list_of_all_covariance_matrices = list_of_all_covariance_matrices
        
        self.dk = self.length_per_dimension / self.number_of_grid_points_per_dimension    
        self.dx = (2 * np.pi) / self.length_per_dimension
        

        # 1. Generate N-dimensional grid for Fourier space
        k_vals = np.linspace(-self.length_per_dimension/2, self.length_per_dimension/2, self.number_of_grid_points_per_dimension, endpoint=False)
        # Create a list of 1D coordinate arrays for meshgrid
        k_grids = [k_vals] * self.number_of_dimensions
        # Using indexing='ij' is crucial for proper multidimensional matrix alignment
        K_grids = np.meshgrid(*k_grids, indexing='ij')
        
        # 2. Apply Mask and Evaluate Function
        # Unpack the list of grids into the mask and evaluation functions
        if rescaled_norm:
            mask = mask_of_support(*K_grids, covariance_matrix=self.covariance_matrix)
        else:
            mask = mask_of_support(*K_grids, covariance_matrix=)
        
        fourier_space = np.zeros(K_grids[0].shape)
        # Extract masked coordinates as a tuple of flattened arrays
        masked_coords = tuple(grid[mask] for grid in K_grids)
        
        fourier_space[mask] = function(*masked_coords, 
                                       bandwidth=self.bandwidth, 
                                       covariance_matrix=self.covariance_matrix, 
                                       list_of_all_covariance_matrices=self.list_of_all_covariance_matrices)
        self.fourier_space = fourier_space

        # 3. Inverse N-Dimensional Fourier Transform
        f_real = fftshift(ifftn(ifftshift(self.fourier_space)))

        # 4. Scaling for continuous IFT in N-dimensions: (dk / 2pi)^ndim * N^ndim
        scaling = (self.dk / (2 * np.pi))**self.ndim * (self.number_of_grid_points_per_dimension**self.ndim)
        self.real_space_grid = f_real.real * scaling

        # 5. Interpolator Setup for N-dimensions
        coord_axis = np.linspace(-self.number_of_grid_points_per_dimension/2, self.number_of_grid_points_per_dimension/2, self.number_of_grid_points_per_dimension, endpoint=False) * self.dx
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
            points: A numpy array of shape (..., ndim) representing coordinates.
        Returns:
            The interpolated real-space values at those points.
        """
        return self._interp(points)


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
                                                 bandwidth_scaling,
                                                 rescaled_norm,
                                                 eigenvalue_calculation)
    kernel_values = np.zeros((M, N))
    for i in range(N):
        kernel_function = FourierTransformerND(charc_func_normal,
                                               circular_mask,
                                               number_of_dimensions=D
                                               bandwidth = h[i],
                                               covariance_matrix=covariance_matrices[i],
                                               list_of_all_covariance_matrices=covariance_matrices,
                                               number_of_grid_points_per_dimension=discrete_fourier_transformation_grid_size,
                                               length_per_dimension=20.0,
                                               rescaled_norm=rescaled_norm)
        kernel_values[:, i] = kernel_function(differences[:, i, :])
    
    return kernel_values
