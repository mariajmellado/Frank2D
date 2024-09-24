from fourier2d import DiscreteFourierTransform2D as ft2d
import numpy as np
from scipy.stats import binned_statistic_2d


class Gridding(object):
    def __init__(self, N, Rmax, unshifted = True):
        self._N = N
        self._Rmax = Rmax

        FT = ft2d(Rmax, N)
        self._u_ft =  FT._u
        self._v_ft =  FT._v
        
        self._unshifted = unshifted

    
    def run(self, u, v, Vis, Weights, type = 'weighted'): 
        print("Gridding...")
        # Calculating bin edges.
        bin_centers = self.edges_centers(self._u_ft)[0]
        bin_edges_u = self.edges_centers(self._u_ft)[1]
        bin_edges_v = self.edges_centers(self._v_ft)[1]

        if type == 'weighted':
            u_gridded, v_gridded, vis_gridded, weights_gridded = self.weighted_gridding(u, v, Vis, Weights, bin_centers, bin_edges_u, bin_edges_v)

        if self._unshifted:
            u_gridded, v_gridded, vis_gridded, weights_gridded = self.unshiftting(u_gridded, v_gridded, vis_gridded, weights_gridded, self._N)

        return u_gridded, v_gridded, vis_gridded, weights_gridded

    def edges_centers(self, freq):
        correction = (freq[1] - freq[0])/2
        bin_centers = np.fft.fftshift(freq)
        edges_=  bin_centers - correction
        edges = np.concatenate((edges_, [edges_[-1] + 2*correction]))
        return bin_centers, edges
    
    
    def weighted_gridding(self, u, v, Vis, Weights, centers, edges_u, edges_v):
        # Calculating grid
        vis_weights_sum_bin, _, _, _ = binned_statistic_2d(u, v, Vis*Weights, 'sum', bins=[edges_u, edges_v], expand_binnumbers = False)
        weights_sum_bin, _, _, _ = binned_statistic_2d(u, v, Weights, 'sum', bins=[edges_u, edges_v], expand_binnumbers = False)
        vis_gridded_matrix =  vis_weights_sum_bin/weights_sum_bin
        weights_gridded_matrix, _, _, _ = binned_statistic_2d(u, v, Weights, 'sum', bins=[edges_u, edges_v], expand_binnumbers = False)
        
    
        u_gridded, v_gridded, vis_gridded, weights_gridded = self.freq_vis_gridded_1d(centers, vis_gridded_matrix, weights_gridded_matrix)
    
        # change Nans by 0 in vis.
        vis_gridded = np.nan_to_num(vis_gridded, nan=0)
        weights_gridded = np.nan_to_num(weights_gridded, nan=0)
    
        # Arrays for UV-table.
        return u_gridded, v_gridded, vis_gridded, weights_gridded

    
    def freq_vis_gridded_1d(self, freqs, vis_matrix, weights_matrix):
        vis_gridded = vis_matrix.flatten()
        weights_gridded = weights_matrix.flatten()
        u_, v_ = np.meshgrid(freqs, freqs, indexing='ij') 
        u_gridded, v_gridded = u_.reshape(-1), v_.reshape(-1)
        return u_gridded, v_gridded, vis_gridded, weights_gridded

    
    def unshiftting(self, u_gridded, v_gridded, vis_gridded, weights_gridded, N):
        # shift in u.
        indexes_pos = np.where(u_gridded>=0)[0]
        indexes_neg = np.where(u_gridded<0)[0]
        new_indexes_u = np.concatenate((indexes_pos, indexes_neg))
        new_u,  new_v, new_vis, new_weights= u_gridded[new_indexes_u], v_gridded[new_indexes_u], vis_gridded[new_indexes_u], weights_gridded[new_indexes_u]
    
        # shift in v.
        new_indexes = self.get_reordered_indices_v_freq(new_v)
        new_u, new_v, new_vis, new_weights =  new_u[new_indexes], new_v[new_indexes], new_vis[new_indexes], new_weights[new_indexes]
    
        new_weights[new_weights==0] = 1e-5 * np.median(new_weights[new_weights!=0])
        
        return new_u, new_v, new_vis, new_weights
    
    def get_reordered_indices_v_freq(self, array):
        # Number of complete blocks in the array
        num_blocks = len(array) // self._N
    
        # Reorder pattern: 0 first, then positives, then negatives
        block_indices = np.arange(self._N)
        reordered_block_indices = np.concatenate((
            block_indices[array[block_indices] == 0],   # Index of 0
            block_indices[array[block_indices] > 0],    # Indices of positives
            block_indices[array[block_indices] < 0]     # Indices of negatives
        ))
    
        # Apply the reordering pattern to all blocks
        reordered_indices = np.concatenate([
            reordered_block_indices + i * self._N for i in range(num_blocks)
        ])
    
        return reordered_indices
    
