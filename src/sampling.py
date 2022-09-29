import numpy as np
from sklearn.neighbors import NearestNeighbors


# Compute the mean distance between each point and its k closest neighbors, and return the median across the PC
def get_med_dist(in_pts, k):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(in_pts)
    total_dist, _ = nbrs.kneighbors(in_pts)
    dist = np.median(np.delete(total_dist, 0, 1).mean(axis=1))
    assert dist >= 1, f'Distance between points can not be lower than 1! Computed: {dist}'
    return dist


# Get the closest previous power of 2
def get_pow2_factor(dist):
    factor = 2 ** (np.floor(np.log2(dist)))
    return factor


# Get the closest previous integer (round down)
def get_int_factor(dist):
    factor = np.floor(dist)
    return factor


# Down-sample PC according to specified factor
def pc_downsampling(in_pts, factor, in_color=None, in_normal=None):
    if factor == 1:
        return in_pts, in_color, in_normal
    # Apply the scale factor to the point coordinates
    out_pts = np.floor(((in_pts + 1) / factor) + 0.5).astype(np.int32) - 1
    # Truncate values
    out_pts[np.where(out_pts < 0)] = 0
    # Remove duplicates
    out_pts, out_idx = np.unique(out_pts, axis=0, return_index=True)  # TODO: Average color for duplicated points
    if in_color is None:
        out_color = None
    else:
        out_color = in_color[out_idx, :]
        
    if in_normal is None:
        out_normal = None
    else:
        out_normal = in_normal[out_idx, :]
    
    return out_pts, out_color, out_normal


# Up-sample PC according to specified factor
def pc_upsampling(in_pts, factor):
    if factor == 1:
        return in_pts
    # Apply the scale factor to the point coordinates
    out_pts = np.floor(((in_pts + 1) * factor) + 0.5).astype(np.int32) - 1
    # Truncate values
    out_pts[np.where(out_pts < 0)] = 0
    return out_pts
