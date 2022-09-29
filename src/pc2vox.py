import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

# Load PC and get coordinates
def load_pc(name, with_color=False, with_normals=False):
    in_pc = PyntCloud.from_file(name)
    points = in_pc.xyz
    points = points.astype(np.uint32)

    colors = None
    if with_color:
        assert 'red' in in_pc.points and 'green' in in_pc.points and 'blue' in in_pc.points, "The input Point Cloud does not have colors!"
        colors = np.transpose([in_pc.points.red, in_pc.points.green, in_pc.points.blue])

    normals = None
    if with_normals:
        assert 'nx' in in_pc.points and 'ny' in in_pc.points and 'nz' in in_pc.points, "The input Point Cloud does not have normals needed for D2 optimization!"
        normals = np.transpose([in_pc.points.nx, in_pc.points.ny, in_pc.points.nz])

    return points, colors, normals


# Write PC to PLY file
def save_pc(name, pts_geom, pts_colo=None):
    pts_geom = pts_geom.astype(np.float32)
    if pts_colo is None:
        geom = {'x': pts_geom[:, 0], 'y': pts_geom[:, 1], 'z': pts_geom[:, 2]}
    else:
        pts_colo = pts_colo.astype(np.uint8)
        geom = {'x': pts_geom[:, 0], 'y': pts_geom[:, 1], 'z': pts_geom[:, 2], 'red': pts_colo[:, 0], 'green': pts_colo[:, 1], 'blue': pts_colo[:, 2]}
    cloud = PyntCloud(pd.DataFrame(data=geom))
    cloud.to_file(name)


# Divide PC into blocks
def pc2blocks(points, blk_size, colors=None, normals=None):
    blk_map_full = points // blk_size
    blk_map, point2blk_idx, blk_len = np.unique(blk_map_full, return_inverse=True, return_counts=True, axis=0)
    num_blocks = blk_map.shape[0]
    blk_p_coord = points % blk_size

    if colors is None and normals is None:
        blocks = [np.zeros((i, 3)) for i in blk_len]
        for i in range(num_blocks):
            blocks[i] = blk_p_coord[point2blk_idx == i, :]

        return blocks, blk_map, None, None
    elif normals is None:
        blocks = [np.zeros((i, 3)) for i in blk_len]
        blk_colors = [np.zeros((i, 3)) for i in blk_len]
        for i in range(num_blocks):
            blocks[i] = blk_p_coord[point2blk_idx == i, :]
            blk_colors[i] = colors[point2blk_idx == i, :]

        return blocks, blk_map, blk_colors, None
    elif colors is None:
        blocks = [np.zeros((i, 3)) for i in blk_len]
        blk_normals = [np.zeros((i, 3)) for i in blk_len]
        for i in range(num_blocks):
            blocks[i] = blk_p_coord[point2blk_idx == i, :]
            blk_normals[i] = normals[point2blk_idx == i, :]

        return blocks, blk_map, None, blk_normals
    else:
        blocks = [np.zeros((i, 3)) for i in blk_len]
        blk_colors = [np.zeros((i, 3)) for i in blk_len]
        blk_normals = [np.zeros((i, 3)) for i in blk_len]
        for i in range(num_blocks):
            blocks[i] = blk_p_coord[point2blk_idx == i, :]
            blk_colors[i] = colors[point2blk_idx == i, :]
            blk_normals[i] = normals[point2blk_idx == i, :]

        return blocks, blk_map, blk_colors, blk_normals


# Convert point coordinates into a block of binary voxels
def point2vox(block, blk_size, block_color=None):
    blk_size = np.int32(blk_size)
    block = block.astype(np.int32)
    if block_color is None:
        output = np.zeros([1, 1, blk_size, blk_size, blk_size], dtype=np.float32)
        output[0, 0, block[:, 0], block[:, 1], block[:, 2]] = 1.0
    else:
        output = np.zeros([1, 4, blk_size, blk_size, blk_size], dtype=np.float32)
        output[0, 0, block[:, 0], block[:, 1], block[:, 2]] = 1.0
        output[0, 1:, block[:, 0], block[:, 1], block[:, 2]] = block_color

    return output


# Convert block of binary voxels into point coordinates
def vox2point(block, with_color=False):
    if not with_color:
        assert len(block.shape) == 3, "Expected 3D shape (3D block)"

        idx_a, idx_b, idx_c = np.where(block)
        points = np.stack((idx_a, idx_b, idx_c), axis=1)

        return points
    else:
        assert len(block.shape) == 4, "Expected 4D shape (3D block with 4 channels)"

        idx_a, idx_b, idx_c = np.where(block[0, :, :, :])
        points = np.stack((idx_a, idx_b, idx_c), axis=1)
        colors = block[1:, idx_a, idx_b, idx_c]

        return points, colors
    
