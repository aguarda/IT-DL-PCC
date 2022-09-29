import pickle
import numpy as np

import pc2vox
import sampling

from pathlib import Path
from torch.utils.data import Dataset


def read_blk_pts(filename):
    """Loads a Pickle (.pkl) file containing a PC block as a list of coordinates."""

    with open(filename, 'rb') as f:
        pts = pickle.load(f)

    # Convert coordinates to 3D block
    blk = pc2vox.point2vox(pts, 64)

    blk = np.squeeze(blk, axis=0)

    return blk


def read_blk_ptscolor(filename):
    """Loads a Pickle (.pkl) file containing a PC block as a list of coordinates and colors."""

    with open(filename, 'rb') as f:
        pts, col = pickle.load(f)

    # Scale colors from [0, 255] to [0, 1]
    col = col / 255

    # Convert coordinates to 3D block
    blk = pc2vox.point2vox(pts, 64, col)

    blk = np.squeeze(blk, axis=0)

    return blk


def read_blk_pts_sr(filename, sfactor, blk_size):
    """Loads a Pickle (.pkl) file containing a PC block as a list of coordinates."""

    with open(filename, 'rb') as f:
        pts = pickle.load(f)

    # Down and Up-sample coordinates
    pts_lr, _, _ = sampling.pc_downsampling(pts, sfactor)
    pts_lr = sampling.pc_upsampling(pts_lr, sfactor)

    # Convert coordinates to 3D block
    blk_in = np.squeeze(pc2vox.point2vox(pts_lr, blk_size), axis=0)
    blk_out = np.squeeze(pc2vox.point2vox(pts, blk_size), axis=0)

    return {"blk_in": blk_in, "blk_out": blk_out}


def read_blk_ptscolor_sr(filename, sfactor, blk_size):
    """Loads a Pickle (.pkl) file containing a PC block as a list of coordinates and colors."""

    with open(filename, 'rb') as f:
        pts, col = pickle.load(f)

    # Scale colors from [0, 255] to [0, 1]
    col = col / 255

    # Down and Up-sample coordinates
    pts_lr, col_lr, _ = sampling.pc_downsampling(pts, sfactor, in_color=col)
    pts_lr = sampling.pc_upsampling(pts_lr, sfactor)

    # Convert coordinates to 3D block
    blk_in = np.squeeze(pc2vox.point2vox(pts_lr, blk_size, col_lr), axis=0)
    blk_out = np.squeeze(pc2vox.point2vox(pts, blk_size, col), axis=0)

    return {"blk_in": blk_in, "blk_out": blk_out}


class PCdataFolder(Dataset):
    """Load a Point Cloud folder database. Training and testing image samples
    are respectively stored in separate directories.
    """

    def __init__(self, root, with_color=False, super_resolution=False, sfactor=2, blk_size=64):
        splitdir = Path(root)

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.with_color = with_color
        self.super_resolution = super_resolution
        self.sf = sfactor
        self.bs = blk_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        if self.super_resolution:
            if self.with_color:
                blk = read_blk_ptscolor_sr(self.samples[index], self.sf, self.bs)
            else:
                blk = read_blk_pts_sr(self.samples[index], self.sf, self.bs)
        else:
            if self.with_color:
                blk = read_blk_ptscolor(self.samples[index])
            else:
                blk = read_blk_pts(self.samples[index])

        return blk

    def __len__(self):
        return len(self.samples)
