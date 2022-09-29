# Copyright (c) 2022, Instituto de Telecomunicações

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
"""Instituto de Telecomunicações Deep Learning-based Point Cloud Codec (IT-DL-PCC).

This is the software of the codecs submitted to the Call for Proposals on JPEG
Pleno Point Cloud Coding issued on January 2022:
    - IT-DL-PCC-G: Geometry-only codec
    - IT-DL-PCC-GC: Joint geometry + color codec

These proposals have been originated by research developed at
Instituto de Telecomunicações (IT), in the context of the project Deep-PCR
entitled “Deep learning-based Point Cloud Representation” (PTDC/EEI-COM/1125/2021),
financed by Fundação para a Ciência e Tecnologia (FCT).


The DL coding model is based on the image compression model published in:
J. Ballé, D. Minnen, S. Singh, S.J. Hwang, N. Johnston:
"Variational Image Compression with a Scale Hyperprior"
Int. Conf. on Learning Representations (ICLR), 2018
https://arxiv.org/abs/1802.01436

"""

import argparse
import sys
import os
import numpy as np
import pickle
import gzip

from absl import app
from absl.flags import argparse_flags

import torch

import loss_functions
import topk
import pc2vox
import sampling
import dl_coding_model
import dl_sr_model


def compress(args):
    """Compresses all test voxel_blocks."""
    # Manage input and output directories
    in_file = args.input_file
    if not in_file.endswith('.ply'):
        raise ValueError("Input must be a PLY file (.ply extension).")

    pc_filename = os.path.splitext(os.path.basename(in_file))[0]
    stream_dir = os.path.join(args.output_dir, pc_filename)
    os.makedirs(stream_dir, exist_ok=True)

    # Load input PC, get list of coordinates
    if 'd2' in args.topk_metrics.lower():
        get_normals = True
    else:
        get_normals = False
    in_points, in_colors, in_normals = pc2vox.load_pc(in_file, with_color=args.with_color, with_normals=get_normals)
    # Convert RGB values from 0-255 to 0-1
    if args.with_color:
        in_colors = in_colors / 255
    # Estimate sampling factor
    if args.scale is None:
        dist = sampling.get_med_dist(in_points, 5)
        sfactor = sampling.get_pow2_factor(dist)
    else:
        sfactor = args.scale

    # Divide PC into blocks of the desired size. Get list of relative coordinates for points in each block
    blocks, blk_map, blocks_colors, blocks_normals = pc2vox.pc2blocks(in_points, args.blk_size*sfactor, colors=in_colors, normals=in_normals)
    num_blk_points_hr = np.array([blk.shape[0] for blk in blocks])
    
    # Load the latest model checkpoint
    if args.with_color:
        model = dl_coding_model.CodingModel(args.num_filters, 4)
    else:
        model = dl_coding_model.CodingModel(args.num_filters, 1)

    device = torch.device("cuda") if args.cuda and torch.cuda.is_available() else torch.device("cpu")

    ckpt = torch.load(args.model_dir, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    if args.cuda:
        model.to(device)
    model.update()
    
    if args.use_sr:
        if args.with_color:
            sr_model = dl_sr_model.SRModel(args.sr_filters, 4)
        else:
            sr_model = dl_sr_model.SRModel(args.sr_filters, 1)

        sr_ckpt = torch.load(args.sr_model_dir, map_location=device)
        sr_model.load_state_dict(sr_ckpt["state_dict"])

        if args.cuda:
            sr_model.to(device)
    
    final_bitstream = []
    final_rho = np.zeros(len(blocks), float)
    final_rho_sr = np.zeros(len(blocks), float)
    final_occupation = np.zeros(len(blocks), np.uint8)
    num_blk_points_lr = np.zeros(len(blocks), np.uint32)

    octants = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])

    geo_metric, color_metric = loss_functions.get_metrics(args.topk_metrics, with_color=args.with_color)
    
    # Iterate all blocks
    for i in range(len(blocks)):
        
        temp_blk_hr = blocks[i]
        if args.with_color:
            temp_col_hr = blocks_colors[i]
        else:
            temp_col_hr = None
        if blocks_normals is not None:
            temp_norm_hr = blocks_normals[i]
        else:
            temp_norm_hr = None

        # Down-sample block
        temp_blk_lr, temp_col_lr, temp_norm_lr = sampling.pc_downsampling(temp_blk_hr, sfactor, in_color=temp_col_hr, in_normal=temp_norm_hr)
        num_blk_points_lr[i] = temp_blk_lr.shape[0]
        
        # Encode block
        x = pc2vox.point2vox(temp_blk_lr, args.blk_size, temp_col_lr)
        with torch.no_grad():
            tensors = model.compress(torch.from_numpy(x).to(torch.device(device)), torch.tensor(args.q_step).to(torch.device(device)))
        final_bitstream.extend([tensors["strings"]])
        # Decode block
        with torch.no_grad():
            x_hat = model.decompress(tensors["strings"], torch.tensor(tensors["shape"]).to(torch.device(device)), torch.tensor(args.q_step).to(torch.device(device))).cpu().numpy()
        # Determine octant occupation
        blk_occupation = np.unique(temp_blk_lr // (args.blk_size/2), axis=0)
        final_occupation[i] = np.packbits((octants[:, None] == blk_occupation).all(-1).any(-1))
        # Sort all block probabilities
        sorted_probabilities = topk.largest_indices(np.squeeze(x_hat[:, 0, :, :, :]), args.blk_size**3)
        tmp_octants = sorted_probabilities // (args.blk_size/2)
        oct_idx = (tmp_octants[:, None] == blk_occupation).all(-1).any(-1)
        sorted_probabilities = sorted_probabilities[oct_idx]

        # Optimize binarization threshold as a Top-k * rho
        if args.use_fast_topk:
            final_rho[i] = topk.fast_topk_optimization(temp_blk_lr, temp_col_lr, temp_norm_lr,
                                                        sorted_probabilities, x_hat[0, 1:, :, :, :],
                                                        num_blk_points_lr[i], geo_metric, color_metric, args.blk_size,
                                                        color_weight=args.color_weight, max_rho=args.max_topk, patience=args.topk_patience)
        else:
            final_rho[i] = topk.full_topk_optimization(temp_blk_lr, temp_col_lr, temp_norm_lr,
                                                        sorted_probabilities, x_hat[0, 1:, :, :, :],
                                                        num_blk_points_lr[i], geo_metric, color_metric, args.blk_size,
                                                        color_weight=args.color_weight, max_rho=args.max_topk, patience=args.topk_patience)

        if args.use_sr and args.sr_topk in ['full','fast']:
            # Generate decoded block
            decoded_x = sorted_probabilities[:round(num_blk_points_lr[i]*final_rho[i])]
            if args.with_color:
                decoded_colors = x_hat[0, 1:, decoded_x[:, 0], decoded_x[:, 1], decoded_x[:, 2]]
            else:
                decoded_colors = None
            # Grid Basic Upsampling block
            dec_x_up = sampling.pc_upsampling(decoded_x, sfactor)
            bin_block = pc2vox.point2vox(dec_x_up, args.blk_size*sfactor, decoded_colors)
            # Avanced Set Upsampling Block
            with torch.no_grad():
                unet_block = sr_model(torch.from_numpy(bin_block).to(torch.device(device))).cpu().numpy()
            # Sort all block probabilities
            sorted_probabilities = topk.largest_indices(np.squeeze(unet_block[:, 0, :, :, :]), (args.blk_size*sfactor)**3)
            tmp_octants = sorted_probabilities // (args.blk_size*sfactor/2)
            oct_idx = (tmp_octants[:, None] == blk_occupation).all(-1).any(-1)
            sorted_probabilities = sorted_probabilities[oct_idx]
            # Optimize binarization threshold as a Top-k * rho              
            if args.sr_topk == 'full':
                final_rho_sr[i] = topk.full_topk_optimization(temp_blk_hr, temp_col_hr, temp_norm_hr,
                                                                sorted_probabilities, unet_block[0, 1:, :, :, :],
                                                                num_blk_points_hr[i], geo_metric, color_metric, args.blk_size*sfactor,
                                                                color_weight=args.color_weight, max_rho=args.sr_max_topk, patience=args.topk_patience)
            else:
                final_rho_sr[i] = topk.fast_topk_optimization(temp_blk_hr, temp_col_hr, temp_norm_hr,
                                                                sorted_probabilities, unet_block[0, 1:, :, :, :],
                                                                num_blk_points_hr[i], geo_metric, color_metric, args.blk_size*sfactor,
                                                                color_weight=args.color_weight, max_rho=args.sr_max_topk, patience=args.topk_patience)


    num_dec_pts = np.around(num_blk_points_lr * final_rho).astype(int)
    
    if not args.use_sr:
        num_dec_pts_sr = None
    elif args.sr_topk in ['full','fast']:
        num_dec_pts_sr = np.around(num_blk_points_hr * final_rho_sr).astype(int)
    else:
        num_dec_pts_sr = np.around(num_blk_points_hr * final_rho).astype(int)
            
    # Write to bitstream
    with gzip.open(os.path.join(stream_dir, pc_filename + ".gz"), "wb") as f:
        pickle.dump([blk_map, num_dec_pts, num_dec_pts_sr, final_occupation, sfactor, args.q_step, tensors["shape"], final_bitstream], f, pickle.HIGHEST_PROTOCOL)


def decompress(args):
    """Decompresses all test voxel_blocks."""
    # Manage input and output directories
    stream_filename = args.input_file
    if not stream_filename.endswith('.gz'):
        raise ValueError("Input bitstream file must have .gz extension.")

    with gzip.open(stream_filename, 'rb') as f:
        blk_map, num_dec_pts, num_dec_pts_sr, final_occupation, sfactor, q_step, z_shape, final_bitstream = pickle.load(f)

    # Check if SR is being used
    if num_dec_pts_sr is None:
        use_sr = False
    else:
        use_sr = True
        if not args.sr_model_dir:
            raise ValueError("If using Super-Resolution, need directory for SR trained model.")

    # Initialize the reconstructed PC (empty)
    pts_geom = np.array([], dtype=np.int32).reshape(0, 3)
    if args.with_color:
        pts_col = np.array([], dtype=np.int32).reshape(0, 3)
    else:
        pts_col = None
        colors = None

    # Load the latest model checkpoint
    if args.with_color:
        model = dl_coding_model.CodingModel(args.num_filters, 4)
    else:
        model = dl_coding_model.CodingModel(args.num_filters, 1)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.model_dir, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    if args.cuda:
        model.to(device)
    model.update()

    if use_sr:
        if args.with_color:
            sr_model = dl_sr_model.SRModel(args.sr_filters, 4)
        else:
            sr_model = dl_sr_model.SRModel(args.sr_filters, 1)

        sr_ckpt = torch.load(args.sr_model_dir, map_location=device)
        sr_model.load_state_dict(sr_ckpt["state_dict"])

        if args.cuda:
            sr_model.to(device)

    # Iterate all blocks
    octants = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    for i in range(len(num_dec_pts)):
        # Unpack string corresponding to the coded block
        strings = final_bitstream[i]
        # Decode block
        with torch.no_grad():
            x_hat = model.decompress(strings, torch.tensor(z_shape).to(torch.device(device)), torch.tensor(q_step).to(torch.device(device))).cpu().numpy()
        blk_size = z_shape[0] * 4 * 8
        # Convert back to point coordinates using a top-k * rho approach, where rho was optimized to minimize distortion
        # Sort all block probabilities, considering only occupied octants
        blk_occupation = octants[np.unpackbits(final_occupation[i]).astype(bool)]
        sorted_probabilities = topk.largest_indices(np.squeeze(x_hat[:, 0, :, :, :]), blk_size**3)
        tmp_octants = sorted_probabilities // (blk_size/2)
        oct_idx = (tmp_octants[:, None] == blk_occupation).all(-1).any(-1)
        sorted_probabilities = sorted_probabilities[oct_idx]
        # Get decoded block points
        points = sorted_probabilities[:num_dec_pts[i]]
        if args.with_color:
            colors = x_hat[0, 1:, points[:, 0], points[:, 1], points[:, 2]]
        
        # Grid Basic Upsampling block
        points = sampling.pc_upsampling(points, sfactor)
        
        if use_sr:
            # Grid Basic Upsampling block
            bin_block = pc2vox.point2vox(points, blk_size*sfactor, colors)
            # Avanced Set Upsampling Block
            with torch.no_grad():
                unet_block = sr_model(torch.from_numpy(bin_block).to(torch.device(device))).cpu().numpy()
            # Sort all block probabilities
            sorted_probabilities = topk.largest_indices(np.squeeze(unet_block[:, 0, :, :, :]), (blk_size*sfactor)**3)
            tmp_octants = sorted_probabilities // (blk_size*sfactor/2)
            oct_idx = (tmp_octants[:, None] == blk_occupation).all(-1).any(-1)
            sorted_probabilities = sorted_probabilities[oct_idx]
            # Binarization
            points = sorted_probabilities[:num_dec_pts_sr[i]]
            if args.with_color:
                colors = unet_block[0, 1:, points[:, 0], points[:, 1], points[:, 2]]
        
        # Merge block points in the fully reconstructed PC
        points = points + (blk_size*sfactor * blk_map[i])
        pts_geom = np.concatenate((pts_geom, points))
        if args.with_color:
            colors = np.around(colors * 255)
            pts_col = np.concatenate((pts_col, colors))
    # Write reconstructed PC to file
    pc2vox.save_pc(stream_filename + ".dec.ply", pts_geom, pts_col)


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument(
        "--with_color", action="store_true",
        help="Jointly compress both geometry and color.")
    parser.add_argument(
        "--num_filters", type=int, default=32,
        help="Number of filters in first convolutional layer of the coding model(default: %(default)s).")
    parser.add_argument(
        "--sr_filters", type=int, default=16,
        help="Number of filters in first convolutional layer of the super-resolution model (default: %(default)s).")
    parser.add_argument(
        "--cuda", action="store_true",
        help="Use cuda")

    subparsers = parser.add_subparsers(
        title="commands", dest="command",
        help="What to do: "
             "'compress' reads the test PC file and writes a compressed binary stream."
             "'decompress' reads the binary stream and reconstructs the PC."
             "input filenames need to be provided. Invoke '<command> -h' for more information.")

    # 'compress' subcommand
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a Point Cloud PLY file, compresses it, and writes the bitstream file.")
    # Mandatory arguments
    compress_cmd.add_argument(
        "input_file",
        help="Input Point Cloud filename (.ply).")
    compress_cmd.add_argument(
        "model_dir",
        help="Directory where to load model checkpoints."
             " For compression, a single directory should be provided: ../models/test/checkpoint_best_loss.pth.tar")
    compress_cmd.add_argument(
        "output_dir",
        help="Directory where the compressed Point Cloud will be saved.")
    # General coding parameters
    compress_cmd.add_argument(
        "--blk_size", type=int, default=128,
        help="Size of the 3D coding block units.")
    compress_cmd.add_argument(
        "--q_step", type=float, default=1,
        help="Explicit quantization step.")
    compress_cmd.add_argument(
        "--scale", type=int, default=None,
        help="Down-sampling scale. If 'None', it is automatically determined.")
    # Top-k optimization parameters
    compress_cmd.add_argument(
        "--topk_metrics", type=str, default="d1yuv",
        help="Metrics to use for the optimized Top-k binarization."
             " Available: 'd1', 'd2', 'd1yuv', 'd2yuv', 'd1rgb', 'd2rgb'."
             " If coding geometry-only, only the geometry metric is used.")
    compress_cmd.add_argument(
        "--color_weight", type=float, default=0.5,
        help="Weight of the color metric in the joint top-k optimization. Between 0 and 1.")
    compress_cmd.add_argument(
        "--use_fast_topk", action="store_true",
        help="Use faster top-k optimization algorithm for the coding model.")
    compress_cmd.add_argument(
        "--max_topk", type=int, default=10,
        help= "Define the maximum factor used by the Top-K optimization algorithms.")
    compress_cmd.add_argument(
        "--topk_patience", type=int, default=5,
        help= "Define the patience for early stopping in the Top-K optimization algorithms.")
    # Super-resolution parameters
    compress_cmd.add_argument(
        "--use_sr", action="store_true",
        help= "Use basic Upsampling (False) or Upsampling with Super-Resolution - SR (True).")
    compress_cmd.add_argument(
        "--sr_model_dir", type=str, default="",
        help="Directory where to load Super-Resolution model."
             " A single directory should be provided for the target down-sampling scale: ../models/SR/checkpoint_best_loss.pth.tar ")
    compress_cmd.add_argument(
        "--sr_topk", type=str, default="full",
        help= "Type of Top-k optimization for the Super-Resolution at the encoder:"
              " 'none' - Use the same as the coding model."
              " 'full' - Use the regular algorithm (default)."
              " 'fast' - Use the faster algorithm.")
    compress_cmd.add_argument(
        "--sr_max_topk", type=int, default=10,
        help= "Define the maximum factor used by the Top-K optimization algorithms.")


    # 'decompress' subcommand
    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a bitstream file, decodes the voxel blocks, and reconstructs the Point Cloud.")
    decompress_cmd.add_argument(
        "input_file",
        help="Input bitstream filename (.gz).")
    decompress_cmd.add_argument(
        "model_dir", 
        help="Directory where to load model checkpoints."
             " For decompression, a single directory should be provided: ../models/test ")
    decompress_cmd.add_argument(
        "--sr_model_dir",
        help="Directory where to load the Super-Resolution model."
             " A single directory should be provided for the target down-sampling scale. ")

    # Parse arguments
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    # Invoke subcommand
    if args.command == "compress":
        if not args.model_dir:
            raise ValueError("Need directory to load model.")
        if not args.input_file or not args.output_dir:
            raise ValueError("Need input PC filename for encoding, and output directory.")
        if args.use_sr and not args.sr_model_dir:
            raise ValueError("If using Super-Resolution, need directory for SR trained model.")
        if args.color_weight < 0 or args.color_weight > 1:
            raise ValueError("Color weight must be a value between 0 and 1.")
        if args.sr_topk not in ['none','full','fast']:
            raise ValueError("Available Top-k Optimization metrics: 'none', 'full', 'fast'.")
        if args.topk_metrics.lower() not in ['d1', 'd2', 'd1yuv', 'd2yuv', 'd1rgb', 'd2rgb']:
            raise ValueError("Available metrics for Top-k optimization are: 'd1' and 'd2' for"
                             " geometry-only coding, 'd1yuv', 'd2yuv', 'd1rgb' and 'd2rgb' for joint coding.")
        compress(args)
    elif args.command == "decompress":
        if not args.model_dir:
            raise ValueError("Need directory to load model.")
        if not args.input_file:
            raise ValueError("Need input bitstream filename for decoding.")
        decompress(args)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
