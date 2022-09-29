# Copyright (c) 2021-2022, InterDigital Communications, Inc
# Copyright (c) 2022, Instituto de Telecomunicações
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of its
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

import argparse
import math
import random
import shutil
import sys
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import dl_coding_model

import loss_functions
from pcdata import PCdataFolder


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, alpha=0.7, gamma=2, omega=0.5, color=False):
        super().__init__()
        self.lmbda = lmbda
        self.alpha = alpha
        self.gamma = gamma
        self.omega = omega
        self.joint = color

    def forward(self, decoded, original):
        out = {}

        num_input_points = original[:, 0, :, :, :].sum()

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_input_points))
            for likelihoods in decoded["likelihoods"].values()
        )
        if self.joint:
            out["mse_loss"] = loss_functions.focal_loss(original[:, 0, :, :, :], decoded["x_hat"][:, 0, :, :, :], gamma=self.gamma, alpha=self.alpha, total=False)
            out["rgb_loss"] = loss_functions.mse_rgb(original, decoded["x_hat"], num_input_points)
            out["loss"] = ((1 - self.omega) * out["mse_loss"] + self.omega * out["rgb_loss"]) + self.lmbda * out["bpp_loss"]
        else:
            out["mse_loss"] = loss_functions.focal_loss(original, decoded["x_hat"], gamma=self.gamma, alpha=self.alpha, total=False)
            out["loss"] = out["mse_loss"] + self.lmbda * out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=1e-4,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=1e-3,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, writer, joint
):
    model.train()
    device = next(model.parameters()).device

    t0 = time.time()

    running_loss = AverageMeter()
    running_mse = AverageMeter()
    running_bpp = AverageMeter()
    running_aux = AverageMeter()
    if joint:
        running_rgb = AverageMeter()
    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        running_loss.update(out_criterion["loss"])
        running_mse.update(out_criterion["mse_loss"])
        running_bpp.update(out_criterion["bpp_loss"])
        running_aux.update(aux_loss)
        if joint:
            running_rgb.update(out_criterion["rgb_loss"])

        if i % 10 == 0:
            if joint:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] | "
                    f'Loss: {running_loss.avg:.5f} | '
                    f'MSE loss: {running_mse.avg:.5f} | '
                    f'RGB loss: {running_rgb.avg:.5f} | '
                    f'Bpp loss: {running_bpp.avg:.3f} | '
                    f'Aux loss: {running_aux.avg:.2f}'
                    , end='\r'
                )
            else:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)] | "
                    f'Loss: {running_loss.avg:.5f} | '
                    f'MSE loss: {running_mse.avg:.5f} | '
                    f'Bpp loss: {running_bpp.avg:.3f} | '
                    f'Aux loss: {running_aux.avg:.2f}'
                    , end='\r'
                )
    
    if joint:
        print(
            f"Train epoch {epoch}: ["
            f"{len(train_dataloader.dataset)}/{len(train_dataloader.dataset)}"
            f" (100%)] | "
            f'Loss: {running_loss.avg:.5f} | '
            f'MSE loss: {running_mse.avg:.5f} | '
            f'RGB loss: {running_rgb.avg:.5f} | '
            f'Bpp loss: {running_bpp.avg:.3f} | '
            f'Aux loss: {running_aux.avg:.2f} | '
            f"Elapsed time: {(time.time() - t0):.2f} seconds"
        )
    else:
        print(
            f"Train epoch {epoch}: ["
            f"{len(train_dataloader.dataset)}/{len(train_dataloader.dataset)}"
            f" (100%)] | "
            f'Loss: {running_loss.avg:.5f} | '
            f'MSE loss: {running_mse.avg:.5f} | '
            f'Bpp loss: {running_bpp.avg:.3f} | '
            f'Aux loss: {running_aux.avg:.2f} | '
            f"Elapsed time: {(time.time() - t0):.2f} seconds"
        )
    # ...log the running losses
    if writer is not None:
        writer.add_scalars('loss', {'train': running_loss.avg}, epoch)
        writer.add_scalars('mse', {'train': running_mse.avg}, epoch)
        writer.add_scalars('bpp', {'train': running_bpp.avg}, epoch)
        if joint:
            writer.add_scalars('rgb', {'train': running_rgb.avg}, epoch)
        writer.close()


def test_epoch(epoch, test_dataloader, model, criterion, writer, joint):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    if joint:
        rgb_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            if joint:
                rgb_loss.update(out_criterion["rgb_loss"])

    if joint:
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.5f} |"
            f"\tMSE loss: {mse_loss.avg:.5f} |"
            f"\tRGB loss: {rgb_loss.avg:.5f} |"
            f"\tBpp loss: {bpp_loss.avg:.3f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )
    else:
            print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.5f} |"
            f"\tMSE loss: {mse_loss.avg:.5f} |"
            f"\tBpp loss: {bpp_loss.avg:.3f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )
    # ...log the validation losses
    if writer is not None:
        writer.add_scalars('loss', {'val': loss.avg}, epoch)
        writer.add_scalars('mse', {'val': mse_loss.avg}, epoch)
        writer.add_scalars('bpp', {'val': bpp_loss.avg}, epoch)
        if joint:
            writer.add_scalars('rgb', {'val': rgb_loss.avg}, epoch)
        writer.close()

    return loss.avg


def save_checkpoint(state, is_best, filedir, filename="checkpoint.pth.tar"):
    os.makedirs(filedir, exist_ok=True)
    torch.save(state, os.path.join(filedir, filename))
    if is_best:
        shutil.copyfile(os.path.join(filedir, filename), os.path.join(filedir, "checkpoint_best_loss.pth.tar"))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Codec training script.")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print model summary.",
    )
    parser.add_argument(
        "--with_color",
        action="store_true",
        help="Jointly compress both geometry and color.",
    )
    parser.add_argument(
        "-d", "--train_data", type=str, required=True, help="Path to training PC data (folder with pickle .pkl files)."
    )
    parser.add_argument(
        "--val_data", type=str, required=True, help="Path to validation PC data (folder with pickle .pkl files)."
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory where to save the model checkpoints."
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=500,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=12,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a previous checkpoint for sequential training.")
    parser.add_argument("--logs", type=str, help="Path to store training logs for TensorBoard.")
    parser.add_argument(
        "--num_filters",
        type=int,
        default=32,
        help="Number of filters in first convolutional layer (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Rate-distortion trade-off parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--fl_alpha",
        type=float,
        default=0.7,
        help="Class balancing weight for Focal Loss (default: %(default)s)",
    )
    parser.add_argument(
        "--fl_gamma",
        type=float,
        default=2.0,
        help="Focusing weight for Focal Loss (default: %(default)s)",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=0.5,
        help="Geometry-color distortion tradeoff parameter (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_dataset = PCdataFolder(args.train_data, args.with_color)
    val_dataset = PCdataFolder(args.val_data, args.with_color)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    if args.with_color:
        net = dl_coding_model.CodingModel(args.num_filters, 4)
    else:
        net = dl_coding_model.CodingModel(args.num_filters, 1)
    
    net = net.to(device)

    if args.verbose:
        if args.with_color:
            summary(net, (4, 64, 64, 64))
        else:
            summary(net, (1, 64, 64, 64))

    if args.logs:
        writer = SummaryWriter(args.logs)
        # writer.add_graph(net)
        # writer.close()
    else:
        writer = None

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda, alpha=args.fl_alpha, gamma=args.fl_gamma, omega=args.omega, color=args.with_color)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    patience = 5
    stop_counter = 0
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            writer,
            args.with_color,
        )
        loss = test_epoch(epoch, val_dataloader, net, criterion, writer, args.with_color)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                args.model_dir,
                filename=f"checkpoint-{epoch:04d}.pth.tar"
            )

        if is_best:
            stop_counter = 0
        else:
            stop_counter = stop_counter + 1
            if stop_counter >= patience:
                print('Early stopping!\n')
                if (net.update()):
                    print('Entropy CDF tables updated.\n')
                break


if __name__ == "__main__":
    main(sys.argv[1:])
