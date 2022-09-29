import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from utils import update_registered_buffers


class IR_Block(nn.Module):

    def __init__(self, num_filters):
        super().__init__()

        self.conv_a1 = nn.Conv3d(num_filters, num_filters//4, kernel_size=(5, 5, 5), stride=1, padding="same", bias=True)

        self.conv_b1 = nn.Conv3d(num_filters, num_filters//4, kernel_size=(3, 3, 3), stride=1, padding="same", bias=True)
        self.conv_b2 = nn.Conv3d(num_filters//4, num_filters//4, kernel_size=(3, 3, 3), stride=1, padding="same", bias=True)

        self.conv_c1 = nn.Conv3d(num_filters, num_filters//4, kernel_size=(1, 1, 1), stride=1, padding="same", bias=True)
        self.conv_c2 = nn.Conv3d(num_filters//4, num_filters//4, kernel_size=(3, 3, 3), stride=1, padding="same", bias=True)
        self.conv_c3 = nn.Conv3d(num_filters//4, num_filters//4, kernel_size=(1, 1, 1), stride=1, padding="same", bias=True)

        self.conv_d1 = nn.Conv3d(num_filters, num_filters//4, kernel_size=(1, 1, 1), stride=1, padding="same", bias=True)
        self.conv_d2 = nn.Conv3d(num_filters//4, num_filters//4, kernel_size=(3, 1, 1), stride=1, padding="same", bias=True)
        self.conv_d3 = nn.Conv3d(num_filters//4, num_filters//4, kernel_size=(1, 3, 1), stride=1, padding="same", bias=True)
        self.conv_d4 = nn.Conv3d(num_filters//4, num_filters//4, kernel_size=(1, 1, 3), stride=1, padding="same", bias=True)


    def forward(self, indata):

        branch1 = F.relu_(self.conv_a1(indata))

        branch2 = F.relu_(self.conv_b1(indata))
        branch2 = F.relu_(self.conv_b2(branch2))

        branch3 = F.relu_(self.conv_c1(indata))
        branch3 = F.relu_(self.conv_c2(branch3))
        branch3 = F.relu_(self.conv_c3(branch3))

        branch4 = F.relu_(self.conv_d1(indata))
        branch4 = F.relu_(self.conv_d2(branch4))
        branch4 = F.relu_(self.conv_d3(branch4))
        branch4 = F.relu_(self.conv_d4(branch4))

        branches = (branch1, branch2, branch3, branch4)

        out = torch.add(indata, torch.cat(branches, dim=1))
        
        return out


class AnalysisTransform(nn.Module):
    """The analysis transform."""

    def __init__(self, num_filters, inout_channels):
        super().__init__()

        self.g_a = nn.Sequential(
            nn.Conv3d(inout_channels, num_filters, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=True),
            nn.ReLU(inplace=True),
            IR_Block(num_filters),
            nn.Conv3d(num_filters, num_filters*2, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=True),
            nn.ReLU(inplace=True),
            IR_Block(num_filters*2),
            nn.Conv3d(num_filters*2, num_filters*4, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=True),
            nn.ReLU(inplace=True),
            IR_Block(num_filters*4),
            nn.Conv3d(num_filters*4, num_filters*4, kernel_size=(1, 1, 1), stride=1, padding="same", bias=False),
        )
        
    def forward(self, indata):
        return self.g_a(indata)


class SynthesisTransform(nn.Module):
    """The synthesis transform."""

    def __init__(self, num_filters, inout_channels):
        super().__init__()

        self.g_s = nn.Sequential(
            nn.ConvTranspose3d(num_filters*4, num_filters*4, kernel_size=(3, 3, 3), stride=2, output_padding=1, padding=(1, 1, 1), bias=True),
            nn.ReLU(inplace=True),
            IR_Block(num_filters*4),
            nn.ConvTranspose3d(num_filters*4, num_filters*2, kernel_size=(3, 3, 3), stride=2, output_padding=1, padding=(1, 1, 1), bias=True),
            nn.ReLU(inplace=True),
            IR_Block(num_filters*2),
            nn.ConvTranspose3d(num_filters*2, num_filters, kernel_size=(3, 3, 3), stride=2, output_padding=1, padding=(1, 1, 1), bias=True),
            nn.ReLU(inplace=True),
            IR_Block(num_filters),
            nn.Conv3d(num_filters, inout_channels, kernel_size=(1, 1, 1), stride=1, padding="same", bias=True),
            nn.Sigmoid(),
        )

    def forward(self, indata):
        return self.g_s(indata)


class HyperAnalysisTransform(nn.Module):
    """The analysis transform for the entropy model parameters."""

    def __init__(self, num_filters):
        super().__init__()

        self.h_a = nn.Sequential(
            nn.Conv3d(num_filters, num_filters, kernel_size=(3, 3, 3), stride=1, padding="same", bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_filters, num_filters, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_filters, num_filters, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=False),
        )

    def forward(self, indata):
        return self.h_a(indata)


class HyperSynthesisTransform(nn.Module):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self, num_filters):
        super().__init__()

        self.h_s = nn.Sequential(
            nn.ConvTranspose3d(num_filters, num_filters, kernel_size=(3, 3, 3), stride=2, output_padding=1, padding=(1, 1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(num_filters, (num_filters*3)//2, kernel_size=(3, 3, 3), stride=2, output_padding=1, padding=(1, 1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d((num_filters*3)//2, num_filters*2, kernel_size=(3, 3, 3), stride=1, padding="same", bias=True),
        )

    def forward(self, indata):
        return self.h_s(indata)


SCALES_MIN = 0.01
SCALES_MAX = 256.
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(torch.log(torch.tensor(min)), torch.log(torch.tensor(max)), levels))


class CodingModel(nn.Module):
    """Main model class."""

    def __init__(self, num_filters, inout_channels):
        super().__init__()
        self.analysis_transform = AnalysisTransform(num_filters, inout_channels)
        self.synthesis_transform = SynthesisTransform(num_filters, inout_channels)
        self.hyper_analysis_transform = HyperAnalysisTransform(num_filters*4)
        self.hyper_synthesis_transform = HyperSynthesisTransform(num_filters*4)

        self.entropy_bottleneck = EntropyBottleneck(num_filters*4)
        self.gaussian_conditional = GaussianConditional(None)

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def forward(self, x):
        y = self.analysis_transform(x)
        z = self.hyper_analysis_transform(y)
        z_hat, side_bits = self.entropy_bottleneck(z)
        gauss_params = self.hyper_synthesis_transform(z_hat)
        scales, means = gauss_params.chunk(2, 1)
        y_hat, bits = self.gaussian_conditional(y, scales, means=means)
        x_hat = self.synthesis_transform(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": bits, "z": side_bits},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv

        if scale_table is None:
            scale_table = get_scale_table()
            
        updated |= self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def compress(self, x, q_step=1):
        """Compresses a 3D PC block."""
        y = self.analysis_transform(x)
        y = torch.div(y, q_step)
        z = self.hyper_analysis_transform(y)

        z_shape = z.size()[-3:]

        side_string = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(side_string, z_shape)

        gauss_params = self.hyper_synthesis_transform(z_hat)
        scales, means = gauss_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales)
        string = self.gaussian_conditional.compress(y, indexes, means=means)

        return {"strings": [string, side_string], "shape": z_shape}

    def decompress(self, strings, z_shape, q_step):
        """Decompresses a 3D PC block."""
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], z_shape)
        gauss_params = self.hyper_synthesis_transform(z_hat)
        scales, means = gauss_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means
        )
        y_hat = torch.mul(y_hat, q_step)
        x_hat = self.synthesis_transform(y_hat).clamp_(0, 1)
        return x_hat
