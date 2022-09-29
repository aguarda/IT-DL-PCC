import torch
import torch.nn as nn
import torch.nn.functional as F


class IR_Block(nn.Module):

    def __init__(self, num_filters):
        super().__init__()

        self.conv_a1 = nn.Conv3d(num_filters, num_filters//4, kernel_size=(3, 3, 3), stride=1, padding="same", bias=True)
        self.conv_a2 = nn.Conv3d(num_filters//4, num_filters//2, kernel_size=(3, 3, 3), stride=1, padding="same", bias=True)

        self.conv_b1 = nn.Conv3d(num_filters, num_filters//4, kernel_size=(1, 1, 1), stride=1, padding="same", bias=True)
        self.conv_b2 = nn.Conv3d(num_filters//4, num_filters//4, kernel_size=(3, 3, 3), stride=1, padding="same", bias=True)
        self.conv_b3 = nn.Conv3d(num_filters//4, num_filters//2, kernel_size=(1, 1, 1), stride=1, padding="same", bias=True)


    def forward(self, indata):

        branch1 = F.relu_(self.conv_a1(indata))
        branch1 = F.relu_(self.conv_a2(branch1))

        branch2 = F.relu_(self.conv_b1(indata))
        branch2 = F.relu_(self.conv_b2(branch2))
        branch2 = F.relu_(self.conv_b3(branch2))

        branches = (branch1, branch2)

        out = torch.add(indata, torch.cat(branches, dim=1))
        
        return out


class IRN(nn.Module):

    def __init__(self, num_filters):
        super().__init__()

        self.irn = nn.Sequential(
            IR_Block(num_filters),
            IR_Block(num_filters),
            IR_Block(num_filters),
        )
        
    def forward(self, indata):
        return self.irn(indata)


class SRModel(nn.Module):

    def __init__(self, num_filters, inout_channels):
        super().__init__()

        # Input Convolution (Features)
        self.convin = nn.Conv3d(inout_channels, num_filters, kernel_size=(3, 3, 3), stride=1, padding="same", bias=True)

        # Downsampling No.1
        self.conv1d = nn.Conv3d(num_filters, num_filters, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=True)
        # IRN No.1
        self.irn1 = IRN(num_filters)
        # Downsampling No.2
        self.conv2d = nn.Conv3d(num_filters, num_filters*2, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=True)
        # IRN No.2
        self.irn2 = IRN(num_filters*2)
        # Downsampling No.3
        self.conv3d = nn.Conv3d(num_filters*2, num_filters*4, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=True)
        # IRN No.3
        self.irn3 = IRN(num_filters*4)
        # Downsampling No.4
        self.conv4d = nn.Conv3d(num_filters*4, num_filters*8, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=True)
        # IRN No.4
        self.irn4 = IRN(num_filters*8)
        # Downsampling No.5
        self.conv5d = nn.Conv3d(num_filters*8, num_filters*8, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=True)
        # IRN No.5
        self.irn5 = IRN(num_filters*8)

        ######### BOTTLENECK #########

        # Upsampling No.1
        self.conv1u = nn.ConvTranspose3d(num_filters*8, num_filters*8, kernel_size=(3, 3, 3), stride=2, output_padding=1, padding=(1, 1, 1), bias=True)
        # IRN No.6
        self.irn6 = IRN(num_filters*16)
        # Reduction Conv. No.1
        self.conv_red1 = nn.Conv3d(num_filters*16, num_filters*8, kernel_size=(3, 3, 3), stride=1, padding="same", bias=True)
        # Upsampling No.2
        self.conv2u = nn.ConvTranspose3d(num_filters*8, num_filters*4, kernel_size=(3, 3, 3), stride=2, output_padding=1, padding=(1, 1, 1), bias=True)
        # IRN No.7
        self.irn7 = IRN(num_filters*8)
        # Reduction Conv. No.2
        self.conv_red2 = nn.Conv3d(num_filters*8, num_filters*4, kernel_size=(3, 3, 3), stride=1, padding="same", bias=True)
        # Upsampling No.3
        self.conv3u = nn.ConvTranspose3d(num_filters*4, num_filters*2, kernel_size=(3, 3, 3), stride=2, output_padding=1, padding=(1, 1, 1), bias=True)
        # IRN No.8
        self.irn8 = IRN(num_filters*4)
        # Reduction Conv. No.3
        self.conv_red3 = nn.Conv3d(num_filters*4, num_filters*2, kernel_size=(3, 3, 3), stride=1, padding="same", bias=True)
        # Upsampling No.4
        self.conv4u = nn.ConvTranspose3d(num_filters*2, num_filters, kernel_size=(3, 3, 3), stride=2, output_padding=1, padding=(1, 1, 1), bias=True)
        # IRN No.9
        self.irn9 = IRN(num_filters*2)
        # Reduction Conv. No.4
        self.conv_red4 = nn.Conv3d(num_filters*2, num_filters, kernel_size=(3, 3, 3), stride=1, padding="same", bias=True)
        # Upsampling No.5
        self.conv5u = nn.ConvTranspose3d(num_filters, num_filters, kernel_size=(3, 3, 3), stride=2, output_padding=1, padding=(1, 1, 1), bias=True)
        # IRN No.10
        self.irn10 = IRN(num_filters*2)
        # Reduction Conv. No.5
        self.conv_red5 = nn.Conv3d(num_filters*2, num_filters, kernel_size=(3, 3, 3), stride=1, padding="same", bias=True)

        # Output Convolution (Probabilities)
        self.convout = nn.Conv3d(num_filters, inout_channels, kernel_size=(3, 3, 3), stride=1, padding="same", bias=True)
    
     
    def forward(self, data):
        
        # Process Input (Input -> features)
        out0 = F.relu_(self.convin(data))
        
        # Downsampling 1
        out1d = F.relu_(self.conv1d(out0))
        # IRN 1
        outirn1d = self.irn1(out1d)
        # Downsampling 2
        out2d = F.relu_(self.conv2d(outirn1d))
        # IRN 2
        outirn2d = self.irn2(out2d)
        # Downsampling 3
        out3d = F.relu_(self.conv3d(outirn2d))
        # IRN 3
        outirn3d = self.irn3(out3d)
        # Downsampling 4
        out4d = F.relu_(self.conv4d(outirn3d))
        # IRN 4
        outirn4d = self.irn4(out4d)
        # Downsampling 5
        out5d = F.relu_(self.conv5d(outirn4d))
        # IRN 5
        outirn5d = self.irn5(out5d)

        ##### Bottleneck Here #####

        # Upsampling 1
        out1u = F.relu_(self.conv1u(outirn5d))
        out1u = torch.cat((outirn4d, out1u), dim=1)
        # IRN 6
        outirn1u = self.irn6(out1u)
        # Reduction Layer 1 to (reduce channels from concat)
        out_red1 = F.relu_(self.conv_red1(outirn1u))
        # Upsampling 2
        out2u = F.relu_(self.conv2u(out_red1))
        out2u = torch.cat((outirn3d, out2u), dim=1)
        # IRN 7
        outirn2u = self.irn7(out2u)
        # Reduction Layer 2
        out_red2 = F.relu_(self.conv_red2(outirn2u))
        # Upsampling 3
        out3u = F.relu_(self.conv3u(out_red2))
        out3u = torch.cat((outirn2d, out3u), dim=1)
        # IRN 8
        outirn3u = self.irn8(out3u)
        # Reduction Layer 3
        out_red3 = F.relu_(self.conv_red3(outirn3u))
        # Upsampling 4
        out4u = F.relu_(self.conv4u(out_red3))
        out4u = torch.cat((outirn1d, out4u), dim=1)
        # IRN 9
        outirn4u = self.irn9(out4u)
        # Reduction Layer 4
        out_red4 = F.relu_(self.conv_red4(outirn4u))
        # Upsampling 5
        out5u = F.relu_(self.conv5u(out_red4))
        out5u = torch.cat((out0, out5u), dim=1)
        # IRN 10
        outirn5u = self.irn10(out5u)
        # Reduction Layer 5
        out_red5 = F.relu_(self.conv_red5(outirn5u))

        # Process Output (features -> probabilities)
        out = torch.sigmoid(self.convout(out_red5))

        return out

