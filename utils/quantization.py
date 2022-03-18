import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
#import tqdm


# used by EST representation
class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in range(1000):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values


class QuantizationLayerEST(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0])
        self.dim = dim

    def forward(self, events):
        # points is a list, since events can have any size
        B = 1 #int((1+events[-1,-1]).item())
        if B < 1:
            B = 1
        num_voxels = int(2 * np.prod(self.dim) * B)
        vox = events[0].new_full([num_voxels,], fill_value=0)
        C, H, W = self.dim

        # get values for each channel
        x, y, t, p = events.t()

        # normalizing timestamps
        t = t / t.max()

        p = (p+1)/2  # maps polarity to 0, 1

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \


        for i_bin in range(C):
           # values = t * self.value_layer.forward(t-i_bin/(C-1))
            values = t * self.value_layer.trilinear_kernel(t-i_bin/(C-1), C)

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)

      #  vox[vox > 0] = 1
        vox = vox.view(-1, 2, C, H, W)
     #   vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

        return vox

class QuantizationLayerVoxGrid(nn.Module):
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim

    def forward(self, events):
        epsilon = 10e-3
        B = 1# int(1+events[-1,-1].item())
        if B < 1:
            B = 1
        num_voxels = int(np.prod(self.dim) * B)
        vox_grid = events[0].new_full([num_voxels,], fill_value=0)
        C, H, W = self.dim

        # get values for each channel
        x, y, t, p = events.t()

        # normalizing timestamps
        t = t / t.max()

        p = (p+1) / 2 # maps polarity to 0, 1

        for i_bin in range(C):
            index = (t > i_bin/C) & (t <= (i_bin+1)/C)
            x1 = x[index]
            y1 = y[index]
            
            idx = x1 + W*y1 + W*H*i_bin
            val = torch.zeros_like(x1) + 1
            vox_grid.put_(idx.long(), val, accumulate=True)

        # normalize
     #   vox_grid = vox_grid / (vox_grid.max() + epsilon)
     #   vox_grid[vox_grid > 0] = 1
        vox_grid = vox_grid.view(-1, C, H, W)
        return vox_grid

# binary representation
class QuantizationLayerBinary(nn.Module):
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim

    def forward(self, events):
        B = 1 #int(1+events[-1,-1].item()) # batch_size
        if B < 1:
            B = 1
        num_voxels = int(2 * np.prod(self.dim) * B)
        vox_binary = events[0].new_full([num_voxels,], fill_value=0)
        C, H, W = self.dim # Time dimension, height, width

        # get values for each channel
        x, y, t, p = events.t()
        b = 1

        # normalizing timestamps
        for bi in range(B):
            if len(t[events[:,-1] == bi]) > 0:
                t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()

        p = (p+1) / 2 # maps polarity to 0, 1
        for i_bin in range(C):
            index = (t > i_bin/C) & (t <= (i_bin+1)/C)
            x1 = x[index]
            y1 = y[index]
            b1 = b[index]
            p1 = p[index]
            idx = x1 + W*y1 + W*H*C*p1 + W*H*C*2*b1 + W*H*i_bin

            val = torch.zeros_like(x1) + 1
            vox_binary.put_(idx.long(), val, accumulate=False) 

        vox_binary = vox_binary.view(-1, 2, C, H, W) # binary voxel
        return vox_binary

class QuantizationLayerEventCount(nn.Module):
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim

    def forward(self, events):
        epsilon = 10e-3 # avoid divide by zero
        B = 1 # int(1+events[-1,-1].item())
        if B < 1:
            B = 1
        num_voxels = int(2 * np.prod(self.dim) * B)
        vox_ec = events[0].new_full([num_voxels,], fill_value=0) # event counts

        H, W = self.dim 

        # get values for each channel
        x, y, t, p = events.t()

        # normalizing timestamps
        t = t / t.max()

        p = (p+1)/2 # maps polarity to 0, 1

        idx = x + W*y + W*H*p 
        val = torch.zeros_like(x) + 1
        vox_ec.put_(idx.long(), val, accumulate=True)

        # normalize 
     #   vox_ec = (vox_ec-vox_ec.mean()) / (vox_ec.max() + epsilon)
        vox_ec[vox_ec > 0] = 1
        vox_ec = vox_ec.view(-1, 2, H, W)
        
        return vox_ec

class QuantizationLayerEventFrame(nn.Module):
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim
        self.quantization_layer = QuantizationLayerEventCount(dim)

    def forward(self, events):
        vox = self.quantization_layer.forward(events)
        event_frame = vox.sum(dim=1)
        H, W = self.dim
        event_frame = event_frame.view(-1, 1, H, W)

        return event_frame
