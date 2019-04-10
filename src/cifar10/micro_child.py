# -*- coding: utf-8 -*-

"""This python file contains PyTorch implementation of ENAS MicroChild. The code 
   is heavily based on the paper's source code, which uses tensorflow library.

   Paper address: https://arxiv.org/abs/1802.03268

   Tensorflow implementation: 
   https://github.com/melodyguan/enas/blob/master/src/cifar10/micro_child.py
   

   Author: Meng Cao
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroChild(nn.Module):
    """A shared CNN graph"""
    
    def __init__(self, config):
        super(MicroChild, self).__init__()
        self.config = config

        self.init_layer = InitLayer(3, config.out_channels * 3, 3)
        self.rcbs = nn.ModuleList([ReluConvBN(config.out_channels * 3, 
                                              config.out_channels) 
                                       for i in range(2)])
        self.pool_layers_indices = self._specify_pool_layers()
        self.layers = self._build_enas_layers()
        if config.use_aux_heads:
            self.aux, self.aux_head_indices = self._build_aux_heads()
        self.proj = self._build_proj()
        

    def _specify_pool_layers(self):
        """Specify which layers are pool layers (with reduction cell).
        """
        pool_distance = self.config.num_layers // 3
        return [pool_distance, pool_distance * 2 + 1]
        

    def _build_enas_layers(self):
        """Build ENAS layers. In every pool layer, the channels are doubled."""
        node_num, out_c = self.config.node_num, self.config.out_channels
        
        layers = nn.ModuleList()
        for i in range(self.config.num_layers):
            if i in self.pool_layers_indices:
                out_c = out_c * 2
            layers.append(ENASLayer(node_num, out_c, 
                                    self.config.fixed))
        return layers

    
    def _build_aux_heads(self):
        """Build auxiliary head for training."""
        pool_layer_num = len(self.pool_layers_indices)
        channels = self.config.out_channels * pool_layer_num**2
        side_length = self.config.image_size // pool_layer_num**2
        
        aux = AuxHeadLayer(channels, side_length)
        # specify which layer to add auxiliary h
        aux_head_indices = [self.pool_layers_indices[-1] + 1]
        
        return aux, aux_head_indices

        
    def _build_proj(self):
        """The final projection layer for logits compution"""
        channels = self.config.out_channels * len(self.pool_layers_indices)**2
        proj = nn.Sequential(
            nn.ReLU(), 
            GlobalAvgPool(), 
            nn.Dropout(p=(1.0 - self.config.keep_prob)), 
            nn.Linear(channels, self.config.class_num)
        )
        return proj


    def _get_dropout_rate(self, layer_id, step_ratio=None):
        """Compute dropout rate for each layer.
        """
        keep_prob = self.config.drop_path_keep_prob
        # layer ratio
        layer_ratio = (layer_id + 1) / self.config.num_layers
        keep_prob = 1.0 - layer_ratio * (1.0 - keep_prob)

        # step ratio
        if step_ratio is not None:
            keep_prob = 1.0 - step_ratio * (1.0 - keep_prob)

        return 1.0 - keep_prob
        

    def forward(self, images, arcs, step_ratio=None):
        """Compute the logits given images.
        
        Args:
            images: input images, [N, C_in, H, W], N is batch size.
            arcs: a tuple of two lists that contain integers represents the architecture 
                of a normal cell and a reduce cell, four integers together in the list as
                a node: (index_1, index_2, op_1, op_2).
            num_train_batches: int, total number of training batches.
        """
        normal_arc, reduce_arc = arcs
        x = self.init_layer(images)
        
        # NOTE: here the implementation is a litte different from Melody's
        inputs = []
        for i in range(len(self.rcbs)):
            inputs.append(self.rcbs[i](x))
        
        # ENAS layers
        aux_logits = None
        for layer_id in range(self.config.num_layers): 
            # arc and dropout rate
            arc = reduce_arc if layer_id in self.pool_layers_indices else normal_arc
            p = self._get_dropout_rate(layer_id, step_ratio)
            
            x = self.layers[layer_id](inputs, arc, p)
            inputs = [inputs[-1], x]
            
            if self.config.use_aux_heads and layer_id in self.aux_head_indices \
                and self.training:
                aux_logits = self.aux(x) # auxiliary head
        
        logits = self.proj(x)
        return logits, aux_logits


class ENASLayer(nn.Module):
    """Implement ENAS layer class, each layer composes B nodes: 2 input nodes and 
       (B - 2) operation nodes. Parameters in one ENAS layers are shared.
       
       One EnasLayer is equivalent to the Convolution or Reduction Cell in the
       paper.
       
       The two input nodes have ID 0 and 1, the first operation node has ID 2 and 
       so on.
    """
    def __init__(self, node_num, out_channels, fixed=False):
        super(ENASLayer, self).__init__()
        self.node_num = node_num
        self.out_channels = out_channels
        
        self.frs = nn.ModuleList([FactorizedReduction(out_channels // 2, out_channels)
                                    for i in range(2)])
        self.final_rcb = ReluConvBN(out_channels * node_num, out_channels)
        
        # build cells (nodes), node_num does NOT include two input nodes
        Cell = ENASCellFixed if fixed else ENASCell
        self.nodes = nn.ModuleList()
        for cell_id in range(2, node_num + 2):
            self.nodes.append(Cell(out_channels, out_channels, cell_id))

        
    def forward(self, inputs, arc, dropout=0.5):
        """Forward two inputs through one ENAS layer, the unused inputs are concatenated.
        
        Args:
            inputs: (h[i], h[i-1]), 
            arc: list of integers, representing the architecture of the cell.
        """
        assert len(arc) == self.node_num * 4, "Oops, the length of arc is {}, which should be a multiple of 4*{}.".format(len(arc), self.node_num)
        assert len(inputs) == 2, "Require exactly 2 inputs."
        
        node_inps = []
        node_inps.extend(self._celibrate_size(inputs)) # assure two inputs have same shape
        
        for i in range(self.node_num):
            # the first operation
            x_id, x_op = arc[4 * i], arc[4 * i + 2]
            x = node_inps[x_id]
            x_out = self.nodes[i](x, x_id, x_op)
            
            # the second operation
            y_id, y_op = arc[4 * i + 1], arc[4 * i + 3]
            y = node_inps[y_id]
            y_out = self.nodes[i](y, y_id, y_op)
            
            # add two op's outputs
            out = x_out + y_out
            node_inps.append(out)
            
        # concatenate all outputs and project
        # NOTE: in the paper this is done for all unused nodes, here we make it sample
        final_output = self.final_rcb(torch.cat(node_inps[2:], dim=1))
        final_output = F.dropout(final_output, p=dropout, training=self.training)
        assert final_output.shape == node_inps[0].shape, "Oops, seems like the final output shape is wrong: {}. Should be equal to {}.".format(final_output.shape, node_inps[0].shape)
        return final_output
        
        
    def _celibrate_size(self, inputs):
        """Because of the reduction cell, the second input might have half WH size 
           and double depth size. This function is to make sure two inputs have the 
           same W and H, and the depth equals to out_channels.
        """
        outs = []
        for i, inp in enumerate(inputs):
            if self._get_C(inp) != self.out_channels:
                outs.append(self.frs[i](inp))
            else:
                outs.append(inp)
                
        assert outs[0].shape == outs[1].shape
        return outs
    
    
    def _get_C(self, x):
        """Get channel size of a given feature map.
        """
        return x.shape[1]
        
        
    def _get_HW(self, x):
        """Get H and W of a given feature map.
        """
        return x.shape[-2], x.shape[-1]


class InitLayer(nn.Module):
    """The initial layer of the network: one convolution layer followed by one
        batch normalization layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, trs=False, bias=False):
        super(InitLayer, self).__init__()
        padding = (kernel_size - 1) // 2 # keep the same size
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size = kernel_size, 
                      padding = padding, 
                      bias = bias), 
            nn.BatchNorm2d(out_channels, 
                           track_running_stats=trs))
        
    def forward(self, x):
        out = self.input_conv(x)
        return out


class FactorizedReduction(nn.Module):
    """Reduces the size of feature map (W and H) by a factor of 2"""
    
    def __init__(self, in_channels, out_channels, trs=False, bias=False):
        super(FactorizedReduction, self).__init__()
        assert out_channels % 2 == 0, "For factorized reduction, output channel number must be even :/"
        self.skip_path_1 = nn.Sequential(
            nn.AvgPool2d(1, stride = 2), 
            nn.Conv2d(in_channels, 
                      out_channels // 2, 
                      kernel_size = 1, 
                      bias = bias))
        
        self.skip_path_2 = copy.deepcopy(self.skip_path_1)
        self.padder = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.bn = nn.BatchNorm2d(out_channels, 
                                 track_running_stats=trs)
        
    def forward(self, x):
        """
        Args:
            x: input feature map with shape [N, C_in, H, W]
            
        Returns:
            out: reudced feature map with shape [N, C_out, H // 2, W // 2]
        """
        out_1 = self.skip_path_1(x) # skip path 1
        out_2 = self.skip_path_2(self.padder(x)[:, :, 1:, 1:]) # skip path 2
        assert out_1.shape == out_2.shape, "Out1's shape {} and out2's shape {} does noe equal :/".format(out_1.shape, out_2.shape)
        out = torch.cat([out_1, out_2], dim=1)
        
        return self.bn(out)


class ReluConvBN(nn.Module):
    """A combination of RELU -> CONV -> BATCH NORM."""

    def __init__(self, in_channels, out_channels, trs=False):
        super(ReluConvBN, self).__init__()
        
        self.rcb = nn.Sequential(
          nn.ReLU(),
          nn.Conv2d(in_channels = in_channels, 
                    out_channels = out_channels, 
                    kernel_size = 1),
          nn.BatchNorm2d(out_channels, 
                         track_running_stats = trs))
        
    def forward(self, inputs):
        return self.rcb(inputs)


class IdentityBranch(nn.Module):
    """The identity branch."""

    def __init__(self):
        super(IdentityBranch, self).__init__()
        
    def forward(self, x):
        return x


class SeparableConv(nn.Module):
    """Implement the depthwise-separable convolution cell."""

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(SeparableConv, self).__init__()
        
        padding = (kernel_size - 1) // 2 # keep the size unchanged
        self.depthwise = nn.Conv2d(in_channels, in_channels, 
                                   kernel_size=kernel_size, 
                                   padding=padding, 
                                   groups=in_channels, 
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        """
        Args:
            x: [N, C_in, H, W]
        
        Return:
            out: [N, C_out, H, W]
        """
        out = self.pointwise(self.depthwise(x))
        return out


class ENASCell(nn.Module):
    """Implement one ENAS cell (or node), each cell can have 5 different operations:
       avg_pool, max_pool, 3*3 conv, 5*5 conv, identity.
    """
    def __init__(self, in_channels, out_channels, node_id):
        super(ENASCell, self).__init__()
        
        self.node_id = node_id
        in_c, out_c = in_channels, out_channels
        self.choices = nn.ModuleDict({
                'conv3': nn.ModuleList([SeparableConv(in_c, out_c, 3)
                                       for i in range(node_id)]), 
                'conv5': nn.ModuleList([SeparableConv(in_c, out_c, 5)
                                       for i in range(node_id)]), 
                'avg_pool': nn.AvgPool2d(3, stride=1, padding=1), 
                'max_pool': nn.MaxPool2d(3, stride=1, padding=1), 
                'identity': IdentityBranch()
        })
        
    def forward(self, x, prev_cell, op_id):
        """
        Args:
            x: input from previous cell.
            prev_cell: integer, the previous cell's ID.
            op_id: integer, indicate which operation to use.
        """
        assert 0 <= op_id <= 4, "Operation ID out of range!"
        assert prev_cell < self.node_id, "Previous cell ID out of range :/"
         
        out = {
          0: lambda x: self.choices['conv3'][prev_cell](x), 
          1: lambda x: self.choices['conv5'][prev_cell](x), 
          2: lambda x: self.choices['avg_pool'](x), 
          3: lambda x: self.choices['max_pool'](x), 
          4: lambda x: self.choices['identity'](x)
        }[op_id](x)
        
        return out


class ENASCellFixed(nn.Module):
    """Implement one ENAS cell (or node), each cell can have 5 different operations:
       avg_pool, max_pool, 3*3 conv, 5*5 conv, identity.
    """
    def __init__(self, in_channels, out_channels, node_id):
        super(ENASCellFixed, self).__init__()
        
        in_c, out_c = in_channels, out_channels
        self.choices = nn.ModuleDict({
                'conv3': SeparableConv(in_c, out_c, 3), 
                'conv5': SeparableConv(in_c, out_c, 5), 
                'avg_pool': nn.AvgPool2d(3, stride=1, padding=1), 
                'max_pool': nn.MaxPool2d(3, stride=1, padding=1), 
                'identity': IdentityBranch()
        })
        
    def forward(self, x, prev_cell, op_id):
        """
        Args:
            x: input from previous cell.
            prev_cell: integer, the previous cell's ID.
            op_id: integer, indicate which operation to use.
        """
        assert 0 <= op_id <= 4, "Operation ID out of range!"
        
        out = {
          0: lambda x: self.choices['conv3'](x), 
          1: lambda x: self.choices['conv5'](x), 
          2: lambda x: self.choices['avg_pool'](x), 
          3: lambda x: self.choices['max_pool'](x), 
          4: lambda x: self.choices['identity'](x)
        }[op_id](x)
        
        return out


class AuxHeadLayer(nn.Module):
    """Auxiliary head for micro child training."""
    
    def __init__(self, in_channels, side_length, trs=False, layer_sizes=[128, 768, 10]):
        super(AuxHeadLayer, self).__init__()
        assert len(layer_sizes) == 3, "Should have exactly 3 layers in the auxiliary head."
        
        self.side_len = math.floor((side_length - 5) / 3 + 1)
        self.aux = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(5, stride=3), 
            ReluConvBN(in_channels, layer_sizes[0]), 
            nn.Conv2d(layer_sizes[0], layer_sizes[1], self.side_len), 
            nn.ReLU(), 
        )
        self.proj = nn.Linear(layer_sizes[1], layer_sizes[-1])
        
    def forward(self, x):
        out = self.aux(x)
        logits = self.proj(torch.squeeze(torch.squeeze(out, dim=-1), dim=-1))
        return logits

    
class GlobalAvgPool(nn.Module):
    """Average all points of an image at each channel."""
    
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
        
    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return torch.mean(torch.mean(x, -1), -1)


