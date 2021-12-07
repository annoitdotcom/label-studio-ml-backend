#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

""" This work is an pytorch implements of Dual Graph Convolutional Network 
for exploiting long-range contextual information for such semantic segmentation. 
The Dual Graph Convolutional Network (DGCNet) models the global context of the 
input feature by modelling two orthogonal graphs in a single framework.
The first component models spatial relationships between pixels in the image, 
while the second models interdependencies along the channel 
dimensions of the network's feature map.
Ref: https://arxiv.org/abs/1909.06121
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    """ Reasoning with Graph Convolution. After projecting the features 
    from coordinate space into the interaction space, we have graph 
    where each node contains feature descriptor. Capturing relations between 
    arbitrary regions in the input is now simplified to capturing 
    interactions between the features of the corresponding nodes. Treating the features 
    as nodes of a fully connected graph, using fully connected graph 
    by learning edge weights that correspond to interactions of the 
    underlying globally-pooled features of each node.

    Arguments:
        num_nodes (int): number of nodes in interaction
        num_state (int): number of states in graph
        bias (boolean): add bias or not
    """

    def __init__(self, num_nodes, num_states, bias=False):
        super(GCN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(num_nodes, num_nodes, kernel_size=1)
        self.conv2 = nn.Conv1d(num_states, num_states,
                               kernel_size=1, bias=bias)

    def forward(self, x):
        # Performs Laplacian smoothing, propagating the node features over the graph.
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x

        # Each node has received all necessary information,
        # update their state through a linear transformation
        h = self.conv2(self.relu(h))
        return h


class DualGraphConv(nn.Module):
    """ Dual graph convolutional network.
    Args:
        in_channels (int):
        inter_channels (int):
        is_normalize (boolean):
    """

    def __init__(self, in_channels, inter_channels, is_normalize=True):
        super(DualGraphConv, self).__init__()

        self.is_normalize = is_normalize
        self.num_nodes = int(1 * inter_channels)
        self.num_states = int(2 * inter_channels)

        # Coordinate space projection
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_theta = nn.Conv2d(
            in_channels, self.num_states, kernel_size=1)
        self.conv_gamma = nn.Conv2d(in_channels, self.num_nodes, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, self.num_nodes, kernel_size=1)

        # Coordinate graph convolution & reprojection
        self.graph_conv_coord = GCN(self.num_nodes, self.num_states)
        self.conv_ws = nn.Conv1d(
            self.num_states, self.num_states, kernel_size=1)

        # Reprojection
        self.block_coord = nn.Conv2d(
            self.num_states, in_channels, kernel_size=1)

        # Feature space projection
        self.conv_ft_theta = nn.Sequential(
            nn.Conv2d(in_channels, self.num_states, kernel_size=1),
            nn.BatchNorm2d(self.num_states),
            nn.ReLU()
        )
        self.conv_ft_phi = nn.Sequential(
            nn.Conv2d(in_channels, self.num_nodes, kernel_size=1),
            nn.BatchNorm2d(self.num_nodes),
            nn.ReLU()
        )

        # Feature graph convolution
        self.graph_conv_space = GCN(self.num_nodes, self.num_states)

        # Reprojection
        self.block_space = nn.Sequential(
            nn.Conv2d(self.num_nodes, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        n = x.size(0)
        x_origin = x
        x = self.avg_pool(x)

        # import pdb; pdb.set_trace()
        # Graph convolution in coordinate space.
        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_theta = self.conv_theta(x).view(n, self.num_states, -1)
        x_gamma = self.conv_gamma(x).view(n, self.num_nodes, -1)
        x_v = self.conv_v(x).view(n, self.num_nodes, -1)

        x_edges = torch.matmul(x_gamma.permute(0, 2, 1), x_v)
        x_edges = F.softmax(x_edges, dim=-1)

        x_graph = torch.matmul(x_theta, x_edges)
        x_graph = self.conv_ws(x_graph)

        # Upsampling & reproject coordinates
        # (n, num_state, h*w) --> (n, num_state, h, w)
        self.upsampling = nn.UpsamplingNearest2d(size=(x_origin.shape[-2:]))
        x_reprojected_coord = self.upsampling(
            x_graph.view(n, self.num_states, *x.size()[2:]))
        x_coord_projection = self.block_coord(x_reprojected_coord)

        # Graph convolution in feature space.
        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_ft_theta_prj = self.conv_ft_theta(
            x_origin).view(n, self.num_states, -1)
        x_ft_phi_prj = self.conv_ft_phi(x_origin).view(n, self.num_nodes, -1)

        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_ft_space = torch.matmul(
            x_ft_theta_prj, x_ft_phi_prj.permute(0, 2, 1))
        # x_ft_space = F.softmax(x_ft_space, dim=-1)

        if self.is_normalize:
            x_ft_space = x_ft_space * (1. / x_ft_phi_prj.size(2))

        # Reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.graph_conv_space(x_ft_space)
        x_reprojected_space = torch.matmul(
            x_ft_theta_prj.permute(0, 2, 1), x_n_rel).permute(0, 2, 1)

        x_reprojected_space = x_reprojected_space.view(
            n, self.num_nodes, *x_origin.size()[2:])
        x_space_projection = self.block_space(x_reprojected_space)

        # Finally refined features
        out = x_origin + x_coord_projection + x_space_projection
        return out


if __name__ == '__main__':
    """ Unit test """

    input_feats = torch.autograd.Variable(torch.randn(4, 2048, 20, 20))
    dual_graph = DualGraphConv(2048, 512, True)
    output = dual_graph(input_feats)
    print(output.shape)
