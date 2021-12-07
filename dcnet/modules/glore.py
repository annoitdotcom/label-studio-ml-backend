#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

""" This work is an pytorch implements of an approach for reasoning 
globally in which a set of features are globally aggregated over the coordinate 
space and then projected to an interaction space where relational reasoning can be 
efficiently computed. After reasoning, relation-aware features are distributed back 
to the original coordinate space for down-stream tasks. 
The Global Reasoning unit (GloRe unit) that implements the coordinate-interaction 
space mapping by weighted global pooling and weighted broadcasting, and 
the relation reasoning via graph convolution on a small graph in interaction space.
Ref: https://arxiv.org/abs/1811.12814
"""

import torch
import torch.nn as nn


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


class GloReUnit(nn.Module):
    """ Graph-Based Global Reasoning Networks.
    Arguments:
        in_channels (int): number of nodes in interaction
        inter_channels (int): number of states in graph
        ConvNd (instance): type of convolution Conv1d|Conv2d|Conv3d
        BatchNormNd (instance): type of batch-norm BatchNorm1d|BatchNorm2d|BatchNorm3d 
        normalize (boolean): True if normalize values of feature maps else False 
    """

    def __init__(
        self,
        in_channels,
        inter_channels,
        ConvNd=nn.Conv2d,
        BatchNormNd=nn.BatchNorm2d,
        is_normalize=False
    ):
        super(GloReUnit, self).__init__()
        self.is_normalize = is_normalize
        self.num_nodes = int(1 * inter_channels)
        self.num_states = int(2 * inter_channels)

        # reduce dim
        self.conv_state = ConvNd(
            in_channels,
            self.num_states,
            kernel_size=1
        )

        # projection map
        self.conv_proj = ConvNd(
            in_channels,
            self.num_nodes,
            kernel_size=1
        )

        # reasoning via graph convolution
        self.gcn = GCN(
            num_nodes=self.num_nodes,
            num_states=self.num_states
        )

        # extend dimension
        self.conv_extend = ConvNd(
            self.num_states,
            in_channels,
            kernel_size=1,
            bias=False
        )
        self.blocker = BatchNormNd(
            in_channels,
            eps=1e-04
        )  # should be zero initialized

    def forward(self, x):
        n = x.size(0)

        # Transform from coordinate to interaction space.
        x_state_reshaped = self.conv_state(x).view(n, self.num_states, -1)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_nodes, -1)
        x_rproj_reshaped = x_proj_reshaped

        # Projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(
            x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))

        if self.is_normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # Reasoning features nodes: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # Reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_states, *x.size()[2:])
        out = x + self.blocker(self.conv_extend(x_state))
        return out


if __name__ == '__main__':
    """ Unit test """

    input_feats = torch.autograd.Variable(torch.randn(2, 32, 14))
    glore_unit = GloReUnit(32, 16, nn.Conv1d, nn.BatchNorm1d, True)
    output = glore_unit(input_feats)
