# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from MFIRSTD.models.MAFEM_block.ms_deform_attn_func import MSDeformAttnFunction
from MFIRSTD.models.network_utils import warp

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=5, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        kernel_size = 3
        self.sampling_offsets = nn.Conv2d(d_model, n_heads * n_levels * n_points * 2, kernel_size=kernel_size, padding=kernel_size//2)
        self.attention_weights = nn.Conv2d(d_model, n_heads * n_levels * n_points, kernel_size=kernel_size, padding=kernel_size//2)
        self.value_proj = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.output_proj = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
    def flow_guid_offset(self,flow_forward,flow_backward,offset):

        N,THW,heads,n_levels,points,_ = offset.shape
        offset = offset.reshape(N,n_levels,-1,heads,n_levels,points,2)
        offset_chunk0,offset_chunk1,offset_chunk2,offset_chunk3,offset_chunk4 = torch.chunk(offset, n_levels, dim=1)

        flow_forward01 = flow_forward[:,0]
        flow_forward12 = flow_forward[:,1]
        flow_forward23 = flow_forward[:,2]
        flow_forward34 = flow_forward[:,3]
        flow_forward02 = flow_forward01 + warp(flow_forward12,flow_forward01)
        flow_forward03 = flow_forward02 + warp(flow_forward23,flow_forward02)
        flow_forward04 = flow_forward03 + warp(flow_forward34,flow_forward03)
        flow_forward13 = flow_forward12 + warp(flow_forward23, flow_forward12)
        flow_forward14 = flow_forward13 + warp(flow_forward34, flow_forward13)
        flow_forward24 = flow_forward23 + warp(flow_forward34, flow_forward23)

        flow_backward10 = flow_backward[:,0]
        flow_backward21 = flow_backward[:,1]
        flow_backward32 = flow_backward[:,2]
        flow_backward43 = flow_backward[:,3]
        flow_backward42 = flow_backward43 + warp(flow_backward32, flow_backward43)
        flow_backward41 = flow_backward42 + warp(flow_backward21, flow_backward42)
        flow_backward40 = flow_backward41 + warp(flow_backward10, flow_backward41)
        flow_backward31 = flow_backward32 + warp(flow_backward21, flow_backward32)
        flow_backward30 = flow_backward31 + warp(flow_backward10, flow_backward31)
        flow_backward20 = flow_backward21 + warp(flow_backward10, flow_backward21)

        b,c,h,w = flow_backward10.shape
        flow_forward01 = flow_forward01.permute(0, 2, 3, 1).reshape(b,-1,c)
        flow_forward02 = flow_forward02.permute(0, 2, 3, 1).reshape(b, -1, c)
        flow_forward03 = flow_forward03.permute(0, 2, 3, 1).reshape(b, -1, c)
        flow_forward04 = flow_forward04.permute(0, 2, 3, 1).reshape(b, -1, c)
        flow_forward12 = flow_forward12.permute(0, 2, 3, 1).reshape(b,-1,c)
        flow_forward13 = flow_forward13.permute(0, 2, 3, 1).reshape(b,-1,c)
        flow_forward14 = flow_forward14.permute(0, 2, 3, 1).reshape(b,-1,c)
        flow_forward23 = flow_forward23.permute(0, 2, 3, 1).reshape(b, -1, c)
        flow_forward24 = flow_forward24.permute(0, 2, 3, 1).reshape(b, -1, c)
        flow_forward34 = flow_forward34.permute(0, 2, 3, 1).reshape(b, -1, c)


        flow_backward10 = flow_backward10.permute(0, 2, 3, 1).reshape(b,-1,c)
        flow_backward20 = flow_backward20.permute(0, 2, 3, 1).reshape(b, -1, c)
        flow_backward30 = flow_backward30.permute(0, 2, 3, 1).reshape(b, -1, c)
        flow_backward40 = flow_backward40.permute(0, 2, 3, 1).reshape(b, -1, c)
        flow_backward21 = flow_backward21.permute(0, 2, 3, 1).reshape(b,-1,c)
        flow_backward31 = flow_backward31.permute(0, 2, 3, 1).reshape(b, -1, c)
        flow_backward41 = flow_backward41.permute(0, 2, 3, 1).reshape(b, -1, c)
        flow_backward32 = flow_backward32.permute(0, 2, 3, 1).reshape(b, -1, c)
        flow_backward42 = flow_backward42.permute(0, 2, 3, 1).reshape(b, -1, c)
        flow_backward43 = flow_backward43.permute(0, 2, 3, 1).reshape(b, -1, c)

        flow_zeros = torch.zeros_like(flow_forward01)

        offset_chunk0 = offset_chunk0 + torch.stack([flow_zeros,flow_forward01,flow_forward02,flow_forward03,flow_forward04],dim=2)[:,None,:,None,:,None,:]
        offset_chunk1 = offset_chunk1 + torch.stack([flow_backward10,flow_zeros,flow_forward12,flow_forward13,flow_forward14],dim=2)[:,None,:,None,:,None,:]
        offset_chunk2 = offset_chunk2 + torch.stack([flow_backward20,flow_backward21,flow_zeros,flow_forward23,flow_forward24],dim=2)[:,None,:,None,:,None,:]
        offset_chunk3 = offset_chunk3 + torch.stack([flow_backward30,flow_backward31,flow_backward32,flow_zeros,flow_forward34],dim=2)[:,None,:,None,:,None,:]
        offset_chunk4 = offset_chunk4 + torch.stack([flow_backward30, flow_backward31, flow_backward32, flow_zeros, flow_forward34], dim=2)[:, None, :, None, :,None, :]
        offset = torch.cat([offset_chunk0,offset_chunk1,offset_chunk2,offset_chunk3,offset_chunk4],dim=1).reshape( N,THW,heads,n_levels,points,2)


        
        
        return offset
    def _reset_offset(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask,flow_forward,flow_backward):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        
        # assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        bs,t,c,h,w = query.shape
        value = self.value_proj(input_flatten.view(bs*t,c,h,w)).view(bs,t,c,h,w)
        sampling_offsets = self.sampling_offsets(query.view(bs*t,c,h,w)).reshape(bs,t,-1,h,w)

        attention_weights = self.attention_weights(query.view(bs*t,c,h,w)).view(bs,t,-1,h,w)

        query = query.flatten(3).transpose(2, 3).contiguous().view(bs,-1,c)
        value = value.flatten(3).transpose(2, 3).contiguous().view(bs,-1,c)
        sampling_offsets = sampling_offsets.flatten(3).transpose(2, 3).contiguous().view(bs,-1,self.n_heads * self.n_levels * self.n_points * 2)
        
        attention_weights = attention_weights.flatten(3).transpose(2, 3).contiguous().view(bs,-1,self.n_heads*self.n_levels * self.n_points)
        N, Len_q, _ = query.shape
        N, Len_in, _ = value.shape
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = sampling_offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        sampling_offsets = self.flow_guid_offset(flow_forward,flow_backward,sampling_offsets)
        attention_weights = attention_weights.view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = output.view(bs,t,h*w,c).transpose(2, 3).contiguous().view(bs*t,c,h,w)
        output = self.output_proj(output)
        return output



