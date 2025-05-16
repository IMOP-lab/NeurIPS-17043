from __future__ import annotations
from collections.abc import Sequence
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep
from dropblock import DropBlock2D, DropBlock3D, LinearScheduler
# from reformer_pytorch import LSHAttention
# from linformer import LinformerSelfAttention
from .utils.utils import get_upsample_layer, get_conv_layer, AdaptivePool, PositionalEncoding
from .utils.uxnet_encoder import uxnet_conv
from .attention.efficient_cross_attention import MultiheadsCrossAttention, Linear_attention
from .attention.sparse_attention import SparseAttention, blocksparse_attention_impl
# from .attention.LSHAttention import LSHattention
# from .attention.LinformerSelfAttention import Linformerattention
from .attention.HEPTAttention import HEPTAttention
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x
        
class TwoConv(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
    ):
        """
        Two conv layers, the first conv layer is used to adjust the number of channels, 
        and the second conv layer is used to integrate feature information.
        para:
        spatial_dims: dimensions of data(2D or 3D).
        in_channels: the channels of the input data.
        out_channels: the channels of the output data.
        """
        super().__init__()
        conv_0 = Convolution(spatial_dims, in_channels, out_channels, act=act, norm=norm, dropout=dropout, bias=bias, padding=1 )   #kernel_size=3
        conv_1 = Convolution(spatial_dims, out_channels, out_channels, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)

class Down(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
    ):
        """
        The downsampling layer includes a pooling downsampling operation to reduce the size of the feature map, 
        and a subsequent Twoconv layer to increase the number of channels.
        para:
        spatial_dims: dimensions of data(2D or 3D).
        in_channels: the channels of the input data.
        out_channels: the channels of the output data.
        """
        super().__init__()
        self.max_pooling = Pool['MAX',spatial_dims](kernel_size=2)
        self.avg_pooling = Pool['AVG',spatial_dims](kernel_size=2)
        self.convs = TwoConv(spatial_dims, in_channels, out_channels, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor):
        max_pooled = self.max_pooling(x)
        avg_pooled = self.avg_pooling(x)
        pooled_sum = max_pooled + avg_pooled
        down_out = self.convs(pooled_sum)
        return down_out

class UpCat(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        cat_channels: int,
        out_channels: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        pre_conv: nn.Module | str | None = "default",
        interp_mode: str = "linear",
        align_corners: bool | None = True,
        halves: bool = True,
        is_pad: bool = True,
    ):
        """
        The upsampling layer includes a upsampling operation to restore the feature map size, a cat operation to fuse original feature,
        and a subsequent Twoconv layer to decrease the number of channels.
        para:
        spatial_dims: dimensions of data(2D or 3D).
        in_channels: the channels of the input data.
        cat_channels: the channels of the cat data. 
        out_channels: the channels of the output data.
        upsample: default is deconv.
        halves: halves in the final upcat layer is set 'False', which doesn't decrease channels in the upsampling operation, others is 'True'. 
        """
        super().__init__()
        if upsample == "nontrainable" and pre_conv is None:
            up_channels = in_channels
        else:
            up_channels = in_channels // 2 if halves else in_channels #up_channels = in_channels // 2
        self.upsample = UpSample(
            spatial_dims,
            in_channels,
            up_channels,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_channels + up_channels, out_channels, act, norm, bias, dropout)
        self.is_pad = is_pad

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        x_0 = self.upsample(x) # upsample x_e to make its shape match x
        # print(f"x_0:{x_0.shape},x:{x.shape},x_e:{x_e.shape}")
        if x_e is not None and torch.jit.isinstance(x_e, torch.Tensor):
            if self.is_pad:
                dimensions = len(x.shape) - 2 # dimensions of the input data
                sp = [0] * (dimensions * 2) # sp is a list of length dimensions * 2 with 0 elements
                for i in range(dimensions): # go through each dimension
                    if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                        sp[i * 2 + 1] = 1
                        print("Using padding!")
                x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1)) # if the shapes of x_e and x_0 are same, no padding is actually performed.
            # print(f"x:,{x.shape}")
        else:
            x = self.convs(x_0)
        return x

class UpCat_Att(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        cat_channels: int,
        out_channels: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        pre_conv: nn.Module | str | None = "default",
        interp_mode: str = "linear",
        align_corners: bool | None = True,
        halves: bool = True,
        is_pad: bool = True,
        num_heads: int = 8,
        att_size :int = [6,6,6],
        c_min: int = 24,
    ):
        """
        The upsampling layer includes a upsampling operation to restore the feature map size, a cat operation to fuse original feature,
        and a subsequent Twoconv layer to decrease the number of channels.
        para:
        spatial_dims: dimensions of data(2D or 3D).
        in_channels: the channels of the input data.
        cat_channels: the channels of the cat data. 
        out_channels: the channels of the output data.
        upsample: default is deconv.
        halves: halves in the final upcat layer is set 'False', which doesn't decrease channels in the upsampling operation, others is 'True'. 
        num_heads: the number of heads in attention calculation.
        att_size: the size of pooling operation.
        """
        super().__init__()
        if upsample == "nontrainable" and pre_conv is None:
            up_channels = in_channels
        else:
            up_channels = in_channels // 2 if halves else in_channels #up_channels = in_channels // 2
        self.upsample = UpSample(
            spatial_dims,
            in_channels,
            up_channels,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_channels + up_channels, out_channels, act, norm, bias, dropout)
        self.is_pad = is_pad
        self.Layer_Attention = Up_Attention(spatial_dims, up_channels, cat_channels, num_heads, att_size, c_min)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        # print('Use Upcat_Att!')
        x_0 = self.upsample(x)
        
        if x_e is not None and torch.jit.isinstance(x_e, torch.Tensor):
            if self.is_pad:
                dimensions = len(x.shape) - 2 # dimensions of the input data
                sp = [0] * (dimensions * 2) # sp is a list of length dimensions * 2 with 0 elements
                for i in range(dimensions): # go through each dimension
                    if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                        sp[i * 2 + 1] = 1
                        print("Using padding!")
                x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(self.Layer_Attention(x_0,x_e))
            # x = self.convs(torch.cat([x_e, x_0], dim=1))
        else:
            x = self.convs(x_0)
        return x

class Down_Attention(nn.Module):
    def __init__(self, spatial_dims, dim, num_heads, att_size, qkv_bias=True, proj_drop=0.,
                 focusing_factor=3):
        """
        Layer attention for the downsampling process.
        para:
        spatial_dims: dimensions of data(2D or 3D).
        dim: the channels of the input1.
        num_heads: the number of heads in attention calculation.
        att_size: the size of pooling operation.
        focusing_factor: focusing factor in attention calculation.
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.att_size = att_size
        
        self.conv2input = get_conv_layer(spatial_dims=spatial_dims, in_channels=dim//2, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.adapt_maxpool = AdaptivePool(spatial_dims, att_size[0], pool_type="max")
        self.adapt_avgpool = AdaptivePool(spatial_dims, att_size[0], pool_type="avg")

        self.focusing_factor = focusing_factor
        # self.proj_q = nn.Linear(dim, head_dim * self.num_heads, bias=False)
        # self.proj_k = nn.Linear(dim, head_dim * self.num_heads, bias=False)
        # self.proj_v = nn.Linear(dim, head_dim * self.num_heads, bias=False)
        # self.proj_o = nn.Linear(head_dim * self.num_heads, dim)
        # self.drop = nn.Dropout(proj_drop)

        self.conv1x1 = get_conv_layer(spatial_dims=spatial_dims, in_channels=head_dim, out_channels=head_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        # self.positional_encoding = PositionalEncoding(spatial_dims=spatial_dims, att_size=att_size, dim=dim)
        self.positional_encoding1 = PositionalEncoding(spatial_dims=spatial_dims, att_size=att_size, dim=dim)
        self.positional_encoding2 = PositionalEncoding(spatial_dims=spatial_dims, att_size=att_size, dim=dim)
        # self.Act = nn.ReLU()
        self.Act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):
        if input1.shape[1] != input2.shape[1]:
            input2 = self.conv2input(input2)
        
        if self.spatial_dims == 2:
            B, C, W, H = input1.shape[0], input1.shape[1], input1.shape[2], input1.shape[3]
            N = self.att_size[0] * self.att_size[1]
        elif self.spatial_dims == 3:
            B, C, W, H, D = input1.shape[0], input1.shape[1], input1.shape[2], input1.shape[3], input1.shape[4]
            N = self.att_size[0] * self.att_size[1] * self.att_size[2]
        else:
            raise ValueError("spatial_dims should be 2 or 3")
        
        # input1 = self.adapt_maxpool(input1)
        # input2 = self.adapt_maxpool(input2)
        
        input1 = self.adapt_maxpool(input1) + self.adapt_avgpool(input1)
        input2 = self.adapt_maxpool(input2) + self.adapt_avgpool(input2)
        
        # reshape the input tensor into a 3D tensor
        q = input1.view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, W*H*D, C)
        k = input2.view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, W*H*D, C)
        v = input2.view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, W*H*D, C)
    
        # q = self.proj_q(q)
        # k = self.proj_k(k)
        # v = self.proj_v(v)
        
        # add an adaptive matrix
        q = self.Act(self.positional_encoding1(q))
        k = self.Act(self.positional_encoding2(k))
        scale = nn.Softplus()(self.scale)   # channel scale
        q = q / scale
        k = k / scale
        
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** self.focusing_factor
        k = k ** self.focusing_factor
        q = (q_norm / q.norm(dim=-1, keepdim=True)) * q
        k = (k_norm / k.norm(dim=-1, keepdim=True)) * k

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3) # (B, heads, W*H*D, ,C/heads)
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        # z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)#(2,8,216,1)
        kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))    # (B, heads, C/heads, C/heads)
        x = q @ kv                                                      # (B, heads, W*H*D,   C/heads)
        # x = q @ kv * z

        x = x.transpose(2,3).reshape(B, C, N)
        if self.spatial_dims == 2:
            v = v.reshape(B * self.num_heads, self.att_size[0], self.att_size[1], -1).permute(0, 3, 1, 2).contiguous()
        elif self.spatial_dims == 3:
            v = v.reshape(B * self.num_heads, self.att_size[0], self.att_size[1], self.att_size[2], -1).permute(0, 4, 1, 2, 3).contiguous()
        else:
            raise ValueError("spatial_dims should be 2 or 3")
        x = x + self.Act(self.conv1x1(v)).reshape(B, C, N).contiguous()

        # x = self.proj_o(x.permute(0,2,1)).permute(0,2,1)
        # attn_output = self.drop(x)
        
        attn_output = x.view_as(input1).contiguous()
        
        if self.spatial_dims == 2:
            upsample_layer = nn.Upsample(size=(W, H), mode='bilinear', align_corners=True)
        else:
            upsample_layer = nn.Upsample(size=(W, H, D), mode='trilinear', align_corners=True)            
        attn_output = upsample_layer(attn_output)

        return self.sigmoid(attn_output)
    
class Up_Attention(nn.Module):
    def __init__(self, spatial_dims, embed_dim1, embed_dim2, num_heads, att_size, c_min):
        """
        Layer attention for the upsampling process.
        para:
        spatial_dims: dimensions of data(2D or 3D).
        embed_dim1: the channels of the input x after upsampling.
        embed_dim2: the channels of the input data from encoder.
        num_heads: the number of heads in attention calculation.
        att_size: the size of pooling operation.
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.embed_dim1 = embed_dim1 #in_channels//2
        self.embed_dim2 = embed_dim2 #cat_channels

        self.num_heads = num_heads
        self.att_size = att_size
        channel_reduce = c_min*2

        self.conv1 = get_conv_layer(spatial_dims=spatial_dims, in_channels=embed_dim1, out_channels=channel_reduce, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = get_conv_layer(spatial_dims=spatial_dims, in_channels=embed_dim2, out_channels=channel_reduce, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv3 = get_conv_layer(spatial_dims=spatial_dims, in_channels=channel_reduce, out_channels=channel_reduce, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv4 = get_conv_layer(spatial_dims=spatial_dims, in_channels=channel_reduce, out_channels=channel_reduce, kernel_size=1, stride=1, padding=0, bias=True)

        self.Norm = nn.GroupNorm(channel_reduce//8, channel_reduce)
        # self.Act = nn.ReLU(inplace=True)
        self.Act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.adapt_maxpool = AdaptivePool(spatial_dims, att_size[0], pool_type="max")
        self.adapt_avgpool = AdaptivePool(spatial_dims, att_size[0], pool_type="avg")
        
        # self.self_attention1 = nn.MultiheadAttention(embed_dim=embed_dim1+embed_dim2, num_heads=num_heads, batch_first= True)
        # self.self_attention2 = nn.MultiheadAttention(embed_dim=embed_dim1+embed_dim2, num_heads=num_heads, batch_first= True)
        self.attention1 = Linear_attention(spatial_dims, dim=channel_reduce, num_heads=num_heads, att_size=att_size)
        self.attention2 = Linear_attention(spatial_dims, dim=channel_reduce, num_heads=num_heads, att_size=att_size)
        # self.LSHAttention1 = LSHAttention(bucket_size = 64,n_hashes = 4,causal = True)
        # self.LSHAttention2 = LSHAttention(bucket_size = 64,n_hashes = 4,causal = True)
        # self.LinformerSelfAttention = LinformerSelfAttention(dim = embed_dim1+embed_dim2, seq_len = att_size[0]*att_size[1], heads = 8, k = 256, one_kv_head = True, share_kv = True)
        # self.LinformerSelfAttention = LinformerSelfAttention(dim = embed_dim1+embed_dim2, seq_len = att_size[0]*att_size[1], heads = 8, k = 256, one_kv_head = True, share_kv = True)
        # self.HEPTAttention1 = HEPTAttention(
        #                     hash_dim=embed_dim1+embed_dim2, 
        #                     h_dim=(embed_dim1+embed_dim2)//num_heads, 
        #                     num_heads=num_heads, 
        #                     block_size=att_size[0],
        #                     n_hashes=8,
        #                     num_w_per_dist=att_size[0]
        #                     )
        # self.HEPTAttention2 = HEPTAttention(
        #                     hash_dim=embed_dim1+embed_dim2, 
        #                     h_dim=(embed_dim1+embed_dim2)//num_heads, 
        #                     num_heads=num_heads, 
        #                     block_size=att_size[0],
        #                     n_hashes=8,
        #                     num_w_per_dist=att_size[0]
        #                     )
                
        self.sigmoid = nn.Sigmoid()
        self.conv_fusion = get_conv_layer(spatial_dims=spatial_dims, in_channels=embed_dim1+embed_dim2, out_channels=channel_reduce, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_final = get_conv_layer(spatial_dims=spatial_dims, in_channels=channel_reduce * 2, out_channels=embed_dim1+embed_dim2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input1, input2):
        fusion = torch.cat((input2, input1),dim=1)
        fusion = self.conv_fusion(fusion)
        
        B, C = fusion.shape[0], fusion.shape[1]

        input1 = self.Act(self.Norm(self.conv1(input1)))
        input1 = self.Act(self.Norm(self.conv3(input1)))
        
        if self.spatial_dims == 2:
            N = self.att_size[0] * self.att_size[1]
        elif self.spatial_dims == 3:
            N = self.att_size[0] * self.att_size[1] * self.att_size[2]
        else:
            raise ValueError("spatial_dims should be 2 or 3")
        temp_input1 = input1

        if self.spatial_dims == 2:
            W1, H1 = input1.shape[2], input1.shape[3]
        else:
            W1, H1, D1 = input1.shape[2], input1.shape[3], input1.shape[4]
        
        input2 = self.Act(self.Norm(self.conv2(input2)))
        input2 = self.Act(self.Norm(self.conv4(input2)))

        temp_input2 = input2
        if self.spatial_dims == 2:
            W2, H2 = input1.shape[2], input1.shape[3]
        else:
            W2, H2, D2 = input1.shape[2], input1.shape[3], input1.shape[4]
        
        input1 = self.adapt_maxpool(input1) + self.adapt_avgpool(input1)
        input2 = self.adapt_maxpool(input2) + self.adapt_avgpool(input2)
        input_kv = self.adapt_maxpool(fusion) + self.adapt_avgpool(fusion)

        # reshape the input tensor into a 3D tensor
        query1 = input1.view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, W*H*D, C)
        query2 = input2.view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, W*H*D, C)
        key = input_kv.view(B, C, -1).permute(0, 2, 1).contiguous()   # (B, W*H*D, C)
        value = input_kv.view(B, C, -1).permute(0, 2, 1).contiguous() # (B, W*H*D, C)

        ### multiple attention mechanism
        'Efficient cross-attention'
        attn_output1 = self.attention1(query1, key, value, B, C, N)
        attn_output2 = self.attention2(query2, key, value, B, C, N)
        'Multi-heads scaled dot product attention'
        # attn_output1,_ = self.self_attention1(query1, key, value)
        # attn_output1  = attn_output1.contiguous()
        # attn_output2,_ = self.self_attention2(query2, key, value)
        # attn_output2  = attn_output2.contiguous()
        'Sparse attention'
        # attn_output1 = blocksparse_attention_impl(query1, key, value, self.num_heads, "all", 32)
        # attn_output2 = blocksparse_attention_impl(query2, key, value, self.num_heads, "all", 32)
        'LSHattention'
        # attn_output1, _, _ = self.LSHAttention1(query1, value)
        # attn_output2, _, _ = self.LSHAttention2(query2, value)  
        'LinformerSelfAttention'
        # attn_output1 = self.LinformerSelfAttention(value, query1)
        # attn_output2 = self.LinformerSelfAttention(value, query2)  
        'LinformerSelfAttention'
        # attn_output1 = self.LinformerSelfAttention(value, query1)
        # attn_output2 = self.LinformerSelfAttention(value, query2)  
        # print(attn_output1.shape,attn_output2.shape)      
        'HEPTAttention'
        # attn_output1 = self.HEPTAttention1(query1, key, value)
        # attn_output2 = self.HEPTAttention2(query2, key, value)  

        if self.spatial_dims == 2:
            attn_output1 = attn_output1.reshape(B, C, self.att_size[0], self.att_size[1]).contiguous()
            attn_output2 = attn_output2.reshape(B, C, self.att_size[0], self.att_size[1]).contiguous()        
        elif self.spatial_dims == 3:
            attn_output1 = attn_output1.reshape(B, C, self.att_size[0], self.att_size[1], self.att_size[2]).contiguous()
            attn_output2 = attn_output2.reshape(B, C, self.att_size[0], self.att_size[1], self.att_size[2]).contiguous()
        else:
            raise ValueError("spatial_dims should be 2 or 3")
        
        # attn_output = F.interpolate(attn_output, size=(W, H), mode='bilinear', align_corners=True)
        if self.spatial_dims == 2:
            upsample_layer1 = nn.Upsample(size=(W1, H1), mode='bilinear', align_corners=True)
            upsample_layer2 = nn.Upsample(size=(W2, H2), mode='bilinear', align_corners=True)
        else:
            upsample_layer1 = nn.Upsample(size=(W1, H1, D1), mode='trilinear', align_corners=True)
            upsample_layer2 = nn.Upsample(size=(W2, H2, D2), mode='trilinear', align_corners=True)  

        attn_output1 = self.sigmoid(upsample_layer1(attn_output1)) * temp_input1
        attn_output2 = self.sigmoid(upsample_layer2(attn_output2)) * temp_input2
        attn_output = self.conv_final(torch.cat((attn_output1, attn_output2),dim=1))

        return attn_output
        
class CCFormer_light_new(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        depths: Sequence[int] = [2, 2, 2, 2],
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),   #instance normalization(实例正则化),在单个样本的每个特征通道上分别进行正则化,忽略了批次中其他样本的信息。可替换为("group", {"num_groups": 8})
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dropblock_prob: float | None = None, # The probability of dropout
        dropblock: str = "dropblock2D",
        use_upcat_att: bool = False,
        feature_size: int = 512,
        mode: str = "L",
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.feature_size = feature_size
        self.mode = mode

        self.act = act
        self.norm = norm
        self.bias = bias
        self.dropout = dropout
        self.upsample = upsample

        self.config = self.get_config()
        self.fea = self.config["features"]

        self.dropout_prob = dropblock_prob
        if dropblock_prob is not None:
            dropblock_dict = {
                "dropout2D": nn.Dropout2d(p=self.dropout_prob),
                "dropout3D": nn.Dropout3d(p=self.dropout_prob),
                "dropblock2D": DropBlock2D(block_size=256, drop_prob=self.dropout_prob),
                "dropblock3D": DropBlock3D(block_size=48, drop_prob=self.dropout_prob),
                "dropblock2D_schedule": LinearScheduler(
                    DropBlock2D(drop_prob=self.dropout_prob, 
                                block_size=256),
                                start_value=0,
                                stop_value=dropblock_prob,
                                nr_steps=400
                ),
                "dropblock3D_schedule": LinearScheduler(
                    DropBlock3D(drop_prob=self.dropout_prob, 
                                block_size=48),
                                start_value=0,
                                stop_value=dropblock_prob,
                                nr_steps=40000
                )
            }
        
            self.dropblock = dropblock_dict.get(dropblock)
            if self.dropblock is None:
                raise ValueError(f'Unsupported dropblock mode: {dropblock}')

        self.conv_init = TwoConv(spatial_dims, in_channels, self.fea[0], act, norm, bias, dropout)
        self.down_layers = nn.ModuleList([
            Down(spatial_dims, self.fea[i], self.fea[i+1], act, norm, bias, dropout)
            for i in range(len(self.fea)-2)
        ])
                
        self.att_layers = nn.ModuleList([
            self.create_attention_layer(i, self.fea[i+1])
            for i in range(len(self.fea)-2)
        ])

        self.upcat_configs = [
            self.get_upcat_config(use_upcat_att, idx=i, in_channels=self.fea[4-i], cat_channels=self.fea[3-i], out_channels=self.fea[3-i])
            for i in range(len(self.fea) - 3)
        ]
        
        self.upcat_layers = nn.ModuleList([
            config['cls'](spatial_dims, **config['params'])
            for config in self.upcat_configs
        ])
        
        self.final_conv1 = Conv["conv", spatial_dims](self.fea[1], self.fea[0], kernel_size=1)
        self.final_conv2 = Conv["conv", spatial_dims](self.fea[0], out_channels, kernel_size=1)
        self.upsample1 = get_upsample_layer(spatial_dims=spatial_dims, in_channels=out_channels, upsample_mode="nontrainable", scale_factor=2)

    def get_config(self):
        if self.mode == "S":
            att_size = [self.feature_size // 16] * 4
            features = (16, 32, 64, 128, 256, 16)
        elif self.mode == "L":
            att_size = [self.feature_size // 2,
                        self.feature_size // 4,
                        self.feature_size // 8,
                        self.feature_size // 16]
            features = {
                2: (32, 64, 128, 256, 512, 32),
                3: (24, 48, 96, 192, 384, 24)
            }[self.spatial_dims]
        elif self.mode == "H":
            att_size = [self.feature_size // 2,
                        self.feature_size // 4,
                        self.feature_size // 8,
                        self.feature_size // 16]
            features = {
                2: (64, 128, 256, 512, 1024, 64),
                3: (32, 64, 128, 256, 512, 32)
            }[self.spatial_dims]
        else:
            raise ValueError(f'Unsupported mode: {self.mode}')
        return {"features": features, "att_size": att_size}

    def create_attention_layer(self, idx, fea):
        att_size = [self.config['att_size'][idx]] * (self.spatial_dims)
        return Down_Attention(
            spatial_dims=self.spatial_dims,
            dim=fea,
            num_heads=8*(2**idx), #8,16,32,64
            att_size=att_size
        )
    
    def get_upcat_config(self, use_upcat_att, idx, in_channels, cat_channels, out_channels):
        base_params = {
            "in_channels": in_channels,
            "cat_channels": cat_channels,
            "out_channels": out_channels,     
            "act": self.act,
            "norm": self.norm,
            "bias": self.bias,
            "dropout": self.dropout,
            "upsample": self.upsample,       
        }
        if use_upcat_att:
            return {
                "cls": UpCat_Att,
                "params": {
                    **base_params,
                    "num_heads": 32//(2**idx),
                    "att_size": [self.config['att_size'][3-idx]] * self.spatial_dims,
                    "c_min": self.fea[1]
                }
            }
        else:
            return {
                "cls": UpCat, 
                "params": base_params
            }
            
    def forward(self, x: torch.Tensor):
        features = [self.conv_init(x)]
            
        for i, (down_layer, att_layer) in enumerate(zip(self.down_layers, self.att_layers)):
            x = down_layer(features[-1]) 
            x = att_layer(x, features[-1]) * x
            features.append(x)
            
        for i, upcat_layer in enumerate(self.upcat_layers):
            x = upcat_layer(x, features[-(i+2)])
            
        x = self.upsample1(x)
        x = self.final_conv1(x) + features[0]
        logits = self.final_conv2(x)
        return logits