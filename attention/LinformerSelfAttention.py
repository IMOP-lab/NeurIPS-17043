import torch
from torch import nn
from linformer import LinformerSelfAttention
from ..utils.utils import get_conv_layer, AdaptivePool

class Linformerattention(nn.Module):
    def __init__(self, spatial_dims, embed_dim, num_heads, att_size):
        """
        Basic self-attention implementation code.
        para:
        spatial_dims: dimensions of data(2D or 3D).
        embed_dim: the channels of the input1.
        num_heads: the number of heads in attention calculation.
        att_size: the size of pooling operation.
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.att_size = att_size
        self.linformer_attention = LinformerSelfAttention(dim = embed_dim, seq_len = att_size[0]*att_size[1], heads = num_heads, k = 256, one_kv_head = True, share_kv = True)
        self.conv2input = get_conv_layer(spatial_dims=spatial_dims, in_channels=embed_dim//2, out_channels=embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.adapt_maxpool = AdaptivePool(spatial_dims, att_size[0], pool_type="max")
        self.adapt_avgpool = AdaptivePool(spatial_dims, att_size[0], pool_type="avg")
        # self.upsample = get_upsample_layer(spatial_dims=spatial_dims, upsample_mode="nontrainable", scale_factor=2)
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

        q = input1.view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, W*H*D, C)
        kv = input2.view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, W*H*D, C)
        
        attn_output= self.linformer_attention(kv, q)
        attn_output  = attn_output.contiguous()

        attn_output = attn_output.view_as(input1).contiguous()

        # attn_output = F.interpolate(attn_output, size=(W, H), mode='bilinear', align_corners=True)
        if self.spatial_dims == 2:
            upsample_layer = nn.Upsample(size=(W, H), mode='bilinear', align_corners=True)
        else:
            upsample_layer = nn.Upsample(size=(W, H, D), mode='trilinear', align_corners=True)            
        attn_output = upsample_layer(attn_output)
        
        return self.sigmoid(attn_output)