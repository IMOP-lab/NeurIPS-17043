import torch
import torch.nn as nn
from ..utils.utils import get_upsample_layer, get_conv_layer, AdaptivePool, PositionalEncoding

class MultiheadsCrossAttention(nn.Module):
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
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first= True)
        self.conv2input = get_conv_layer(spatial_dims=spatial_dims, in_channels=embed_dim//2, out_channels=embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.adapt_maxpool = AdaptivePool(spatial_dims, att_size[0], pool_type="max")
        self.adapt_avgpool = AdaptivePool(spatial_dims, att_size[0], pool_type="avg")
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
        # print(input1.shape,input2.shape,N)

        q = input1.view(B, C, -1).permute(2, 0, 1).contiguous()  # (B, W*H*D, C)
        k = input2.view(B, C, -1).permute(2, 0, 1).contiguous()  # (B, W*H*D, C)
        v = input2.view(B, C, -1).permute(2, 0, 1).contiguous()  # (B, W*H*D, C)
    
        attn_output,_ = self.self_attention(q, k, v)
        attn_output  = attn_output.contiguous()

        attn_output = attn_output.view_as(input1).contiguous()

        # attn_output = F.interpolate(attn_output, size=(W, H), mode='bilinear', align_corners=True)
        if self.spatial_dims == 2:
            upsample_layer = nn.Upsample(size=(W, H), mode='bilinear', align_corners=True)
        else:
            upsample_layer = nn.Upsample(size=(W, H, D), mode='trilinear', align_corners=True)            
        attn_output = upsample_layer(attn_output)
        
        return self.sigmoid(attn_output)
    
class Linear_attention(nn.Module):
    def __init__(self, spatial_dims, dim, num_heads, att_size, proj_drop=0., focusing_factor=3):
        """
        Linear attention which is the basic of Layer attention.
        para:
        spatial_dims: dimensions of data(2D or 3D).
        dim: the channels of the input data(k,q,v).
        num_heads: the number of heads in attention calculation.
        att_size: the size of pooling operation.
        focusing_factor: focusing factor in attention calculation.
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.att_size = att_size
    
        self.focusing_factor = focusing_factor
        # self.proj_q = nn.Linear(dim, head_dim * self.num_heads, bias=False)
        # self.proj_k = nn.Linear(dim, head_dim * self.num_heads, bias=False)
        # self.proj_v = nn.Linear(dim, head_dim * self.num_heads, bias=False)
        # self.proj_o = nn.Linear(head_dim * self.num_heads, dim)
        # self.drop = nn.Dropout(proj_drop)

        # self.dwc = nn.Conv3d(in_channels=head_dim, out_channels=head_dim, kernel_size=5, groups=head_dim, padding=2)
        # self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=5, groups=head_dim, padding=2)
        self.conv1x1 = get_conv_layer(spatial_dims=spatial_dims, in_channels=head_dim, out_channels=head_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        # self.positional_encoding = PositionalEncoding(spatial_dims=spatial_dims, att_size=att_size, dim=dim)
        self.positional_encoding1 = PositionalEncoding(spatial_dims=spatial_dims, att_size=att_size, dim=dim)
        self.positional_encoding2 = PositionalEncoding(spatial_dims=spatial_dims, att_size=att_size, dim=dim)
        # self.Act = nn.ReLU()
        self.Act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, q, k, v, B, C, N):#q,k(B, W*H*D, C)
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
            # v = v.reshape(B, self.num_heads, self.att_size[0], self.att_size[1], -1).permute(0, 1, 4, 2, 3).view(B, -1, self.att_size[0], self.att_size[1]).contiguous()
        elif self.spatial_dims == 3:
            v = v.reshape(B * self.num_heads, self.att_size[0], self.att_size[1], self.att_size[2], -1).permute(0, 4, 1, 2, 3).contiguous()
        else:
            raise ValueError("spatial_dims should be 2 or 3")
        attn_output = x + self.Act(self.conv1x1(v)).reshape(B, C, N).contiguous()
                
        # attn_output = self.proj_o(attn_output.permute(0,2,1)).permute(0,2,1)
        # attn_output = self.drop(x)
        
        return attn_output