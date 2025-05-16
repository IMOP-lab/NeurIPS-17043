import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ..utils.utils import get_conv_layer, AdaptivePool

def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == 'all':
        b = torch.tril(torch.ones([n, n]))
    elif attn_mode == 'local':
        bandwidth = local_attn_ctx
        ctx = min(n - 1, bandwidth - 1)
        b = torch.tril(torch.ones([n, n]), ctx)
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
        y = torch.transpose(x, 0, 1)
        z = torch.zeros([n, n], dtype=torch.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = torch.eq(torch.fmod(q - k, stride), 0)
        c3 = torch.logical_and(c1, c2)
        b = c3.float()
    else:
        raise ValueError('Not yet implemented')
    b = torch.reshape(b, [1, 1, n, n])
    return b

def strided_transpose(x, n_ctx, local_attn_ctx, blocksize):
    bT_ctx = n_ctx // local_attn_ctx
    assert bT_ctx % blocksize == 0, f'{bT_ctx}, {blocksize}'
    n, t, embd = x.size()
    x = torch.reshape(x, [n, bT_ctx, local_attn_ctx, embd])
    x = torch.transpose(x, 0, 2, 1, 3)
    x = torch.reshape(x, [n, t, embd])
    return x

def split_heads(x, n):
    return torch.transpose(split_states(x, n), 0, 2, 1, 3)

def merge_heads(x):
    return merge_states(torch.transpose(x, 0, 2, 1, 3))

def split_states(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = x.size()
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return torch.reshape(x, new_x_shape)


    return torch.reshape(x, new_x_shape)

def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = x.size()
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return torch.reshape(x, new_x_shape)

def attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None):
    q = split_heads(q, heads)
    k = split_heads(k, heads)
    v = split_heads(v, heads)
    n_timesteps = k.size()[2]
    mask = get_attn_mask(n_timesteps, attn_mode, local_attn_ctx).float()
    w = torch.matmul(q, k.transpose(-2, -1))
    scale_amount = 1.0 / np.sqrt(q.size()[-1])
    w = w * scale_amount
    w = w * mask + -1e9 * (1 - mask)
    w = F.softmax(w, dim=-1)
    a = torch.matmul(w, v)
    a = merge_heads(a)
    return a

def blocksparse_attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None, blocksize=32, num_verts=None, vertsize=None):
    n_ctx = q.size()[1]
    if attn_mode == 'strided':
        q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
        k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
        v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)
    n_state = q.size()[-1] // heads
    scale_amount = 1.0 / np.sqrt(n_state)
    w = torch.matmul(q, k.transpose(-2, -1))
    w = F.softmax(w * scale_amount, dim=-1)
    a = torch.matmul(w, v)
    if attn_mode == 'strided':
        n, t, embd = a.size()
        bT_ctx = n_ctx // local_attn_ctx
        a = torch.reshape(a, [n, local_attn_ctx, bT_ctx, embd])
        a = torch.transpose(a, 0, 2, 1, 3)
        a = torch.reshape(a, [n, t, embd])
    return a

class SparseAttention(nn.Module):
    def __init__(self, spatial_dims, embed_dim, num_heads, att_size, attn_mode, local_attn_ctx=None, blocksize=32):
        super(SparseAttention, self).__init__()
        self.spatial_dims = spatial_dims
        self.embed_dim = embed_dim
        self.heads = num_heads
        self.att_size = att_size
        self.attn_mode = attn_mode
        self.local_attn_ctx = local_attn_ctx
        self.blocksize = blocksize
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

        q = input1.view(B, C, -1).permute(2, 0, 1).contiguous()  # (B, W*H*D, C)
        k = input2.view(B, C, -1).permute(2, 0, 1).contiguous()  # (B, W*H*D, C)
        v = input2.view(B, C, -1).permute(2, 0, 1).contiguous()  # (B, W*H*D, C)
        
        spare_attention = blocksparse_attention_impl(q, k, v, self.heads, self.attn_mode, self.local_attn_ctx)
        attn_output = spare_attention.permute(1, 2, 0).contiguous()   # (C, B, W*H*D)
        if self.spatial_dims == 2:
            attn_output = attn_output.view(B, C, self.att_size[0], self.att_size[1])
            upsample_layer = nn.Upsample(size=(W, H), mode='bilinear', align_corners=True)
        else:
            attn_output = attn_output.view(B, C, self.att_size[0], self.att_size[1], self.att_size[2])
            upsample_layer = nn.Upsample(size=(W, H, D), mode='trilinear', align_corners=True)            
        attn_output = upsample_layer(attn_output)
        
        return self.sigmoid(attn_output)
