import torch
import torch.nn as nn

def get_upsample_layer(
    spatial_dims: int, 
    in_channels: int, 
    upsample_mode: str = "nontrainable", 
    scale_factor: int = 2,
    align_corners: bool = True
):
    """
    Create an upper sampling layer that supports both 2D and 3D data.
    para:
    spatial_dims: dimension of data(2D or 3D).
    in_channels: channels of input data.
    upsample_mode: upsample mode, support "nontrainable", "deconv", "pixelshuffle".
    scale_factor: scale factor of upsampling.
    align_corners: whether to align corners when interpolating.
    """
    if upsample_mode == "nontrainable":# interpolation upsampling
        if spatial_dims == 2:
            return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=align_corners)
        elif spatial_dims == 3:
            return nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=align_corners)
        else:
            raise ValueError("spatial_dims should be 2 or 3")

    elif upsample_mode == "deconv":# convTranspose upsampling
        kernel_size = scale_factor
        stride = scale_factor
        if spatial_dims == 2:
            return nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride)
        elif spatial_dims == 3:
            return nn.ConvTranspose3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride)
        else:
            raise ValueError("spatial_dims should be 2 or 3")

    elif upsample_mode == "pixelshuffle":# only 2D
        if spatial_dims == 2:
            return nn.PixelShuffle(upscale_factor=scale_factor)
        elif spatial_dims == 3:
            raise NotImplementedError("PixelShuffle is only implemented for 2D operations in PyTorch.")
        else:
            raise ValueError("spatial_dims should be 2 or 3")

    else:
        raise ValueError("Invalid upsample_mode. Choose from 'nontrainable', 'deconv', 'pixelshuffle'.")
    
def get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
    """
    return 2D or 3D conv layer. 
    para:
    spatial_dims: dimension of data(2D or 3D).
    in_channels: channels of the input data.
    out_channels: channels of the output data.
    kernel_size: kernel size.
    stride: stride.
    padding: padding.
    bias: whether use bias.
    """
    if spatial_dims == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    elif spatial_dims == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    else:
        raise ValueError("spatial_dims should be 2 or 3!")
    
class AdaptivePool(nn.Module):
    def __init__(self, spatial_dims, att_size, pool_type="max"):
        """
        Create an adaptive pooling layer that supports both 2D and 3D data.
        para:
        spatial_dims: 2D pooling or 3D pooling.
        att_size: pooling size.
        pool_type: max pooling or avg pooling.
        """
        super().__init__()
        
        if spatial_dims == 2:
            if pool_type == "max":
                self.adapt_pool = nn.AdaptiveMaxPool2d(output_size=(att_size, att_size))
            elif pool_type == "avg":
                self.adapt_pool = nn.AdaptiveAvgPool2d(output_size=(att_size, att_size))
            else:
                raise ValueError("pool_type should be 'max' or 'avg'!")
        elif spatial_dims == 3:
            if pool_type == "max":
                self.adapt_pool = nn.AdaptiveMaxPool3d(output_size=(att_size, att_size, att_size))
            elif pool_type == "avg":
                self.adapt_pool = nn.AdaptiveAvgPool3d(output_size=(att_size, att_size, att_size))
            else:
                raise ValueError("pool_type should be 'max' or 'avg'!")
        else:
            raise ValueError("spatial_dims should be 2 or 3!")

    def forward(self, x: torch.Tensor):
        return self.adapt_pool(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, spatial_dims, att_size, dim):
        """
        Initialize a 2D or 3D positional encoding parameter.
        para:
        spatial_dims: dimension of data(2D or 3D).
        att_size: the size of positional encoding parameter.
        dim: the last dimension of positional encoding parameter(equal to channels).
        """
        super().__init__()
        
        if spatial_dims == 2:
            self.positional_encoding = nn.Parameter(torch.zeros(size=(1, att_size[0] * att_size[1], dim)))
        elif spatial_dims == 3:
            self.positional_encoding = nn.Parameter(torch.zeros(size=(1, att_size[0] * att_size[1] * att_size[2], dim)))
        else:
            raise ValueError("spatial_dims should be 2 or 3")

    def forward(self, x: torch.Tensor):
        return x + self.positional_encoding