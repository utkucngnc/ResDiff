import math
import torch as th
import torch.nn as nn

__all__ = ['SimpleSR', 'SimpleSR_2x', 'SimpleSR_4x', 'SimpleSR_8x']

class SimpleSR(nn.Module):
    def __init__(self, 
                 in_ch: int = 3, 
                 channels: int = 64, 
                 out_ch: int = 3, 
                 scale_factor: int = 4) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        hidden_channels = channels // 2
        out_channels = int(out_ch * (scale_factor ** 2))

        # Feature Mapping
        self.feat_map = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Sub-pixel Convolution
        self.sub_pixel_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(scale_factor)
        )

        # Initial model weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 32:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    0.001)
                    nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                    nn.init.zeros_(module.bias.data)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__forward_impl(x)
    
    def __forward_impl(self, x: th.Tensor) -> th.Tensor:
        res = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)(x)
        x = self.feat_map(x)
        x = self.sub_pixel_conv(x).float()
        #out = th.clamp_(x + res, 0.0, 1.0)
        return th.clamp_(x, 0.0, 1.0)

def SimpleSR_2x(**kwargs) -> SimpleSR:
    return SimpleSR(scale_factor=2, **kwargs)

def SimpleSR_4x(**kwargs) -> SimpleSR:
    return SimpleSR(scale_factor=4, **kwargs)

def SimpleSR_8x(**kwargs) -> SimpleSR:
    return SimpleSR(scale_factor=8, **kwargs)