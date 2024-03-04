import torch as th
import torch.nn as nn
from pytorch_wavelets import DWTForward

class CNNLoss(nn.Module):
    def __init__(self) -> None:
        super(CNNLoss, self).__init__()
    
    def forward(self, x: th.Tensor, y: th.Tensor, alpha: float = 0.12, beta: float = 0.10):
        total_loss = alpha * self.FFT_loss(x, y) + beta * self.DWT_loss(x, y) + self.Spatial_loss(x, y)
        return total_loss.float()
    
    def FFT_loss(self, input, target):
        input_fft = th.fft.fftn(input)
        target_fft = th.fft.fftn(target)
        return th.mean(th.abs(input_fft - target_fft) ** 2)
    
    def DWT_loss(self, input, target, J=3):
        xfm = DWTForward(J=J, mode="zero", wave='db3').to(input.device)
        _, H_in = xfm(input.float())
        _, H_tar = xfm(target.float())
        H_in = H_in[0][0][0]
        H_tar = H_tar[0][0][0]
        loss = 0.0
        for i in range(J):
            loss += th.mean(th.abs(H_in[i] - H_tar[i]) ** 2)
        return loss
    
    def Spatial_loss(self, input, target):
        return th.mean(th.abs(input - target) ** 2)