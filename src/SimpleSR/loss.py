import torch as th
import torch.nn as nn

import pywt
import ptwt
import numpy as np
from typing import Union

class DiscreteWaveletLoss(nn.Module):
    """The torch.nn.Module class that implements discrete wavelet loss - a
    wavelet domain loss function for optimizing SuperResolution models.
    
    Args:
        loss_weight (float): weight for discrete wavelet loss. Default: 1.0
        wavelet (str): wavelet name. Default: "db3"
        level (int): decomposition level. Default: 1
    """
    def __init__(self, loss_weight: float = 1.0, wavelet: str = "db3", level: int = 1) -> None:
        super(DiscreteWaveletLoss, self).__init__()
        assert wavelet in pywt.wavelist(), f"Invalid wavelet name: {wavelet}"
        self.wavelet = pywt.Wavelet(wavelet)
        self.level = level
        self.loss_weight = loss_weight
    
    def forward_dwt(self, x: th.Tensor) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        coefficients = ptwt.wavedec2(x, wavelet=self.wavelet , level=self.level, mode="constant") # Returns a list of [cA, (cH, cV, cD)]  
        return coefficients[1]
    
    def forward(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        coeff_x = self.forward_dwt(x)
        coeff_y = self.forward_dwt(y)

        loss = th.stack([th.mean(th.abs(coeff_x[i] - coeff_y[i]) ** 2) for i in range(3)]).sum()
        #print(loss)
        return self.loss_weight * loss

class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x: th.Tensor):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = th.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        freq = th.fft.fft2(y, norm='ortho')
        freq = th.stack([freq.real, freq.imag], -1)

        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = th.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = th.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[th.isnan(matrix_tmp)] = 0.0
            matrix_tmp = th.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return th.mean(loss)

    def forward(self, pred: th.Tensor, target: th.Tensor, matrix : th.Tensor = None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = th.mean(pred_freq, 0, keepdim=True)
            target_freq = th.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        loss = self.loss_formulation(pred_freq, target_freq, matrix)
        #print(loss)
        return loss * self.loss_weight

class L2Loss(nn.Module):
    def __init__(self) -> None:
        super(L2Loss, self).__init__()
    
    def forward(self, x: th.Tensor, y: th.Tensor):
        loss = th.mean(th.abs(x - y) ** 2)
        #print(loss)
        return loss
        
class CNNLoss(nn.Module):
    def __init__(self) -> None:
        super(CNNLoss, self).__init__()
    
    def forward(self, x: th.Tensor, y: th.Tensor, alpha: float = 0.12, beta: float = 0.10):
        fft_loss = FocalFrequencyLoss(loss_weight=alpha)(x, y)
        wavelet_loss = DiscreteWaveletLoss(loss_weight=beta)(x, y)
        spatial_loss = L2Loss()(x, y)
        total_loss = fft_loss + wavelet_loss + spatial_loss
        return L2Loss()(x, y)