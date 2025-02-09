import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def extract(value, time, x_shape):
    device = time.device
    out = torch.gather(value, index=time, dim=0).float().to(device)
    return out.view([time.shape[0]] + [1] * (len(x_shape)-1))  # reshape到x的形状


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        
        self.model = model
        self.T = T
        
        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T).double())
        
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))
        
    def forward(self, x_0):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        
        loss = F.mse_loss(self.model(x_t, t), noise, reduction="none")
        return loss
    
    
    
    
    