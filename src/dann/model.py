# src/dann/model.py
import torch
import torch.nn as nn
from typing import Tuple
from monai.networks.nets import UNet

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) for DANN.
    During the forward pass, it's an identity function. During the backward
    pass, it reverses the gradient by multiplying it with a negative constant.
    """
    @staticmethod
    def forward(ctx, x, lambda_p):
        ctx.lambda_p = lambda_p
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_p
        return output, None

def grad_reverse(x, lambda_p=1.0):
    return GradientReversalFunction.apply(x, lambda_p)

class DANNUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 5,
        channels: Tuple = (16, 32, 64, 128, 256),
        strides: Tuple = (2, 2, 2, 2),
        num_res_units: int = 2,
        domain_hidden_dim: int = 512,
    ):
        super().__init__()
        
        self.unet = UNet(
            spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
            channels=channels, strides=strides, num_res_units=num_res_units,
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(out_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, domain_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(domain_hidden_dim, domain_hidden_dim // 2), 
            nn.ReLU(),                                           
            nn.Dropout(0.5),                                     
            nn.Linear(domain_hidden_dim // 2, 2)
        )
    
    def forward(self, x, lambda_p=1.0, return_domain_logits=False):
        seg_logits = self.unet(x)
        
        if not return_domain_logits:
            return seg_logits
            
        features = self.feature_extractor(seg_logits.detach()) 
        
        features_reversed = grad_reverse(features, lambda_p)
        domain_logits = self.domain_classifier(features_reversed)
        
        return seg_logits, domain_logits

def load_unet_weights(dann_model: DANNUNet, path: str, device: torch.device):
    """
    Loads weights from a standard UNet checkpoint into the `unet` submodule
    of the DANNUNet model.
    """
    try:
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Create a new state dict with the 'unet.' prefix for matching
        unet_state_dict = {}
        for key, value in state_dict.items():
            unet_state_dict[f"unet.{key}"] = value

        dann_model.load_state_dict(unet_state_dict, strict=False)
        print(f"Successfully loaded UNet weights from: {path}")
    except FileNotFoundError:
        print(f"Warning: Pre-trained model not found at '{path}'. UNet will be trained from scratch.")
    except Exception as e:
        print(f"Error loading UNet weights: {e}. UNet will be trained from scratch.")