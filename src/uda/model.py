# src/uda/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet

class MultiLevelUDANet(nn.Module):
    """
    A UNet wrapper that extracts multi-level encoder features using hooks.
    These features are then passed through projector heads for alignment.
    """
    def __init__(self, pretrained_unet: UNet, feature_dims: int = 32):
        super().__init__()
        self.unet = pretrained_unet
        self._hook_features = {}

        def get_hook(name):
            def hook(model, input, output):
                self._hook_features[name] = output
            return hook

        # These paths correspond to the encoder blocks in the MONAI UNet implementation.
        # This structure is specific to the MONAI version used.
        encoder_blocks = [
            self.unet.model[0],                                                   # Level 1
            self.unet.model[1].submodule[0],                                      # Level 2
            self.unet.model[1].submodule[1].submodule[0],                         # Level 3
            self.unet.model[1].submodule[1].submodule[1].submodule[0],            # Level 4
            self.unet.model[1].submodule[1].submodule[1].submodule[1].submodule   # Level 5 (Bottleneck)
        ]
        
        in_channels = [16, 32, 64, 128, 256]
        
        self.projectors = nn.ModuleList()
        for i, (block, channels) in enumerate(zip(encoder_blocks, in_channels)):
            level_name = f'level_{i+1}'
            block.register_forward_hook(get_hook(level_name))
            
            projector = nn.Sequential(
                nn.Conv3d(channels, feature_dims, kernel_size=1),
                nn.ReLU(),
                nn.Conv3d(feature_dims, feature_dims, kernel_size=1)
            )
            self.projectors.append(projector)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for inference."""
        return self.unet(x)

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass that returns both the final segmentation logits and a list
        of projected intermediate features for use in the alignment loss.
        """
        seg_output = self.unet(x)
        
        projected_features = []
        try:
            for i in range(len(self.projectors)):
                level_name = f'level_{i+1}'
                if level_name not in self._hook_features:
                    raise RuntimeError(f"Hook for {level_name} did not capture features.")
                
                features = self._hook_features[level_name]
                projected = self.projectors[i](features)
                pooled = F.adaptive_avg_pool3d(projected, 1).flatten(1)
                projected_features.append(pooled)
        finally:
            self._hook_features.clear() # Ensure hooks are cleared after each pass
            
        return seg_output, projected_features