# src/training/model.py
import torch
from monai.networks.nets import UNet

def build_model(config: object) -> UNet:
    """
    Builds and returns a 3D UNet model based on the configuration.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=config.NUM_CLASSES,
        channels=config.MODEL_CHANNELS,
        strides=config.MODEL_STRIDES,
        num_res_units=2
    )
    return model

def load_pretrained_weights(model: UNet, path: str, device: torch.device):
    """
    Loads pre-trained weights into a model, handling different checkpoint formats.
    """
    try:
        checkpoint = torch.load(path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print(f"Successfully loaded pre-trained weights from: {path}")
    except FileNotFoundError:
        print(f"Warning: Pre-trained model not found at '{path}'. The model will be trained from scratch.")
    except Exception as e:
        print(f"Error loading weights: {e}. The model will be trained from scratch.")