# src/training/optimizer.py
import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR
from monai.networks.nets import UNet

def get_finetune_optimizer_scheduler(model: UNet, config: object):
    """
    Sets up an AdamW optimizer with differential learning rates for the encoder
    and decoder, along with a CosineAnnealingLR scheduler.

    Decoder layers are identified as any module containing a ConvTranspose3d layer,
    plus the final output convolution.
    """
    decoder_param_ids = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ConvTranspose3d):
            for p in module.parameters():
                decoder_param_ids.add(id(p))

    # The final output convolution is also part of the decoder path
    final_conv = model.model[-1]
    for p in final_conv.parameters():
        decoder_param_ids.add(id(p))

    encoder_params = [p for p in model.parameters() if id(p) not in decoder_param_ids]
    decoder_params = [p for p in model.parameters() if id(p) in decoder_param_ids]
    
    optimizer = AdamW([
        {'params': encoder_params, 'lr': config.FINETUNE_LR_ENCODER},
        {'params': decoder_params, 'lr': config.FINETUNE_LR_DECODER}
    ], weight_decay=config.FINETUNE_WEIGHT_DECAY)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config.FINETUNE_MAX_EPOCHS, eta_min=1e-7)
    
    return optimizer, scheduler

def get_scratch_optimizer_scheduler(model: UNet, config: object):
    """
    Sets up a standard SGD optimizer and PolynomialLR scheduler for training from scratch.
    """
    optimizer = SGD(
        model.parameters(), lr=config.SCRATCH_INITIAL_LR, momentum=0.99,
        weight_decay=config.SCRATCH_WEIGHT_DECAY, nesterov=True
    )
    scheduler = PolynomialLR(optimizer, total_iters=config.SCRATCH_MAX_EPOCHS, power=0.9)
    return optimizer, scheduler