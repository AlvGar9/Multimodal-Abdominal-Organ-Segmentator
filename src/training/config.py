# src/training/config.py
import torch

# -- General settings --
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4
PIN_MEMORY = True

# -- Data settings --
PREPROCESSED_DATA_DIR = './preprocessed_data/AMOS'

# -- Model settings --
NUM_CLASSES = 5
ORGAN_NAMES = ["Background", "Spleen", "Right Kidney", "Left Kidney", "Liver"]
MODEL_CHANNELS = (16, 32, 64, 128, 256)
MODEL_STRIDES = (2, 2, 2, 2)

# -- Parameters for Training from Scratch --
SCRATCH_MAX_EPOCHS = 200
SCRATCH_BATCH_SIZE = 2
SCRATCH_INITIAL_LR = 1e-2
SCRATCH_WEIGHT_DECAY = 1e-4

# -- Parameters for Fine-Tuning --
FINETUNE_MAX_EPOCHS = 150
PRETRAINED_CT_PATH = './best_model_ct_full.pth' # Path to the trained CT model
FINETUNE_LR_DECODER = 1e-3
FINETUNE_LR_ENCODER = 1e-5
FINETUNE_WEIGHT_DECAY = 1e-5

# -- Parameters for DANN --
DANN_MAX_EPOCHS = 200

# -- General training loop settings --
VAL_INTERVAL = 10
EARLY_STOP_PATIENCE = 5
MIN_IMPROVEMENT_DELTA = 1e-4

# In src/training/config.py

# # -- CHAOS Data settings -
CHAOS_MRI_IMAGES_DIR = './preprocessed_data/CHAOS/MR/images'
CHAOS_MRI_LABELS_DIR = './preprocessed_data/CHAOS/MR/labels'

# Base output directory for all evaluation results
EVALUATION_OUTPUT_DIR = './evaluation_results'

# Dictionary mapping model names to their file paths
# Assumes a 'models/' directory in the project root holds all .pth files
MODELS_TO_EVALUATE = {
    # Baselines (from train.py)
    "UNet_CT_Baseline": "models/best_model_ct_full.pth",
    "UNet_MR_5_samples": "models/best_model_mr_5_samples.pth",
    "UNet_MR_15_samples": "models/best_model_mr_15_samples.pth",
    "UNet_MR_30_samples": "models/best_model_mr_30_samples.pth",
    "UNet_MR_all_samples": "models/best_model_mr_all_samples.pth",
    
    # Fine-Tuning (from finetune.py)
    "FineTune_MR_5_samples": "models/finetuned_model_mr_5_samples.pth",
    "FineTune_MR_15_samples": "models/finetuned_model_mr_15_samples.pth",
    "FineTune_MR_30_samples": "models/finetuned_model_mr_30_samples.pth",
    "FineTune_MR_all_samples": "models/finetuned_model_mr_full_samples.pth",

    # DANN (from train_dann.py)
    "DANN_MR_5_samples": "models/dann_model_5_mri.pth",
    "DANN_MR_15_samples": "models/dann_model_15_mri.pth",
    "DANN_MR_30_samples": "models/dann_model_30_mri.pth",
    "DANN_MR_all_samples": "models/dann_model_all_mri.pth",
    
    # UDA (from train_uda.py)
    "UDA_MR_5_samples": "models/uda_model_5_mri.pth",
    "UDA_MR_15_samples": "models/uda_model_15_mri.pth",
    "UDA_MR_30_samples": "models/uda_model_30_mri.pth",
    "UDA_MR_54_samples": "models/uda_model_54_mri.pth",
}