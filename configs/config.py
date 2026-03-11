"""
config.py
---------
Central configuration for the Multimodal Phishing Detection System.
Modify these values to adjust model behaviour, training hyperparameters,
and file-system paths without touching any other source file.
"""

import os
import torch


class Config:
    # ------------------------------------------------------------------ #
    #  Paths                                                               #
    # ------------------------------------------------------------------ #
    ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Dataset root – override with an absolute path if needed.
    DATASET_DIR: str = os.path.join(ROOT_DIR, "dataset", "trainval")

    # Where to save / load model checkpoints.
    CHECKPOINT_DIR: str = os.path.join(ROOT_DIR, "checkpoints")

    

    # ------------------------------------------------------------------ #
    #  Dataset                                                             #
    # ------------------------------------------------------------------ #
    # Fraction of the dataset used for validation (rest goes to training).
    VAL_SPLIT: float = 0.2

    # Fixed seed for reproducible train/val splits.
    RANDOM_SEED: int = 42

    # Sub-folder names inside each sample directory (must match the dataset).
    URL_SUBDIR: str = "URL"
    URL_FILENAME: str = "url.txt"

    SCREENSHOT_SUBDIR: str = "SCREEN-SHOT"
    SCREENSHOT_FILENAME: str = "screen_shoot.png"

    LABEL_SUBDIR: str = "Label"
    LABEL_FILENAME: str = "label.txt"

    # String values expected inside label.txt (case-insensitive).
    PHISHING_LABEL: str = "phishing"      # → class 1
    LEGITIMATE_LABEL: str = "legitimate"  # → class 0

    # ------------------------------------------------------------------ #
    #  Text Encoder (BERT)                                                 #
    # ------------------------------------------------------------------ #
    TEXT_MODEL_NAME: str = "bert-base-uncased"

    # Maximum token length fed to BERT (URLs are typically short).
    MAX_URL_LENGTH: int = 128

    # Dimensionality of the [CLS] embedding produced by BERT.
    TEXT_FEATURE_DIM: int = 768

    # ------------------------------------------------------------------ #
    #  Image Encoder (ResNet-50)                                           #
    # ------------------------------------------------------------------ #
    # Input image dimensions expected by the CNN.
    IMAGE_SIZE: int = 224

    # ImageNet normalisation statistics used by torchvision pretrained models.
    IMAGE_MEAN: tuple = (0.485, 0.456, 0.406)
    IMAGE_STD: tuple = (0.229, 0.224, 0.225)

    # Dimensionality of the feature vector extracted by ResNet-50's avgpool.
    IMAGE_FEATURE_DIM: int = 2048

    # ------------------------------------------------------------------ #
    #  Fusion & Classifier                                                 #
    # ------------------------------------------------------------------ #
    # Combined dimension after concatenating text + image features.
    FUSION_DIM: int = TEXT_FEATURE_DIM + IMAGE_FEATURE_DIM  # 2816

    # Hidden layer sizes in the MLP classifier head.
    CLASSIFIER_HIDDEN_DIMS: tuple = (512, 256)

    # Dropout probability applied in the classifier.
    DROPOUT_RATE: float = 0.3

    # Number of output classes (phishing=1, legitimate=0).
    NUM_CLASSES: int = 2

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 10
    LEARNING_RATE: float = 2e-5       # Warm LR suitable for fine-tuning BERT.
    WEIGHT_DECAY: float = 1e-2

    # Gradient clipping to stabilise transformer training.
    MAX_GRAD_NORM: float = 1.0

    # Number of DataLoader worker processes (set 0 for debugging).
    NUM_WORKERS: int = 4

    # ------------------------------------------------------------------ #
    #  Hardware                                                            #
    # ------------------------------------------------------------------ #
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #
    # How many batches between console log lines.
    LOG_INTERVAL: int = 10


# Singleton instance – import this object everywhere else.
config = Config()