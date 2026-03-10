"""
model.py
--------
Multimodal Phishing Detection Model.

Architecture
------------
┌─────────────────────────┐    ┌──────────────────────────┐
│   URL Text (tokens)     │    │  Screenshot (RGB image)  │
└────────────┬────────────┘    └────────────┬─────────────┘
             │                              │
     ┌───────▼───────┐            ┌─────────▼──────────┐
     │  TextEncoder  │            │   ImageEncoder     │
     │  (BERT-base)  │            │   (ResNet-50)      │
     │  → 768-d vec  │            │   → 2048-d vec     │
     └───────┬───────┘            └─────────┬──────────┘
             │                              │
             └─────────────┬────────────────┘
                           │  concat → 2816-d
                  ┌────────▼────────┐
                  │  FusionLayer    │
                  │  (BN + Dropout) │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  ClassifierHead │
                  │  (MLP → 2-way)  │
                  └─────────────────┘
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import models

try:
    from configs.config import config
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from configs.config import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Sub-modules                                                                 #
# --------------------------------------------------------------------------- #

class TextEncoder(nn.Module):
    """
    Wraps a pretrained BERT model and returns the [CLS] token embedding.

    The entire BERT backbone is kept trainable so the model can adapt its
    URL-domain language understanding during fine-tuning.  If GPU memory is
    tight, freeze the first N BERT layers via `freeze_layers(n)`.
    """

    def __init__(self, model_name: str = config.TEXT_MODEL_NAME) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.output_dim = self.bert.config.hidden_size  # 768 for bert-base

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, seq_len)
        attention_mask: torch.Tensor,  # (B, seq_len)
    ) -> torch.Tensor:                 # (B, 768)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state: (B, seq_len, hidden)
        # We take the [CLS] token (position 0) as the sentence-level embedding.
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def freeze_layers(self, num_layers: int) -> None:
        """Freeze the first `num_layers` transformer encoder layers."""
        # Always freeze the embedding layer when freezing any layers.
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        encoder_layers = self.bert.encoder.layer
        for i, layer in enumerate(encoder_layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        logger.info("Froze BERT embeddings + first %d encoder layers.", num_layers)


class ImageEncoder(nn.Module):
    """
    Wraps a pretrained ResNet-50 backbone.

    The final fully-connected classification head is replaced with an
    Identity layer so that the 2048-dimensional feature vector is exposed
    directly.  Lower convolutional layers are optionally frozen.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the classification head – keep everything up to avgpool.
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.output_dim = 2048  # ResNet-50 avgpool output channels

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # (B, 3, H, W) → (B, 2048)
        x = self.features(image)      # (B, 2048, 1, 1)
        x = torch.flatten(x, start_dim=1)  # (B, 2048)
        return x

    def freeze_backbone(self, unfreeze_layer4: bool = True) -> None:
        """
        Freeze all ResNet layers except (optionally) layer4.

        Keeping layer4 trainable lets the model adapt high-level visual
        features to the phishing detection domain without full retraining.
        """
        for param in self.features.parameters():
            param.requires_grad = False

        if unfreeze_layer4:
            # layer4 is the 7th child (index 6) of the original ResNet.
            # In our Sequential it lives at index 7 (after layer1-4 + maxpool).
            for param in self.features[7].parameters():  # layer4
                param.requires_grad = True
            logger.info("ResNet frozen except layer4.")
        else:
            logger.info("Entire ResNet backbone frozen.")


class FusionLayer(nn.Module):
    """
    Fuses text and image feature vectors via concatenation followed by
    Batch Normalisation and Dropout for regularisation.
    """

    def __init__(self, fusion_dim: int = config.FUSION_DIM, dropout: float = config.DROPOUT_RATE) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(fusion_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, text_feat: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
        # Concatenate along the feature dimension.
        fused = torch.cat([text_feat, img_feat], dim=1)  # (B, 2816)
        fused = self.bn(fused)
        fused = self.dropout(fused)
        return fused


class ClassifierHead(nn.Module):
    """
    Multi-layer perceptron that maps the fused representation to class logits.

    Topology (default): 2816 → 512 → 256 → 2
    """

    def __init__(
        self,
        input_dim: int = config.FUSION_DIM,
        hidden_dims: Tuple[int, ...] = config.CLASSIFIER_HIDDEN_DIMS,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT_RATE,
    ) -> None:
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            prev_dim = hidden_dim

        # Final projection to class logits (no activation – handled by loss fn).
        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, input_dim) → (B, num_classes)
        return self.mlp(x)


# --------------------------------------------------------------------------- #
#  Top-level model                                                             #
# --------------------------------------------------------------------------- #

class MultimodalPhishingDetector(nn.Module):
    """
    End-to-end multimodal phishing detection model.

    Accepts tokenised URL text AND a webpage screenshot, fuses the feature
    representations from both modalities, and outputs class logits.

    Args:
        text_model_name  : HuggingFace model identifier for the text encoder.
        freeze_bert_layers: Number of initial BERT layers to freeze (0 = none).
        freeze_resnet    : Whether to freeze most ResNet layers.
        unfreeze_layer4  : (Only used when freeze_resnet=True) keep layer4 trainable.
    """

    def __init__(
        self,
        text_model_name: str = config.TEXT_MODEL_NAME,
        freeze_bert_layers: int = 0,
        freeze_resnet: bool = False,
        unfreeze_layer4: bool = True,
    ) -> None:
        super().__init__()

        # --- Encoders ---
        self.text_encoder = TextEncoder(model_name=text_model_name)
        self.image_encoder = ImageEncoder(pretrained=True)

        # Optionally freeze parts of the backbones.
        if freeze_bert_layers > 0:
            self.text_encoder.freeze_layers(freeze_bert_layers)
        if freeze_resnet:
            self.image_encoder.freeze_backbone(unfreeze_layer4=unfreeze_layer4)

        # --- Fusion ---
        fusion_dim = self.text_encoder.output_dim + self.image_encoder.output_dim
        self.fusion = FusionLayer(fusion_dim=fusion_dim, dropout=config.DROPOUT_RATE)

        # --- Classifier ---
        self.classifier = ClassifierHead(
            input_dim=fusion_dim,
            hidden_dims=config.CLASSIFIER_HIDDEN_DIMS,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT_RATE,
        )

        logger.info(
            "MultimodalPhishingDetector initialised – "
            "text_dim=%d | image_dim=%d | fusion_dim=%d",
            self.text_encoder.output_dim,
            self.image_encoder.output_dim,
            fusion_dim,
        )

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, seq_len)
        attention_mask: torch.Tensor,   # (B, seq_len)
        image: torch.Tensor,            # (B, 3, H, W)
    ) -> torch.Tensor:                  # (B, num_classes)

        # Encode each modality independently.
        text_features = self.text_encoder(input_ids, attention_mask)  # (B, 768)
        image_features = self.image_encoder(image)                     # (B, 2048)

        # Fuse and classify.
        fused = self.fusion(text_features, image_features)             # (B, 2816)
        logits = self.classifier(fused)                                # (B, 2)

        return logits

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def count_parameters(self) -> Dict[str, int]:
        """Return trainable / total parameter counts per sub-module."""
        counts = {}
        for name, module in [
            ("text_encoder", self.text_encoder),
            ("image_encoder", self.image_encoder),
            ("fusion", self.fusion),
            ("classifier", self.classifier),
        ]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {"trainable": trainable, "total": total}
        return counts

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convenience method for inference.

        Args:
            batch: Dict with keys 'input_ids', 'attention_mask', 'image'.

        Returns:
            Predicted class indices (B,).
        """
        self.eval()
        with torch.no_grad():
            logits = self(
                batch["input_ids"],
                batch["attention_mask"],
                batch["image"],
            )
        return logits.argmax(dim=1)