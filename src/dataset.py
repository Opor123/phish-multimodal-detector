"""
dataset.py
----------
PyTorch Dataset and DataLoader factory for the Multimodal Phishing Detection
dataset.

Expected on-disk layout (one folder per sample):

    dataset/trainval/
        L0001_legitimate/
            URL/url.txt
            SCREEN-SHOT/screen_shoot.png
            Label/label.txt
        P0001_phishing/
            URL/url.txt
            SCREEN-SHOT/screen_shoot.png
            Label/label.txt
        ...
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms

# Allow importing config whether this file is run standalone or as a module.
try:
    from configs.config import config
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from configs.config import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Image pre-processing pipeline                                               #
# --------------------------------------------------------------------------- #

def build_image_transform(image_size: int, mean: tuple, std: tuple) -> transforms.Compose:
    """Return a torchvision transform for inference/validation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_train_image_transform(image_size: int, mean: tuple, std: tuple) -> transforms.Compose:
    """Return an augmented torchvision transform for training."""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# --------------------------------------------------------------------------- #
#  Core Dataset class                                                          #
# --------------------------------------------------------------------------- #

class PhishingDataset(Dataset):
    """
    Loads URL text, webpage screenshot, and binary phishing/legitimate label
    for each sample folder found under `root_dir`.

    Args:
        root_dir   : Path to the folder that contains per-sample sub-folders.
        tokenizer  : HuggingFace tokenizer for URL text.
        max_length : Maximum token length for the tokenizer.
        transform  : torchvision transform applied to the screenshot.
    """

    def __init__(
        self,
        root_dir: str,
        tokenizer: AutoTokenizer,
        max_length: int = config.MAX_URL_LENGTH,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform or build_image_transform(
            config.IMAGE_SIZE, config.IMAGE_MEAN, config.IMAGE_STD
        )

        # Collect all valid sample paths.
        self.samples: List[Dict] = self._scan_dataset()
        logger.info("Loaded %d samples from %s", len(self.samples), self.root_dir)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _scan_dataset(self) -> List[Dict]:
        """Walk root_dir and collect paths for every valid sample."""
        samples = []
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        for sample_dir in sorted(self.root_dir.iterdir()):
            if not sample_dir.is_dir():
                continue

            url_path = sample_dir / config.URL_SUBDIR / config.URL_FILENAME
            img_path = sample_dir / config.SCREENSHOT_SUBDIR / config.SCREENSHOT_FILENAME
            lbl_path = sample_dir / config.LABEL_SUBDIR / config.LABEL_FILENAME

            # Skip incomplete samples but warn so the user can investigate.
            missing = [p for p in (url_path, img_path, lbl_path) if not p.exists()]
            if missing:
                logger.warning("Skipping %s – missing files: %s", sample_dir.name, missing)
                continue

            label = self._parse_label(lbl_path)
            if label is None:
                logger.warning("Skipping %s – unrecognised label.", sample_dir.name)
                continue

            samples.append({
                "sample_id": sample_dir.name,
                "url_path": url_path,
                "img_path": img_path,
                "label": label,
            })

        if not samples:
            raise RuntimeError(f"No valid samples found in {self.root_dir}.")

        return samples

    @staticmethod
    def _parse_label(label_path: Path) -> Optional[int]:
        """Read label.txt and return 1 for phishing, 0 for legitimate, None on error."""
        text = label_path.read_text(encoding="utf-8").strip().lower()
        if config.PHISHING_LABEL in text:
            return 1
        if config.LEGITIMATE_LABEL in text:
            return 0
        return None

    @staticmethod
    def _load_url(url_path: Path) -> str:
        """Read url.txt and return the URL string, stripped of whitespace."""
        return url_path.read_text(encoding="utf-8").strip()

    def _load_image(self, img_path: Path) -> torch.Tensor:
        """Open a screenshot, convert to RGB, and apply the transform pipeline."""
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.error("Cannot open image %s: %s – using blank image.", img_path, exc)
            image = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE))
        return self.transform(image)

    def _tokenize_url(self, url: str) -> Dict[str, torch.Tensor]:
        """Tokenise a URL string using the HuggingFace tokenizer."""
        encoding = self.tokenizer(
            url,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Squeeze the batch dimension added by return_tensors="pt".
        return {k: v.squeeze(0) for k, v in encoding.items()}

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        url = self._load_url(sample["url_path"])
        image = self._load_image(sample["img_path"])
        token_dict = self._tokenize_url(url)

        return {
            "input_ids": token_dict["input_ids"],           # (max_length,)
            "attention_mask": token_dict["attention_mask"], # (max_length,)
            "image": image,                                 # (3, H, W)
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "sample_id": sample["sample_id"],               # useful for debugging
        }

    def get_label_distribution(self) -> Dict[str, int]:
        """Return a count of phishing vs legitimate samples."""
        counts: Dict[str, int] = {"legitimate": 0, "phishing": 0}
        for s in self.samples:
            key = "phishing" if s["label"] == 1 else "legitimate"
            counts[key] += 1
        return counts


# --------------------------------------------------------------------------- #
#  DataLoader factory                                                          #
# --------------------------------------------------------------------------- #

def build_dataloaders(
    root_dir: str = config.DATASET_DIR,
    val_split: float = config.VAL_SPLIT,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from a single dataset directory.

    Steps
    -----
    1. Instantiate PhishingDataset with augmented transform for training.
    2. Split into train / val subsets using a fixed random seed.
    3. Re-wrap the val subset with a plain (non-augmenting) transform.
    4. Return (train_loader, val_loader).

    Returns
    -------
    train_loader, val_loader
    """
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    # Use augmented transforms for the full dataset scan first.
    train_transform = build_train_image_transform(
        config.IMAGE_SIZE, config.IMAGE_MEAN, config.IMAGE_STD
    )
    val_transform = build_image_transform(
        config.IMAGE_SIZE, config.IMAGE_MEAN, config.IMAGE_STD
    )

    # Full dataset with training augmentation.
    full_dataset = PhishingDataset(
        root_dir=root_dir,
        tokenizer=tokenizer,
        transform=train_transform,
    )

    # Deterministic split.
    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Override the transform for the validation subset.
    # We do this by wrapping the subset in a lightweight adapter.
    val_subset.dataset = _TransformOverrideDataset(full_dataset, val_transform)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "DataLoaders ready – train: %d batches | val: %d batches",
        len(train_loader),
        len(val_loader),
    )
    return train_loader, val_loader


class _TransformOverrideDataset(Dataset):
    """
    Thin wrapper around PhishingDataset that swaps out the image transform.
    Used internally to apply non-augmenting transforms on the validation split.
    """

    def __init__(self, base_dataset: PhishingDataset, transform: transforms.Compose) -> None:
        self.base = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Temporarily swap the transform, fetch the item, then restore.
        original_transform = self.base.transform
        self.base.transform = self.transform
        item = self.base[idx]
        self.base.transform = original_transform
        return item