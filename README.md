# Multimodal Phishing Website Detection System

A production-quality PyTorch system that combines **NLP (BERT)** and **Computer Vision (ResNet-50)** to classify websites as phishing or legitimate using URL text and webpage screenshots.

---

## Project Structure

```
phishing_detector/
├── configs/
│   └── config.py          # All hyperparameters & paths – edit here first
├── src/
│   ├── dataset.py          # Dataset loader + DataLoader factory
│   ├── model.py            # Multimodal neural network
│   ├── train.py            # Training + evaluation pipeline
│   └── utils.py            # Metrics, checkpointing, early stopping
└── requirements.txt
```

---

## Dataset Layout

```
dataset/trainval/
    L0001_legitimate/
        URL/url.txt
        SCREEN-SHOT/screen_shoot.png
        Label/label.txt          ← contains "legitimate"
    P0001_phishing/
        URL/url.txt
        SCREEN-SHOT/screen_shoot.png
        Label/label.txt          ← contains "phishing"
```

---

## Quick Start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Configure paths

Edit `configs/config.py`:

```python
DATASET_DIR = "/path/to/your/dataset/trainval"
```

### 3 · Train

```bash
# From the project root:
python src/train.py

# With optional flags:
python src/train.py \
    --freeze-bert-layers 6 \   # freeze first 6 BERT layers (saves VRAM)
    --freeze-resnet \          # freeze most ResNet layers
    --patience 5               # early stopping patience
```

### 4 · Resume from checkpoint

```bash
python src/train.py --resume checkpoints/best_model.pt
```

---

## Model Architecture

```
URL text ──► TextEncoder (BERT-base)  ──► 768-d
                                                 \
                                                  ├──► FusionLayer ──► ClassifierHead ──► logits
                                                 /     (BN+Drop)       (MLP 2816→512→256→2)
Screenshot ──► ImageEncoder (ResNet50) ──► 2048-d
```

| Component       | Backbone       | Output dim |
|-----------------|----------------|------------|
| TextEncoder     | bert-base-uncased | 768      |
| ImageEncoder    | ResNet-50      | 2048       |
| FusionLayer     | Concat + BN    | 2816       |
| ClassifierHead  | MLP            | 2          |

---

## Key Configuration Options (`configs/config.py`)

| Parameter             | Default            | Description                        |
|-----------------------|--------------------|------------------------------------|
| `TEXT_MODEL_NAME`     | bert-base-uncased  | Any HuggingFace transformer        |
| `MAX_URL_LENGTH`      | 128                | Token limit for URLs               |
| `IMAGE_SIZE`          | 224                | Screenshot resize target           |
| `BATCH_SIZE`          | 16                 | Reduce if GPU OOM                  |
| `NUM_EPOCHS`          | 10                 | Training epochs                    |
| `LEARNING_RATE`       | 2e-5               | Applied to BERT; ×10 for the rest  |
| `CLASSIFIER_HIDDEN_DIMS` | (512, 256)      | MLP hidden layer sizes             |
| `DROPOUT_RATE`        | 0.3                | Dropout in fusion + classifier     |
| `VAL_SPLIT`           | 0.2                | Fraction held out for validation   |

---

## Google Colab Tips

```python
# Check GPU
import torch; print(torch.cuda.get_device_name(0))

# Install deps
!pip install -r requirements.txt -q

# Mount Drive (if dataset lives there)
from google.colab import drive
drive.mount('/content/drive')

# Update config
from configs.config import config
config.DATASET_DIR = "/content/drive/MyDrive/dataset/trainval"
config.BATCH_SIZE  = 8   # reduce for smaller GPUs

# Train
%run src/train.py
```