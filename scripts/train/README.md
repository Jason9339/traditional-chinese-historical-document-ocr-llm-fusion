# Training Scripts

This directory contains training scripts for both baseline and finetuned TrOCR models.

## Prerequisites

### 1. Prepare Tokenizer

Before training, you need a tokenizer. You can either:
- Download from the baseline model checkpoint (recommended)
- Build your own tokenizer

Place the tokenizer in: `models/tokenizer/`

### 2. (Optional) WandB Setup

To enable Weights & Biases logging:

```bash
export WANDB_API_KEY="your_api_key_here"
export WANDB_PROJECT="your_project_name"
```

If not set, training will proceed without WandB logging.

---

## Baseline Training

Train TrOCR from scratch on 4.1M synthetic dataset.

### Quick Start

```bash
cd scripts/train
python train_baseline.py
```

### Configuration

- **Dataset**: Automatically loads from HuggingFace (`ZihCiLin/traditional-chinese-ocr-synthetic`)
- **Target**: 100k training steps (5 checkpoints at 20k intervals)
- **Output**: `models/baseline_checkpoint/`

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Batch Size (Train) | 64 | Per-device train batch size |
| Batch Size (Eval) | 32 | Per-device eval batch size |
| Learning Rate | 5e-5 | Initial learning rate |
| Epochs | 3 | Number of epochs |
| Save Steps | 20000 | Save checkpoint every N steps |

### Expected Training Time

- **GPU**: ~10-15 hours on A100 (single GPU)
- **Checkpoints**: 5 total (at 20k, 40k, 60k, 80k, 100k steps)

---

## Finetune Training

Finetune baseline checkpoint on Lo Chia-Luen historical manuscripts (690 train / 46 val).

### Quick Start

```bash
cd scripts/train
python train_finetune.py
```

### Configuration

- **Dataset**: Automatically loads from HuggingFace (`ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen`)
- **Base Model**: `models/baseline_checkpoint/` (or specify with `--checkpoint`)
- **Output**: `models/finetune_checkpoint/`

### Command-Line Arguments

```bash
python train_finetune.py \
  --checkpoint models/baseline_checkpoint \
  --tokenizer models/tokenizer \
  --epochs 30 \
  --batch_size_train 8 \
  --batch_size_eval 16 \
  --learning_rate 2e-5
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `models/baseline_checkpoint` | Baseline model checkpoint path |
| `--tokenizer` | `models/tokenizer` | Tokenizer path |
| `--epochs` | 30 | Training epochs |
| `--batch_size_train` | 8 | Train batch size |
| `--batch_size_eval` | 16 | Eval batch size |
| `--learning_rate` | 2e-5 | Learning rate |

### Data Augmentation

- **Epochs 1-20**: Augmented (ColorJitter, RandomAffine, GaussianBlur)
- **Epochs 21-30**: Clean (no augmentation)

This strategy helps the model learn robust features early, then fine-tune on clean data.

### Expected Training Time

- **GPU**: ~30-60 minutes on A100 (single GPU, 30 epochs)
- **Checkpoints**: Saved per epoch, best model selected automatically

---

## Using Pre-trained Models (Recommended)

Instead of training from scratch, you can download our pre-trained models from HuggingFace:

```python
from transformers import VisionEncoderDecoderModel

# Baseline model
baseline = VisionEncoderDecoderModel.from_pretrained("ZihCiLin/trocr-traditional-chinese-baseline")

# Finetuned model
finetuned = VisionEncoderDecoderModel.from_pretrained("ZihCiLin/trocr-traditional-chinese-historical-finetune")
```

---

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```bash
# For baseline
# Edit train_baseline.py: per_device_train_batch_size=32

# For finetune
python train_finetune.py --batch_size_train 4
```

### Dataset Download Issues

If HuggingFace dataset download is slow or fails:
1. Check internet connection
2. Try manual download and use local path
3. Use HuggingFace cache: `~/.cache/huggingface/datasets/`

### Tokenizer Not Found

Ensure tokenizer exists at `models/tokenizer/`. You can:
1. Download from our baseline model
2. Copy from original experiment directory
3. Build your own using `build_tokenizer.py`

---

## Model Output Structure

After training, the checkpoint directory contains:

```
models/baseline_checkpoint/  (or finetune_checkpoint/)
├── config.json              # Model configuration
├── pytorch_model.bin        # Model weights
├── preprocessor_config.json # Image processor config
├── tokenizer_config.json    # Tokenizer config
├── tokenizer.json           # Tokenizer vocabulary
└── training_args.bin        # Training arguments
```

---

## Next Steps

After training:
1. **Evaluate**: Use evaluation scripts in `scripts/evaluate/`
2. **Test**: Run experiments in `experiments/`
3. **Deploy**: Upload to HuggingFace Model Hub (see Phase 2.5 in `claude.md`)
