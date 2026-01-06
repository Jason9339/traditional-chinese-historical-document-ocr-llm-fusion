#!/usr/bin/env python3
"""
Small-scale test script - Finetune TrOCR Training
Quick validation of finetune flow (only 2 epochs)
"""

import os
import sys

# Set environment variables
os.environ["WANDB_MODE"] = "disabled"

import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderModel,
    PreTrainedTokenizerFast,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from dataclasses import dataclass
from typing import Any, Dict, List

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from gfd.image_preprocessing import OCRImageTransform
from huggingface_hub import hf_hub_download

print("=" * 80)
print("🧪 FINETUNE TRAINING TEST - Small Scale")
print("=" * 80)
print("Configuration:")
print("  - Epochs: 2 (instead of 30)")
print("  - Full Lo Chia-Luen dataset (690 train / 46 val)")
print("  - Batch size: 8")
print("  - No data augmentation (for speed)")
print("=" * 80)

set_seed(42)

# Output directory
output_dir = os.path.join(PROJECT_ROOT, "models", "test_finetune_checkpoint")
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# CER Calculation
# ============================================================
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def cer_chinese(references: List[str], predictions: List[str]) -> float:
    total_dist = 0
    total_chars = 0
    for ref, pred in zip(references, predictions):
        ref_str = str(ref)
        pred_str = str(pred)
        dist = levenshtein_distance(ref_str, pred_str)
        total_dist += dist
        total_chars += len(ref_str)
    return total_dist / total_chars if total_chars > 0 else 0.0

_tokenizer_for_metrics = None

def compute_metrics(pred):
    global _tokenizer_for_metrics
    import numpy as np

    predictions_ids = pred.predictions
    if isinstance(predictions_ids, tuple):
        predictions_ids = predictions_ids[0]

    vocab_size = _tokenizer_for_metrics.vocab_size
    unk_token_id = _tokenizer_for_metrics.unk_token_id

    is_valid = (predictions_ids >= 0) & (predictions_ids < vocab_size)
    safe_predictions_ids = np.where(is_valid, predictions_ids, unk_token_id)

    decoded_preds = _tokenizer_for_metrics.batch_decode(
        safe_predictions_ids.tolist(), skip_special_tokens=True
    )

    labels = pred.label_ids.copy()
    labels[labels == -100] = _tokenizer_for_metrics.pad_token_id
    decoded_labels = _tokenizer_for_metrics.batch_decode(
        labels.tolist(), skip_special_tokens=True
    )

    cer_score = cer_chinese(decoded_labels, decoded_preds)
    return {"cer": cer_score}

# ============================================================
# Data Collator
# ============================================================
@dataclass
class DataCollatorForOCR:
    processor: TrOCRProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        label_input_ids = [f["labels"] for f in features]

        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_input_ids},
            padding="longest",
            return_attention_mask=True,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"]
        labels = labels.masked_fill(labels_batch.attention_mask.ne(1), -100)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

# ============================================================
# Dataset Wrapper
# ============================================================
class HFHistoricalDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, transform=None, max_label_length=128, repo_id=None):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_label_length = max_label_length
        self.repo_id = repo_id

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        text = sample['text']
        crop_path = sample['crop_path']

        if self.repo_id:
            image_file = hf_hub_download(
                repo_id=self.repo_id,
                filename=crop_path,
                repo_type='dataset'
            )
            image = Image.open(image_file)
        else:
            image = Image.open(crop_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_str = text + self.tokenizer.eos_token
        tokenized_output = self.tokenizer(
            label_str,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_label_length,
        )
        labels_input_ids = tokenized_output.input_ids[0].tolist()

        return {
            "pixel_values": image,
            "labels": labels_input_ids,
        }

# ============================================================
# Main
# ============================================================
def main():
    global _tokenizer_for_metrics

    # Load Tokenizer
    tokenizer_path = os.path.join(PROJECT_ROOT, "models", "tokenizer")
    print(f"\n✓ Loading tokenizer from: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.pad_token or "<pad>"
    tokenizer.bos_token = tokenizer.bos_token or "<s>"
    tokenizer.eos_token = tokenizer.eos_token or "</s>"
    tokenizer.unk_token = tokenizer.unk_token or "<unk>"
    _tokenizer_for_metrics = tokenizer

    print("✓ Loading TrOCR processor...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
    processor.tokenizer = tokenizer

    # Image Transform (NO AUGMENTATION FOR SPEED)
    transform = OCRImageTransform(
        target_size=(384, 384),
        auto_rotate_vertical=True,
        normalize=True,
    )

    # Load Model from HuggingFace (use uploaded baseline)
    print("\n✓ Loading baseline model from HuggingFace...")
    print("  Model: ZihCiLin/trocr-traditional-chinese-baseline")

    try:
        model = VisionEncoderDecoderModel.from_pretrained(
            "ZihCiLin/trocr-traditional-chinese-baseline"
        )
        print("  ✓ Loaded from HuggingFace")
    except:
        print("  ⚠ HuggingFace load failed, trying local checkpoint...")
        local_checkpoint = os.environ.get('LOCAL_BASELINE_CHECKPOINT', './checkpoint-100000')
        if os.path.exists(local_checkpoint):
            model = VisionEncoderDecoderModel.from_pretrained(local_checkpoint)
            print(f"  ✓ Loaded from local: {local_checkpoint}")
        else:
            print("  ❌ No baseline model found!")
            return 1

    model.config.decoder_start_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model.to(device)

    # Load Historical Dataset from HuggingFace
    print("\n✓ Loading historical dataset from HuggingFace...")
    dataset_repo = "ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen"
    print(f"  Dataset: {dataset_repo}")

    dataset = load_dataset(dataset_repo)
    train_hf = dataset['train']
    val_hf = dataset['validation']

    print(f"  Train samples: {len(train_hf)}")
    print(f"  Val samples: {len(val_hf)}")

    train_dataset = HFHistoricalDataset(
        train_hf,
        tokenizer,
        transform,
        repo_id=dataset_repo
    )
    val_dataset = HFHistoricalDataset(
        val_hf,
        tokenizer,
        transform,
        repo_id=dataset_repo
    )
    data_collator = DataCollatorForOCR(processor=processor)

    # Training Arguments (SMALL SCALE)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=2,  # Only 2 epochs (instead of 30)

        logging_strategy="steps",
        logging_steps=10,

        save_strategy="epoch",
        save_total_limit=2,

        evaluation_strategy="epoch",

        learning_rate=2e-5,
        warmup_ratio=0.05,

        predict_with_generate=True,
        generation_max_length=128,
        fp16=torch.cuda.is_available(),

        report_to="none",
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n" + "=" * 80)
    print("🚀 Starting test finetuning (2 epochs)...")
    print("=" * 80)

    try:
        train_result = trainer.train()

        # Evaluate
        print("\n✓ Evaluating...")
        metrics = trainer.evaluate(eval_dataset=val_dataset)

        print("\n" + "=" * 80)
        print("✅ Test finetuning completed successfully!")
        print("=" * 80)
        print(f"\nCheckpoint saved to: {output_dir}")
        print(f"Final CER: {metrics['eval_cer']:.4f}")
        print("\n✓ Finetune training script works correctly!")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
