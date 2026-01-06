#!/usr/bin/env python3
"""
Baseline TrOCR Training Script
Train TrOCR from scratch on 4.1M synthetic Traditional Chinese OCR dataset
Target: 100k training steps
"""

import os
import sys
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderModel,
    PreTrainedTokenizerFast,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
import wandb
from dataclasses import dataclass
from typing import Any, Dict, List

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from gfd.image_preprocessing import OCRImageTransform

# ============================================================
# Configuration
# ============================================================
set_seed(42)

# Output directory
output_dir = os.path.join(PROJECT_ROOT, "models", "baseline_checkpoint")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/logs", exist_ok=True)

# ============================================================
# WandB Setup (Optional - User should set their own API key)
# ============================================================
# To use WandB, set these environment variables:
# export WANDB_API_KEY="your_api_key"
# export WANDB_PROJECT="TrOCR-Chinese-Baseline"
if "WANDB_API_KEY" in os.environ:
    try:
        wandb.login()
        print("✓ WandB initialized")
    except Exception as e:
        print(f"⚠ WandB login failed: {e}")
        print("  Continuing without WandB logging")
else:
    print("ℹ WandB not configured. To enable, set WANDB_API_KEY environment variable.")
    os.environ["WANDB_MODE"] = "disabled"

# ============================================================
# CER Calculation
# ============================================================
def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
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
    """Calculate Character Error Rate for Chinese text"""
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
    """Compute CER metric during evaluation"""
    global _tokenizer_for_metrics
    if _tokenizer_for_metrics is None:
        raise ValueError("Tokenizer for metrics is not set.")

    import numpy as np

    try:
        # Process predictions
        predictions_ids = pred.predictions
        if isinstance(predictions_ids, tuple):
            predictions_ids = predictions_ids[0]

        # Filter invalid token IDs
        vocab_size = _tokenizer_for_metrics.vocab_size
        unk_token_id = _tokenizer_for_metrics.unk_token_id
        is_valid = (predictions_ids >= 0) & (predictions_ids < vocab_size)
        safe_predictions_ids = np.where(is_valid, predictions_ids, unk_token_id)

        # Decode predictions
        decoded_preds = _tokenizer_for_metrics.batch_decode(
            safe_predictions_ids.tolist(), skip_special_tokens=True
        )

        # Process labels
        labels = pred.label_ids.copy()
        labels[labels == -100] = _tokenizer_for_metrics.pad_token_id
        decoded_labels = _tokenizer_for_metrics.batch_decode(
            labels.tolist(), skip_special_tokens=True
        )

        # Calculate CER
        cer_score = cer_chinese(decoded_labels, decoded_preds)
        return {"cer": cer_score}

    except Exception as e:
        print(f"\n⚠ Exception in compute_metrics: {e}")
        import traceback
        traceback.print_exc()
        raise e


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
# HuggingFace Dataset Wrapper
# ============================================================
class HFOCRDataset(Dataset):
    """
    Wrapper for HuggingFace OCR dataset
    Loads from: ZihCiLin/traditional-chinese-ocr-synthetic
    """

    def __init__(self, hf_dataset, tokenizer, transform=None, max_label_length=128):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = sample['image']  # PIL Image
        text = sample['text']     # String

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transform
        if self.transform:
            image = self.transform(image)

        # Tokenize label
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
# Main Training Function
# ============================================================
def main():
    print("=" * 80)
    print("Baseline TrOCR Training - 4.1M Synthetic Dataset")
    print("=" * 80)

    # ============================================================
    # Load Tokenizer
    # ============================================================
    # Note: User should prepare tokenizer in advance
    # For now, we'll use a placeholder path
    tokenizer_path = os.path.join(PROJECT_ROOT, "models", "tokenizer")
    
    if not os.path.exists(tokenizer_path):
        print(f"\n⚠ Tokenizer not found at: {tokenizer_path}")
        print("  Please prepare the tokenizer first.")
        print("  You can download it from the baseline model or train it separately.")
        return

    print(f"\n Loading tokenizer from: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.pad_token or "<pad>"
    tokenizer.bos_token = tokenizer.bos_token or "<s>"
    tokenizer.eos_token = tokenizer.eos_token or "</s>"
    tokenizer.unk_token = tokenizer.unk_token or "<unk>"

    global _tokenizer_for_metrics
    _tokenizer_for_metrics = tokenizer

    print("✓ Loading TrOCR processor...")
    try:
        processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-stage1", tokenizer=tokenizer
        )
    except:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
        processor.tokenizer = tokenizer
    print("✓ Processor loaded")

    # ============================================================
    # Image Transform
    # ============================================================
    IMG_HEIGHT = 384
    IMG_WIDTH = 384
    transform = OCRImageTransform(
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        auto_rotate_vertical=True,
        normalize=True,
    )

    # ============================================================
    # Load Model
    # ============================================================
    print("\n✓ Loading pre-trained TrOCR model...")
    model_name = "microsoft/trocr-base-stage1"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Resize token embeddings
    print(f"  Resizing embeddings: {model.decoder.config.vocab_size} → {len(tokenizer)}")
    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.decoder.vocab_size = len(tokenizer)

    # Configure special tokens
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Moving model to: {device}")
    model.to(device)

    # ============================================================
    # Load Dataset from HuggingFace
    # ============================================================
    print("\n✓ Loading dataset from HuggingFace...")
    print("  Dataset: ZihCiLin/traditional-chinese-ocr-synthetic")
    
    dataset = load_dataset("ZihCiLin/traditional-chinese-ocr-synthetic")
    
    print(f"\n  Available splits: {list(dataset.keys())}")
    
    # Use train split
    full_dataset = dataset['train']
    print(f"  Total samples: {len(full_dataset):,}")

    # Split into train/val (95/5)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    print(f"\n  Dataset split:")
    print(f"    Training:   {train_size:,} samples (95%)")
    print(f"    Validation: {val_size:,} samples (5%)")

    # Create train/val splits
    train_hf = full_dataset.select(range(train_size))
    val_hf = full_dataset.select(range(train_size, len(full_dataset)))

    # Wrap in custom dataset
    train_dataset = HFOCRDataset(
        hf_dataset=train_hf,
        tokenizer=tokenizer,
        transform=transform,
    )

    val_dataset = HFOCRDataset(
        hf_dataset=val_hf,
        tokenizer=tokenizer,
        transform=transform,
    )

    data_collator = DataCollatorForOCR(processor=processor)

    # ============================================================
    # Training Arguments
    # ============================================================
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=3,  # Adjust based on dataset size

        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=2000,

        save_strategy="steps",
        save_steps=20000,  # Save every 20k steps (target: 100k steps total)
        save_total_limit=3,

        evaluation_strategy="steps",
        eval_steps=20000,

        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,

        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",

        dataloader_num_workers=12,
        dataloader_pin_memory=True,

        remove_unused_columns=False,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,

        fp16=torch.cuda.is_available(),

        report_to="wandb" if "WANDB_API_KEY" in os.environ else "none",
        run_name="trocr-baseline-100k",
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=20,
        early_stopping_threshold=0.0005,
    )

    # ============================================================
    # Initialize Trainer
    # ============================================================
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback] if training_args.load_best_model_at_end else [],
    )

    # ============================================================
    # Start Training
    # ============================================================
    print("\n" + "=" * 80)
    print("Starting training...")
    print("Target: 100k steps (5 checkpoints at 20k intervals)")
    print("=" * 80 + "\n")

    try:
        train_result = trainer.train()

        # Save final model
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        # Evaluate best model
        if training_args.load_best_model_at_end:
            print("\n✓ Evaluating best model on validation set...")
            metrics = trainer.evaluate(eval_dataset=val_dataset)
            trainer.log_metrics("eval_best", metrics)
            trainer.save_metrics("eval_best", metrics)
            print(f"\nBest model metrics: {metrics}")

        print("\n" + "=" * 80)
        print("✓ Training completed successfully!")
        print(f"  Model saved to: {output_dir}")
        print("=" * 80)

    except Exception as e:
        print(f"\n⚠ Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if "WANDB_API_KEY" in os.environ:
            wandb.finish()


if __name__ == "__main__":
    main()
