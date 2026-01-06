#!/usr/bin/env python3
"""
Finetune TrOCR Training Script
Finetune baseline TrOCR on Lo Chia-Luen historical manuscript dataset
Dataset: 690 train / 46 val samples
"""

import os
import sys
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from transformers import (
    VisionEncoderDecoderModel,
    PreTrainedTokenizerFast,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed,
)
import wandb
from dataclasses import dataclass
from typing import Any, Dict, List
import argparse

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from gfd.image_preprocessing import OCRImageTransform
from huggingface_hub import hf_hub_download

# ============================================================
# Configuration
# ============================================================
set_seed(42)

# Output directory
output_dir = os.path.join(PROJECT_ROOT, "models", "finetune_checkpoint")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/logs", exist_ok=True)

# ============================================================
# WandB Setup (Optional - User should set their own API key)
# ============================================================
if "WANDB_API_KEY" in os.environ:
    try:
        wandb.login()
        print("✓ WandB initialized")
    except Exception as e:
        print(f"⚠ WandB login failed: {e}")
        os.environ["WANDB_MODE"] = "disabled"
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
class HFHistoricalDataset(Dataset):
    """
    Wrapper for HuggingFace Historical OCR dataset
    Loads from: ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen
    """

    def __init__(self, hf_dataset, tokenizer, transform=None, max_label_length=128, repo_id=None):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_label_length = max_label_length
        self.repo_id = repo_id

    def set_transform(self, new_transform):
        """Allow dynamic transform change (for data augmentation switch)"""
        self.transform = new_transform

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
# Data Augmentation Switch Callback
# ============================================================
class DataAugmentationSwitchCallback(TrainerCallback):
    """
    Switch from augmented to clean transform after specified epoch
    Default: Use augmentation for first 20 epochs, then clean
    """

    def __init__(self, train_dataset, clean_transform, switch_epoch=20):
        self.train_dataset = train_dataset
        self.clean_transform = clean_transform
        self.switch_epoch = switch_epoch
        self.has_switched = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not self.has_switched and state.epoch >= self.switch_epoch:
            print(f"\n[DataAugmentation] Epoch {state.epoch}: Switching to clean transform")
            self.train_dataset.set_transform(self.clean_transform)
            self.has_switched = True


# ============================================================
# Augmented Transform
# ============================================================
class AugmentedOCRTransform:
    """Apply data augmentation before base OCR transform"""

    def __init__(self, base_transform):
        self.base_transform = base_transform
        self.aug_pipeline = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=2, shear=2, scale=(0.98, 1.02)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ])

    def __call__(self, img):
        img = self.aug_pipeline(img)
        return self.base_transform(img)


# ============================================================
# Main Training Function
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Finetune TrOCR on Historical Dataset")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Baseline checkpoint path (default: models/baseline_checkpoint)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path (default: models/tokenizer)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size_train", type=int, default=8, help="Train batch size")
    parser.add_argument("--batch_size_eval", type=int, default=16, help="Eval batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")

    args = parser.parse_args()

    # Set default paths if not provided
    if args.checkpoint is None:
        args.checkpoint = os.path.join(PROJECT_ROOT, "models", "baseline_checkpoint")
    if args.tokenizer is None:
        args.tokenizer = os.path.join(PROJECT_ROOT, "models", "tokenizer")

    print("=" * 80)
    print("Finetune TrOCR on Historical Manuscripts")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Baseline checkpoint: {args.checkpoint}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Train batch size: {args.batch_size_train}")
    print(f"  Learning rate: {args.learning_rate}")

    # ============================================================
    # Load Tokenizer
    # ============================================================
    if not os.path.exists(args.tokenizer):
        print(f"\n⚠ Tokenizer not found at: {args.tokenizer}")
        print("  Please prepare the tokenizer first.")
        return

    print(f"\n✓ Loading tokenizer from: {args.tokenizer}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.pad_token or "<pad>"
    tokenizer.bos_token = tokenizer.bos_token or "<s>"
    tokenizer.eos_token = tokenizer.eos_token or "</s>"
    tokenizer.unk_token = tokenizer.unk_token or "<unk>"

    global _tokenizer_for_metrics
    _tokenizer_for_metrics = tokenizer

    print("✓ Loading TrOCR processor...")
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1", tokenizer=tokenizer)
    except:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
        processor.tokenizer = tokenizer

    # ============================================================
    # Image Transforms
    # ============================================================
    IMG_HEIGHT = 384
    IMG_WIDTH = 384

    # Clean transform (no augmentation)
    clean_transform = OCRImageTransform(
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        auto_rotate_vertical=True,
        normalize=True,
    )

    # Augmented transform (for training)
    train_transform = AugmentedOCRTransform(clean_transform)

    # ============================================================
    # Load Model from Checkpoint
    # ============================================================
    if not os.path.exists(args.checkpoint):
        print(f"\n⚠ Checkpoint not found at: {args.checkpoint}")
        print("  Please train baseline model first or download from HuggingFace.")
        return

    print(f"\n✓ Loading model from checkpoint: {args.checkpoint}")
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint)

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
    print("\n✓ Loading historical dataset from HuggingFace...")
    dataset_repo = "ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen"
    print(f"  Dataset: {dataset_repo}")

    dataset = load_dataset(dataset_repo)
    train_hf = dataset['train']
    val_hf = dataset['validation']

    print(f"\n  Dataset statistics:")
    print(f"    Training:   {len(train_hf)} samples")
    print(f"    Validation: {len(val_hf)} samples")

    # Wrap in custom dataset
    train_dataset = HFHistoricalDataset(
        hf_dataset=train_hf,
        tokenizer=tokenizer,
        transform=train_transform,
        repo_id=dataset_repo,
    )

    val_dataset = HFHistoricalDataset(
        hf_dataset=val_hf,
        tokenizer=tokenizer,
        transform=clean_transform,
        repo_id=dataset_repo,
    )

    data_collator = DataCollatorForOCR(processor=processor)

    # ============================================================
    # Training Arguments
    # ============================================================
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size_train,
        per_device_eval_batch_size=args.batch_size_eval,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,

        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,

        save_strategy="epoch",
        save_total_limit=3,

        evaluation_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,

        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",

        dataloader_num_workers=8,
        dataloader_pin_memory=True,

        predict_with_generate=True,
        generation_max_length=128,
        fp16=torch.cuda.is_available(),

        report_to="wandb" if "WANDB_API_KEY" in os.environ else "none",
        run_name=f"trocr-finetune-bs{args.batch_size_train}-ep{args.epochs}",
    )

    # Data augmentation switch callback
    # Switch from augmented to clean after 20 epochs
    switch_aug_epoch = 20
    aug_callback = DataAugmentationSwitchCallback(
        train_dataset=train_dataset,
        clean_transform=clean_transform,
        switch_epoch=switch_aug_epoch,
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), aug_callback],
    )

    # ============================================================
    # Start Training
    # ============================================================
    print("\n" + "=" * 80)
    print(f"Starting finetuning... (Epochs={args.epochs}, BS={args.batch_size_train})")
    print(f"Data augmentation will be disabled after epoch {switch_aug_epoch}")
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
            print("\n✓ Evaluating best model...")
            metrics = trainer.evaluate(eval_dataset=val_dataset)
            trainer.log_metrics("eval_best", metrics)
            trainer.save_metrics("eval_best", metrics)
            print(f"\nBest model metrics: {metrics}")

        print("\n" + "=" * 80)
        print("✓ Finetuning completed successfully!")
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
