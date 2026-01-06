#!/usr/bin/env python3
"""
Test Set 1: Synthetic Random Validation Set (test_random)
Evaluates both baseline and finetune models with Pure OCR and GFD modes
Dataset: HuggingFace - ZihCiLin/traditional-chinese-ocr-synthetic (test_random split)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import re
import string
import numpy as np
from datasets import load_dataset

# Add project root to path to import gfd module
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfd.gfd_chinese import ChineseOCRBreezper
from gfd.utils import process_config, combine_config

def remove_punctuation(text: str) -> str:
    chinese_punctuation = '，。！？；：「」『』（）〔〕【】《》〈〉、·—…～'
    english_punctuation = string.punctuation
    all_punctuation = chinese_punctuation + english_punctuation
    pattern = f"[{re.escape(all_punctuation)}]"
    return re.sub(pattern, '', text)

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def compute_cer(ref, pred, remove_punct=True, remove_space=True) -> float:
    ref_str = str(ref)
    pred_str = str(pred)

    if remove_space:
        ref_str = re.sub(r"\s+", "", ref_str)
        pred_str = re.sub(r"\s+", "", pred_str)

    if remove_punct:
        ref_str = remove_punctuation(ref_str)
        pred_str = remove_punctuation(pred_str)

    if len(ref_str) == 0:
        return 0.0

    return levenshtein_distance(ref_str, pred_str) / len(ref_str)

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = sample['image']
        text = sample['text']

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'text': text, 'idx': idx}

def evaluate_model(dataset, model_name, fusing_r=0.0, num_beams=5):
    model_config_path = PROJECT_ROOT / 'config_files/model/gfd-ocr-phase2-zhtw.yaml'
    prompt_config_path = PROJECT_ROOT / 'config_files/prompt/ocr-zhtw-prompt.yaml'

    model_config = process_config(str(model_config_path))
    prompt_config = process_config(str(prompt_config_path))

    model_config = model_config._replace(
        ocr_model_path=model_name,
        ocr_tokenizer_path=model_name
    )

    combined_config = combine_config(prompt_config, model_config)
    combined_config = combined_config._replace(fusing_r=fusing_r)

    print(f"  Loading model: {model_name}")
    print(f"  Fusing ratio: {fusing_r} ({'Pure OCR' if fusing_r == 0.0 else 'GFD'})")

    model = ChineseOCRBreezper(combined_config)

    # Store prompts from config
    ocr_prompt = getattr(combined_config, 'ocr_prompt', '')
    llm_prompt = getattr(combined_config, 'llm_prompt', '')

    results = []
    total_cer = 0.0
    perfect_matches = 0

    print(f"  Running inference on {len(dataset)} samples...")

    first_sample_debug = True
    for sample in tqdm(dataset, desc="  Progress"):
        # Convert HuggingFace Image to PIL.Image
        image = sample['image']

        # Debug output for first sample
        if first_sample_debug:
            print(f"\n[DEBUG] First sample image info:")
            print(f"  Type: {type(image)}")
            print(f"  Is PIL.Image: {isinstance(image, Image.Image)}")
            if hasattr(image, 'shape'):
                print(f"  Shape: {image.shape}")
            if hasattr(image, 'dtype'):
                print(f"  Dtype: {image.dtype}")
            if hasattr(image, 'mode'):
                print(f"  Mode: {image.mode}")

        if not isinstance(image, Image.Image):
            # Handle PyTorch Tensor
            if hasattr(image, 'numpy'):
                img_array = image.numpy()
                if first_sample_debug:
                    print(f"  Converted to numpy - shape: {img_array.shape}, dtype: {img_array.dtype}")
                # Convert from float32 [0, 1] to uint8 [0, 255]
                if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                    img_array = (img_array * 255).astype(np.uint8)
                    if first_sample_debug:
                        print(f"  After denormalization - shape: {img_array.shape}, dtype: {img_array.dtype}")
                # Remove extra dimensions if present
                img_array = np.squeeze(img_array)
                if first_sample_debug:
                    print(f"  After squeeze - shape: {img_array.shape}, dtype: {img_array.dtype}")
                # Handle different array formats
                if img_array.ndim == 3 and img_array.shape[0] in [1, 3, 4]:
                    # CHW format -> HWC format
                    img_array = np.transpose(img_array, (1, 2, 0))
                    if first_sample_debug:
                        print(f"  After CHW->HWC transpose - shape: {img_array.shape}, dtype: {img_array.dtype}")
                # Handle grayscale by converting to RGB
                if img_array.ndim == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                    if first_sample_debug:
                        print(f"  After RGB conversion - shape: {img_array.shape}, dtype: {img_array.dtype}")
                elif img_array.ndim == 3 and img_array.shape[2] == 1:
                    img_array = np.repeat(img_array, 3, axis=2)
                    if first_sample_debug:
                        print(f"  After grayscale->RGB conversion - shape: {img_array.shape}, dtype: {img_array.dtype}")
                image = Image.fromarray(img_array)
                if first_sample_debug:
                    print(f"  Final PIL.Image - mode: {image.mode}, size: {image.size}")
            # Handle numpy array
            elif hasattr(image, '__array_interface__'):
                img_array = np.array(image)
                if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                    img_array = (img_array * 255).astype(np.uint8)
                img_array = np.squeeze(img_array)
                if img_array.ndim == 3 and img_array.shape[0] in [1, 3, 4]:
                    img_array = np.transpose(img_array, (1, 2, 0))
                if img_array.ndim == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.ndim == 3 and img_array.shape[2] == 1:
                    img_array = np.repeat(img_array, 3, axis=2)
                image = Image.fromarray(img_array)
            # Handle other PIL-like objects
            elif hasattr(image, 'convert'):
                image = image.convert("RGB")

        if first_sample_debug:
            first_sample_debug = False
            print(f"[DEBUG] Proceeding with inference...\n")

        gt_text = sample['text']

        pred_text = model.get_transcription(
            image,
            num_beams=num_beams,
            ocr_prompt=ocr_prompt,
            llm_prompt=llm_prompt
        )
        cer = compute_cer(gt_text, pred_text)

        total_cer += cer
        if gt_text == pred_text:
            perfect_matches += 1

        results.append({
            'idx': sample['idx'],
            'ground_truth': gt_text,
            'prediction': pred_text,
            'cer': cer,
            'match': gt_text == pred_text
        })

    avg_cer = total_cer / len(dataset)

    return {
        'results': results,
        'avg_cer': avg_cer,
        'perfect_matches': perfect_matches,
        'total_samples': len(dataset)
    }

def main():
    parser = argparse.ArgumentParser(description='Test Set 1: Synthetic Random Validation')
    parser.add_argument('--dataset_repo', type=str,
                        default='ZihCiLin/traditional-chinese-ocr-synthetic',
                        help='HuggingFace dataset repository')
    parser.add_argument('--split', type=str, default='test_random',
                        help='Dataset split')
    parser.add_argument('--baseline_model', type=str,
                        default='ZihCiLin/trocr-traditional-chinese-baseline',
                        help='Baseline model from HuggingFace')
    parser.add_argument('--finetune_model', type=str,
                        default='ZihCiLin/trocr-traditional-chinese-historical-finetune',
                        help='Finetune model from HuggingFace')
    parser.add_argument('--num_beams', type=int, default=5, help='Beam search size')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Test Set 1: Synthetic Random Validation (test_random)")
    print("=" * 80)
    print(f"Dataset: {args.dataset_repo} (split: {args.split})")
    print(f"Baseline model: {args.baseline_model}")
    print(f"Finetune model: {args.finetune_model}")
    print(f"Num beams: {args.num_beams}")
    print(f"Output directory: {args.output_dir}")
    print()

    print(f"Loading dataset from HuggingFace: {args.dataset_repo} ({args.split})...")
    hf_dataset = load_dataset(args.dataset_repo, split=args.split)
    dataset = HFDatasetWrapper(hf_dataset, transform=None)
    print(f"  Total samples: {len(dataset)}")
    print()

    experiments = [
        ('baseline', 'pure_ocr', args.baseline_model, 0.0),
        ('baseline', 'gfd', args.baseline_model, 0.3),
        ('finetune', 'pure_ocr', args.finetune_model, 0.0),
        ('finetune', 'gfd', args.finetune_model, 0.3),
    ]

    all_results = {}

    for model_type, mode, model_name, fusing_r in experiments:
        print(f"\n{'=' * 80}")
        print(f"Experiment: {model_type.upper()} + {mode.upper()}")
        print(f"{'=' * 80}\n")

        stats = evaluate_model(dataset, model_name, fusing_r, args.num_beams)

        key = f"{model_type}_{mode}"
        all_results[key] = stats

        print(f"\n  Results:")
        print(f"    Average CER: {stats['avg_cer']:.4f}")
        print(f"    Perfect matches: {stats['perfect_matches']}/{stats['total_samples']}")

        output_file = os.path.join(args.output_dir, f"{key}_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    summary_file = os.path.join(args.output_dir, 'summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        summary = {k: {'avg_cer': v['avg_cer'],
                       'perfect_matches': v['perfect_matches'],
                       'total_samples': v['total_samples']}
                   for k, v in all_results.items()}
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}\n")

    for model_type in ['baseline', 'finetune']:
        for mode in ['pure_ocr', 'gfd']:
            key = f"{model_type}_{mode}"
            stats = all_results[key]
            print(f"  {model_type} + {mode}: CER={stats['avg_cer']:.4f}")
    print()

    print(f"Results saved to: {args.output_dir}")
    print(f"Summary: {summary_file}")

if __name__ == '__main__':
    main()
