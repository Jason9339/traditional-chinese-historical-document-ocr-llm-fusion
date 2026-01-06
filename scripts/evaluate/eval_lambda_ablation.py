#!/usr/bin/env python3
"""
Lambda Ablation Study
Tests different fusion weights (λ = fusing_r) on historical test set
Model: Finetune only
Lambda values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
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
from huggingface_hub import hf_hub_download

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

    return min(levenshtein_distance(ref_str, pred_str) / len(ref_str), 1.0)

class HistoricalDataset(Dataset):
    def __init__(self, hf_dataset, repo_id, transform=None):
        self.hf_dataset = hf_dataset
        self.repo_id = repo_id
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        text = sample['text']
        crop_path = sample['crop_path']

        image_file = hf_hub_download(
            repo_id=self.repo_id,
            filename=crop_path,
            repo_type='dataset'
        )
        image = Image.open(image_file).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'text': text, 'idx': idx}

def evaluate_with_lambda(dataset, model_name, fusing_r, num_beams=5):
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

    print(f"  Loading model with λ={fusing_r}...")
    model = ChineseOCRBreezper(combined_config)

    # Store prompts from config
    ocr_prompt = getattr(combined_config, 'ocr_prompt', '')
    llm_prompt = getattr(combined_config, 'llm_prompt', '')

    results = []
    total_cer = 0.0
    perfect_matches = 0

    print(f"  Running inference...")
    for sample in tqdm(dataset, desc=f"  λ={fusing_r}"):
        image = sample['image']
        gt_text = sample['text']

        # Convert tensor back to PIL.Image if needed
        if not isinstance(image, Image.Image):
            if hasattr(image, 'numpy'):
                img_array = image.numpy()
                # Convert from float32 [0, 1] to uint8 [0, 255]
                if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                    img_array = (img_array * 255).astype(np.uint8)
                # Remove extra dimensions if present
                img_array = np.squeeze(img_array)
                # Handle different array formats
                if img_array.ndim == 3 and img_array.shape[0] in [1, 3, 4]:
                    # CHW format -> HWC format
                    img_array = np.transpose(img_array, (1, 2, 0))
                # Handle grayscale by converting to RGB
                if img_array.ndim == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.ndim == 3 and img_array.shape[2] == 1:
                    img_array = np.repeat(img_array, 3, axis=2)
                image = Image.fromarray(img_array)
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
            elif hasattr(image, 'convert'):
                image = image.convert("RGB")

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
        'fusing_r': fusing_r,
        'results': results,
        'avg_cer': avg_cer,
        'cer_percent': avg_cer * 100,
        'perfect_matches': perfect_matches,
        'total_samples': len(dataset)
    }

def main():
    parser = argparse.ArgumentParser(description='Lambda Ablation Study')
    parser.add_argument('--dataset_repo', type=str,
                        default='ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen',
                        help='HuggingFace dataset repository')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate')
    parser.add_argument('--model', type=str,
                        default='ZihCiLin/trocr-traditional-chinese-historical-finetune',
                        help='Finetune model from HuggingFace')
    parser.add_argument('--lambda_values', type=str,
                        default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0',
                        help='Comma-separated lambda values to test')
    parser.add_argument('--num_beams', type=int, default=5, help='Beam search size')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    lambda_values = [float(x) for x in args.lambda_values.split(',')]

    print("=" * 80)
    print("Lambda Ablation Study")
    print("=" * 80)
    print(f"Dataset: {args.dataset_repo} (split: {args.split})")
    print(f"Model: {args.model}")
    print(f"Lambda values: {lambda_values}")
    print(f"Num beams: {args.num_beams}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    print()

    print("Loading historical dataset from HuggingFace...")
    hf_dataset = load_dataset(args.dataset_repo, split=args.split)
    dataset = HistoricalDataset(hf_dataset, args.dataset_repo, transform=None)
    print(f"  Total samples: {len(dataset)}")
    print()

    all_results = {}

    for idx, lambda_val in enumerate(lambda_values, 1):
        print(f"\n{'=' * 80}")
        print(f"Experiment {idx}/{len(lambda_values)}: λ = {lambda_val}")
        print(f"{'=' * 80}\n")

        stats = evaluate_with_lambda(dataset, args.model, lambda_val, args.num_beams)
        all_results[f"lambda_{lambda_val}"] = stats

        print(f"\n  Results:")
        print(f"    Average CER: {stats['avg_cer']:.4f} ({stats['cer_percent']:.2f}%)")
        print(f"    Perfect matches: {stats['perfect_matches']}/{stats['total_samples']}")

        output_file = os.path.join(args.output_dir, f"lambda_{lambda_val}_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    summary_file = os.path.join(args.output_dir, 'ablation_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        summary = {k: {'fusing_r': v['fusing_r'],
                       'avg_cer': v['avg_cer'],
                       'cer_percent': v['cer_percent'],
                       'perfect_matches': v['perfect_matches'],
                       'total_samples': v['total_samples']}
                   for k, v in all_results.items()}
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_file = os.path.join(args.output_dir, 'ablation_summary.csv')
    with open(csv_file, 'w') as f:
        f.write("lambda,cer,cer_percent,perfect_matches,total_samples\n")
        for lambda_val in lambda_values:
            stats = all_results[f"lambda_{lambda_val}"]
            f.write(f"{lambda_val},{stats['avg_cer']:.4f},{stats['cer_percent']:.2f}%,")
            f.write(f"{stats['perfect_matches']},{stats['total_samples']}\n")

    print(f"\n{'=' * 80}")
    print("Summary Table")
    print(f"{'=' * 80}\n")
    print(f"{'λ':>6} | {'CER':>8} | {'Perfect':>7}")
    print("-" * 30)
    for lambda_val in lambda_values:
        stats = all_results[f"lambda_{lambda_val}"]
        pm_rate = stats['perfect_matches'] / stats['total_samples'] * 100
        print(f"{lambda_val:>6.1f} | {stats['cer_percent']:>7.2f}% | {pm_rate:>6.1f}%")
    print()

    print(f"Results saved to: {args.output_dir}")
    print(f"Summary JSON: {summary_file}")
    print(f"Summary CSV: {csv_file}")

if __name__ == '__main__':
    main()
