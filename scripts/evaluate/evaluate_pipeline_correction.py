"""
Pipeline OCR Post-correction Evaluation
OCR Top-3 candidate generation + LLM correction
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime
import argparse
import re
import string

# Add project root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import torch
from tqdm import tqdm
import pandas as pd

from gfd.ocr_with_candidates import create_ocr_with_candidates_from_config
from gfd.llm_corrector import create_llm_corrector_from_config


def remove_punctuation(text: str) -> str:
    """Remove punctuation (Chinese and English)"""
    chinese_punctuation = '，。！？；：「」『』（）〔〕【】《》〈〉、·—…～'
    english_punctuation = string.punctuation
    all_punctuation = chinese_punctuation + english_punctuation
    pattern = f"[{re.escape(all_punctuation)}]"
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance"""
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


def compute_cer(ground_truth: str, prediction: str, remove_punct: bool = True, remove_space: bool = True) -> float:
    """
    Calculate Character Error Rate (CER)
    Using Levenshtein distance
    - Remove whitespace characters
    - Remove Chinese and English punctuation

    Args:
        ground_truth: ground truth text
        prediction: predicted text
        remove_punct: whether to remove punctuation
        remove_space: whether to remove spaces

    Returns:
        CER (0.0 ~ 1.0)
    """
    ref_str = str(ground_truth)
    pred_str = str(prediction)

    # Remove all whitespace
    if remove_space:
        ref_str = re.sub(r"\s+", "", ref_str)
        pred_str = re.sub(r"\s+", "", pred_str)

    # Remove punctuation
    if remove_punct:
        ref_str = remove_punctuation(ref_str)
        pred_str = remove_punctuation(pred_str)

    if len(ref_str) == 0:
        return 0.0 if len(pred_str) == 0 else 1.0

    dist = levenshtein_distance(ref_str, pred_str)
    return min(dist / len(ref_str), 1.0)


def analyze_top3_coverage(ground_truth: str, candidates: List[Dict]) -> Dict:
    """
    Analyze Top-3 candidate coverage

    Args:
        ground_truth: ground truth text
        candidates: candidate list

    Returns:
        {
            "total_chars": int,
            "top1_correct": int,
            "top3_contains": int,
            "top3_coverage_rate": float
        }
    """
    total_chars = len(ground_truth)
    top1_correct = 0
    top3_contains = 0

    for i, gt_char in enumerate(ground_truth):
        if i >= len(candidates):
            break

        cand = candidates[i]

        # Check if Top-1 is correct
        if 'char' in cand and cand['char'] == gt_char:
            top1_correct += 1

        # Check if Top-3 contains correct character
        if 'top_k' in cand:
            top3_chars = [item.get('char', '') for item in cand['top_k']]
            if gt_char in top3_chars:
                top3_contains += 1

    return {
        "total_chars": total_chars,
        "top1_correct": top1_correct,
        "top3_contains": top3_contains,
        "top3_coverage_rate": top3_contains / total_chars if total_chars > 0 else 0.0
    }


def evaluate_pipeline_method(
    dataset,
    ocr_with_candidates,
    llm_corrector,
    dataset_name: str = "dataset",
    num_beams: int = 5,
    save_dir: str = None,
    metadata: Dict[str, Any] = None
) -> List[Dict]:
    """
    Execute Pipeline evaluation

    Args:
        dataset: dataset (iterable)
        ocr_with_candidates: OCR Top-3 generator
        llm_corrector: LLM corrector
        dataset_name: dataset name
        num_beams: Beam search size
        save_dir: Results save directory
        metadata: experiment configuration metadata

    Returns:
        evaluation result list
    """
    results = []

    # Prepare metadata (create default if not provided)
    if metadata is None:
        metadata = {}

    print(f"\n{'='*70}")
    print(f"Starting Pipeline evaluation: {dataset_name}")
    print(f"{'='*70}")
    print(f"  Mode: {llm_corrector.mode}")
    print(f"  LLM type: {llm_corrector.llm_type}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Num Beams: {num_beams}\n")

    for idx in tqdm(range(len(dataset)), desc="Pipeline inference"):
        try:
            sample = dataset[idx]

            # Phase 1: OCR with Top-3
            ocr_result = ocr_with_candidates.predict_with_candidates(
                sample['image'],
                num_beams=num_beams
            )

            # Phase 2: LLM Correction
            corrected = llm_corrector.correct(ocr_result)

            # Phase 3: Evaluation
            ground_truth = sample['text']

            cer_ocr = compute_cer(ground_truth, ocr_result['final_text'])
            cer_corrected = compute_cer(ground_truth, corrected['corrected_text'])
            improvement = cer_ocr - cer_corrected

            # Record results
            result = {
                # === Core data ===
                'sample_id': sample.get('id', sample.get('sample_id', idx)),
                'ground_truth': ground_truth,
                'ocr_prediction': ocr_result['final_text'],
                'corrected_prediction': corrected['corrected_text'],
                'raw_response': corrected.get('raw_response', ''),  # LLM raw output
                'prompt': corrected.get('prompt', ''),  # Full prompt input

                # === Top-K candidate info ===
                'candidates': ocr_result['candidates'],

                # === Evaluation metrics ===
                'cer_ocr': cer_ocr,
                'cer_corrected': cer_corrected,
                'improvement': improvement,
                'perfect_match_ocr': cer_ocr == 0.0,
                'perfect_match_corrected': cer_corrected == 0.0,
                'parse_success': corrected.get('parse_success', False),
                'parse_mode': corrected.get('parse_mode', 'json'),

                # === Correction statistics ===
                'num_corrections': len(corrected.get('corrections', [])),
                'corrections': corrected.get('corrections', []),

                # === Experiment config (Metadata) ===
                'metadata': metadata.copy()  # Save full metadata for each sample
            }

            # Additional analysis: Top-3 coverage
            top3_analysis = analyze_top3_coverage(ground_truth, ocr_result['candidates'])
            result.update({
                f'top3_{k}': v for k, v in top3_analysis.items()
            })

            results.append(result)

        except Exception as e:
            print(f"\n❌ Sample {idx} processing failed: {e}")
            import traceback
            traceback.print_exc()

            # Record failed sample
            results.append({
                'sample_id': sample.get('id', sample.get('sample_id', idx)),
                'ground_truth': sample.get('text', 'N/A'),
                'ocr_prediction': f"ERROR: {str(e)}",
                'corrected_prediction': f"ERROR: {str(e)}",
                'cer_ocr': 1.0,
                'cer_corrected': 1.0,
                'improvement': 0.0,
                'error': str(e),
                'parse_success': False
            })
            continue

    # Calculate overall statistics
    print_summary(results, dataset_name)

    # Save results
    if save_dir:
        save_results(results, save_dir, dataset_name, llm_corrector.mode)

    return results


def print_summary(results: List[Dict], dataset_name: str):
    """Print evaluation summary"""
    print(f"\n{'='*70}")
    print(f"Evaluation summary: {dataset_name}")
    print(f"{'='*70}")

    total = len(results)
    successful = sum(1 for r in results if 'error' not in r)

    print(f"Total samples: {total}")
    print(f"Successfully processed: {successful} ({successful/total*100:.1f}%)")

    if successful == 0:
        print("⚠️  No successfully processed samples")
        return

    # Filter successful samples
    valid_results = [r for r in results if 'error' not in r]

    # CER Statistics
    avg_cer_ocr = sum(r['cer_ocr'] for r in valid_results) / len(valid_results)
    avg_cer_corrected = sum(r['cer_corrected'] for r in valid_results) / len(valid_results)
    avg_improvement = sum(r['improvement'] for r in valid_results) / len(valid_results)

    print(f"\nCER Statistics:")
    print(f"  OCR CER:       {avg_cer_ocr:.4f}")
    print(f"  Corrected CER: {avg_cer_corrected:.4f}")
    print(f"  Improvement:   {avg_improvement:+.4f}")

    # Perfect Match
    perfect_ocr = sum(1 for r in valid_results if r['perfect_match_ocr'])
    perfect_corrected = sum(1 for r in valid_results if r['perfect_match_corrected'])

    print(f"\nPerfect Match:")
    print(f"  OCR:       {perfect_ocr}/{successful} ({perfect_ocr/successful*100:.1f}%)")
    print(f"  Corrected: {perfect_corrected}/{successful} ({perfect_corrected/successful*100:.1f}%)")

    # Top-3 coverage
    if 'top3_coverage_rate' in valid_results[0]:
        avg_top3_coverage = sum(r['top3_coverage_rate'] for r in valid_results) / len(valid_results)
        print(f"\nTop-3 coverage: {avg_top3_coverage:.2%}")

    # Correction statistics
    total_corrections = sum(r['num_corrections'] for r in valid_results)
    avg_corrections = total_corrections / len(valid_results)
    print(f"\nCorrection statistics:")
    print(f"  Total corrections: {total_corrections}")
    print(f"  Average per sample: {avg_corrections:.2f}")

    # JSON parsing success rate
    parse_success = sum(1 for r in valid_results if r.get('parse_success', False))
    print(f"\nJSON parsing success rate: {parse_success}/{successful} ({parse_success/successful*100:.1f}%)")


def save_results(results: List[Dict], save_dir: str, dataset_name: str, mode: str):
    """Save evaluation results"""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{dataset_name}_{mode}_{timestamp}"

    # 1. Save detailed results (JSON)
    json_path = os.path.join(save_dir, f"{prefix}_detailed.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Detailed results saved: {json_path}")

    # 2. Save comparison results (CSV)
    comparison_data = []
    for r in results:
        comparison_data.append({
            'sample_id': r['sample_id'],
            'ground_truth': r['ground_truth'],
            'ocr_prediction': r['ocr_prediction'],
            'corrected_prediction': r['corrected_prediction'],
            'cer_ocr': r['cer_ocr'],
            'cer_corrected': r['cer_corrected'],
            'improvement': r['improvement'],
            'num_corrections': r.get('num_corrections', 0),
            'perfect_match_ocr': r.get('perfect_match_ocr', False),
            'perfect_match_corrected': r.get('perfect_match_corrected', False),
            'top3_coverage_rate': r.get('top3_coverage_rate', 0.0)
        })

    df_comparison = pd.DataFrame(comparison_data)
    csv_path = os.path.join(save_dir, f"{prefix}_comparison.csv")
    df_comparison.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Comparison results saved: {csv_path}")

    # 3. Save summary statistics (CSV)
    valid_results = [r for r in results if 'error' not in r]
    if len(valid_results) > 0:
        summary = {
            'dataset': dataset_name,
            'mode': mode,
            'total_samples': len(results),
            'successful_samples': len(valid_results),
            'avg_cer_ocr': sum(r['cer_ocr'] for r in valid_results) / len(valid_results),
            'avg_cer_corrected': sum(r['cer_corrected'] for r in valid_results) / len(valid_results),
            'avg_improvement': sum(r['improvement'] for r in valid_results) / len(valid_results),
            'perfect_match_ocr': sum(1 for r in valid_results if r['perfect_match_ocr']),
            'perfect_match_corrected': sum(1 for r in valid_results if r['perfect_match_corrected']),
            'avg_top3_coverage': sum(r.get('top3_coverage_rate', 0) for r in valid_results) / len(valid_results),
            'total_corrections': sum(r['num_corrections'] for r in valid_results),
            'parse_success_rate': sum(1 for r in valid_results if r.get('parse_success', False)) / len(valid_results)
        }

        df_summary = pd.DataFrame([summary])
        summary_path = os.path.join(save_dir, f"{prefix}_summary.csv")
        df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"✓ Summary statistics saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Pipeline OCR Post-correction Evaluation')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='historical',
                        choices=['historical', 'artical', 'validation'],
                        help='Dataset selection')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split (train/val/test)')

    # OCR model parameters
    parser.add_argument('--ocr_checkpoint', type=str,
                        default='/mnt/whliao/experiment/output_phase1_phase2_from_scratch/checkpoint-100000',
                        help='OCR model checkpoint path')
    parser.add_argument('--ocr_tokenizer', type=str,
                        default='/mnt/whliao/experiment/tokenizer',
                        help='OCR tokenizer path')
    parser.add_argument('--num_beams', type=int, default=5,
                        help='Beam search size')

    # LLM parameters
    parser.add_argument('--llm_type', type=str, default='local',
                        choices=['local', 'api'],
                        help='LLM type')
    parser.add_argument('--llm_model', type=str,
                        default='/mnt/whliao/MediaTek-Research-Breeze-7B-Instruct-v1_0',
                        help='LLM model path or API model name')
    parser.add_argument('--mode', type=str, default='strict',
                        choices=['baseline', 'strict', 'loose'],
                        help='Correction mode')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key (required for API mode)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='LLM generation temperature')

    # Output parameters
    parser.add_argument('--save_dir', type=str,
                        default='/mnt/whliao/experiment/generative-fusion-decoding-OCR/examples/results_pipeline',
                        help='Results save directory')

    args = parser.parse_args()

    # Loading dataset
    print(f"\nLoading dataset: {args.dataset} ({args.split})")
    dataset = load_dataset(args.dataset, args.split)
    print(f"✓ Dataset loaded: {len(dataset)} Sample")

    # Loading OCR Top-3 generator
    print(f"\nLoading OCR Top-3 generator...")
    ocr = create_ocr_with_candidates_from_config(
        checkpoint_path=args.ocr_checkpoint,
        tokenizer_path=args.ocr_tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=None  # Use float32 to avoid dtype issues
    )

    # Loading LLM corrector
    print(f"\nLoading LLM corrector...")
    llm_corrector = create_llm_corrector_from_config(
        llm_type=args.llm_type,
        llm_model_path=args.llm_model,
        mode=args.mode,
        api_key=args.api_key,
        temperature=args.temperature
    )

    # Prepare experiment configuration metadata
    from datetime import datetime
    metadata = {
        "llm_type": args.llm_type,
        "llm_model": args.llm_model,
        "ocr_model": args.ocr_checkpoint,
        "mode": args.mode,
        "temperature": args.temperature,
        "num_beams": args.num_beams,
        "dataset": args.dataset,
        "split": args.split,
        "timestamp": datetime.now().isoformat()
    }

    # Execute evaluation
    results = evaluate_pipeline_method(
        dataset=dataset,
        ocr_with_candidates=ocr,
        llm_corrector=llm_corrector,
        dataset_name=f"{args.dataset}_{args.split}",
        num_beams=args.num_beams,
        save_dir=args.save_dir,
        metadata=metadata
    )

    print(f"\n{'='*70}")
    print("Evaluation complete!")
    print(f"{'='*70}\n")


def load_dataset(dataset_name: str, split: str = 'test', max_samples: int = None):
    """
    Load dataset using HuggingFace datasets

    Args:
        dataset_name: dataset name ('historical', 'artical', 'validation')
        split: dataset split (historical only)
        max_samples: max samples (validation only)

    Returns:
        Dataset
    """
    if dataset_name == 'historical':
        # Use HuggingFace dataset instead of local files
        from datasets import load_dataset as hf_load_dataset
        from huggingface_hub import hf_hub_download
        from PIL import Image

        repo_id = 'ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen'
        print(f"Loading HuggingFace dataset: {repo_id} (split: {split})")
        hf_dataset = hf_load_dataset(repo_id, split=split)

        # Wrap HuggingFace dataset to match expected interface
        class HFDatasetWrapper:
            def __init__(self, hf_dataset, repo_id):
                self.hf_dataset = hf_dataset
                self.repo_id = repo_id
                print(f"Loaded {len(hf_dataset)} samples from HuggingFace")

            def __len__(self):
                return len(self.hf_dataset)

            def __getitem__(self, idx):
                sample = self.hf_dataset[idx]
                text = sample['text']
                crop_path = sample['crop_path']

                # Download image file from HuggingFace dataset repo
                image_file = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=crop_path,
                    repo_type='dataset'
                )
                image = Image.open(image_file).convert('RGB')

                return {
                    'image': image,
                    'text': text,
                    'text_direction': sample.get('text_direction', 'ltr'),
                    'metadata': {
                        'id': sample.get('id', f'sample_{idx}'),
                        'rec_id': sample.get('rec_id', ''),
                        'page_id': sample.get('page_id', ''),
                        'shape_id': sample.get('shape_id', ''),
                        'group_id': sample.get('group_id', ''),
                        'order': sample.get('order', 0),
                        'source_image': sample.get('source_image', ''),
                        'split': split
                    }
                }

        dataset = HFDatasetWrapper(hf_dataset, repo_id)
    elif dataset_name == 'artical':
        raise NotImplementedError("Artical dataset not available in open-source version. Use historical dataset instead.")
    elif dataset_name == 'validation':
        raise NotImplementedError("Validation dataset not available in open-source version. Use historical dataset instead.")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


if __name__ == "__main__":
    main()
