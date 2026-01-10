#!/usr/bin/env python3
"""
Post-correction Evaluation
Pipeline OCR with LLM post-correction
Tests: 2 OCR models × 2 LLMs × 3 modes = 12 experiments
Dataset: Historical test set (185 samples)
"""

import sys
import os
from pathlib import Path

# Add project root to path to import gfd module
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Import from same directory
from evaluate_pipeline_correction import main as original_main
import argparse

def main():
    parser = argparse.ArgumentParser(description='Post-correction Evaluation (Simplified)')

    parser.add_argument('--ocr_model', type=str,
                        choices=['baseline', 'finetune'],
                        default='finetune',
                        help='OCR model to use')
    parser.add_argument('--llm', type=str,
                        choices=['breeze', 'gpt'],
                        default='breeze',
                        help='LLM to use for correction')
    parser.add_argument('--mode', type=str,
                        choices=['baseline', 'strict', 'loose'],
                        default='strict',
                        help='Correction mode')
    parser.add_argument('--num_beams', type=int, default=5,
                        help='Beam search size')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key (for GPT models)')

    args = parser.parse_args()

    if args.ocr_model == 'baseline':
        ocr_checkpoint = 'ZihCiLin/trocr-traditional-chinese-baseline'
        ocr_tokenizer = 'ZihCiLin/trocr-traditional-chinese-baseline'
    else:
        ocr_checkpoint = 'ZihCiLin/trocr-traditional-chinese-historical-finetune'
        ocr_tokenizer = 'ZihCiLin/trocr-traditional-chinese-historical-finetune'

    if args.llm == 'breeze':
        llm_type = 'local'
        llm_model = 'MediaTek-Research/Breeze-7B-32k-Base-v1_0'
        api_key = None
    else:
        llm_type = 'api'
        llm_model = 'gpt-5.2'
        api_key = args.openai_api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required for GPT models. Set --openai_api_key or OPENAI_API_KEY env var.")

    print("=" * 80)
    print("Post-correction Evaluation")
    print("=" * 80)
    print(f"OCR Model: {args.ocr_model} ({ocr_checkpoint})")
    print(f"LLM: {args.llm} ({llm_model})")
    print(f"Mode: {args.mode}")
    print(f"Num beams: {args.num_beams}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    print()

    original_args = [
        '--dataset', 'historical',
        '--split', 'test',
        '--ocr_checkpoint', ocr_checkpoint,
        '--ocr_tokenizer', ocr_tokenizer,
        '--num_beams', str(args.num_beams),
        '--llm_type', llm_type,
        '--llm_model', llm_model,
        '--mode', args.mode,
        '--temperature', '0.0',
        '--save_dir', args.output_dir,
    ]

    if api_key:
        original_args.extend(['--api_key', api_key])

    sys.argv = ['evaluate_pipeline_correction.py'] + original_args
    original_main()

if __name__ == '__main__':
    main()
