# Evaluation Scripts

Simplified evaluation scripts for reproducing ICPR 2026 paper results.

## Overview

All scripts load models from HuggingFace Model Hub:
- **Baseline Model**: `ZihCiLin/trocr-traditional-chinese-baseline`
- **Finetune Model**: `ZihCiLin/trocr-traditional-chinese-historical-finetune`

## Scripts

### Test Set 1: test_rand
**Script**: `eval_testset1.py`
**Dataset**: HuggingFace `ZihCiLin/traditional-chinese-ocr-synthetic` (test_rand split)
**Experiments**: 4 (2 models × 2 modes)

```bash
python scripts/evaluate/eval_testset1.py \
    --num_beams 5 \
    --output_dir results/testset1
```

### Test Set 2: test_semantic
**Script**: `eval_testset2.py`
**Dataset**: HuggingFace `ZihCiLin/traditional-chinese-ocr-synthetic` (test_semantic split, 395 samples)
**Experiments**: 4 (2 models × 2 modes)

```bash
python scripts/evaluate/eval_testset2.py \
    --num_beams 5 \
    --output_dir results/testset2
```

### Test Set 3: historical
**Script**: `eval_testset3.py`
**Dataset**: 185 historical manuscript samples (HuggingFace)
**Experiments**: 4 (2 models × 2 modes)

```bash
python scripts/evaluate/eval_testset3.py \
    --split test \
    --num_beams 5 \
    --output_dir results/testset3
```

### Lambda Ablation
**Script**: `eval_lambda_ablation.py`
**Dataset**: Historical test set (185 samples)
**Experiments**: 11 (λ from 0.0 to 1.0)

```bash
python scripts/evaluate/eval_lambda_ablation.py \
    --split test \
    --num_beams 5 \
    --output_dir results/lambda_ablation
```

### Post-correction
**Script**: `eval_postcorrection.py`
**Dataset**: Historical test set (185 samples)
**Experiments**: 1 (single configuration)

```bash
# Example: Finetune + Breeze + Strict mode
python scripts/evaluate/eval_postcorrection.py \
    --ocr_model finetune \
    --llm breeze \
    --mode strict \
    --num_beams 5 \
    --output_dir results/postcorrection

# For GPT models, set API key:
export OPENAI_API_KEY='your-key-here'
python scripts/evaluate/eval_postcorrection.py \
    --ocr_model baseline \
    --llm gpt \
    --mode loose \
    --num_beams 5 \
    --output_dir results/postcorrection_gpt
```

## Output Format

All scripts generate:
- **Detailed JSON**: Per-sample predictions and CER scores
- **Summary JSON**: Aggregated statistics
- **CSV** (ablation only): Table-ready results

## Common Parameters

- `--num_beams`: Beam search size (default: 5)
- `--output_dir`: Output directory (required)
- `--baseline_model`: Override baseline model
- `--finetune_model`: Override finetune model

## Dependencies

All scripts depend on:
- `gfd` module from `generative-fusion-decoding-OCR`
- HuggingFace `transformers` and `datasets`
- All datasets will be automatically downloaded from HuggingFace on first run
- For post-correction: Breeze-7B (local) or OpenAI API access
