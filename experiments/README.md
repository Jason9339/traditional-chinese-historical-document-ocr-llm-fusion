# Evaluation Experiments

This directory contains simplified evaluation scripts for reproducing paper results (ICPR 2026).

## Test Sets

### Test Set 1: Synthetic Random (test_rand)
- **Script**: `run_testset1.sh`
- **Dataset**: HuggingFace `ZihCiLin/traditional-chinese-ocr-synthetic` (test_rand split)
- **Models**: Baseline + Finetune
- **Modes**: Pure OCR + GFD
- **Reproduces**: Table 1 & Table 2 (test_rand rows)

```bash
bash experiments/run_testset1.sh
```

### Test Set 2: Semantic Article (test_semantic)
- **Script**: `run_testset2.sh`
- **Dataset**: HuggingFace `ZihCiLin/traditional-chinese-ocr-synthetic` (test_semantic split, 395 samples)
- **Models**: Baseline + Finetune
- **Modes**: Pure OCR + GFD
- **Reproduces**: Table 1 & Table 2 (test_semantic rows)

```bash
bash experiments/run_testset2.sh
```

### Test Set 3: Historical Documents ⚠️
- **Script**: `run_testset3.sh`
- **Dataset**: HuggingFace `ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen` (test split, 185 samples)
- **Models**: Baseline + Finetune
- **Modes**: Pure OCR + GFD
- **Reproduces**: Table 1 & Table 2 (historical rows)

**⚠️ Gated Dataset**: This dataset contains library materials and requires access approval. Visit [dataset page](https://huggingface.co/datasets/ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen) to request access, then login with `huggingface-cli login`.

```bash
bash experiments/run_testset3.sh
```

### Lambda Ablation Study
- **Script**: `run_lambda_ablation.sh`
- **Dataset**: Historical test set (185 samples)
- **Model**: Finetune only
- **Tests**: Fusion weight λ ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 0.9}
- **Reproduces**: Table 3 (ablation study)

```bash
bash experiments/run_lambda_ablation.sh
```

### Post-correction Experiments
- **Script**: `run_postcorrection.sh`
- **Dataset**: Historical test set (185 samples)
- **Configuration**: Finetune OCR + ChatGPT
- **Modes**: baseline + strict + loose (3 experiments)
- **Reproduces**: Table 4 (pipeline post-correction)

**Note**: Requires OpenAI API key:
```bash
export OPENAI_API_KEY='your-key-here'
bash experiments/run_postcorrection.sh
```

## Requirements

### Datasets
Datasets are automatically downloaded from HuggingFace on first run:
- `ZihCiLin/traditional-chinese-ocr-synthetic` (Test Set 1 & 2) - **Public**
- `ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen` (Test Set 3) - **Gated** ⚠️

**Important**: The Lo Chia-Luen dataset requires access approval. Request access at the [dataset page](https://huggingface.co/datasets/ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen) before running Test Set 3.

### Models
Models are automatically loaded from HuggingFace Hub:
- Baseline: `ZihCiLin/trocr-traditional-chinese-baseline`
- Finetune: `ZihCiLin/trocr-traditional-chinese-historical`

## Output

All results are saved to `results/testsetN_TIMESTAMP/` directories with:
- JSON files containing detailed predictions and CER scores
- Summary statistics matching paper tables
