#!/usr/bin/env bash
# Test Set 1: Synthetic Random Validation Set (test_random)
# Evaluates baseline and finetune models on synthetic random character images
# Dataset: HuggingFace - ZihCiLin/traditional-chinese-ocr-synthetic (test_random split)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

OUTPUT_DIR="$PROJECT_ROOT/results/testset1_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "Test Set 1: test_rand Evaluation"
echo "========================================"
echo "Dataset: HuggingFace (test_rand split)"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

cd "$PROJECT_ROOT" || exit 1

python scripts/evaluate/eval_testset1.py \
    --num_beams 5 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Results: $OUTPUT_DIR"
echo "========================================"
