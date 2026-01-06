#!/usr/bin/env bash
# Test Set 2: Semantic Article Dataset (test_semantic)
# Evaluates baseline and finetune models on semantic article dataset images
# Dataset: HuggingFace - ZihCiLin/traditional-chinese-ocr-synthetic (test_semantic split)
# 395 samples

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

OUTPUT_DIR="$PROJECT_ROOT/results/testset2_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "Test Set 2: test_semantic Evaluation"
echo "========================================"
echo "Dataset: HuggingFace (test_semantic split)"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

cd "$PROJECT_ROOT" || exit 1

python scripts/evaluate/eval_testset2.py \
    --num_beams 5 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Results: $OUTPUT_DIR"
echo "========================================"
