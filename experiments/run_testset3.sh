#!/usr/bin/env bash
# Test Set 3: Historical Documents
# Evaluates baseline and finetune models on real historical manuscript images
# Dataset: 185 samples from HuggingFace (test split)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

OUTPUT_DIR="$PROJECT_ROOT/results/testset3_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "Test Set 3: Historical Documents"
echo "========================================"
echo "Dataset: HF (185 test samples)"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

cd "$PROJECT_ROOT" || exit 1

python scripts/evaluate/eval_testset3.py \
    --split test \
    --num_beams 5 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Results: $OUTPUT_DIR"
echo "========================================"
