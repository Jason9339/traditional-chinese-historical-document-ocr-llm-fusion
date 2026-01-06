#!/usr/bin/env bash
# Lambda Ablation Study
# Tests different fusion weights (λ) from 0.0 to 1.0
# Model: Finetune model only
# Dataset: Historical test set (185 samples)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

OUTPUT_DIR="$PROJECT_ROOT/results/lambda_ablation_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "Lambda Ablation Study"
echo "========================================"
echo "Testing λ ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 0.9}"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

cd "$PROJECT_ROOT" || exit 1

python scripts/evaluate/eval_lambda_ablation.py \
    --lambda_values "0.0,0.1,0.3,0.5,0.7,0.9" \
    --split test \
    --num_beams 5 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Ablation study complete!"
echo "Results: $OUTPUT_DIR"
echo "========================================"
