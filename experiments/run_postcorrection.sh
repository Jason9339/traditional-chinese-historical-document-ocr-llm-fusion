#!/usr/bin/env bash
# Post-correction Experiments
# ChatGPT + Finetune OCR × 3 prompt modes
# Dataset: Historical test set (185 samples)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="$PROJECT_ROOT/results/postcorrection_$TIMESTAMP"

echo "========================================"
echo "Post-correction Experiments"
echo "========================================"
echo "OCR: Finetune | LLM: ChatGPT"
echo "Modes: baseline, strict, loose"
echo "Output: $OUTPUT_BASE"
echo "========================================"
echo ""

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set."
    echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

cd "$PROJECT_ROOT" || exit 1

MODES=("baseline" "strict" "loose")

for MODE in "${MODES[@]}"; do
    echo ""
    echo "========================================"
    echo "Mode: $MODE"
    echo "========================================"

    OUTPUT_DIR="$OUTPUT_BASE/${MODE}"

    python scripts/evaluate/eval_postcorrection.py \
        --ocr_model finetune \
        --llm gpt \
        --mode "$MODE" \
        --num_beams 5 \
        --output_dir "$OUTPUT_DIR"

    if [ $? -eq 0 ]; then
        echo "✓ Mode $MODE complete"
    else
        echo "✗ Mode $MODE failed"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "All experiments complete!"
echo "Results: $OUTPUT_BASE"
echo "========================================"
