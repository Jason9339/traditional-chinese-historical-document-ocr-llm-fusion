# Decoding-Time Fusion of OCR and Large Language Models for Traditional Chinese Historical Document Recognition

Official implementation of the paper: **"Decoding-Time Fusion of OCR and Large Language Models for Traditional Chinese Historical Document Recognition"**.

## Overview

This project proposes **Decoding-Time OCR–LLM Fusion** for Traditional Chinese historical document OCR. We use **TrOCR** as the visual recognition model and **Breeze-7B** as a semantic scorer, integrating them during **beam search** decoding. To address tokenization mismatches between OCR and LLM, we employ **UTF-8 byte space alignment** and **byte-prefix marginalization**, enabling semantic guidance without joint training.

We provide reproducible research resources including:
- A configurable **synthetic data generator** and **4.1M dataset**
- A **web-based annotation system**
- A benchmark of **921 manually annotated samples** from the Lo Chia-Luen Collection with geometric metadata

Experiments show that fusion decoding consistently reduces CER on both semantic text and real historical manuscripts, outperforming LLM-based post-correction.

---

## Resources

### Paper
- Paper: Under review

### Tools & Datasets
- **Synthetic Generator**: [GitHub: ocr-synth-generator](https://github.com/Jason9339/ocr-synth-generator)
- **Synthetic Dataset (4.1M)**: [Hugging Face: traditional-chinese-ocr-synthetic](https://huggingface.co/datasets/ZihCiLin/traditional-chinese-ocr-synthetic)
- **Annotation System**: [GitHub: document-ocr-annotation-system](https://github.com/Jason9339/document-ocr-annotation-system)
- **Lo Chia-Luen Benchmark (921)**: [Hugging Face: traditional-chinese-historical-ocr-lo-chia-luen](https://huggingface.co/datasets/ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen)

### Models
- **Baseline TrOCR**: [ZihCiLin/trocr-traditional-chinese-baseline](https://huggingface.co/ZihCiLin/trocr-traditional-chinese-baseline)
- **Finetuned TrOCR**: [ZihCiLin/trocr-traditional-chinese-historical-finetune](https://huggingface.co/ZihCiLin/trocr-traditional-chinese-historical-finetune)
- **Breeze-7B LLM**: [MediaTek-Research/Breeze-7B-32k-Base-v1_0](https://huggingface.co/MediaTek-Research/Breeze-7B-32k-Base-v1_0) (loaded automatically)

---

## Installation

### System Requirements
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA 11.7+ support
  - Minimum 8GB VRAM (for inference)
  - 16GB+ VRAM recommended (for training)
- **RAM**: 16GB+ system memory
- **Disk Space**: ~10GB for models and cache

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/Jason9339/traditional-chinese-historical-document-ocr-llm-fusion.git
cd traditional-chinese-historical-document-ocr-llm-fusion

# Install dependencies
pip install -r requirements.txt
```

**Note**: The `requirements.txt` includes PyTorch 2.0.1 with CUDA 11.7 runtime libraries. If you have a different CUDA version, you may need to install PyTorch separately.

### Troubleshooting

**CUDA Version Mismatch:**
If you encounter CUDA-related errors, verify your CUDA version:
```bash
nvidia-smi  # Check CUDA driver version
python -c "import torch; print(torch.cuda.is_available())"
```

**Dependency Conflicts:**
In a clean environment, some packages may show version warnings. These can usually be ignored if the evaluation scripts run successfully.

**transformer-engine Conflicts:**
If you see `transformer-engine` errors, uninstall it:
```bash
pip uninstall transformer-engine -y
```

**Gated Dataset Access:**
If you see "gated dataset" or "must be authenticated" errors:
```
Dataset 'ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen' is a gated dataset
```
This means you need to:
1. Request access at the [dataset page](https://huggingface.co/datasets/ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen)
2. Login with `huggingface-cli login` after approval

---

## Quickstart

### ⚠️ Important: Dataset Access

**Lo Chia-Luen Benchmark is a Gated Dataset**

Test Set 3 uses the Lo Chia-Luen historical document dataset, which contains library materials and requires access approval:

1. **Request Access**: Visit [ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen](https://huggingface.co/datasets/ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen)
2. **Click "Request Access"** and accept the terms
3. **Wait for Approval** (usually within 24 hours)
4. **Login to HuggingFace**:
   ```bash
   huggingface-cli login
   # Or set your token: export HF_TOKEN="your-token"
   ```

**Note**: Test Set 1 & 2 (synthetic datasets) are publicly accessible and do not require approval.

### Download Models and Data

```bash
# Test Set 1 & 2 are publicly accessible
# Dataset: ZihCiLin/traditional-chinese-ocr-synthetic

# Test Set 3 requires access approval (see above)
# Dataset: ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen

# Models are automatically downloaded when running evaluation scripts
```

---

## Reproduce Paper Tables

All experiments can be reproduced using the provided scripts in `experiments/`.

### Table 1: Baseline TrOCR Results

```bash
# Test Set 1 (Synthetic Random)
bash experiments/run_testset1.sh --model baseline

# Test Set 2 (Synthetic Semantic)
bash experiments/run_testset2.sh --model baseline

# Test Set 3 (Real Historical)
bash experiments/run_testset3.sh --model baseline
```

### Table 2: Finetuned TrOCR Results

```bash
# Test Set 1-3 with finetuned model
bash experiments/run_testset1.sh --model finetune
bash experiments/run_testset2.sh --model finetune
bash experiments/run_testset3.sh --model finetune
```

### Table 3: Post-Correction vs Fusion

```bash
# Comparison with LLM-based post-correction
bash experiments/run_postcorrection.sh
# Note: Requires OpenAI API key for GPT-based experiments
```

### Table 4: Lambda Ablation Study

```bash
# Fusion weight λ sensitivity (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
bash experiments/run_lambda_ablation.sh
```

---

## Project Structure

```
traditional-chinese-historical-document-ocr-llm-fusion/
├── gfd/                    # Core fusion decoding implementation
├── scripts/                # Evaluation scripts
│   ├── train/             # Training scripts
│   └── evaluate/          # Evaluation scripts
├── config_files/          # Configuration files
│   ├── model/            # Model configs
│   └── prompt/           # Prompt templates
├── experiments/           # Experiment scripts for paper reproduction
├── models/                # Tokenizer files
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Training

To finetune TrOCR on Lo Chia-Luen dataset:

```bash
python scripts/train/finetune_trocr.py \
  --dataset ZihCiLin/traditional-chinese-historical-ocr-lo-chia-luen \
  --output_dir models/trocr_finetuned \
  --num_train_epochs 10
```

---

## Citation

Citation will be updated after acceptance.

---

## License

- **Code**: MIT License
- **Synthetic Dataset**: CC BY 4.0
- **Lo Chia-Luen Benchmark**: CC BY-NC 4.0 (non-commercial use only)

---

## Acknowledgments

This work is based on the Lo Chia-Luen Manuscripts from the Special Collection Center of NCCU Libraries and leverages the [Generative Fusion Decoding](https://github.com/itsnamgyu/generative-fusion-decoding) framework.

---

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Zih-Ci Lin: 111703004@g.nccu.edu.tw
