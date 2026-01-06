"""
OCR with Top-K Candidates Generator
Candidate character generator for Pipeline OCR post-correction method

Features:
- Execute OCR inference and output final prediction
- Extract Top-K candidate characters and their probabilities for each position
- Support Beam Search
"""

from typing import Dict, List, Any, Union, Tuple
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class OCRWithCandidates:
    """
    OCR inference engine with Top-K candidate character output

    Usage:
        ocr = OCRWithCandidates(model, tokenizer, processor)
        result = ocr.predict_with_candidates(image, num_beams=5, top_k=3)

        # Result contains:
        # - final_text: Final predicted text
        # - final_ids: Final predicted token IDs
        # - candidates: Top-K candidate characters for each position
    """

    def __init__(
        self,
        model: VisionEncoderDecoderModel,
        tokenizer,
        processor: TrOCRProcessor = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize OCR candidate generator

        Args:
            model: TrOCR model (VisionEncoderDecoderModel)
            tokenizer: Tokenizer (supports decode/encode)
            processor: TrOCR Processor (for image preprocessing, optional)
            device: Device ("cuda" or "cpu")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device

        # Ensure model is on the correct device
        self.model.to(self.device)
        self.model.eval()

        print(f"✓ OCRWithCandidates initialization completed (device: {self.device})")

    def predict_with_candidates(
        self,
        image: Union[Image.Image, torch.Tensor],
        num_beams: int = 5,
        top_k: int = 3,
        max_length: int = 128,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Execute OCR inference and return Top-K candidates

        Args:
            image: PIL Image or preprocessed tensor
            num_beams: Number of beams for beam search
            top_k: Top-K candidates for each position
            max_length: Maximum generation length
            return_attention: Whether to return attention weights

        Returns:
            {
                "final_text": str,              # Final predicted text
                "final_ids": List[int],         # Final predicted token IDs
                "candidates": List[Dict],       # Top-K candidates for each position
                "beam_scores": List[float],     # (Optional) beam scores
                "attention_weights": Tensor     # (Optional) attention weights
            }
        """
        # 1. Image preprocessing
        pixel_values = self._preprocess_image(image)

        # 2. Model inference (with scores)
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                num_beams=num_beams,
                max_length=max_length,
                output_scores=True,
                return_dict_in_generate=True,
                output_attentions=return_attention,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                decoder_start_token_id=self.tokenizer.eos_token_id  # TrOCR uses EOS as start token
            )

        # 3. Decode final result
        final_ids = outputs.sequences[0].tolist()
        final_text = self.tokenizer.decode(final_ids, skip_special_tokens=True)

        # 4. Extract Top-K candidates for each position
        candidates = self._extract_top_k_candidates(
            outputs.scores,
            final_ids,
            top_k=top_k
        )

        # 5. Build return result
        result = {
            "final_text": final_text,
            "final_ids": final_ids,
            "candidates": candidates,
        }

        # Optional additional information
        if hasattr(outputs, 'sequences_scores'):
            result["beam_scores"] = outputs.sequences_scores.tolist()

        if return_attention and hasattr(outputs, 'attentions'):
            result["attention_weights"] = outputs.attentions

        return result

    def _preprocess_image(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess image (using same preprocessing as training)

        Args:
            image: PIL Image or preprocessed tensor

        Returns:
            pixel_values: Tensor of shape (1, C, H, W)
        """
        if isinstance(image, torch.Tensor):
            # Already a tensor, ensure it's on correct device and dtype
            pixel_values = image.to(self.device)
        else:
            # PIL Image, use same OCRImageTransform as training
            try:
                # Try to use training-time OCRImageTransform
                import sys
                import os
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                from image_preprocessing import OCRImageTransform

                transform = OCRImageTransform(
                    target_size=(384, 384),
                    auto_rotate_vertical=True,  # Auto-rotate vertical text
                    normalize=True
                )
                pixel_values = transform(image).unsqueeze(0).to(self.device)

            except Exception as e:
                print(f"⚠  Unable to use OCRImageTransform, using fallback preprocessing: {e}")
                # Fallback: use simple preprocessing
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                pixel_values = transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        # Ensure dtype matches model
        model_dtype = next(self.model.parameters()).dtype
        if pixel_values.dtype != model_dtype:
            pixel_values = pixel_values.to(model_dtype)

        return pixel_values

    def _extract_top_k_candidates(
        self,
        scores: Tuple[torch.Tensor, ...],
        final_ids: List[int],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Extract Top-K candidates for each position from scores

        Args:
            scores: Tuple of tensors, each tensor shape = (batch_size, vocab_size)
                   Length = number of generated tokens
            final_ids: Final selected token IDs
            top_k: Top-K candidates

        Returns:
            List of candidate dicts, each dict contains:
            {
                "position": int,        # Position index (starting from 0)
                "char": str,            # Final selected character
                "token_id": int,        # Final selected token ID
                "top_k": [              # Top-K candidates
                    {
                        "char": str,
                        "token_id": int,
                        "prob": float,
                        "logit": float,
                        "rank": int
                    },
                    ...
                ]
            }
        """
        candidates = []

        # Filter out special tokens (BOS/EOS/PAD)
        special_tokens = {
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else None,
        }
        special_tokens.discard(None)

        for pos, score_tensor in enumerate(scores):
            # score_tensor shape: (batch_size=1, vocab_size)
            # Take first batch
            logits = score_tensor[0]  # shape: (vocab_size,)

            # Calculate probability distribution
            probs = torch.softmax(logits, dim=-1)

            # Get Top-K
            top_k_probs, top_k_ids = torch.topk(probs, k=min(top_k, len(probs)))

            # Get corresponding logits
            top_k_logits = logits[top_k_ids]

            # Convert to characters
            top_k_items = []
            for rank, (token_id, prob, logit) in enumerate(zip(top_k_ids, top_k_probs, top_k_logits), 1):
                token_id_int = token_id.item()

                # Decode character
                char = self.tokenizer.decode([token_id_int], skip_special_tokens=False)

                top_k_items.append({
                    "char": char,
                    "token_id": token_id_int,
                    "prob": prob.item(),
                    "logit": logit.item(),
                    "rank": rank
                })

            # Final selected character (from final_ids)
            # Note: final_ids contains all tokens (including decoder_start_token/BOS)
            # scores[0] corresponds to final_ids[1] (skip decoder_start_token)
            # Therefore scores[pos] corresponds to final_ids[pos + 1]
            if pos + 1 < len(final_ids):
                final_token_id = final_ids[pos + 1]  # Fix: +1 offset
                final_char = self.tokenizer.decode([final_token_id], skip_special_tokens=False)
            else:
                # Out of range, use Top-1
                final_token_id = top_k_items[0]["token_id"]
                final_char = top_k_items[0]["char"]

            candidates.append({
                "position": pos,
                "char": final_char,
                "token_id": final_token_id,
                "top_k": top_k_items
            })

        return candidates

    def batch_predict_with_candidates(
        self,
        images: List[Image.Image],
        num_beams: int = 5,
        top_k: int = 3,
        max_length: int = 128
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction (multiple images)

        Args:
            images: List of PIL Images
            num_beams: Number of beams for beam search
            top_k: Top-K candidates for each position
            max_length: Maximum generation length

        Returns:
            List of prediction results
        """
        results = []
        for image in images:
            result = self.predict_with_candidates(
                image=image,
                num_beams=num_beams,
                top_k=top_k,
                max_length=max_length
            )
            results.append(result)

        return results


def create_ocr_with_candidates_from_config(
    checkpoint_path: str,
    tokenizer_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype = None
) -> OCRWithCandidates:
    """
    Create OCRWithCandidates instance from config (convenience function)

    Args:
        checkpoint_path: OCR model checkpoint path
        tokenizer_path: Tokenizer path
        device: Device
        torch_dtype: Model data type (e.g., torch.float16)

    Returns:
        OCRWithCandidates instance
    """
    from gfd.tokenizer_chinese_adapter import ChineseCharTokenizerAdapter

    # Load processor
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
        print("✓ TrOCR Processor loaded successfully")
    except Exception as e:
        print(f"⚠  Unable to load processor: {e}")
        processor = None

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = ChineseCharTokenizerAdapter.from_pretrained(tokenizer_path)
    print("✓ Tokenizer loaded successfully")

    # Load model
    print(f"Loading OCR model: {checkpoint_path}")
    if torch_dtype is not None:
        model = VisionEncoderDecoderModel.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            use_safetensors=True
        )
    else:
        model = VisionEncoderDecoderModel.from_pretrained(
            checkpoint_path,
            use_safetensors=True
        )

    # Set model token IDs
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    print("✓ OCR model loaded successfully")

    # Create OCRWithCandidates
    ocr_with_candidates = OCRWithCandidates(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        device=device
    )

    return ocr_with_candidates
