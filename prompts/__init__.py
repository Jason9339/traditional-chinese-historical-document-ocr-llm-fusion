"""
Prompt Templates for Pipeline OCR Correction
"""

from .ocr_correction_prompts import (
    STRICT_PROMPT_TEMPLATE,
    LOOSE_PROMPT_TEMPLATE,
    build_strict_prompt,
    build_loose_prompt
)

__all__ = [
    'STRICT_PROMPT_TEMPLATE',
    'LOOSE_PROMPT_TEMPLATE',
    'build_strict_prompt',
    'build_loose_prompt'
]
