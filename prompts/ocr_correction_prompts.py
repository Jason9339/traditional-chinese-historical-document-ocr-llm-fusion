"""
OCR Correction Prompt Templates
Prompt templates for Pipeline OCR post-correction method

Includes three strategies:
- Baseline: No candidates provided, relies purely on LLM semantic understanding
- Strict: Forces selection from Top-3 candidates
- Loose: References Top-3 but does not enforce restrictions
"""

from typing import List, Dict, Any


# ==================================================
# Baseline Prompt Template (no candidates provided)
# Tests LLM's correction ability based purely on semantic understanding
# ==================================================

BASELINE_PROMPT_TEMPLATE = """以下是OCR辨識結果，可能有錯誤。請輸出校正後的文字。

{ocr_text}

校正後："""


# ==================================================
# Strict Prompt Template (simplified version for Base models)
# Forces selection from Top-3 candidates
# ==================================================

STRICT_PROMPT_TEMPLATE = """修正以下OCR文字錯誤（可從候選字中選擇更正確的字）。只輸出修正後文字。

{ocr_text}

候選：{candidates_simple}

修正後："""


# ==================================================
# Loose Prompt Template
# References Top-3 but allows free correction
# ==================================================

LOOSE_PROMPT_TEMPLATE = """你是專業的繁體中文OCR校對助手。

【任務】
根據OCR識別結果和候選字參考，校正可能的錯誤。

【指導原則】
1. 參考提供的 Top-3 候選字，但不強制限制
2. 如果 Top-3 都不正確，可以選擇其他更合適的字
3. 必須考慮上下文語義，確保文意通順
4. 保持原文的語氣和風格
5. 對於不確定的位置，保持原字不變

【OCR識別結果】
{ocr_text}

【每個位置的候選字參考】
{candidates_detail}

【輸出格式要求】
請以 JSON 格式輸出，必須包含以下欄位：

{{
  "corrected_text": "校正後的完整文字（字串）",
  "corrections": [
    {{
      "position": 位置索引（整數，從0開始）,
      "original": "原始字（字串）",
      "corrected": "校正後的字（字串）",
      "source": "top1" 或 "top2" 或 "top3" 或 "llm_suggestion"（字串）
    }}
  ]
}}

【注意事項】
- corrections 陣列只包含有修改的位置
- 如果沒有任何修正，corrections 為空陣列
- source 可以是 "top1"、"top2"、"top3" 或 "llm_suggestion"
- llm_suggestion 表示候選字都不合適，由你自行建議

請嚴格按照上述 JSON 格式輸出："""


# ==================================================
# Helper Functions
# ==================================================

def format_candidates_simple(candidates: List[Dict[str, Any]], max_positions: int = None) -> str:
    """
    Format candidates in minimal format (for Base models)

    Args:
        candidates: List of candidate characters
        max_positions: Maximum number of positions to display

    Returns:
        Simplified format: 位置0[亞/壘/弤] 位置1[美/奚/羔]
    """
    if max_positions is not None:
        candidates = candidates[:max_positions]

    parts = []
    for cand in candidates:
        pos = cand['position']
        if 'top_k' in cand and len(cand['top_k']) > 0:
            chars = '/'.join([item.get('char', '') for item in cand['top_k'][:3]])
            parts.append(f"位置{pos}[{chars}]")

    return ' '.join(parts)


def format_candidates_detail(candidates: List[Dict[str, Any]], max_positions: int = None) -> str:
    """
    Format candidate information into readable text (detailed version)

    Args:
        candidates: List of candidates, each dict contains position, char, top_k
        max_positions: Maximum positions to display (to avoid overly long prompts)

    Returns:
        Formatted candidate list text
    """
    lines = []

    # Limit number of positions to display
    if max_positions is not None:
        candidates_to_show = candidates[:max_positions]
        if len(candidates) > max_positions:
            lines.append(f"（顯示前 {max_positions} 個位置，共 {len(candidates)} 個位置）\n")
    else:
        candidates_to_show = candidates

    for cand in candidates_to_show:
        pos = cand['position']
        char = cand.get('char', '')

        # Format header
        lines.append(f"位置 {pos}: 當前='{char}'")

        # Format Top-K
        if 'top_k' in cand and len(cand['top_k']) > 0:
            for item in cand['top_k']:
                rank = item.get('rank', 0)
                char_item = item.get('char', '')
                prob = item.get('prob', 0.0)
                lines.append(f"  {rank}. '{char_item}' (機率: {prob:.2%})")
        else:
            lines.append("  （無候選字資訊）")

        lines.append("")  # Empty line separator

    return "\n".join(lines)


def build_baseline_prompt(ocr_text: str, candidates: List[Dict[str, Any]] = None, max_positions: int = None) -> str:
    """
    Build baseline prompt (without using candidates)

    Args:
        ocr_text: OCR recognition result text
        candidates: Candidate list (ignored, kept for interface consistency)
        max_positions: Maximum positions to display (ignored, kept for interface consistency)

    Returns:
        Complete prompt string
    """
    prompt = BASELINE_PROMPT_TEMPLATE.format(
        ocr_text=ocr_text
    )

    return prompt


def build_strict_prompt(ocr_text: str, candidates: List[Dict[str, Any]], max_positions: int = 100) -> str:
    """
    Build strict prompt (simplified version for Base models)

    Args:
        ocr_text: OCR recognition result text
        candidates: Candidate list
        max_positions: Maximum positions to display

    Returns:
        Complete prompt string
    """
    # Use simplified format
    candidates_simple = format_candidates_simple(candidates, max_positions)

    prompt = STRICT_PROMPT_TEMPLATE.format(
        ocr_text=ocr_text,
        candidates_simple=candidates_simple
    )

    return prompt


def build_loose_prompt(ocr_text: str, candidates: List[Dict[str, Any]], max_positions: int = 100) -> str:
    """
    Build loose prompt

    Args:
        ocr_text: OCR recognition result text
        candidates: Candidate list
        max_positions: Maximum positions to display

    Returns:
        Complete prompt string
    """
    candidates_detail = format_candidates_detail(candidates, max_positions)

    prompt = LOOSE_PROMPT_TEMPLATE.format(
        ocr_text=ocr_text,
        candidates_detail=candidates_detail
    )

    return prompt


# ==================================================
# Test Examples
# ==================================================

if __name__ == "__main__":
    # Test example
    test_ocr_text = "亞荧會議之目的"

    test_candidates = [
        {
            "position": 0,
            "char": "亞",
            "top_k": [
                {"rank": 1, "char": "亞", "prob": 0.92},
                {"rank": 2, "char": "垔", "prob": 0.05},
                {"rank": 3, "char": "西", "prob": 0.02}
            ]
        },
        {
            "position": 1,
            "char": "荧",
            "top_k": [
                {"rank": 1, "char": "荧", "prob": 0.45},
                {"rank": 2, "char": "美", "prob": 0.42},
                {"rank": 3, "char": "菱", "prob": 0.08}
            ]
        },
        {
            "position": 2,
            "char": "會",
            "top_k": [
                {"rank": 1, "char": "會", "prob": 0.98},
                {"rank": 2, "char": "曾", "prob": 0.01},
                {"rank": 3, "char": "晝", "prob": 0.005}
            ]
        }
    ]

    print("=" * 70)
    print("Strict Prompt Example")
    print("=" * 70)
    strict_prompt = build_strict_prompt(test_ocr_text, test_candidates)
    print(strict_prompt)

    print("\n\n")
    print("=" * 70)
    print("Loose Prompt Example")
    print("=" * 70)
    loose_prompt = build_loose_prompt(test_ocr_text, test_candidates)
    print(loose_prompt)
