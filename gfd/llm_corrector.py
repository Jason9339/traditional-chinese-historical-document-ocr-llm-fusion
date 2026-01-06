"""
LLM Corrector for Pipeline OCR Post-Correction
Supports local LLM (Breeze-7B) and API (OpenAI/Claude)
"""

import json
import re
from typing import Dict, Any, Optional, List
import torch


class LLMCorrector:
    """
    LLM corrector for Pipeline OCR post-correction

    Supports:
    - Local LLM: Breeze-7B-Instruct
    - API: OpenAI GPT-4 / Claude 3.5 Sonnet
    - Three modes: baseline / strict / loose
    """

    def __init__(
        self,
        llm_type: str = "local",  # "local" or "api"
        llm_model_path: str = None,  # Local model path or API model name
        mode: str = "strict",  # "baseline" or "strict" or "loose"
        api_key: str = None,  # API key (if using API)
        temperature: float = 0.0,
        max_tokens: int = 2048,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize LLM corrector

        Args:
            llm_type: LLM type ("local" or "api")
            llm_model_path: Model path or API model name
            mode: Correction mode ("baseline", "strict", or "loose")
            api_key: API key (required for API mode)
            temperature: Generation temperature
            max_tokens: Maximum generation tokens
            device: Device ("cuda" or "cpu")
        """
        self.llm_type = llm_type
        self.llm_model_path = llm_model_path
        self.mode = mode
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device

        # Initialize corresponding LLM
        if llm_type == "local":
            self._init_local_llm(llm_model_path)
        elif llm_type == "api":
            self._init_api_client(llm_model_path, api_key)
        else:
            raise ValueError(f"Unknown llm_type: {llm_type}")

        print(f"✓ LLMCorrector initialization completed")
        print(f"  - Type: {llm_type}")
        print(f"  - Mode: {mode}")
        print(f"  - Temperature: {temperature}")

    def _init_local_llm(self, model_path: str):
        """Initialize local LLM (Breeze-7B)"""
        if model_path is None:
            raise ValueError("Local LLM requires model_path")

        print(f"Loading local LLM: {model_path}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.eval()

        print(f"✓ Local LLM loaded successfully")

    def _init_api_client(self, model_name: str, api_key: str):
        """Initialize API client (OpenAI/Claude)"""
        if model_name is None:
            raise ValueError("API mode requires model_name")

        if api_key is None:
            raise ValueError("API mode requires api_key")

        self.model_name = model_name

        # Determine API type based on model name
        if "gpt" in model_name.lower() or "o1" in model_name.lower():
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.api_type = "openai"
            print(f"✓ OpenAI API client initialized: {model_name}")

        elif "claude" in model_name.lower():
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.api_type = "claude"
            print(f"✓ Claude API client initialized: {model_name}")

        else:
            raise ValueError(f"Unsupported API model: {model_name}")

    def correct(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute correction

        Args:
            ocr_result: OCR result containing:
                {
                    "final_text": str,
                    "candidates": List[Dict]
                }

        Returns:
            {
                "corrected_text": str,
                "corrections": List[Dict],
                "raw_response": str,
                "prompt": str
            }
        """
        # 1. Build prompt
        prompt = self._build_prompt(ocr_result)

        # 2. Call LLM
        if self.llm_type == "local":
            response = self._call_local_llm(prompt)
        else:
            response = self._call_api(prompt)

        # 3. Parse response
        result = self._parse_response(response, ocr_result["final_text"])

        # 4. Add prompt to result
        result["prompt"] = prompt

        return result

    def _build_prompt(self, ocr_result: Dict[str, Any]) -> str:
        """Build correction prompt"""
        from prompts.ocr_correction_prompts import build_baseline_prompt, build_strict_prompt, build_loose_prompt
        import re

        ocr_text = ocr_result["final_text"]
        candidates = ocr_result["candidates"]

        # Remove all spaces from OCR text
        ocr_text_no_space = re.sub(r"\s+", "", ocr_text)

        # Select prompt template
        if self.mode == "baseline":
            prompt = build_baseline_prompt(ocr_text_no_space, candidates)
        elif self.mode == "strict":
            prompt = build_strict_prompt(ocr_text_no_space, candidates)
        elif self.mode == "loose":
            prompt = build_loose_prompt(ocr_text_no_space, candidates)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'baseline', 'strict', or 'loose'")

        return prompt

    def _call_local_llm(self, prompt: str) -> str:
        """Call local LLM"""
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]

        # Check if tokenizer has chat_template
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # If no chat template, use simple format
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response

    def _call_api(self, prompt: str) -> str:
        """Call API"""
        if self.api_type == "openai":
            # OpenAI API
            # GPT-5.x uses max_completion_tokens, older models use max_tokens
            # Removed response_format to support plain text output (strict mode no longer requires JSON)
            if "gpt-5" in self.model_name.lower():
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            return response.choices[0].message.content

        elif self.api_type == "claude":
            # Claude API
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        else:
            raise ValueError(f"Unknown API type: {self.api_type}")

    def _parse_response(self, response: str, original_text: str) -> Dict[str, Any]:
        """
        Parse LLM response to structured format (supports JSON or plain text)

        Args:
            response: Raw LLM response
            original_text: Original OCR text (for fallback)

        Returns:
            {
                "corrected_text": str,
                "corrections": List[Dict],
                "raw_response": str,
                "parse_success": bool
            }
        """
        # First try JSON parsing (maintain backward compatibility)
        try:
            result = json.loads(response)

            # Ensure required fields exist
            if "corrected_text" not in result:
                result["corrected_text"] = original_text
            if "corrections" not in result:
                result["corrections"] = []

            result["raw_response"] = response
            result["parse_success"] = True
            return result

        except json.JSONDecodeError:
            pass

        # Try to extract JSON block
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if "corrected_text" not in result:
                    result["corrected_text"] = original_text
                if "corrections" not in result:
                    result["corrections"] = []
                result["raw_response"] = response
                result["parse_success"] = True
                return result
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object
        json_pattern2 = r'\{[^{}]*"corrected_text"[^{}]*\}'
        match = re.search(json_pattern2, response, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                if "corrections" not in result:
                    result["corrections"] = []
                result["raw_response"] = response
                result["parse_success"] = True
                return result
            except json.JSONDecodeError:
                pass

        # JSON parsing failed -> Try plain text parsing
        # Clean response: remove leading/trailing whitespace, take first line only (if multiple lines)
        cleaned = response.strip()

        # If response is empty or too long (possibly nonsense), return original text
        if not cleaned or len(cleaned) > len(original_text) * 3:
            return {
                "corrected_text": original_text,
                "corrections": [],
                "raw_response": response,
                "parse_success": False,
                "error": "Response empty or too long"
            }

        # Take first line as correction result
        first_line = cleaned.split('\n')[0].strip()

        if first_line:
            return {
                "corrected_text": first_line,
                "corrections": [],  # Plain text mode cannot provide detailed correction info
                "raw_response": response,
                "parse_success": True,  # Successfully extracted text
                "parse_mode": "plain_text"
            }

        # Complete failure, return original text
        return {
            "corrected_text": original_text,
            "corrections": [],
            "raw_response": response,
            "parse_success": False,
            "error": "No valid text found in response"
        }

    def batch_correct(
        self,
        ocr_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Batch correction

        Args:
            ocr_results: List of OCR results

        Returns:
            List of correction results
        """
        results = []
        for ocr_result in ocr_results:
            try:
                result = self.correct(ocr_result)
                results.append(result)
            except Exception as e:
                # Error handling: return original text
                print(f"⚠  Correction failed: {e}")
                results.append({
                    "corrected_text": ocr_result["final_text"],
                    "corrections": [],
                    "raw_response": "",
                    "parse_success": False,
                    "error": str(e)
                })

        return results


def create_llm_corrector_from_config(
    llm_type: str,
    llm_model_path: str,
    mode: str = "strict",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048
) -> LLMCorrector:
    """
    Create LLMCorrector instance from config (convenience function)

    Args:
        llm_type: "local" or "api"
        llm_model_path: Model path or API model name
        mode: "strict" or "loose"
        api_key: API key (optional)
        temperature: Generation temperature
        max_tokens: Maximum generation tokens

    Returns:
        LLMCorrector instance
    """
    corrector = LLMCorrector(
        llm_type=llm_type,
        llm_model_path=llm_model_path,
        mode=mode,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return corrector
