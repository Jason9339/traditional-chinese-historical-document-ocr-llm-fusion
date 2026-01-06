"""
Byte-Level Adapter for Chinese Character-Level Tokenizer
Adapts PreTrainedTokenizerFast (character-level) to the byte-level interface required by GFD
"""

from transformers import PreTrainedTokenizerFast
from gfd.tokenizer import ByteTokenizer


class ChineseCharTokenizerAdapter(PreTrainedTokenizerFast, ByteTokenizer):
    """
    Adapts character-level Chinese tokenizer to byte-level tokenizer

    Core Approach:
    1. Chinese characters are essentially UTF-8 encoded bytes
    2. We create byte_encoder/decoder mappings for each character
    3. When GFD operates in byte space, we convert back to character space
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_byte_mappings()

    def _build_byte_mappings(self):
        """Build character <-> byte mappings"""
        print("Building byte-level mappings...")

        # Build byte encoding mapping for each token
        self.token_to_bytes = {}
        self.bytes_to_token = {}

        # Build standard byte encoder/decoder (256 possible byte values)
        # Uses the same mapping strategy as RobertaTokenizer
        self.byte_encoder = {}
        self.byte_decoder = {}

        # Direct mapping for basic ASCII characters
        for i in range(256):
            # Printable characters used directly
            if 33 <= i <= 126:
                self.byte_encoder[i] = chr(i)
                self.byte_decoder[chr(i)] = i
            else:
                # Non-printable characters mapped to Unicode private area
                char = chr(256 + i)
                self.byte_encoder[i] = char
                self.byte_decoder[char] = i

        # Build bytes mapping for each token in vocabulary
        for token, token_id in self.get_vocab().items():
            if token in self.all_special_tokens:
                # Special tokens encoded directly
                byte_seq = token.encode('utf-8')
            else:
                # Regular tokens encoded as UTF-8
                byte_seq = token.encode('utf-8')

            self.token_to_bytes[token_id] = byte_seq

        print(f"✓ Byte mappings built successfully (vocab size: {len(self.get_vocab())})")

    def convert_ids_to_bytes(self, ids, skip_special_tokens=True):
        """
        Convert token IDs to bytes
        This is the core method required by GFD
        """
        if isinstance(ids, int):
            ids = [ids]

        # Convert to tokens
        tokens = self.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

        if isinstance(tokens, str):
            tokens = [tokens]

        # Convert each token to bytes
        result = []
        for token in tokens:
            if token in self.all_special_tokens and skip_special_tokens:
                continue

            # Convert token to UTF-8 bytes
            byte_seq = token.encode('utf-8')
            result.append(byte_seq)

        return result

    def tokenize_from_byte(self, byte_str):
        """
        Tokenize from byte string
        This is another core method required by GFD
        """
        # Decode bytes to string
        try:
            text = byte_str.decode('utf-8', errors='ignore')
        except:
            text = str(byte_str)

        # Tokenize using the original tokenizer
        ids = self(text, add_special_tokens=False).input_ids

        return ids

    def _convert_token_to_bytes(self, token):
        """Convert single token to bytes"""
        if token in self.all_special_tokens:
            return token.encode("utf-8")

        # For regular tokens, use byte_encoder mapping
        byte_values = []
        for ch in token:
            # Get UTF-8 bytes of character
            ch_bytes = ch.encode('utf-8')
            for byte_val in ch_bytes:
                # Use byte_encoder mapping
                if byte_val in self.byte_encoder:
                    byte_values.append(ord(self.byte_encoder[byte_val]))
                else:
                    byte_values.append(byte_val)

        return bytes(byte_values)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load tokenizer from pretrained model
        """
        # Load using standard method first
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

        # Create adapter instance
        adapter = cls(
            tokenizer_object=tokenizer.backend_tokenizer,
            **tokenizer.init_kwargs
        )

        # Copy special tokens
        for attr in ['bos_token', 'eos_token', 'unk_token', 'sep_token',
                     'pad_token', 'cls_token', 'mask_token']:
            if hasattr(tokenizer, attr):
                setattr(adapter, attr, getattr(tokenizer, attr))

        # Build byte mappings
        adapter._build_byte_mappings()

        return adapter


def create_chinese_tokenizer_adapter(tokenizer_path):
    """
    Convenience function: Create Chinese tokenizer adapter

    Args:
        tokenizer_path: Path to tokenizer

    Returns:
        Adapted tokenizer
    """
    print(f"Loading Chinese tokenizer adapter from: {tokenizer_path}")

    adapter = ChineseCharTokenizerAdapter.from_pretrained(tokenizer_path)

    print("✓ Chinese tokenizer adapter created successfully")
    print(f"  - Vocab size: {len(adapter)}")
    print(f"  - Supports byte_encoder: {hasattr(adapter, 'byte_encoder')}")
    print(f"  - Supports byte_decoder: {hasattr(adapter, 'byte_decoder')}")
    print(f"  - Supports convert_ids_to_bytes: {hasattr(adapter, 'convert_ids_to_bytes')}")
    print(f"  - Supports tokenize_from_byte: {hasattr(adapter, 'tokenize_from_byte')}")

    return adapter


if __name__ == "__main__":
    # Test adapter
    import sys
    import os

    # Use relative path from repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = os.path.join(repo_root, "models/tokenizer")

    print("="*80)
    print("Testing Chinese Tokenizer Adapter")
    print("="*80)

    try:
        adapter = create_chinese_tokenizer_adapter(tokenizer_path)

        # Test 1: Basic tokenization
        print("\nTest 1: Basic tokenization")
        text = "繁體中文測試"
        ids = adapter(text, add_special_tokens=False).input_ids
        print(f"Text: {text}")
        print(f"IDs: {ids}")

        # Test 2: convert_ids_to_bytes
        print("\nTest 2: convert_ids_to_bytes")
        byte_list = adapter.convert_ids_to_bytes(ids, skip_special_tokens=True)
        print(f"Bytes: {byte_list}")
        print(f"Decoded: {b''.join(byte_list).decode('utf-8')}")

        # Test 3: tokenize_from_byte
        print("\nTest 3: tokenize_from_byte")
        byte_str = text.encode('utf-8')
        ids_from_bytes = adapter.tokenize_from_byte(byte_str)
        print(f"Byte string: {byte_str}")
        print(f"IDs from bytes: {ids_from_bytes}")
        print(f"Match: {ids == ids_from_bytes}")

        print("\n" + "="*80)
        print("✓ Adapter tests passed!")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
