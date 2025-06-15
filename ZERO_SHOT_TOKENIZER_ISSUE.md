# Zero-Shot Voice Cloning - Tokenizer Compatibility Issue

## Problem

The current implementation encounters an error when loading the Orpheus model tokenizer:
```
ERROR: Failed to load Orpheus model: data did not match any variant of untagged enum ModelWrapper at line 1509159 column 3
```

This error occurs because the tokenizer format used by the `canopylabs/orpheus-3b-0.1-pretrained` model is incompatible with the current version of the transformers library.

## Root Cause

The Orpheus model uses a custom tokenizer format that may have been created with an older or modified version of the tokenizers library. The tokenizer.json file contains structures that the current transformers library cannot parse.

## Workarounds

### Option 1: Use the GGUF Model Directly

Since the standard TTS works with GGUF models through the API, you could modify the implementation to:
1. Use the API for standard inference (already working)
2. For zero-shot, implement a custom solution that works with GGUF models

### Option 2: Use a Different Model

Try using a different model that has compatible tokenizers:
```env
ORPHEUS_MODEL_HF=meta-llama/Llama-2-7b-hf  # Example - would need TTS fine-tuning
```

### Option 3: Custom Tokenizer Implementation

Implement a custom tokenizer that bypasses the problematic tokenizer.json file:

```python
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

# Load tokenizer manually
tokenizer = Tokenizer.from_file("path/to/tokenizer.json")
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    # Add other special tokens as needed
)
```

### Option 4: Use Older Transformers Version

The model might work with an older version of transformers:
```bash
pip install transformers==4.30.0
```

## Recommended Solution

For production use, the best approach would be to:

1. **Contact the Model Authors**: Reach out to Canopy Labs for an updated model/tokenizer
2. **Use API Mode**: Continue using the API-based inference which works well
3. **Alternative Implementation**: Implement zero-shot using the GGUF model directly with llama.cpp modifications

## Technical Details

The issue appears to be in the tokenizer.json file structure, specifically:
- The file contains an "untagged enum ModelWrapper" that doesn't match any known format
- This might be due to custom modifications or version incompatibility
- The model itself might load fine, but without a working tokenizer, text processing is impossible

## Next Steps

1. Check if there's an updated version of the model on Hugging Face
2. Try loading just the model weights without the tokenizer
3. Implement a custom tokenizer based on the special tokens we know
4. Consider using a different approach for zero-shot voice cloning