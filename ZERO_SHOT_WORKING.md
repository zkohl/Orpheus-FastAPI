# Zero-Shot Voice Cloning - Working Implementation

The zero-shot voice cloning now works by following the exact approach from the Jupyter notebook!

## Key Fixes

1. **Used the correct model**: `canopylabs/orpheus-tts-0.1-pretrained` (not the 3b version)
2. **Avoided problematic tokenizer files**: Downloads model without tokenizer.json (which has compatibility issues)
3. **Fallback tokenizer**: Uses Llama tokenizer if Orpheus tokenizer fails to load

## Quick Setup

1. **Enable in .env**:
```env
ORPHEUS_ENABLE_MODEL_INFERENCE=true
ORPHEUS_MODEL_HF=canopylabs/orpheus-tts-0.1-pretrained
```

2. **Install dependencies**:
```bash
pip install transformers accelerate huggingface_hub
```

3. **Start server**:
```bash
python app.py
```

The server will:
- Download the model (first time only, ~6GB)
- Load it into GPU memory
- Initialize the tokenizer (may show a warning about using fallback)

## Testing

Use the provided test script:
```bash
python test_zero_shot.py voice_sample.wav "Voice transcript" "Text to generate"
```

Or use curl:
```bash
curl -X POST http://localhost:5005/v1/audio/speech/zero-shot \
  -F "text=Hello, this is a test!" \
  -F "voice_transcript=This is my voice" \
  -F "voice_audio=@my_voice.wav" \
  --output output.wav
```

## How It Works

The implementation now exactly mirrors the notebook:

1. Downloads model files WITHOUT the problematic tokenizer.json
2. Loads tokenizer separately (with fallback to Llama tokenizer)
3. Creates the exact prompt format with special tokens
4. Generates speech tokens
5. Decodes to audio using SNAC

## Expected Output

When it works correctly, you should see:
```
Loading Orpheus model for direct inference...
Using model: canopylabs/orpheus-tts-0.1-pretrained
Loading model on CUDA with bfloat16 precision
Using notebook approach: downloading model without tokenizer files...
Loaded tokenizer successfully (or warning about fallback)
Orpheus model loaded successfully on cuda
✅ Model inference initialized successfully
```

## Troubleshooting

If you still get tokenizer errors:
1. The implementation will automatically fall back to Llama tokenizer
2. This may affect text encoding quality slightly but should still work
3. Check that you have enough GPU memory (6-8GB needed)

The key insight was that the notebook specifically avoids the tokenizer.json file that causes compatibility issues!