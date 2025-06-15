# Zero-Shot Voice Cloning on CPU - Performance Warning

## The Issue

You're running the Orpheus 3B model on CPU, which is causing the generation to be extremely slow. The logs show:
- "CPU only (No CUDA GPU detected)"
- The model loaded successfully but generation is taking a very long time

## Expected Performance

- **GPU (CUDA)**: 10-30 seconds for generation
- **CPU**: 5-20+ MINUTES for generation (100x slower!)

## Solutions

### 1. Use GPU (Recommended)
If you have an NVIDIA GPU:
- Install CUDA drivers
- Install PyTorch with CUDA support:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

### 2. For CPU-Only Systems

The model has been updated to:
- Limit generation to 500 tokens on CPU (instead of 8192)
- Add progress logging
- Use KV cache for faster generation

### 3. Use Smaller Model (Future)
Consider using a smaller model designed for CPU inference.

### 4. Use the API Mode
The standard TTS using the GGUF model through llama.cpp is much faster on CPU:
```bash
# This will be fast even on CPU
curl -X POST http://localhost:5005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "tara"}'
```

## What's Happening

When you see:
```
Starting model generation (this may take a while on CPU)...
Device: CPU
CPU mode: Limiting generation to 500 tokens for faster response
Generating with max_new_tokens=500...
```

The model IS running, but each token takes ~1-2 seconds to generate on CPU. With 500 tokens, expect 8-15 minutes of waiting.

## Recommendation

For testing zero-shot voice cloning:
1. Use a GPU-enabled system
2. Or use Google Colab with GPU runtime
3. Or be prepared to wait 10-20 minutes for CPU generation

The regular API-based TTS remains fast because it uses optimized GGUF models with llama.cpp.