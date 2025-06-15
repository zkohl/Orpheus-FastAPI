# Fix GPU Detection for RTX 6000 Blackwell

Your RTX 6000 Blackwell with 96GB VRAM is not being detected by PyTorch. This is because you likely have the CPU-only version of PyTorch installed.

## Quick Fix

1. **Uninstall current PyTorch**:
```bash
pip uninstall torch torchvision torchaudio -y
```

2. **Install PyTorch with CUDA support**:

For CUDA 12.1 (latest, recommended for Blackwell):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8 (if you have older CUDA drivers):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Verify GPU detection**:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

You should see:
```
CUDA available: True
GPU: NVIDIA RTX 6000 Ada Generation  # or similar
```

## Check CUDA Installation

If PyTorch still doesn't detect your GPU:

1. **Check NVIDIA drivers**:
```bash
nvidia-smi
```

2. **Check CUDA version**:
```bash
nvcc --version
```

## After Fixing

Once PyTorch detects your GPU:
- The model will load in bfloat16 precision
- Generation will be 100x+ faster (seconds instead of minutes)
- You'll be able to use the full 8192 token limit
- Zero-shot voice cloning will work in real-time

## Expected Performance with RTX 6000

With your 96GB VRAM GPU:
- Model loading: 5-10 seconds
- Zero-shot generation: 5-15 seconds
- Can handle multiple concurrent requests
- Can load much larger models if needed

Your GPU is one of the best available for AI inference!