# Zero-Shot Voice Cloning Setup Guide

This guide explains how to enable and use zero-shot voice cloning in Orpheus FastAPI.

## Prerequisites

1. **Hardware Requirements**:
   - **GPU (Recommended)**: 6-8 GB VRAM for bfloat16 inference
   - **CPU (Slower)**: 12-16 GB RAM for float32 inference

2. **Dependencies**:
   ```bash
   pip install transformers accelerate
   ```

## Setup

1. **Enable Model Inference**:
   Edit your `.env` file and add:
   ```env
   ORPHEUS_ENABLE_MODEL_INFERENCE=true
   ORPHEUS_MODEL_HF=canopylabs/orpheus-3b-0.1-pretrained
   ```

2. **Start the Server**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 5005
   ```

   On first startup with model inference enabled, the server will:
   - Download the Orpheus model (~6GB)
   - Load it into memory
   - This may take a few minutes

3. **Verify Setup**:
   Check the server logs for:
   ```
   ✅ Model inference initialized successfully
   ```

## Usage

### API Endpoint

**POST** `/v1/audio/speech/zero-shot`

```bash
curl -X POST http://localhost:5005/v1/audio/speech/zero-shot \
  -F "text=Hello, this is a test of zero-shot voice cloning!" \
  -F "voice_transcript=This is my voice" \
  -F "voice_audio=@my_voice.wav" \
  --output cloned_speech.wav
```

### Python Example

```python
import requests

url = "http://localhost:5005/v1/audio/speech/zero-shot"

files = {
    'voice_audio': ('voice.wav', open('my_voice.wav', 'rb'), 'audio/wav')
}
data = {
    'text': 'Hello, this is a test of zero-shot voice cloning!',
    'voice_transcript': 'This is my voice'
}

response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    with open('cloned_speech.wav', 'wb') as f:
        f.write(response.content)
    print("Success!")
else:
    print(f"Error: {response.json()}")
```

## How It Works

1. **Voice Analysis**: Your voice sample is encoded into tokens using SNAC
2. **Prompt Creation**: A special prompt is created with your voice tokens and transcript
3. **Generation**: The Orpheus model generates speech matching your voice
4. **Decoding**: Generated tokens are converted back to audio

## Tips for Best Results

1. **Voice Sample Quality**:
   - Use a clear recording without background noise
   - 3-10 seconds of natural speech
   - Consistent volume and tone

2. **Voice Transcript**:
   - Must accurately match what's said in the audio
   - Include punctuation for better results

3. **Performance**:
   - First request will be slower (model loading)
   - GPU inference is much faster than CPU
   - Longer texts take proportionally more time

## Troubleshooting

### "Zero-shot voice cloning is not available"
- Ensure `ORPHEUS_ENABLE_MODEL_INFERENCE=true` in `.env`
- Restart the server after changing settings
- Check server logs for initialization errors

### Out of Memory Errors
- Reduce other memory usage
- Use a GPU with sufficient VRAM
- Consider using a smaller model variant

### Slow Generation
- GPU acceleration is highly recommended
- CPU inference is 10-20x slower
- Consider shorter text inputs for testing

## Architecture

The implementation uses a hybrid approach:
- **Standard voices**: Use the existing API backend (fast, efficient)
- **Zero-shot cloning**: Load model directly (more memory, but enables advanced features)

This allows you to use both modes in the same server, optimizing for different use cases.