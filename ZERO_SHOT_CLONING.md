# Zero-Shot Voice Cloning with Orpheus FastAPI

This document describes how to use the zero-shot voice cloning feature that has been added to Orpheus FastAPI.

## Overview

Zero-shot voice cloning allows you to synthesize speech in any voice by providing just a short audio sample (3-10 seconds) of the target voice. This is based on the implementation shown in the `Orpheus_0_1_Pretrain_Inference.ipynb` notebook.

## API Endpoint

### POST `/v1/audio/speech/zero-shot`

Generate speech using zero-shot voice cloning.

**Request Format:** `multipart/form-data`

**Parameters:**
- `text` (string, required): The text to synthesize
- `voice_transcript` (string, required): The transcript of the voice audio sample
- `voice_audio` (file, required): Audio file containing the voice to clone (WAV format recommended)

**Response:** Audio file (WAV format)

**Response Headers:**
- `X-Generation-Time`: Time taken to generate the audio (in seconds)
- `X-Voice-Type`: Set to "zero-shot"

## Usage Example

### Using cURL

```bash
curl -X POST http://localhost:8000/v1/audio/speech/zero-shot \
  -F "text=I finally got into the university of my dreams!" \
  -F "voice_transcript=Hello, this is my voice" \
  -F "voice_audio=@my_voice_sample.wav" \
  --output output.wav
```

### Using Python

```python
import requests

# API endpoint
url = "http://localhost:8000/v1/audio/speech/zero-shot"

# Prepare the request
files = {
    'voice_audio': ('voice.wav', open('my_voice_sample.wav', 'rb'), 'audio/wav')
}
data = {
    'text': 'I finally got into the university of my dreams!',
    'voice_transcript': 'Hello, this is my voice'
}

# Send request
response = requests.post(url, files=files, data=data)

# Save the output
if response.status_code == 200:
    with open('output.wav', 'wb') as f:
        f.write(response.content)
    print("Success! Audio saved to output.wav")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### Using the Test Script

A test script is provided for easy testing:

```bash
python test_zero_shot.py voice_sample.wav "Hello world" "Text to synthesize"
```

## Voice Sample Requirements

For best results, the voice audio sample should:

1. **Duration**: 3-10 seconds of clear speech
2. **Quality**: High-quality recording without background noise
3. **Content**: Natural speech that matches the transcript
4. **Format**: WAV format (24kHz sample rate preferred)
5. **Speech Style**: Similar to how you want the synthesized speech to sound

## How It Works

1. The voice audio is tokenized using the SNAC model
2. These tokens, along with the transcript, create a voice prompt
3. The Orpheus model uses this prompt to generate speech in the cloned voice
4. The generated tokens are decoded back to audio

## Technical Details

The implementation follows the pattern from the Jupyter notebook:

1. **Audio Tokenization**: Uses SNAC model to convert voice audio to tokens
2. **Prompt Formation**: Creates a special prompt format with voice tokens and transcript
3. **Generation**: Uses the Orpheus model to generate speech tokens
4. **Decoding**: Converts tokens back to audio using SNAC decoder

## Limitations

- Voice quality depends on the quality of the input sample
- Works best with clear, noise-free recordings
- The voice transcript should accurately match the audio
- Longer texts may take more time to generate

## Performance

- First request may be slower due to model loading
- Subsequent requests are faster due to model caching
- GPU acceleration significantly improves performance

## Error Handling

Common errors and solutions:

1. **Missing voice audio**: Ensure the audio file is included in the request
2. **Invalid audio format**: Convert to WAV format for best compatibility
3. **Model loading errors**: Check that SNAC model can be downloaded
4. **Memory errors**: Ensure sufficient GPU/CPU memory is available