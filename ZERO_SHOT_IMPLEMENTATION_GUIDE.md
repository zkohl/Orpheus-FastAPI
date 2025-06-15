# Zero-Shot Voice Cloning Implementation Guide

This guide explains how to properly implement zero-shot voice cloning for Orpheus, addressing the current limitations.

## The Problem

The current Orpheus FastAPI implementation cannot support zero-shot voice cloning because:

1. It uses an external inference API (`/v1/completions`) that only accepts text prompts
2. Zero-shot voice cloning requires passing raw token IDs to the model
3. Standard inference servers (llama.cpp, vLLM, etc.) don't support token ID inputs

## Solution: Direct Model Implementation

Here's how to implement zero-shot voice cloning by loading the model directly:

### 1. Create a New Implementation File

Create `tts_engine/zero_shot_inference.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import librosa
import numpy as np
from typing import List, Optional
import os

class ZeroShotTTS:
    def __init__(self, model_name="canopylabs/orpheus-3b-0.1-pretrained"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Load SNAC for audio encoding/decoding
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.snac_model.to(self.device)
        self.snac_model.eval()
        
    def tokenize_audio(self, audio_path: str) -> List[int]:
        """Convert audio file to token IDs"""
        # Load audio at 24kHz
        audio_array, _ = librosa.load(audio_path, sr=24000)
        
        # Convert to tensor
        waveform = torch.from_numpy(audio_array).unsqueeze(0).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32, device=self.device)
        
        # Encode with SNAC
        with torch.inference_mode():
            codes = self.snac_model.encode(waveform)
        
        # Convert to flat token sequence
        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item() + 128266)
            all_codes.append(codes[1][0][2*i].item() + 128266 + 4096)
            all_codes.append(codes[2][0][4*i].item() + 128266 + (2*4096))
            all_codes.append(codes[2][0][(4*i)+1].item() + 128266 + (3*4096))
            all_codes.append(codes[1][0][(2*i)+1].item() + 128266 + (4*4096))
            all_codes.append(codes[2][0][(4*i)+2].item() + 128266 + (5*4096))
            all_codes.append(codes[2][0][(4*i)+3].item() + 128266 + (6*4096))
            
        return all_codes
    
    def generate_zero_shot(self, 
                          target_text: str, 
                          voice_audio_path: str, 
                          voice_transcript: str,
                          temperature: float = 0.5,
                          top_p: float = 0.9) -> bytes:
        """Generate speech with zero-shot voice cloning"""
        
        # Tokenize voice audio
        voice_tokens = self.tokenize_audio(voice_audio_path)
        
        # Tokenize texts
        voice_transcript_ids = self.tokenizer(voice_transcript, return_tensors="pt").input_ids[0]
        target_text_ids = self.tokenizer(target_text, return_tensors="pt").input_ids[0]
        
        # Special tokens
        SOH = torch.tensor([128259])
        SOT = torch.tensor([128261])
        EOT = torch.tensor([128257])
        SOS = torch.tensor([128260])
        EOS = torch.tensor([128009])
        EOAI = torch.tensor([128262])
        EOH = torch.tensor([128258])
        
        # Build zero-shot prompt
        # Format: SOH SOT voice_transcript EOT SOS voice_tokens EOS EOAI SOH SOT target_text EOT
        input_ids = torch.cat([
            SOH, SOT, voice_transcript_ids, EOT, 
            SOS, torch.tensor(voice_tokens), EOS, EOAI,
            SOH, SOT, target_text_ids, EOT
        ]).unsqueeze(0).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=990,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                eos_token_id=128258,
            )
        
        # Extract generated tokens (after the last EOT)
        generated_tokens = generated_ids[0][input_ids.shape[1]:]
        
        # Convert tokens to audio
        audio = self.decode_tokens_to_audio(generated_tokens)
        
        return audio
    
    def decode_tokens_to_audio(self, tokens: torch.Tensor) -> bytes:
        """Convert generated tokens back to audio"""
        # Remove special tokens and convert to list
        tokens_list = tokens.cpu().tolist()
        
        # Find speech tokens (between SOS and EOS/EOH)
        speech_tokens = []
        in_speech = False
        
        for token in tokens_list:
            if token == 128260:  # SOS
                in_speech = True
                continue
            elif token in [128009, 128258]:  # EOS or EOH
                break
            elif in_speech and token >= 128266:
                speech_tokens.append(token - 128266)
        
        if not speech_tokens:
            return b''
        
        # Redistribute to SNAC layers
        layer_1, layer_2, layer_3 = [], [], []
        for i in range(len(speech_tokens) // 7):
            idx = i * 7
            layer_1.append(speech_tokens[idx])
            layer_2.append(speech_tokens[idx+1] - 4096)
            layer_3.append(speech_tokens[idx+2] - 2*4096)
            layer_3.append(speech_tokens[idx+3] - 3*4096)
            layer_2.append(speech_tokens[idx+4] - 4*4096)
            layer_3.append(speech_tokens[idx+5] - 5*4096)
            layer_3.append(speech_tokens[idx+6] - 6*4096)
        
        # Decode with SNAC
        codes = [
            torch.tensor([layer_1], device=self.device),
            torch.tensor([layer_2], device=self.device),
            torch.tensor([layer_3], device=self.device)
        ]
        
        with torch.inference_mode():
            audio_hat = self.snac_model.decode(codes)
        
        # Convert to bytes
        audio_np = audio_hat.squeeze().cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        return audio_int16.tobytes()
```

### 2. Update the FastAPI Endpoint

Modify `app.py` to use the direct implementation:

```python
# Add at the top
zero_shot_model = None

def get_zero_shot_model():
    global zero_shot_model
    if zero_shot_model is None:
        from tts_engine.zero_shot_inference import ZeroShotTTS
        zero_shot_model = ZeroShotTTS()
    return zero_shot_model

@app.post("/v1/audio/speech/zero-shot")
async def create_zero_shot_speech(
    text: str = Form(...),
    voice_transcript: str = Form(...),
    voice_audio: UploadFile = File(...)
):
    """Generate speech using zero-shot voice cloning (direct model implementation)"""
    
    # Save uploaded audio temporarily
    temp_audio_path = f"temp_{voice_audio.filename}"
    with open(temp_audio_path, "wb") as f:
        content = await voice_audio.read()
        f.write(content)
    
    try:
        # Get or initialize model
        model = get_zero_shot_model()
        
        # Generate audio
        audio_bytes = model.generate_zero_shot(
            target_text=text,
            voice_audio_path=temp_audio_path,
            voice_transcript=voice_transcript
        )
        
        # Create WAV file
        import wave
        output_path = f"outputs/zero_shot_{int(time.time())}.wav"
        with wave.open(output_path, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(audio_bytes)
        
        return FileResponse(output_path, media_type="audio/wav")
        
    finally:
        # Cleanup
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
```

### 3. Memory Requirements

This approach requires loading the full model in memory:
- GPU: ~6-8 GB VRAM for the model in bfloat16
- CPU: ~12-16 GB RAM for the model in float32
- Additional memory for SNAC model and processing

### 4. Alternative: Hybrid Approach

Keep using the API for standard voices, but load the model only for zero-shot:

```python
# In app.py
USE_DIRECT_MODEL_FOR_ZERO_SHOT = os.getenv("ENABLE_ZERO_SHOT", "false").lower() == "true"

@app.post("/v1/audio/speech/zero-shot")
async def create_zero_shot_speech(...):
    if not USE_DIRECT_MODEL_FOR_ZERO_SHOT:
        raise HTTPException(
            status_code=501,
            detail="Zero-shot voice cloning requires ENABLE_ZERO_SHOT=true in .env"
        )
    # ... rest of implementation
```

## Conclusion

Zero-shot voice cloning requires direct model access because it needs to pass specific token IDs that standard inference APIs don't support. The implementation above shows how to properly integrate this feature while maintaining compatibility with the existing API-based approach for standard voices.