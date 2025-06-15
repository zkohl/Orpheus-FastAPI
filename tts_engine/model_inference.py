"""
Direct model inference for Orpheus TTS
This module handles loading the model directly for zero-shot voice cloning
"""

import torch
import numpy as np
from typing import List, Optional, Generator, Tuple
import time
import os
import sys

# Helper to detect if running in Uvicorn's reloader
def is_reloader_process():
    """Check if the current process is a uvicorn reloader"""
    return (sys.argv[0].endswith('_continuation.py') or 
            os.environ.get('UVICORN_STARTED') == 'true')

IS_RELOADER = is_reloader_process()

# Global model instances
_orpheus_model = None
_tokenizer = None
_model_loaded = False
_use_simple_tokenizer = False  # Flag to use simple tokenization

# Llama tokenizer as fallback
_llama_tokenizer = None

def get_llama_tokenizer():
    """Get a Llama tokenizer as fallback"""
    global _llama_tokenizer
    if _llama_tokenizer is None:
        from transformers import LlamaTokenizer
        # Try to use a known working Llama tokenizer
        try:
            _llama_tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
        except:
            # If that fails, create a basic one
            _llama_tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    return _llama_tokenizer

def simple_tokenize_text(text: str) -> torch.Tensor:
    """Simple text tokenization when full tokenizer is not available"""
    # Use Llama tokenizer as fallback
    tokenizer = get_llama_tokenizer()
    return tokenizer(text, return_tensors="pt").input_ids[0]

def get_orpheus_model():
    """Get or initialize the Orpheus model for direct inference"""
    global _orpheus_model, _tokenizer, _model_loaded, _use_simple_tokenizer
    
    if not _model_loaded:
        try:
            if not IS_RELOADER:
                print("Loading Orpheus model for direct inference...")
            
            from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
            from huggingface_hub import snapshot_download
            
            # Use the correct model name from the notebook
            model_name = os.environ.get("ORPHEUS_MODEL_HF", "canopylabs/orpheus-tts-0.1-pretrained")
            
            if not IS_RELOADER:
                print(f"Using model: {model_name}")
            
            # Determine device and dtype first
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.bfloat16
                if not IS_RELOADER:
                    print(f"Loading model on CUDA with bfloat16 precision")
                    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            else:
                device = "cpu"
                dtype = torch.float32
                if not IS_RELOADER:
                    print(f"Loading model on CPU with float32 precision")
                    print("WARNING: CPU inference will be slow. GPU recommended for production use.")
                    print("\nGPU NOT DETECTED! This is likely because:")
                    print("1. You have the CPU-only version of PyTorch installed")
                    print("2. Run: pip uninstall torch torchvision torchaudio -y")
                    print("3. Then: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                    print("\nYour system has an RTX 6000 - it should be detected!")
            
            # Handle Orpheus models specially (like the notebook)
            if "orpheus" in model_name.lower():
                if not IS_RELOADER:
                    print("Using notebook approach: downloading model without tokenizer files...")
                
                # Download model files without tokenizer (like the notebook does)
                model_path = snapshot_download(
                    repo_id=model_name,
                    allow_patterns=[
                        "config.json",
                        "*.safetensors",
                        "model.safetensors.index.json",
                    ],
                    ignore_patterns=[
                        "optimizer.pt",
                        "pytorch_model.bin",
                        "training_args.bin",
                        "scheduler.pt",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                        "vocab.json",
                        "merges.txt",
                        "tokenizer.*"
                    ]
                )
                
                # Try to load tokenizer separately
                try:
                    # First try the standard way
                    _tokenizer = AutoTokenizer.from_pretrained(model_name)
                    if not IS_RELOADER:
                        print("Loaded tokenizer successfully")
                except:
                    # If that fails, use fallback
                    if not IS_RELOADER:
                        print("WARNING: Could not load Orpheus tokenizer, using Llama tokenizer as fallback")
                    _tokenizer = get_llama_tokenizer()
                    _use_simple_tokenizer = True
                
                # Load model from downloaded path
                # Handle rope_scaling compatibility
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name)
                
                # Fix rope_scaling for older transformers versions
                if hasattr(config, 'rope_scaling') and isinstance(config.rope_scaling, dict):
                    if 'rope_type' in config.rope_scaling and config.rope_scaling['rope_type'] == 'llama3':
                        # Convert to older format
                        if not IS_RELOADER:
                            print("Converting Llama3 rope_scaling to compatible format...")
                        config.rope_scaling = {
                            'type': 'linear',
                            'factor': config.rope_scaling.get('factor', 32.0)
                        }
                
                _orpheus_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=config,  # Use modified config
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None
                )
                
            else:
                # For non-Orpheus models, use standard loading
                _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                _orpheus_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True
                )
            
            if device == "cpu":
                _orpheus_model = _orpheus_model.to(device)
            
            _orpheus_model.eval()
            _model_loaded = True
            
            if not IS_RELOADER:
                print(f"Orpheus model loaded successfully on {device}")
            
        except Exception as e:
            print(f"ERROR: Failed to load Orpheus model: {e}")
            print("Zero-shot voice cloning will not be available.")
            _model_loaded = False
            raise
    
    return _orpheus_model, _tokenizer

def generate_with_model(
    input_ids: torch.Tensor,
    max_new_tokens: int = 990,
    temperature: float = 0.5,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1
) -> torch.Tensor:
    """Generate tokens using the loaded model"""
    model, _ = get_orpheus_model()
    
    # Add attention mask to avoid warnings
    attention_mask = torch.ones_like(input_ids)
    
    print(f"Generating with max_new_tokens={max_new_tokens}...")
    
    # For CPU, add a simple progress indicator
    if not torch.cuda.is_available():
        print("CPU generation started. Progress dots will appear (each = ~10 tokens):")
        print("[", end="", flush=True)
        
        class ProgressCallback:
            def __init__(self):
                self.count = 0
            
            def __call__(self, *args, **kwargs):
                self.count += 1
                if self.count % 10 == 0:
                    print(".", end="", flush=True)
                return False  # Don't stop generation
        
        progress = ProgressCallback()
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=128258,
                pad_token_id=128258,
                use_cache=True,
                stopping_criteria=[progress] if not torch.cuda.is_available() else None
            )
        
        if not torch.cuda.is_available():
            print("]")
    else:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=128258,
                pad_token_id=128258,
                use_cache=True
            )
    
    return generated_ids

def create_zero_shot_prompt(
    target_text: str,
    voice_tokens: List[int],
    voice_transcript: str
) -> torch.Tensor:
    """Create the input IDs for zero-shot voice cloning"""
    global _use_simple_tokenizer
    
    _, tokenizer = get_orpheus_model()
    
    # Tokenize texts
    if _use_simple_tokenizer or tokenizer is None:
        # Use fallback tokenization
        print("Using fallback tokenizer for text encoding")
        voice_transcript_ids = simple_tokenize_text(voice_transcript)
        target_text_ids = simple_tokenize_text(target_text)
    else:
        # Use the loaded tokenizer
        voice_transcript_ids = tokenizer(voice_transcript, return_tensors="pt").input_ids[0]
        target_text_ids = tokenizer(target_text, return_tensors="pt").input_ids[0]
    
    # Special tokens
    SOH = torch.tensor([128259])
    SOT = torch.tensor([128261])
    EOT = torch.tensor([128257])
    SOS = torch.tensor([128260])
    EOS = torch.tensor([128009])
    EOAI = torch.tensor([128262])
    EOH = torch.tensor([128258])
    
    # Build zero-shot prompt following the notebook pattern
    # Format: SOH SOT voice_transcript EOT SOS voice_tokens EOS EOAI SOH SOT target_text EOT
    input_ids = torch.cat([
        SOH, SOT, voice_transcript_ids, EOT,
        SOS, torch.tensor(voice_tokens), EOS, EOAI,
        SOH, SOT, target_text_ids, EOT
    ]).unsqueeze(0)
    
    # Move to same device as model
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    
    return input_ids

def extract_speech_tokens(generated_ids: torch.Tensor, input_length: int) -> List[int]:
    """Extract speech tokens from generated output"""
    # Get only the newly generated tokens
    generated_tokens = generated_ids[0][input_length:].cpu().tolist()
    
    # Find speech tokens (after SOS token)
    speech_tokens = []
    found_sos = False
    
    for token in generated_tokens:
        if token == 128260:  # SOS
            found_sos = True
            continue
        elif token in [128009, 128258, 128257]:  # EOS, EOH, EOT
            if found_sos:
                break
        elif found_sos and token >= 128266:  # Speech tokens start at 128266
            speech_tokens.append(token)
    
    return speech_tokens

def tokens_to_speech_codes(tokens: List[int]) -> Tuple[List[int], List[int], List[int]]:
    """Convert flat token list to SNAC layer codes"""
    # Remove offset and redistribute to layers
    adjusted_tokens = [t - 128266 for t in tokens if t >= 128266]
    
    # Ensure we have complete frames (multiples of 7)
    num_frames = len(adjusted_tokens) // 7
    adjusted_tokens = adjusted_tokens[:num_frames * 7]
    
    if not adjusted_tokens:
        return [], [], []
    
    layer_1, layer_2, layer_3 = [], [], []
    
    for i in range(num_frames):
        idx = i * 7
        layer_1.append(adjusted_tokens[idx])
        layer_2.append(adjusted_tokens[idx + 1] - 4096)
        layer_3.append(adjusted_tokens[idx + 2] - 2*4096)
        layer_3.append(adjusted_tokens[idx + 3] - 3*4096)
        layer_2.append(adjusted_tokens[idx + 4] - 4*4096)
        layer_3.append(adjusted_tokens[idx + 5] - 5*4096)
        layer_3.append(adjusted_tokens[idx + 6] - 6*4096)
    
    return layer_1, layer_2, layer_3

def decode_speech_tokens_to_audio(speech_tokens: List[int]) -> Optional[bytes]:
    """Decode speech tokens to audio bytes using SNAC"""
    if not speech_tokens:
        return None
    
    # Convert tokens to SNAC codes
    layer_1, layer_2, layer_3 = tokens_to_speech_codes(speech_tokens)
    
    if not layer_1:
        return None
    
    # Import SNAC model (from speechpipe)
    from .speechpipe import model as snac_model, snac_device
    
    # Create code tensors
    codes = [
        torch.tensor([layer_1], dtype=torch.int32, device=snac_device),
        torch.tensor([layer_2], dtype=torch.int32, device=snac_device),
        torch.tensor([layer_3], dtype=torch.int32, device=snac_device)
    ]
    
    # Decode with SNAC
    with torch.inference_mode():
        audio_hat = snac_model.decode(codes)
    
    # Convert to audio bytes
    audio_np = audio_hat.squeeze().cpu().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    return audio_int16.tobytes()

def generate_zero_shot_speech(
    target_text: str,
    voice_tokens: List[int],
    voice_transcript: str,
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_tokens: int = 8192
) -> Generator[bytes, None, None]:
    """Generate speech using zero-shot voice cloning with streaming"""
    
    print(f"Generating zero-shot speech for: {target_text[:50]}...")
    print(f"Voice transcript: {voice_transcript}")
    print(f"Voice tokens: {len(voice_tokens)} tokens")
    
    # Create input prompt
    input_ids = create_zero_shot_prompt(target_text, voice_tokens, voice_transcript)
    input_length = input_ids.shape[1]
    
    print(f"Input prompt length: {input_length} tokens")
    
    # Generate tokens
    start_time = time.time()
    print(f"Starting model generation (this may take a while on CPU)...")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # For CPU, use fewer tokens to avoid long waits
    if not torch.cuda.is_available():
        max_tokens = min(max_tokens, 500)  # Limit to 500 tokens on CPU
        print(f"CPU mode: Limiting generation to {max_tokens} tokens for faster response")
    
    try:
        generated_ids = generate_with_model(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        generation_time = time.time() - start_time
        print(f"Model generation completed in {generation_time:.2f}s")
        print(f"Generated shape: {generated_ids.shape}")
    except Exception as e:
        print(f"ERROR during generation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Extract speech tokens
    speech_tokens = extract_speech_tokens(generated_ids, input_length)
    print(f"Extracted {len(speech_tokens)} speech tokens")
    
    if not speech_tokens:
        print("WARNING: No speech tokens generated")
        return
    
    # Process tokens in chunks for streaming
    chunk_size = 49  # 7 tokens * 7 frames
    
    for i in range(0, len(speech_tokens), chunk_size):
        chunk = speech_tokens[i:i + chunk_size]
        if len(chunk) >= 7:  # Need at least one complete frame
            audio_bytes = decode_speech_tokens_to_audio(chunk)
            if audio_bytes:
                yield audio_bytes

def is_model_available() -> bool:
    """Check if direct model inference is available"""
    return _model_loaded

def initialize_model():
    """Pre-initialize the model (called during startup)"""
    if os.environ.get("ORPHEUS_ENABLE_MODEL_INFERENCE", "false").lower() == "true":
        try:
            get_orpheus_model()
            return True
        except Exception as e:
            print(f"Failed to initialize model: {e}")
            return False
    return False