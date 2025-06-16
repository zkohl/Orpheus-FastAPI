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
                
                # Try to load tokenizer with different strategies
                tokenizer_loaded = False
                
                # Strategy 1: Try loading from model name without special token handling
                try:
                    _tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        use_fast=True,  # Try fast tokenizer first
                        add_prefix_space=False
                    )
                    tokenizer_loaded = True
                    if not IS_RELOADER:
                        print("Loaded Orpheus tokenizer (fast version)")
                except Exception as e1:
                    if not IS_RELOADER:
                        print(f"Fast tokenizer failed: {e1}")
                    
                    # Strategy 2: Try slow tokenizer
                    try:
                        _tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            use_fast=False
                        )
                        tokenizer_loaded = True
                        if not IS_RELOADER:
                            print("Loaded Orpheus tokenizer (slow version)")
                    except Exception as e2:
                        if not IS_RELOADER:
                            print(f"Slow tokenizer failed: {e2}")
                        
                        # Strategy 3: Try to get tokenizer config and build manually
                        try:
                            from transformers import LlamaTokenizerFast
                            # Orpheus uses a Llama-based tokenizer
                            _tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
                            tokenizer_loaded = True
                            if not IS_RELOADER:
                                print("Loaded Orpheus tokenizer as LlamaTokenizerFast")
                        except Exception as e3:
                            if not IS_RELOADER:
                                print(f"LlamaTokenizerFast failed: {e3}")
                                print("\nWARNING: Could not load Orpheus tokenizer!")
                                print("Zero-shot voice cloning will NOT work with fallback tokenizer.")
                                print("The vocabulary mapping will be incorrect.")
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
    """Create the input IDs for zero-shot voice cloning - EXACTLY like notebook"""
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
    
    # Special tokens - matching notebook exactly
    # From notebook: 
    # start_tokens = torch.tensor([[ 128259]], dtype=torch.int64)  # SOH
    # end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)  # EOS, SOS, SOT, EOT
    # final_tokens = torch.tensor([[128258, 128262]], dtype=torch.int64)  # EOH, EOAI
    
    # Let's match the notebook's structure exactly
    start_tokens = torch.tensor([128259])  # SOH
    end_tokens = torch.tensor([128009, 128260, 128261, 128257])  # EOS, SOS, SOT, EOT
    final_tokens = torch.tensor([128258, 128262])  # EOH, EOAI
    
    # Build prompt EXACTLY like notebook:
    # zeroprompt_input_ids = torch.cat([start_tokens, input_ids, end_tokens, torch.tensor([myts]), final_tokens], dim=1)
    # Where input_ids is voice_transcript tokenized
    # Then: second_input_ids = torch.cat([zeroprompt_input_ids, start_tokens, input_ids, end_tokens], dim=1)
    # Where second input_ids is target_text tokenized
    
    # First build the zero-shot prompt part
    zeroprompt_part = torch.cat([
        start_tokens,            # SOH (128259)
        voice_transcript_ids,    # tokenized voice transcript
        end_tokens,             # EOS, SOS, SOT, EOT (128009, 128260, 128261, 128257)
        torch.tensor(voice_tokens),  # voice audio tokens
        final_tokens            # EOH, EOAI (128258, 128262)
    ])
    
    # Then add the target text part
    input_ids = torch.cat([
        zeroprompt_part,
        start_tokens,           # SOH (128259)
        target_text_ids,        # tokenized target text  
        end_tokens             # EOS, SOS, SOT, EOT (128009, 128260, 128261, 128257)
    ]).unsqueeze(0)
    
    # Move to same device as model
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    
    return input_ids

def extract_speech_tokens(generated_ids: torch.Tensor, input_length: int) -> List[int]:
    """Extract speech tokens from generated output - EXACTLY like notebook"""
    # Get the full generated sequence
    full_sequence = generated_ids[0].cpu().tolist()
    
    print(f"[DEBUG] Full sequence length: {len(full_sequence)}")
    print(f"[DEBUG] Input length: {input_length}")
    
    # Notebook code:
    # token_to_find = 128257  # EOT
    # token_to_remove = 128258  # EOH
    token_to_find = 128257
    token_to_remove = 128258
    
    # Find all occurrences of EOT - notebook: token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)
    eot_positions = [i for i, token in enumerate(full_sequence) if token == token_to_find]
    print(f"[DEBUG] EOT (128257) positions found: {eot_positions}")
    
    # Also check for EOS/EOH to stop early
    eos_eoh_positions = [i for i, token in enumerate(full_sequence) if token in [128009, 128258]]
    print(f"[DEBUG] EOS/EOH positions: {eos_eoh_positions[:10]}...")  # First 10
    
    # Notebook: if len(token_indices[1]) > 0: last_occurrence_idx = token_indices[1][-1].item()
    if eot_positions:
        last_eot_idx = eot_positions[-1]
        # Notebook: cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
        cropped_tokens = full_sequence[last_eot_idx + 1:]
        
        # Find first EOS/EOH after last EOT to stop there
        stop_positions = [i - last_eot_idx - 1 for i in eos_eoh_positions if i > last_eot_idx]
        if stop_positions and stop_positions[0] < len(cropped_tokens):
            print(f"[DEBUG] Found EOS/EOH at position {stop_positions[0]} after EOT, truncating")
            cropped_tokens = cropped_tokens[:stop_positions[0]]
        
        print(f"[DEBUG] Tokens after last EOT: {len(cropped_tokens)}")
        print(f"[DEBUG] First 20 tokens after EOT: {cropped_tokens[:20]}")
    else:
        # Notebook: else: cropped_tensor = generated_ids
        cropped_tokens = full_sequence
        print(f"[DEBUG] No EOT found, using full sequence")
    
    # Notebook: mask = cropped_tensor != token_to_remove
    # Remove EOH tokens (128258)
    filtered_tokens = [t for t in cropped_tokens if t != token_to_remove]
    print(f"[DEBUG] After removing EOH (128258): {len(filtered_tokens)} tokens")
    
    # IMPORTANT: The notebook does NOT filter by >= 128266!
    # It returns ALL tokens after EOT (minus EOH tokens)
    print(f"[DEBUG] Returning all {len(filtered_tokens)} tokens (notebook doesn't filter by >= 128266)")
    if filtered_tokens:
        print(f"[DEBUG] First 10 tokens: {filtered_tokens[:10]}")
        print(f"[DEBUG] Token range: {min(filtered_tokens)} - {max(filtered_tokens)}")
    
    return filtered_tokens

def tokens_to_speech_codes(tokens: List[int]) -> Tuple[List[int], List[int], List[int]]:
    """Convert flat token list to SNAC layer codes - EXACTLY like notebook"""
    print(f"[DEBUG] tokens_to_speech_codes: Processing {len(tokens)} tokens")
    
    # Notebook code:
    # new_length = (row_length // 7) * 7
    # trimmed_row = row[:new_length]
    # trimmed_row = [t - 128266 for t in trimmed_row]
    
    # First, ensure we have complete frames (multiples of 7)
    new_length = (len(tokens) // 7) * 7
    trimmed_tokens = tokens[:new_length]
    print(f"[DEBUG] Trimmed to {new_length} tokens (multiple of 7)")
    
    # Subtract 128266 from ALL tokens (notebook does this to ALL tokens, not just >= 128266)
    adjusted_tokens = [t - 128266 for t in trimmed_tokens]
    print(f"[DEBUG] After subtracting 128266, first 10 adjusted tokens: {adjusted_tokens[:10] if adjusted_tokens else []}")
    print(f"[DEBUG] Adjusted token range: {min(adjusted_tokens) if adjusted_tokens else 'N/A'} - {max(adjusted_tokens) if adjusted_tokens else 'N/A'}")
    
    if not adjusted_tokens:
        return [], [], []
    
    # Notebook's redistribute_codes function:
    layer_1, layer_2, layer_3 = [], [], []
    
    # Notebook: for i in range((len(code_list)+1)//7):
    # But we already trimmed, so use len(adjusted_tokens)//7
    num_frames = len(adjusted_tokens) // 7
    print(f"[DEBUG] Processing {num_frames} frames")
    
    for i in range(num_frames):
        idx = i * 7
        # Notebook code exactly:
        # layer_1.append(code_list[7*i])
        # layer_2.append(code_list[7*i+1]-4096)
        # layer_3.append(code_list[7*i+2]-(2*4096))
        # layer_3.append(code_list[7*i+3]-(3*4096))
        # layer_2.append(code_list[7*i+4]-(4*4096))
        # layer_3.append(code_list[7*i+5]-(5*4096))
        # layer_3.append(code_list[7*i+6]-(6*4096))
        
        try:
            layer_1.append(adjusted_tokens[idx])
            layer_2.append(adjusted_tokens[idx + 1] - 4096)
            layer_3.append(adjusted_tokens[idx + 2] - 2*4096)
            layer_3.append(adjusted_tokens[idx + 3] - 3*4096)
            layer_2.append(adjusted_tokens[idx + 4] - 4*4096)
            layer_3.append(adjusted_tokens[idx + 5] - 5*4096)
            layer_3.append(adjusted_tokens[idx + 6] - 6*4096)
        except IndexError as e:
            print(f"[ERROR] Index error at frame {i}, idx {idx}: {e}")
            break
    
    # Validate the codes are in valid ranges for SNAC
    print(f"[DEBUG] Layer 1 size: {len(layer_1)}, range: {min(layer_1) if layer_1 else 'N/A'} - {max(layer_1) if layer_1 else 'N/A'}")
    print(f"[DEBUG] Layer 2 size: {len(layer_2)}, range: {min(layer_2) if layer_2 else 'N/A'} - {max(layer_2) if layer_2 else 'N/A'}")
    print(f"[DEBUG] Layer 3 size: {len(layer_3)}, range: {min(layer_3) if layer_3 else 'N/A'} - {max(layer_3) if layer_3 else 'N/A'}")
    
    # SNAC expects values in range 0-4095 for each layer
    # Check for out of bounds values
    oob_l1 = [v for v in layer_1 if v < 0 or v > 4095]
    oob_l2 = [v for v in layer_2 if v < 0 or v > 4095]
    oob_l3 = [v for v in layer_3 if v < 0 or v > 4095]
    
    if oob_l1 or oob_l2 or oob_l3:
        print(f"[WARNING] Out of bounds values detected!")
        print(f"[WARNING] Layer 1 OOB: {len(oob_l1)} values")
        print(f"[WARNING] Layer 2 OOB: {len(oob_l2)} values")
        print(f"[WARNING] Layer 3 OOB: {len(oob_l3)} values")
        
        # Clamp values to valid range (0-4095)
        layer_1 = [max(0, min(4095, v)) for v in layer_1]
        layer_2 = [max(0, min(4095, v)) for v in layer_2]
        layer_3 = [max(0, min(4095, v)) for v in layer_3]
        print(f"[DEBUG] Clamped values to valid range [0, 4095]")
    
    return layer_1, layer_2, layer_3

def decode_speech_tokens_to_audio(speech_tokens: List[int]) -> Optional[bytes]:
    """Decode speech tokens to audio bytes using SNAC - like notebook"""
    print(f"[DECODE] Decoding {len(speech_tokens)} tokens to audio")
    
    if not speech_tokens:
        print(f"[DECODE] No tokens to decode")
        return None
    
    # Convert tokens to SNAC codes
    layer_1, layer_2, layer_3 = tokens_to_speech_codes(speech_tokens)
    
    if not layer_1:
        print(f"[DECODE] No layer_1 codes after processing")
        return None
    
    print(f"[DECODE] Code tensor sizes - L1: {len(layer_1)}, L2: {len(layer_2)}, L3: {len(layer_3)}")
    
    # Import SNAC model (from speechpipe)
    from .speechpipe import model as snac_model, snac_device
    
    # Create code tensors - notebook: codes = [torch.tensor(layer_1).unsqueeze(0), ...]
    try:
        codes = [
            torch.tensor([layer_1], dtype=torch.int32, device=snac_device),
            torch.tensor([layer_2], dtype=torch.int32, device=snac_device),
            torch.tensor([layer_3], dtype=torch.int32, device=snac_device)
        ]
        print(f"[DECODE] Created code tensors on device: {snac_device}")
    except Exception as e:
        print(f"[ERROR] Failed to create code tensors: {e}")
        return None
    
    # Decode with SNAC - notebook: audio_hat = snac_model.decode(codes)
    try:
        with torch.inference_mode():
            audio_hat = snac_model.decode(codes)
        print(f"[DECODE] SNAC decode successful, audio shape: {audio_hat.shape}")
    except Exception as e:
        print(f"[ERROR] SNAC decode failed: {e}")
        print(f"[ERROR] This often means token values are out of range")
        return None
    
    # Convert to audio bytes
    try:
        audio_np = audio_hat.squeeze().cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        print(f"[DECODE] Converted to {len(audio_bytes)} bytes of audio")
        return audio_bytes
    except Exception as e:
        print(f"[ERROR] Failed to convert audio to bytes: {e}")
        return None

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
    
    # Create input prompt - fix the argument order to match function signature
    input_ids = create_zero_shot_prompt(target_text, voice_tokens, voice_transcript)
    input_length = input_ids.shape[1]
    
    print(f"Input prompt length: {input_length} tokens")
    
    # Generate tokens
    start_time = time.time()
    print(f"Starting model generation (this may take a while on CPU)...")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Notebook uses max_new_tokens=990, let's match that
    max_tokens = min(max_tokens, 990)  # Notebook limit
    print(f"[MAIN] Limiting generation to {max_tokens} tokens (notebook uses 990)")
    
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
    print(f"[MAIN] Extracted {len(speech_tokens)} tokens")
    
    if not speech_tokens:
        print("[ERROR] No tokens extracted after EOT!")
        print("This means either:")
        print("- No EOT token was found in generated sequence")
        print("- No tokens were generated after EOT")
        print("- All tokens were EOH (128258) and got filtered")
        return
    
    # Process tokens in chunks for streaming (like notebook processes in batches)
    chunk_size = 49  # 7 tokens * 7 frames
    total_audio_bytes = 0
    chunk_count = 0
    
    print(f"[MAIN] Processing {len(speech_tokens)} tokens in chunks of {chunk_size}")
    
    for i in range(0, len(speech_tokens), chunk_size):
        chunk = speech_tokens[i:i + chunk_size]
        print(f"[CHUNK {chunk_count+1}] Processing tokens {i} to {i+len(chunk)}")
        
        if len(chunk) >= 7:  # Need at least one complete frame
            try:
                audio_bytes = decode_speech_tokens_to_audio(chunk)
                if audio_bytes:
                    chunk_count += 1
                    total_audio_bytes += len(audio_bytes)
                    print(f"[CHUNK {chunk_count}] Generated {len(audio_bytes)} audio bytes")
                    yield audio_bytes
                else:
                    print(f"[CHUNK {chunk_count+1}] No audio bytes generated")
            except Exception as e:
                print(f"[ERROR] Failed to decode chunk {chunk_count+1}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[CHUNK] Skipping final {len(chunk)} tokens (need at least 7)")
    
    print(f"[MAIN] Total audio generated: {total_audio_bytes} bytes in {chunk_count} chunks")
    if total_audio_bytes == 0:
        print("[ERROR] No audio bytes were generated from tokens!")

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