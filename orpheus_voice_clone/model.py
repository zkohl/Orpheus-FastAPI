"""Main voice cloning model class."""

import os
import time
from typing import List, Optional, Union
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, scan_cache_dir
from snac import SNAC

from .config import ModelConfig
from .audio_utils import load_audio, save_audio_to_wav, validate_audio_duration
from .tokenizers import AudioTokenizer, prepare_prompt_tokens, extract_generated_audio_tokens


class VoiceCloneModel:
    """
    Production-ready voice cloning model using Orpheus 3B.
    
    This class manages model initialization and provides a clean interface
    for voice cloning inference. It's designed to be instantiated once
    and reused for multiple inference calls in a server environment.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the voice cloning model.
        
        Args:
            config: Model configuration. If None, uses default config.
        """
        self.config = config or ModelConfig()
        self._initialized = False
        
        # Model components (lazy loaded)
        self._text_model = None
        self._snac_model = None
        self._tokenizer = None
        self._audio_tokenizer = None
        
        # Cache directory for models
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        
        print(f"Initialized VoiceCloneModel on device: {self.config.device}")
        print(f"Model cache directory: {self.cache_dir}")
    
    def _ensure_initialized(self):
        """Ensure models are loaded (lazy initialization)."""
        if not self._initialized:
            self._load_models()
    
    def _check_model_cached(self, repo_id: str) -> bool:
        """
        Check if a model is already cached locally.
        
        Args:
            repo_id: HuggingFace repository ID
            
        Returns:
            True if model is cached, False otherwise
        """
        try:
            # Use HuggingFace's cache scanning to check if model exists
            cache_info = scan_cache_dir(self.cache_dir)
            for repo in cache_info.repos:
                if repo.repo_id == repo_id and repo.size_on_disk > 0:
                    print(f"✓ Model {repo_id} already cached ({repo.size_on_disk_str})")
                    return True
            return False
        except Exception as e:
            print(f"Warning: Could not scan cache: {e}")
            return False
    
    def _download_with_retry(self, repo_id: str, max_retries: int = 3) -> str:
        """
        Download model with retry logic to handle rate limiting.
        
        Args:
            repo_id: Repository ID to download
            max_retries: Maximum number of retry attempts
            
        Returns:
            Path to downloaded model
        """
        for attempt in range(max_retries):
            try:
                print(f"Downloading {repo_id} (attempt {attempt + 1}/{max_retries})...")
                
                model_path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=self.cache_dir,
                    local_files_only=False,
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
                
                print(f"✓ Successfully downloaded {repo_id}")
                return model_path
                
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    wait_time = min(60 * (2 ** attempt), 300)  # Exponential backoff, max 5 minutes
                    print(f"⚠️ Rate limited on attempt {attempt + 1}. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Download error on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(5)  # Brief pause before retry
        
        raise Exception(f"Failed to download {repo_id} after {max_retries} attempts")
    
    def _load_models(self):
        """Load all required models with caching support."""
        print("Loading models...")
        
        # Try to load from cache first
        try:
            # Load tokenizer (usually small and fast)
            print(f"Loading tokenizer: {self.config.model_name}...")
            try:
                # First try loading from local cache
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=True
                )
                print("✓ Loaded tokenizer from cache")
            except Exception:
                # If not in cache, download it
                print("Tokenizer not in cache, downloading...")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=False
                )
            
            # Load SNAC model
            print(f"Loading SNAC audio model: {self.config.snac_model_name}...")
            try:
                # Try local cache first
                self._snac_model = SNAC.from_pretrained(
                    self.config.snac_model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=True
                )
                print("✓ Loaded SNAC model from cache")
            except Exception:
                # Download if not cached
                print("SNAC model not in cache, downloading...")
                self._snac_model = SNAC.from_pretrained(
                    self.config.snac_model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=False
                )
            
            self._audio_tokenizer = AudioTokenizer(self._snac_model)
            
            # Load main model
            print(f"Loading main model: {self.config.model_name}...")
            
            # Check if model is already cached
            if self._check_model_cached(self.config.model_name):
                try:
                    # Load from local cache
                    model_path = snapshot_download(
                        repo_id=self.config.model_name,
                        cache_dir=self.cache_dir,
                        local_files_only=True,
                        allow_patterns=[
                            "config.json",
                            "*.safetensors",
                            "model.safetensors.index.json",
                        ]
                    )
                    print("✓ Using cached model files")
                except Exception as e:
                    print(f"Warning: Cache check passed but loading failed: {e}")
                    # Fall back to downloading
                    model_path = self._download_with_retry(self.config.model_name)
            else:
                # Download model with retry logic
                model_path = self._download_with_retry(self.config.model_name)
            
            # Load the model
            print("Loading text generation model into memory...")
            self._text_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir,
                local_files_only=True  # Use local files since we just downloaded/verified them
            )
            self._text_model.to(self.config.device)
            
            self._initialized = True
            print("✅ All models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            self._initialized = False
            raise
    
    def clone_voice(
        self,
        voice_sample_path: str,
        voice_transcript: str,
        target_texts: Union[str, List[str]],
        output_dir: str = "output_audio",
        **generation_kwargs
    ) -> List[str]:
        """
        Clone a voice to speak new text.
        
        Args:
            voice_sample_path: Path to the voice sample audio file
            voice_transcript: Transcript of what's said in the voice sample
            target_texts: Text(s) to generate with the cloned voice
            output_dir: Directory to save output audio files
            **generation_kwargs: Additional generation parameters
        
        Returns:
            List of paths to generated audio files
        
        Raises:
            FileNotFoundError: If voice sample doesn't exist
            ValueError: If inputs are invalid
        """
        # Ensure models are loaded
        self._ensure_initialized()
        
        # Validate inputs
        if not os.path.exists(voice_sample_path):
            raise FileNotFoundError(f"Voice sample not found: {voice_sample_path}")
        
        if isinstance(target_texts, str):
            target_texts = [target_texts]
        
        if not target_texts:
            raise ValueError("No target texts provided")
        
        # Load and validate audio
        print(f"Loading voice sample: {voice_sample_path}")
        audio_array, sample_rate = load_audio(voice_sample_path, self.config.sample_rate)
        validate_audio_duration(audio_array, sample_rate)
        
        # Tokenize audio
        print("Tokenizing audio...")
        audio_tokens = self._audio_tokenizer.tokenize_audio(audio_array)
        
        # Prepare prompts
        print("Preparing prompts...")
        input_ids, attention_mask = prepare_prompt_tokens(
            self._tokenizer,
            audio_tokens,
            voice_transcript,
            target_texts
        )
        
        # Move to device
        input_ids = input_ids.to(self.config.device)
        attention_mask = attention_mask.to(self.config.device)
        
        # Generate
        print(f"Generating {len(target_texts)} audio samples...")
        generated_audio_paths = self._generate_audio(
            input_ids,
            attention_mask,
            output_dir,
            **generation_kwargs
        )
        
        return generated_audio_paths
    
    def _generate_audio(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_dir: str,
        **generation_kwargs
    ) -> List[str]:
        """
        Generate audio from prepared inputs.
        
        Args:
            input_ids: Prepared input token IDs
            attention_mask: Attention mask
            output_dir: Directory to save outputs
            **generation_kwargs: Generation parameters
        
        Returns:
            List of generated audio file paths
        """
        # Debug information
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Device: {input_ids.device}")
        
        # Prepare generation parameters
        gen_params = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": True,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "num_return_sequences": 1,
            "eos_token_id": 128258,
            "pad_token_id": 128263,  # Explicitly set pad_token_id
        }
        gen_params.update(generation_kwargs)
        
        print(f"Generation parameters: {gen_params}")
        
        # Generate
        try:
            with torch.no_grad():
                generated_ids = self._text_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_params
                )
            
            print(f"Generated IDs shape: {generated_ids.shape}")
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Extract audio tokens
        print("Extracting audio tokens from generated output...")
        audio_token_lists = extract_generated_audio_tokens(generated_ids)
        
        if not audio_token_lists:
            raise ValueError("No audio tokens extracted from generated output")
        
        print(f"Extracted {len(audio_token_lists)} audio token sequences")
        
        # Decode and save audio
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for i, token_list in enumerate(audio_token_lists):
            if not token_list:
                print(f"Warning: Empty token list for sample {i}")
                continue
            
            print(f"Processing audio tokens for sample {i}: {len(token_list)} tokens")
            
            try:
                # Decode audio
                audio_tensor = self._audio_tokenizer.detokenize_audio(token_list)
                
                if audio_tensor is None:
                    print(f"Warning: Failed to decode audio for sample {i}")
                    continue
                
                # Save to file
                output_path = os.path.join(output_dir, f"generated_{i:03d}.wav")
                save_audio_to_wav(audio_tensor, output_path, self.config.sample_rate)
                output_paths.append(output_path)
                
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        if not output_paths:
            raise ValueError("Failed to generate any valid audio outputs")
        
        return output_paths
    
    def process_batch(
        self,
        voice_sample_path: str,
        voice_transcript: str,
        target_texts_batch: List[List[str]],
        output_base_dir: str = "output_audio",
        **generation_kwargs
    ) -> List[List[str]]:
        """
        Process multiple batches of target texts with the same voice.
        
        This is more efficient than calling clone_voice multiple times
        as it reuses the voice encoding.
        
        Args:
            voice_sample_path: Path to voice sample
            voice_transcript: Transcript of voice sample
            target_texts_batch: List of text lists to generate
            output_base_dir: Base directory for outputs
            **generation_kwargs: Generation parameters
        
        Returns:
            List of lists of generated audio paths
        """
        # Ensure models are loaded
        self._ensure_initialized()
        
        # Load and tokenize audio once
        audio_array, _ = load_audio(voice_sample_path, self.config.sample_rate)
        audio_tokens = self._audio_tokenizer.tokenize_audio(audio_array)
        
        all_outputs = []
        
        for batch_idx, target_texts in enumerate(target_texts_batch):
            output_dir = os.path.join(output_base_dir, f"batch_{batch_idx:03d}")
            
            # Prepare prompts for this batch
            input_ids, attention_mask = prepare_prompt_tokens(
                self._tokenizer,
                audio_tokens,
                voice_transcript,
                target_texts
            )
            
            # Generate
            input_ids = input_ids.to(self.config.device)
            attention_mask = attention_mask.to(self.config.device)
            
            output_paths = self._generate_audio(
                input_ids,
                attention_mask,
                output_dir,
                **generation_kwargs
            )
            
            all_outputs.append(output_paths)
        
        return all_outputs
    
    def warmup(self):
        """
        Warmup the model by loading all components.
        
        Useful for server deployments to ensure the model is ready
        before serving requests.
        """
        self._ensure_initialized()
        print("Model warmed up and ready!")
    
    def clear_cache(self):
        """
        Clear the model cache if needed (useful for updates or space management).
        
        Note: This will require re-downloading models on next use.
        """
        if hasattr(self, '_text_model') and self._text_model is not None:
            del self._text_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if hasattr(self, '_snac_model') and self._snac_model is not None:
            del self._snac_model
        
        self._initialized = False
        print("Model cache cleared")