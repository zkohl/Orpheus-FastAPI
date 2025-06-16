"""Main voice cloning model class."""

import os
from typing import List, Optional, Union
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
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
        
        print(f"Initialized VoiceCloneModel on device: {self.config.device}")
    
    def _ensure_initialized(self):
        """Ensure models are loaded (lazy initialization)."""
        if not self._initialized:
            self._load_models()
    
    def _load_models(self):
        """Load all required models."""
        print("Loading models...")
        
        # Load tokenizer
        print("Loading tokenizer...")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        except Exception as e:
            print(f"Failed to load tokenizer with AutoTokenizer: {e}")
            print("Trying alternative loading method...")
            from transformers import LlamaTokenizerFast
            self._tokenizer = LlamaTokenizerFast.from_pretrained(self.config.model_name)
        
        # Load SNAC model
        print("Loading SNAC audio model...")
        self._snac_model = SNAC.from_pretrained(self.config.snac_model_name)
        self._audio_tokenizer = AudioTokenizer(self._snac_model)
        
        # Download model files efficiently
        print("Downloading model files...")
        model_path = self._download_model_files()
        
        # Load text generation model
        print("Loading text generation model...")
        self._text_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
        self._text_model.to(self.config.device)
        
        self._initialized = True
        print("Models loaded successfully!")
    
    def _download_model_files(self) -> str:
        """Download model files efficiently, excluding unnecessary files."""
        return snapshot_download(
            repo_id=self.config.model_name,
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
        # Prepare generation parameters
        gen_params = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": True,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "num_return_sequences": 1,
            "eos_token_id": 128258,
        }
        gen_params.update(generation_kwargs)
        
        # Generate
        with torch.no_grad():
            generated_ids = self._text_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_params
            )
        
        # Extract audio tokens
        audio_token_lists = extract_generated_audio_tokens(generated_ids)
        
        # Decode and save audio
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for i, token_list in enumerate(audio_token_lists):
            if not token_list:
                print(f"Warning: Empty token list for sample {i}")
                continue
            
            # Decode audio
            audio_tensor = self._audio_tokenizer.detokenize_audio(token_list)
            
            # Save to file
            output_path = os.path.join(output_dir, f"generated_{i:03d}.wav")
            save_audio_to_wav(audio_tensor, output_path, self.config.sample_rate)
            output_paths.append(output_path)
        
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