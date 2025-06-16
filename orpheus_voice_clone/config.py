"""Configuration management for Orpheus Voice Clone."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the Orpheus Voice Clone model."""
    
    # Model settings
    model_name: str = "canopylabs/orpheus-3b-0.1-pretrained"
    snac_model_name: str = "hubertsiuzdak/snac_24khz"
    
    # Audio settings
    sample_rate: int = 24000
    
    # Generation settings (with environment variable defaults)
    max_new_tokens: int = None
    temperature: float = None
    top_p: float = None
    repetition_penalty: float = None
    
    # Token settings
    huggingface_token: Optional[str] = None
    
    # Device settings (auto-detected if not specified)
    device: Optional[str] = None
    
    def __post_init__(self):
        """Initialize configuration after dataclass creation."""
        # Set HuggingFace token from environment if not provided
        if self.huggingface_token is None:
            self.huggingface_token = os.environ.get("HF_TOKEN")
        
        # Set generation parameters from environment if not provided
        if self.max_new_tokens is None:
            self.max_new_tokens = int(os.environ.get("VOICE_CLONE_MAX_NEW_TOKENS", "990"))
        if self.temperature is None:
            self.temperature = float(os.environ.get("VOICE_CLONE_TEMPERATURE", "0.5"))
        if self.top_p is None:
            self.top_p = float(os.environ.get("VOICE_CLONE_TOP_P", "0.9"))
        if self.repetition_penalty is None:
            self.repetition_penalty = float(os.environ.get("VOICE_CLONE_REPETITION_PENALTY", "1.1"))
        
        # Auto-detect device if not specified
        if self.device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        # Set environment variables
        if self.huggingface_token:
            os.environ["HF_TOKEN"] = self.huggingface_token
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"