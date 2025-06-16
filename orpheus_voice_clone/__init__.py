"""Orpheus Voice Clone - Zero-shot voice cloning module."""

from .model import VoiceCloneModel
from .config import ModelConfig
from .audio_utils import load_audio, save_audio_to_wav, validate_audio_duration
from .tokenizers import AudioTokenizer, prepare_prompt_tokens, extract_generated_audio_tokens

__all__ = [
    "VoiceCloneModel",
    "ModelConfig",
    "load_audio",
    "save_audio_to_wav",
    "validate_audio_duration",
    "AudioTokenizer",
    "prepare_prompt_tokens",
    "extract_generated_audio_tokens"
]

__version__ = "1.0.0"