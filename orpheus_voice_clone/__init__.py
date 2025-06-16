"""Orpheus Voice Clone - Zero-shot voice cloning module."""

from .config import ModelConfig
from .model import VoiceCloneModel
from .tokenizers import AudioTokenizer
from .audio_utils import load_audio, save_audio_to_wav, validate_audio_duration

__all__ = [
    "ModelConfig",
    "VoiceCloneModel",
    "AudioTokenizer",
    "load_audio",
    "save_audio_to_wav",
    "validate_audio_duration"
]

__version__ = "1.0.0"