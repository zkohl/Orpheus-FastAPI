"""Orpheus Voice Clone - Production-ready voice cloning with Orpheus 3B model."""

from .model import VoiceCloneModel
from .config import ModelConfig

__version__ = "0.1.0"
__all__ = ["VoiceCloneModel", "ModelConfig"]