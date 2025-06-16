"""Audio processing utilities for voice cloning."""

import os
from typing import Tuple

import numpy as np
import torch
import librosa
import soundfile as sf


def load_audio(file_path: str, target_sample_rate: int = 24000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        target_sample_rate: Target sample rate (default: 24000 Hz)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # Load audio with librosa (handles various formats)
    audio, sr = librosa.load(file_path, sr=target_sample_rate, mono=True)
    
    # Normalize audio to [-1, 1] range
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio, target_sample_rate


def save_audio_to_wav(audio_tensor: torch.Tensor, output_path: str, sample_rate: int = 24000):
    """
    Save audio tensor to WAV file.
    
    Args:
        audio_tensor: Audio data as torch tensor
        output_path: Path to save WAV file
        sample_rate: Audio sample rate
    """
    # Convert to numpy and ensure proper shape
    if isinstance(audio_tensor, torch.Tensor):
        audio_np = audio_tensor.squeeze().cpu().numpy()
    else:
        audio_np = np.array(audio_tensor).squeeze()
    
    # Ensure 1D array
    if len(audio_np.shape) > 1:
        audio_np = audio_np.squeeze()
    
    # Normalize to prevent clipping
    if np.max(np.abs(audio_np)) > 1.0:
        audio_np = audio_np / np.max(np.abs(audio_np))
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as WAV
    sf.write(output_path, audio_np, sample_rate)
    print(f"Saved audio to: {output_path}")


def validate_audio_duration(audio_array: np.ndarray, sample_rate: int, min_duration: float = 0.5, max_duration: float = 30.0):
    """
    Validate audio duration is within acceptable range.
    
    Args:
        audio_array: Audio data array
        sample_rate: Sample rate
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
    
    Raises:
        ValueError: If audio duration is outside acceptable range
    """
    duration = len(audio_array) / sample_rate
    
    if duration < min_duration:
        raise ValueError(f"Audio too short: {duration:.2f}s (minimum: {min_duration}s)")
    
    if duration > max_duration:
        raise ValueError(f"Audio too long: {duration:.2f}s (maximum: {max_duration}s)")
    
    print(f"Audio duration: {duration:.2f}s (valid)")