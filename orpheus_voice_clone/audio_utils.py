"""Audio processing utilities for voice cloning."""

import os
from typing import Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf


def load_audio(file_path: Union[str, Path], target_sample_rate: int = 24000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        target_sample_rate: Target sample rate (default: 24000)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # Load audio using torchaudio
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    
    # Convert to numpy array
    audio_array = waveform.squeeze().numpy()
    
    return audio_array, target_sample_rate


def save_audio_to_wav(audio_tensor: torch.Tensor, output_path: Union[str, Path], sample_rate: int = 24000):
    """
    Save audio tensor to WAV file.
    
    Args:
        audio_tensor: Audio tensor to save
        output_path: Path to save the audio file
        sample_rate: Sample rate of the audio
    """
    # Ensure the tensor is detached from computation graph and on CPU
    audio_np = audio_tensor.squeeze().detach().cpu().numpy()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save using soundfile
    sf.write(output_path, audio_np, sample_rate)
    print(f"Saved audio to: {output_path}")


def validate_audio_duration(audio_array: np.ndarray, sample_rate: int, min_duration: float = 0.5, max_duration: float = 30.0):
    """
    Validate audio duration is within acceptable range.
    
    Args:
        audio_array: Audio samples
        sample_rate: Sample rate
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
    
    Raises:
        ValueError: If duration is outside acceptable range
    """
    duration = len(audio_array) / sample_rate
    
    if duration < min_duration:
        raise ValueError(f"Audio too short: {duration:.2f}s (minimum: {min_duration}s)")
    
    if duration > max_duration:
        raise ValueError(f"Audio too long: {duration:.2f}s (maximum: {max_duration}s)")
    
    print(f"Audio duration: {duration:.2f}s (valid)")