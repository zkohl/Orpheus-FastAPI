"""Audio processing utilities for voice cloning."""

import os
import wave
from typing import Union, Tuple

import numpy as np
import torch
import librosa


def load_audio(
    file_path: str,
    sample_rate: int = 24000
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to the audio file
        sample_rate: Target sample rate (default: 24000)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    
    Raises:
        FileNotFoundError: If the audio file doesn't exist
        ValueError: If the audio file is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        audio_array, sr = librosa.load(file_path, sr=sample_rate)
        return audio_array, sample_rate
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {e}")


def save_audio_to_wav(
    audio: Union[torch.Tensor, np.ndarray],
    output_path: str,
    sample_rate: int = 24000
) -> None:
    """
    Save audio tensor or numpy array as WAV file.
    
    Args:
        audio: 1D audio array in range [-1.0, 1.0]
        output_path: Destination .wav file path
        sample_rate: Sample rate in Hz
    
    Raises:
        ValueError: If input audio is not 1D
        IOError: If writing fails
    """
    # Convert to numpy if input is a torch tensor
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().squeeze().cpu().numpy()
    
    # Validate shape
    if audio.ndim != 1:
        raise ValueError(f"Expected 1D mono audio array, got shape {audio.shape}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Normalize to int16
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    
    # Define WAV parameters
    params = (
        1,  # n_channels
        2,  # sampwidth (int16)
        sample_rate,  # framerate
        len(audio_int16),  # n_frames
        "NONE",  # comptype
        "not compressed"  # compname
    )
    
    # Write to WAV
    try:
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setparams(params)
            wav_file.writeframes(audio_int16.tobytes())
        
        # Log file info
        file_size = os.path.getsize(output_path)
        duration = len(audio_int16) / sample_rate
        
        print(f"✅ Saved: {output_path}")
        print(f"📦 Size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        
        if file_size < 1000:
            print("⚠️  WARNING: Output file is suspiciously small!")
            
    except Exception as e:
        raise IOError(f"Failed to write audio file: {e}")


def validate_audio_duration(
    audio_array: np.ndarray,
    sample_rate: int,
    min_duration: float = 0.1,
    max_duration: float = 300.0
) -> None:
    """
    Validate audio duration is within acceptable bounds.
    
    Args:
        audio_array: Audio samples
        sample_rate: Sample rate in Hz
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
    
    Raises:
        ValueError: If duration is outside bounds
    """
    duration = len(audio_array) / sample_rate
    
    if duration < min_duration:
        raise ValueError(f"Audio too short: {duration:.2f}s < {min_duration}s")
    
    if duration > max_duration:
        raise ValueError(f"Audio too long: {duration:.2f}s > {max_duration}s")