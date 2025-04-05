"""
Audio preprocessing utilities for speech recognition models.

This module contains functions for preprocessing audio data for speech recognition tasks,
including resampling, normalization, and feature extraction.
"""

import librosa
import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000
) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate (default: 16kHz for Whisper)
        
    Returns:
        Resampled audio array
    """
    if orig_sr != target_sr:
        return librosa.resample(
            y=audio,
            orig_sr=orig_sr,
            target_sr=target_sr
        )
    return audio


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to have zero mean and unit variance.
    
    Args:
        audio: Audio array
        
    Returns:
        Normalized audio array
    """
    if len(audio) == 0:
        return audio
    
    # Normalize to [-1, 1] range
    audio = audio / (np.max(np.abs(audio)) + 1e-10)
    return audio


def pad_or_trim(
    array: np.ndarray,
    length: int,
    pad_value: float = 0.0
) -> np.ndarray:
    """
    Pad or trim an array to the specified length.
    
    Args:
        array: Input array
        length: Target length
        pad_value: Value to use for padding
        
    Returns:
        Padded or trimmed array
    """
    if len(array) == length:
        return array
    
    if len(array) > length:
        return array[:length]
    
    return np.pad(
        array,
        (0, length - len(array)),
        mode='constant',
        constant_values=pad_value
    )


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    fmin: int = 0,
    fmax: Optional[int] = 8000
) -> np.ndarray:
    """
    Extract mel spectrogram features from audio.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate of audio
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Mel spectrogram features
    """
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec


def preprocess_audio_sample(
    audio_sample: Dict,
    target_sr: int = 16000,
    max_length: Optional[int] = None
) -> Dict:
    """
    Preprocess a single audio sample for Whisper model.
    
    Args:
        audio_sample: Dictionary containing audio data and metadata
        target_sr: Target sample rate
        max_length: Maximum length of audio in samples (will pad/trim if specified)
        
    Returns:
        Preprocessed audio sample
    """
    # Extract audio array and original sample rate
    audio_array = audio_sample["audio"]["array"]
    orig_sr = audio_sample["audio"]["sampling_rate"]
    
    # Resample audio to target sample rate
    audio_resampled = resample_audio(audio_array, orig_sr, target_sr)
    
    # Normalize audio
    audio_normalized = normalize_audio(audio_resampled)
    
    # Pad or trim if max_length is specified
    if max_length is not None:
        audio_normalized = pad_or_trim(audio_normalized, max_length)
    
    # Convert to tensor
    audio_tensor = torch.tensor(audio_normalized)
    
    # Update the sample with preprocessed audio
    processed_sample = {
        "audio": audio_tensor,
        "sampling_rate": target_sr,
        "text": audio_sample["text"]
    }
    
    return processed_sample


def create_preprocessing_pipeline(
    target_sr: int = 16000,
    max_length: Optional[int] = None,
    extract_features: bool = False,
    n_mels: int = 80
) -> callable:
    """
    Create a preprocessing pipeline function for dataset mapping.
    
    Args:
        target_sr: Target sample rate
        max_length: Maximum length of audio in samples
        extract_features: Whether to extract mel spectrogram features
        n_mels: Number of mel bands (if extract_features is True)
        
    Returns:
        Preprocessing function for dataset mapping
    """
    def preprocess_fn(batch):
        # Resample audio to 16kHz
        audio_array = resample_audio(
            batch["audio"]["array"],
            batch["audio"]["sampling_rate"],
            target_sr
        )
        
        # Normalize audio
        audio_normalized = normalize_audio(audio_array)
        
        # Pad or trim if max_length is specified
        if max_length is not None:
            audio_normalized = pad_or_trim(audio_normalized, max_length)
        
        # Extract features if requested
        if extract_features:
            features = extract_mel_spectrogram(
                audio_normalized,
                sample_rate=target_sr,
                n_mels=n_mels
            )
            return {
                "audio": torch.tensor(audio_normalized),
                "features": torch.tensor(features),
                "text": batch["text"]
            }
        
        return {
            "audio": torch.tensor(audio_normalized),
            "text": batch["text"]
        }
    
    return preprocess_fn


def is_valid_sample(batch: Dict) -> bool:
    """
    Check if a sample is valid (non-empty audio and text).
    
    Args:
        batch: Audio sample batch
        
    Returns:
        True if sample is valid, False otherwise
    """
    # Check if audio exists and is not empty
    has_audio = (
        "audio" in batch and
        batch["audio"] is not None and
        isinstance(batch["audio"], dict) and
        "array" in batch["audio"] and
        len(batch["audio"]["array"]) > 0
    )
    
    # Check if text exists and is not empty
    has_text = (
        "text" in batch and
        batch["text"] is not None and
        len(batch["text"]) > 0
    )
    
    return has_audio and has_text
