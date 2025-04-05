"""
Data augmentation techniques for speech recognition models.

This module contains functions for augmenting audio data to improve model robustness,
including time stretching, pitch shifting, adding noise, and more.
"""

import numpy as np
import librosa
from typing import Dict, Tuple, Optional, Union, List


def time_stretch(
    audio: np.ndarray,
    rate: float = 1.0
) -> np.ndarray:
    """
    Time stretch the audio by a rate.
    
    Args:
        audio: Audio array
        rate: Stretch factor (1.0 = no stretch, >1.0 = faster, <1.0 = slower)
        
    Returns:
        Time-stretched audio
    """
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(
    audio: np.ndarray,
    sr: int,
    n_steps: float
) -> np.ndarray:
    """
    Shift the pitch of audio by n_steps semitones.
    
    Args:
        audio: Audio array
        sr: Sample rate
        n_steps: Number of semitones to shift (can be fractional)
        
    Returns:
        Pitch-shifted audio
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def add_white_noise(
    audio: np.ndarray,
    noise_level: float = 0.005
) -> np.ndarray:
    """
    Add white noise to the audio.
    
    Args:
        audio: Audio array
        noise_level: Standard deviation of the noise
        
    Returns:
        Noisy audio
    """
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise


def add_background_noise(
    audio: np.ndarray,
    noise_audio: np.ndarray,
    snr_db: float = 10.0
) -> np.ndarray:
    """
    Add background noise to the audio at a specified signal-to-noise ratio.
    
    Args:
        audio: Audio array
        noise_audio: Noise audio array
        snr_db: Signal-to-noise ratio in dB
        
    Returns:
        Audio with background noise
    """
    # Calculate signal and noise power
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise_audio ** 2)
    
    # Calculate scaling factor for noise
    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(signal_power / (noise_power * snr_linear))
    
    # Ensure noise is at least as long as the audio
    if len(noise_audio) < len(audio):
        # Repeat noise to match audio length
        repeats = int(np.ceil(len(audio) / len(noise_audio)))
        noise_audio = np.tile(noise_audio, repeats)
    
    # Trim noise to match audio length
    noise_audio = noise_audio[:len(audio)]
    
    # Add scaled noise to audio
    return audio + scale * noise_audio


def apply_room_impulse_response(
    audio: np.ndarray,
    impulse_response: np.ndarray
) -> np.ndarray:
    """
    Apply a room impulse response to simulate room acoustics.
    
    Args:
        audio: Audio array
        impulse_response: Room impulse response array
        
    Returns:
        Audio with room acoustics
    """
    # Convolve audio with impulse response
    convolved = np.convolve(audio, impulse_response)
    
    # Trim to original length
    return convolved[:len(audio)]


def speed_up(
    audio: np.ndarray,
    factor: float = 1.1
) -> np.ndarray:
    """
    Speed up audio by resampling.
    
    Args:
        audio: Audio array
        factor: Speed factor (>1.0 = faster, <1.0 = slower)
        
    Returns:
        Sped up audio
    """
    # Calculate new length
    new_length = int(len(audio) / factor)
    
    # Resample to new length
    return librosa.resample(audio, orig_sr=1, target_sr=1/factor)


def random_crop(
    audio: np.ndarray,
    crop_length: int
) -> np.ndarray:
    """
    Randomly crop audio to specified length.
    
    Args:
        audio: Audio array
        crop_length: Length of crop in samples
        
    Returns:
        Cropped audio
    """
    if len(audio) <= crop_length:
        return audio
    
    # Random start index
    start = np.random.randint(0, len(audio) - crop_length)
    
    return audio[start:start + crop_length]


def apply_random_augmentations(
    audio: np.ndarray,
    sr: int,
    augmentation_types: List[str] = None,
    p: float = 0.5
) -> np.ndarray:
    """
    Apply random augmentations to audio with probability p.
    
    Args:
        audio: Audio array
        sr: Sample rate
        augmentation_types: List of augmentation types to apply
        p: Probability of applying each augmentation
        
    Returns:
        Augmented audio
    """
    if augmentation_types is None:
        augmentation_types = ['time_stretch', 'pitch_shift', 'white_noise']
    
    augmented_audio = audio.copy()
    
    for aug_type in augmentation_types:
        if np.random.random() < p:
            if aug_type == 'time_stretch':
                rate = np.random.uniform(0.9, 1.1)
                augmented_audio = time_stretch(augmented_audio, rate)
            
            elif aug_type == 'pitch_shift':
                n_steps = np.random.uniform(-2, 2)
                augmented_audio = pitch_shift(augmented_audio, sr, n_steps)
            
            elif aug_type == 'white_noise':
                noise_level = np.random.uniform(0.001, 0.01)
                augmented_audio = add_white_noise(augmented_audio, noise_level)
            
            elif aug_type == 'speed_up':
                factor = np.random.uniform(0.9, 1.1)
                augmented_audio = speed_up(augmented_audio, factor)
    
    return augmented_audio


def create_augmentation_pipeline(
    augmentation_types: List[str] = None,
    p: float = 0.5
) -> callable:
    """
    Create an augmentation pipeline function for dataset mapping.
    
    Args:
        augmentation_types: List of augmentation types to apply
        p: Probability of applying each augmentation
        
    Returns:
        Augmentation function for dataset mapping
    """
    if augmentation_types is None:
        augmentation_types = ['time_stretch', 'pitch_shift', 'white_noise']
    
    def augment_fn(batch):
        audio = batch["audio"]
        sr = batch.get("sampling_rate", 16000)
        
        # Convert to numpy array if it's a tensor
        if hasattr(audio, 'numpy'):
            audio_np = audio.numpy()
        else:
            audio_np = audio
        
        # Apply augmentations
        augmented_audio = apply_random_augmentations(
            audio_np, sr, augmentation_types, p
        )
        
        # Update batch with augmented audio
        if hasattr(audio, 'numpy'):
            # Convert back to tensor of same type
            import torch
            batch["audio"] = torch.tensor(augmented_audio, dtype=audio.dtype)
        else:
            batch["audio"] = augmented_audio
        
        return batch
    
    return augment_fn
