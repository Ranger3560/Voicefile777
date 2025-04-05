"""
Data utilities for speech recognition models.

This module contains utility functions for working with speech data,
including data format conversion, batch processing, and dataset management.
"""

import os
import torch
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import json
import logging
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, DatasetDict, Audio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_audio_format(
    input_file: str,
    output_file: str,
    target_sr: int = 16000,
    target_format: str = "wav"
) -> str:
    """
    Convert audio file to target format and sample rate.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        target_sr: Target sample rate
        target_format: Target audio format
        
    Returns:
        Path to output audio file
    """
    # Load audio
    audio, sr = librosa.load(input_file, sr=None)
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Save to target format
    sf.write(output_file, audio, target_sr)
    
    return output_file


def batch_convert_audio_files(
    input_dir: str,
    output_dir: str,
    target_sr: int = 16000,
    target_format: str = "wav",
    recursive: bool = True,
    extensions: List[str] = ["mp3", "wav", "flac", "m4a", "ogg"]
) -> List[str]:
    """
    Batch convert audio files to target format and sample rate.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        target_sr: Target sample rate
        target_format: Target audio format
        recursive: Whether to search recursively
        extensions: List of audio extensions to process
        
    Returns:
        List of output file paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find audio files
    audio_files = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(f".{ext}") for ext in extensions):
                    audio_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(f".{ext}") for ext in extensions):
                audio_files.append(os.path.join(input_dir, file))
    
    logger.info(f"Found {len(audio_files)} audio files to convert")
    
    # Convert files
    output_files = []
    for input_file in tqdm(audio_files, desc="Converting audio files"):
        # Create output file path
        rel_path = os.path.relpath(input_file, input_dir)
        output_file = os.path.join(output_dir, os.path.splitext(rel_path)[0] + f".{target_format}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert file
        try:
            convert_audio_format(input_file, output_file, target_sr, target_format)
            output_files.append(output_file)
        except Exception as e:
            logger.error(f"Error converting {input_file}: {e}")
    
    logger.info(f"Converted {len(output_files)} audio files")
    
    return output_files


def create_dataset_from_directory(
    audio_dir: str,
    transcript_file: Optional[str] = None,
    transcript_format: str = "csv",
    audio_format: str = "wav",
    target_sr: int = 16000,
    split: Optional[Union[Dict[str, float], str]] = None
) -> Union[Dataset, DatasetDict]:
    """
    Create a dataset from a directory of audio files and optional transcript file.
    
    Args:
        audio_dir: Directory containing audio files
        transcript_file: Path to transcript file (optional)
        transcript_format: Format of transcript file ("csv", "json", "txt")
        audio_format: Audio file format
        target_sr: Target sample rate
        split: Dataset split configuration
        
    Returns:
        Dataset or DatasetDict
    """
    # Load transcripts if provided
    transcripts = {}
    if transcript_file:
        if transcript_format == "csv":
            df = pd.read_csv(transcript_file)
            for _, row in df.iterrows():
                if "file_id" in df.columns and "text" in df.columns:
                    transcripts[row["file_id"]] = row["text"]
                elif "filename" in df.columns and "text" in df.columns:
                    transcripts[row["filename"]] = row["text"]
                elif "path" in df.columns and "text" in df.columns:
                    transcripts[row["path"]] = row["text"]
                elif len(df.columns) >= 2:
                    # Assume first column is file_id and second is text
                    transcripts[row[df.columns[0]]] = row[df.columns[1]]
        elif transcript_format == "json":
            with open(transcript_file, "r") as f:
                transcripts = json.load(f)
        elif transcript_format == "txt":
            with open(transcript_file, "r") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        file_id, text = parts
                        transcripts[file_id] = text
    
    # Find audio files
    audio_files = []
    for file in os.listdir(audio_dir):
        if file.endswith(f".{audio_format}"):
            audio_files.append(os.path.join(audio_dir, file))
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Create dataset
    data = []
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        file_id = os.path.splitext(os.path.basename(audio_file))[0]
        
        # Get transcript if available
        text = transcripts.get(file_id, "")
        
        # Create sample
        sample = {
            "file_id": file_id,
            "audio": audio_file,
            "text": text
        }
        
        data.append(sample)
    
    # Create dataset
    dataset = Dataset.from_dict({
        "file_id": [sample["file_id"] for sample in data],
        "audio": [sample["audio"] for sample in data],
        "text": [sample["text"] for sample in data]
    })
    
    # Cast audio column
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sr))
    
    # Split dataset if requested
    if split:
        if isinstance(split, dict):
            dataset = dataset.train_test_split(**split)
        elif split == "speaker":
            # Try to split by speaker ID if file_id format is like "speaker_id-utterance_id"
            speaker_ids = [file_id.split("-")[0] if "-" in file_id else file_id 
                          for file_id in dataset["file_id"]]
            unique_speakers = list(set(speaker_ids))
            
            # Split speakers into train/test
            np.random.shuffle(unique_speakers)
            train_speakers = set(unique_speakers[:int(len(unique_speakers) * 0.8)])
            
            # Create split indices
            train_indices = [i for i, speaker in enumerate(speaker_ids) if speaker in train_speakers]
            test_indices = [i for i, speaker in enumerate(speaker_ids) if speaker not in train_speakers]
            
            # Split dataset
            train_dataset = dataset.select(train_indices)
            test_dataset = dataset.select(test_indices)
            
            dataset = DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })
    
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    return dataset


def merge_datasets(datasets: List[Dataset], strategy: str = "concat") -> Dataset:
    """
    Merge multiple datasets into one.
    
    Args:
        datasets: List of datasets to merge
        strategy: Merge strategy ("concat" or "interleave")
        
    Returns:
        Merged dataset
    """
    if not datasets:
        raise ValueError("No datasets provided")
    
    if len(datasets) == 1:
        return datasets[0]
    
    if strategy == "concat":
        # Concatenate datasets
        merged_dataset = datasets[0]
        for dataset in datasets[1:]:
            merged_dataset = concatenate_datasets([merged_dataset, dataset])
        
        return merged_dataset
    
    elif strategy == "interleave":
        # Interleave datasets
        from itertools import zip_longest
        
        # Get samples from each dataset
        all_samples = []
        for dataset in datasets:
            all_samples.append([dataset[i] for i in range(len(dataset))])
        
        # Interleave samples
        interleaved_samples = []
        for samples in zip_longest(*all_samples):
            interleaved_samples.extend([s for s in samples if s is not None])
        
        # Create new dataset
        merged_dataset = Dataset.from_dict({
            key: [sample[key] for sample in interleaved_samples if key in sample]
            for key in interleaved_samples[0].keys()
        })
        
        return merged_dataset
    
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")


def filter_dataset_by_duration(
    dataset: Dataset,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    duration_key: str = "audio"
) -> Dataset:
    """
    Filter dataset by audio duration.
    
    Args:
        dataset: Dataset to filter
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        duration_key: Key for audio data
        
    Returns:
        Filtered dataset
    """
    def get_duration(sample):
        if isinstance(sample[duration_key], dict) and "array" in sample[duration_key]:
            return len(sample[duration_key]["array"]) / sample[duration_key]["sampling_rate"]
        elif torch.is_tensor(sample[duration_key]):
            # Assume 16kHz sampling rate if not specified
            return len(sample[duration_key]) / 16000
        else:
            return 0
    
    # Compute durations
    durations = [get_duration(dataset[i]) for i in range(len(dataset))]
    
    # Create filter indices
    indices = []
    for i, duration in enumerate(durations):
        if (min_duration is None or duration >= min_duration) and \
           (max_duration is None or duration <= max_duration):
            indices.append(i)
    
    # Filter dataset
    filtered_dataset = dataset.select(indices)
    
    logger.info(f"Filtered dataset from {len(dataset)} to {len(filtered_dataset)} samples")
    
    return filtered_dataset


def balance_dataset_by_duration(
    dataset: Dataset,
    target_duration: float,
    duration_key: str = "audio",
    strategy: str = "subsample"
) -> Dataset:
    """
    Balance dataset to have approximately equal duration per class.
    
    Args:
        dataset: Dataset to balance
        target_duration: Target duration per class in seconds
        duration_key: Key for audio data
        strategy: Balancing strategy ("subsample" or "oversample")
        
    Returns:
        Balanced dataset
    """
    # Check if dataset has a "label" column
    if "label" not in dataset.column_names:
        logger.warning("Dataset does not have a 'label' column, cannot balance by class")
        return dataset
    
    # Get unique labels
    labels = dataset["label"]
    unique_labels = list(set(labels))
    
    # Compute durations per sample
    def get_duration(sample):
        if isinstance(sample[duration_key], dict) and "array" in sample[duration_key]:
            return len(sample[duration_key]["array"]) / sample[duration_key]["sampling_rate"]
        elif torch.is_tensor(sample[duration_key]):
            # Assume 16kHz sampling rate if not specified
            return len(sample[duration_key]) / 16000
        else:
            return 0
    
    durations = [get_duration(dataset[i]) for i in range(len(dataset))]
    
    # Group samples by label
    label_indices = {label: [] for label in unique_labels}
    label_durations = {label: 0 for label in unique_labels}
    
    for i, (label, duration) in enumerate(zip(labels, durations)):
        label_indices[label].append(i)
        label_durations[label] += duration
    
    # Balance dataset
    balanced_indices = []
    
    if strategy == "subsample":
        # Subsample classes with more than target duration
        for label in unique_labels:
            indices = label_indices[label]
            total_duration = label_durations[label]
            
            if total_duration <= target_duration:
                # Keep all samples
                balanced_indices.extend(indices)
            else:
                # Subsample to target duration
                cumulative_duration = 0
                for i in indices:
                    duration = durations[i]
                    if cumulative_duration + duration <= target_duration:
                        balanced_indices.append(i)
                        cumulative_duration += duration
                    else:
                        # Add with probability proportional to remaining duration
                        remaining = target_duration - cumulative_duration
                        if remaining > 0 and np.random.random() < remaining / duration:
                            balanced_indices.append(i)
                            cumulative_duration += duration
                        
                        if cumulative_duration >= target_duration:
                            break
    
    elif strategy == "oversample":
        # Oversample classes with less than target duration
        for label in unique_labels:
            indices = label_indices[label]
            total_duration = label_durations[label]
            
            if total_duration >= target_duration:
                # Subsample to target duration
                cumulative_duration = 0
                for i in indices:
                    duration = durations[i]
                    if cumulative_duration + duration <= target_duration:
                        balanced_indices.append(i)
                        cumulative_duration += duration
                    else:
                        # Add with probability proportional to remaining duration
                        remaining = target_duration - cumulative_duration
                        if remaining > 0 and np.random.random() < remaining / duration:
                            balanced_indices.append(i)
                            cumulative_duration += duration
                        
                        if cumulative_duration >= target_duration:
                            break
            else:
                # Oversample to target duration
                while total_duration < target_duration:
                    # Add random sample
                    i = np.random.choice(indices)
                    balanced_indices.append(i)
                    total_duration += durations[i]
    
    # Create balanced dataset
    balanced_dataset = dataset.select(balanced_indices)
    
    logger.info(f"Balanced dataset from {len(dataset)} to {len(balanced_dataset)} samples")
    
    return balanced_dataset


def concatenate_datasets(datasets: List[Dataset]) -> Dataset:
    """
    Concatenate multiple datasets into one.
    
    Args:
        datasets: List of datasets to concatenate
        
    Returns:
        Concatenated dataset
    """
    from datasets import concatenate_datasets as hf_concatenate_datasets
    
    return hf_concatenate_datasets(datasets)


def save_dataset_to_disk(
    dataset: Union[Dataset, DatasetDict],
    output_dir: str,
    save_audio: bool = False,
    audio_format: str = "wav"
) -> str:
    """
    Save dataset to disk.
    
    Args:
      
(Content truncated due to size limit. Use line ranges to read in chunks)